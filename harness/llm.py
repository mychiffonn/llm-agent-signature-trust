"""Model backends that return structured JSON, with a shared on-disk cache.

  APIBackend  Anthropic Python SDK (needs ANTHROPIC_API_KEY or an `ant auth login` profile)
  CLIBackend  the `claude` CLI in headless mode (uses your Claude Code login)
  OpenRouterBackend  any OpenRouter model with structured outputs (default deepseek/deepseek-v4.1-flash).
              Key from OPENROUTER_API_KEY, or the file named by OPENROUTER_KEY_FILE (default ~/openrouter_key.txt).
              Supports temperature and a per-call seed, so seeds are real independent samples.
  DeepSeekBackend, GeminiBackend, HFBackend  the same chat-completions client against DeepSeek's API, Google's
              OpenAI-compatible endpoint and the Hugging Face Inference Providers router (HF_TOKEN).

Both have one method, json_call(system, user, schema, sample). It returns
{"output": dict | None, "tool_log": [], "stop_reason": str, "cost_usd": float | None}.
Tool use (experiment P3) is handled by the simulation as a JSON round-trip, so it
behaves the same on both backends.
Every call is cached in SQLite, keyed by backend, model, effort, prompts, schema and
sample index. A rerun costs nothing.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import tempfile
from dataclasses import dataclass

DEFAULT_MODEL = os.environ.get("EPI_MODEL", "claude-opus-5-5")
DEFAULT_EFFORT = os.environ.get("EPI_EFFORT", "low")
CACHE_PATH = os.environ.get("EPI_CACHE", os.path.join(os.path.dirname(__file__), "..", "cache.sqlite"))
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    refusals: int = 0
    errors: int = 0


class Cache:
    def __init__(self, path: str = CACHE_PATH):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, k: str):
        row = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v) -> None:
        self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
        self.db.commit()


class _Base:
    name = "base"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, concurrency: int = 8,
                 temperature: float | None = None):
        self.model, self.effort, self.temperature = model, effort, temperature
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = Cache()
        self.usage = Usage()

    async def json_call(self, system: str, user: str, schema: dict, sample: int = 0) -> dict:
        key = hashlib.sha256(json.dumps(
            dict(b=self.name, m=self.model, e=self.effort, s=system, u=user, sc=schema, n=sample,
                 **({"t": self.temperature} if self.temperature is not None else {})),
            sort_keys=True).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            self.usage.cached += 1
            return hit
        async with self.sem:
            res = await self._call(system, user, schema, sample)
        res.setdefault("tool_log", [])
        if res["stop_reason"] == "refusal":
            self.usage.refusals += 1
        if res["output"] is None and res["stop_reason"] != "refusal":
            self.usage.errors += 1
            return res  # don't cache failures
        self.usage.calls += 1
        self.cache.put(key, res)
        return res

    async def preflight(self) -> None:
        pass

    async def _call(self, system: str, user: str, schema: dict, sample: int = 0) -> dict:
        raise NotImplementedError


class APIBackend(_Base):
    name = "api"

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.AsyncAnthropic(max_retries=6)

    async def preflight(self) -> None:
        a = self.anthropic
        try:
            await self.client.models.retrieve(self.model)
        except TypeError as e:
            raise SystemExit(f"No Anthropic credentials found ({e}). Set ANTHROPIC_API_KEY, or use --backend cli.")
        except a.AuthenticationError:
            raise SystemExit("Anthropic credentials were rejected (401).")
        except a.NotFoundError:
            raise SystemExit(f"Model {self.model!r} not found for this account.")

    async def _call(self, system, user, schema, sample=0) -> dict:
        a = self.anthropic
        params = dict(model=self.model, max_tokens=4000, system=system,
                      messages=[{"role": "user", "content": user}],
                      output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}})
        if self.model in FALLBACK_MODELS:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        try:
            resp = await self.client.beta.messages.create(**params)
        except a.APIStatusError as e:
            return {"output": None, "stop_reason": f"api_error_{e.status_code}", "cost_usd": None}
        except a.APIConnectionError:
            return {"output": None, "stop_reason": "connection_error", "cost_usd": None}
        self.usage.input_tokens += resp.usage.input_tokens
        self.usage.output_tokens += resp.usage.output_tokens
        out = None
        if resp.stop_reason != "refusal":
            text = next((b.text for b in resp.content if b.type == "text"), None)
            try:
                out = json.loads(text) if text else None
            except json.JSONDecodeError:
                out = None
        return {"output": out, "stop_reason": resp.stop_reason, "cost_usd": None}


class CLIBackend(_Base):
    """Runs `claude -p` with Claude Code's own tools, MCP servers and settings turned off.
    Your Claude Code login is used. Claude Code still adds roughly 1k tokens of its own
    context (an environment block and account info). That is the same in every
    condition. The API backend adds none of it."""
    name = "cli"

    def __init__(self, *a, timeout: float = 300, retries: int = 2, **kw):
        super().__init__(*a, **kw)
        self.timeout, self.retries = timeout, retries
        self.cwd = tempfile.mkdtemp(prefix="epi_cli_")  # empty dir: no CLAUDE.md, no project settings

    def _args(self, system: str, schema: dict) -> list[str]:
        return ["claude", "-p", "--model", self.model, "--effort", self.effort,
                "--output-format", "json", "--json-schema", json.dumps(schema),
                "--system-prompt", system, "--tools", "", "--strict-mcp-config",
                "--setting-sources", "project", "--no-session-persistence"]

    async def preflight(self) -> None:
        r = await self._call("Reply with ok=true.", "ping",
                             {"type": "object", "properties": {"ok": {"type": "boolean"}},
                              "required": ["ok"], "additionalProperties": False})
        if r["output"] is None:
            raise SystemExit(f"claude CLI preflight failed: {r['stop_reason']}")

    async def _call(self, system, user, schema, sample=0) -> dict:
        last = "unknown"
        for _ in range(self.retries + 1):
            try:
                proc = await asyncio.create_subprocess_exec(
                    *self._args(system, schema), cwd=self.cwd,
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                out, err = await asyncio.wait_for(proc.communicate(user.encode()), self.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                last = "timeout"
                continue
            except FileNotFoundError:
                raise SystemExit("`claude` CLI not found on PATH.")
            try:
                d = json.loads(out)
            except json.JSONDecodeError:
                last = f"bad_cli_output: {(err or out)[:200]!r}"
                continue
            u = d.get("usage") or {}
            self.usage.input_tokens += (u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0)
                                        + u.get("cache_creation_input_tokens", 0))
            self.usage.output_tokens += u.get("output_tokens", 0)
            self.usage.cost_usd += d.get("total_cost_usd") or 0.0
            if d.get("is_error"):
                last = f"cli_error: {d.get('subtype')} {d.get('api_error_status')}"
                if d.get("stop_reason") == "refusal":
                    return {"output": None, "stop_reason": "refusal", "cost_usd": d.get("total_cost_usd")}
                continue
            return {"output": d.get("structured_output"), "stop_reason": d.get("stop_reason"),
                    "cost_usd": d.get("total_cost_usd")}
        return {"output": None, "stop_reason": last, "cost_usd": None}


def _dotenv(name: str) -> str | None:
    """Read NAME from the environment, else from the repo's .env file (KEY=value lines)."""
    if os.environ.get(name):
        return os.environ[name]
    path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(path):
        for line in open(path):
            k, _, v = line.strip().partition("=")
            if k.strip() == name and v.strip():
                return v.strip().strip('"').strip("'")
    return None


class OpenRouterBackend(_Base):
    """OpenAI-compatible chat completions (OpenRouter by default; subclasses set other providers).
    temperature defaults to 0.7 and, where the provider supports it, the per-call `seed` is the sample index,
    so different seeds are independent samples and a rerun hits the cache."""
    name = "openrouter"
    URL = "https://openrouter.ai/api/v1/chat/completions"
    KEY_ENV = "OPENROUTER_API_KEY"
    SEED = True          # provider accepts a seed
    STRICT_SCHEMA = True  # provider accepts response_format json_schema (else json_object + schema in the prompt)

    def _key(self) -> str | None:
        key = _dotenv(self.KEY_ENV)
        if not key:
            path = os.path.expanduser(os.environ.get("OPENROUTER_KEY_FILE", "~/openrouter_key.txt"))
            key = open(path).read().strip() if os.path.exists(path) else None
        return key

    def _body(self, system, user, schema, sample) -> dict:
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": 6000,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}},
                "usage": {"include": True}, "provider": {"require_parameters": True}, "seed": int(sample)}
        if self.effort in ("none", "off"):
            body["reasoning"] = {"enabled": False}
        elif self.effort:
            body["reasoning"] = {"effort": self.effort}
        return body

    def _cost(self, usage: dict, provider: str) -> float | None:
        """Cost when the provider does not report it (None = unknown)."""
        return None

    def __init__(self, *a, timeout: float = 180, retries: int = 4, **kw):
        super().__init__(*a, **kw)
        if self.temperature is None:
            self.temperature = 0.7
        import httpx
        key = self._key()
        if not key:
            raise SystemExit(f"No API key for {self.name}: set {self.KEY_ENV} (environment or .env).")
        self.client = httpx.AsyncClient(timeout=timeout, headers={"Authorization": f"Bearer {key}"})
        self.retries = retries
        self.providers: dict[str, int] = {}

    async def preflight(self) -> None:
        r = await self._call("Reply with ok=true.", "ping",
                             {"type": "object", "properties": {"ok": {"type": "boolean"}},
                              "required": ["ok"], "additionalProperties": False})
        if r["output"] is None:
            raise SystemExit(f"{self.name} preflight failed: {r['stop_reason']}")

    async def _call(self, system, user, schema, sample=0) -> dict:
        body = self._body(system, user, schema, sample)
        last = "unknown"
        for attempt in range(self.retries + 1):
            try:
                r = await self.client.post(self.URL, json=body)
            except Exception as e:  # network errors: back off and retry
                last = f"connection_error {type(e).__name__}"
                await asyncio.sleep(2 * (attempt + 1))
                continue
            if r.status_code in (408, 429, 500, 502, 503, 504):
                last = f"api_error_{r.status_code}"
                await asyncio.sleep(3 * (attempt + 1))
                continue
            try:
                d = r.json()
            except ValueError:
                last = f"bad_json_{r.status_code}"
                continue
            if "error" in d or not d.get("choices"):
                last = f"api_error_{(d.get('error') or {}).get('code', r.status_code)}"
                await asyncio.sleep(2 * (attempt + 1))
                continue
            u = d.get("usage") or {}
            self.usage.input_tokens += u.get("prompt_tokens", 0)
            self.usage.output_tokens += u.get("completion_tokens", 0)
            prov = d.get("provider") or r.headers.get("x-inference-provider") or "?"
            cost = u.get("cost") if u.get("cost") is not None else self._cost(u, prov)
            self.usage.cost_usd += cost or 0.0
            self.providers[prov] = self.providers.get(prov, 0) + 1
            ch = d["choices"][0]
            text = (ch.get("message") or {}).get("content") or ""
            try:
                out = json.loads(text)
            except json.JSONDecodeError:
                # some providers wrap JSON in prose or code fences: take the outermost object
                i, j = text.find("{"), text.rfind("}")
                try:
                    out = json.loads(text[i:j + 1]) if i >= 0 < j else None
                except json.JSONDecodeError:
                    out = None
            if out is None or not isinstance(out, dict) or (
                    not self.STRICT_SCHEMA and not set(schema.get("required", [])) <= set(out)):
                last = f"unparsable_output ({ch.get('finish_reason')})"
                continue
            stop = "refusal" if ch.get("finish_reason") == "content_filter" else ch.get("finish_reason")
            return {"output": out, "stop_reason": stop, "cost_usd": cost, "provider": prov}
        return {"output": None, "stop_reason": last, "cost_usd": None}


class DeepSeekBackend(OpenRouterBackend):
    """DeepSeek's own API (DEEPSEEK_API_KEY). Same weights as deepseek/deepseek-v4.1-flash on OpenRouter, but it only
    offers JSON mode, so the schema goes in the system prompt and required keys are checked on return."""
    name = "deepseek"
    URL = "https://api.deepseek.com/chat/completions"
    KEY_ENV = "DEEPSEEK_API_KEY"
    STRICT_SCHEMA = False

    def _key(self):
        return _dotenv(self.KEY_ENV)

    def _body(self, system, user, schema, sample) -> dict:
        sys2 = system + "\n\nReply with a single JSON object that matches this JSON schema:\n" + json.dumps(schema)
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": 6000, "seed": int(sample),
                "messages": [{"role": "system", "content": sys2}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}}
        if self.effort and self.effort not in ("none", "off"):
            body["reasoning_effort"] = self.effort
        return body


class GeminiBackend(OpenRouterBackend):
    """Google's OpenAI-compatible endpoint (GEMINI_API_KEY). Strict JSON schema; no seed parameter, so repeated
    seeds differ only through sampling (the cache key still includes the sample index)."""
    name = "gemini"
    URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    KEY_ENV = "GEMINI_API_KEY"
    SEED = False

    def _key(self):
        return _dotenv(self.KEY_ENV)

    def _body(self, system, user, schema, sample) -> dict:
        body = {"model": self.model, "temperature": self.temperature, "max_tokens": 6000,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}}}
        if self.effort and self.effort not in ("none", "off"):
            body["reasoning_effort"] = self.effort
        return body


class HFBackend(OpenRouterBackend):
    """Hugging Face Inference Providers router (HF_TOKEN), OpenAI-compatible, billed to the HF account.
    Preflight reads the router's model catalogue and pins the first live provider that supports JSON-schema output
    (or the one given as "model:provider"), so every call in a run goes to the same provider. Models without
    structured output get the schema in the system prompt and their reply is parsed as JSON. The router doesn't return cost, so
    it is computed from the catalogue's per-token prices for the pinned provider."""
    name = "hf"
    URL = "https://router.huggingface.co/v1/chat/completions"
    KEY_ENV = "HF_TOKEN"

    def _key(self):
        return _dotenv(self.KEY_ENV)

    def _body(self, system, user, schema, sample) -> dict:
        model = f"{self.model}:{self.provider}" if getattr(self, "provider", None) else self.model
        body = {"model": model, "temperature": self.temperature, "max_tokens": 6000, "seed": int(sample),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.STRICT_SCHEMA:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "strict": True, "schema": schema}}
        else:
            body["messages"][0]["content"] = (system + "\n\nReply with a single JSON object that matches this JSON "
                                              "schema:\n" + json.dumps(schema))  # no response_format: some
            # providers reject json_object too; the reply is parsed and its required keys checked
        if self.effort and self.effort not in ("none", "off"):
            body["reasoning_effort"] = self.effort
        return body

    def _cost(self, usage, provider):
        p = (getattr(self, "pricing", None) or {})
        if not p or p.get("input") is None:
            return None
        return (usage.get("prompt_tokens", 0) * p["input"] + usage.get("completion_tokens", 0) * p.get("output", 0)) / 1e6

    async def preflight(self) -> None:
        if ":" in self.model:
            self.model, want = self.model.split(":", 1)
        else:
            want = None
        r = await self.client.get("https://router.huggingface.co/v1/models")
        cat = {m["id"]: m for m in r.json().get("data", [])}
        if self.model not in cat:
            raise SystemExit(f"{self.model} is not served by the Hugging Face router")
        live = [p for p in cat[self.model].get("providers", []) if p.get("status") == "live"]
        if want:
            live = [p for p in live if p["provider"] == want]
        if not live:
            raise SystemExit(f"no live provider for {self.model}" + (f" named {want}" if want else ""))
        pick = next((p for p in live if p.get("supports_structured_output")), live[0])
        self.provider, self.pricing = pick["provider"], pick.get("pricing") or {}
        self.STRICT_SCHEMA = bool(pick.get("supports_structured_output"))
        self.model_tag = f"{self.model}:{self.provider}"
        await super().preflight()


DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4.1-flash"
DEFAULT_MODELS = {"openrouter": DEFAULT_OPENROUTER_MODEL, "deepseek": "deepseek-flash", "gemini": "gemini-3.8-flash",
                  "hf": "google/gemma-4-31B-it"}


def make_backend(kind: str, model: str | None = None, effort: str | None = None, concurrency: int = 8,
                 temperature: float | None = None):
    cls = {"api": APIBackend, "cli": CLIBackend, "openrouter": OpenRouterBackend, "deepseek": DeepSeekBackend,
           "gemini": GeminiBackend, "hf": HFBackend}[kind]
    default = DEFAULT_MODELS.get(kind, DEFAULT_MODEL)
    return cls(model=model or default, effort=effort or DEFAULT_EFFORT, concurrency=concurrency,
               temperature=temperature)
