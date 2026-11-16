"""What the agents wrote: selected quotes and simple text codings of their messages.

Both summary.html (report.py) and the paper exporter (paper.py) read from here, so every quote and rate on either
surface comes from the results files. Quotes are selected by condition and seed, never typed in by hand.
Only confirmatory cases (seed >= 2000) are used.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONFIRM_SEED0 = 2000
MODEL_OF_TAG = {"gemini-flash": "gemini-3.8-flash", "gemini-flash-lite": "gemini-3.5-flash-lite",
                "deepseek": "deepseek-flash", "haiku": "claude-haiku-5-5"}

INVALID_WORDS = re.compile(r"\binvalid\b", re.I)
UNSIGNED_WORDS = re.compile(r"\bunsigned\b|\bnot signed\b|\bno signatures?\b|none (of which )?are signed", re.I)


def packet_rows():
    """{model id: [rows]} for the confirmatory single-agent runs."""
    out = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "packet_conf_*.jsonl"))):
        tag = os.path.basename(f)[len("packet_conf_"):].rsplit("_", 1)[0]
        for line in open(f):
            r = json.loads(line)
            if r["config"]["seed"] >= CONFIRM_SEED0 and r.get("message"):
                out[MODEL_OF_TAG.get(tag, tag)].append(r)
    for m, rows in out.items():  # the same condition can appear in two files: keep one
        out[m] = list({json.dumps(r["config"], sort_keys=True): r for r in rows}.values())
    return out


def network_rows(prefix="net_floorfix_"):
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "results", prefix + "*.jsonl"))):
        meta_f = f.rsplit(".", 1)[0] + ".meta.json"
        model = json.load(open(meta_f)).get("model") if os.path.exists(meta_f) else os.path.basename(f)
        out[model] = [json.loads(line) for line in open(f)]
    return out


def _match(c, want):
    return all(c.get(k, False if k == "attribution_note" else None) == v for k, v in want.items())


# id, model, condition (config fields), what it illustrates
QUOTES = [
    ("verified", "gemini-3.8-flash",
     dict(experiment="P2", packet="none", signing="false", state_reliability=False, attribution_note=False, seed=2000),
     "Only the two reports for the wrong answer are signed. Gemini treats them as decisive."),
    ("cantverify", "claude-haiku-5-5",
     dict(experiment="P2", packet="none", signing="false", state_reliability=False, attribution_note=False, seed=2001),
     "Same condition, Haiku: unsigned is read as unverifiable."),
    ("attrignored", "claude-haiku-5-5",
     dict(experiment="P2", packet="none", signing="false", state_reliability=False, attribution_note=True, seed=2001),
     "Haiku has been told every report is correctly attributed, and still discounts the unsigned ones."),
    ("attrused", "gemini-3.8-flash",
     dict(experiment="P2", packet="none", signing="false", state_reliability=True, attribution_note=True, seed=2000),
     "With the same sentence, Gemini goes back to counting reports."),
    ("puppets", "gemini-3.8-flash",
     dict(experiment="P2", packet="sock3", signing="puppets", state_reliability=False, attribution_note=False,
          seed=2002),
     "Only the three fake witnesses sign. Their signatures become the argument."),
    ("forged", "deepseek-flash",
     dict(experiment="P2", packet="forged", signing="all", state_reliability=True, attribution_note=False, seed=2000),
     "A report forged in a real witness's name shows as INVALID and is dropped."),
    ("forgedunsigned", "gemini-3.8-flash",
     dict(experiment="P2U", packet="forged", state_reliability=False, seed=2000),
     "Without signatures the same forgery counts as a sixth witness."),
    ("registry", "deepseek-flash",
     dict(experiment="P3", packet="none", signing="all", state_reliability=False, registry_coverage=0.6, seed=2000),
     "No attack, but the registry misses two honest witnesses. The check tool removes them and the answer flips."),
    ("nothing", "claude-haiku-5-5",
     dict(experiment="P3", packet="none", signing="none", state_reliability=False, seed=2002),
     "No attack, nobody signs. The check finds zero accredited witnesses."),
    ("captured", "claude-haiku-5-5",
     dict(experiment="P3", packet="captured", signing="all", state_reliability=False, registry_coverage=1.0, seed=2000),
     "A real witness lies with its real key. Every check passes."),
]
NET_QUOTE = ("network", "gemini-3.8-flash", dict(experiment="P2", adversary="sockpuppet", signing="puppets", seed=2000),
             "agent_32", "Network, only the two fake agents sign. A downstream agent with no evidence of its own.")


def quotes(pk=None, net=None):
    """[(id, model, text, P(truth), note)]. Raises if a selected trial is missing, so a quote can't silently vanish."""
    pk = pk if pk is not None else packet_rows()
    out = []
    for qid, model, want, note in QUOTES:
        hits = [r for r in pk.get(model, []) if _match(r["config"], want)]
        if len(hits) != 1:
            raise ValueError(f"quote {qid}: {len(hits)} matching trials")
        out.append((qid, model, hits[0]["message"].strip(), hits[0]["p_truth"], note))
    net = net if net is not None else network_rows()
    qid, model, want, agent, note = NET_QUOTE
    for r in net.get(model, []):
        if _match(r["config"], want):
            last = [e for e in r["log"] if e["agent"] == agent and e["round"] == r["config"]["rounds"] - 1]
            if last:
                p = last[0]["belief_A"] if r["scenario"]["truth"] == "A" else 1 - last[0]["belief_A"]
                out.append((qid, model, last[0]["message"].strip(), p, note))
    return out


def _rate(rows, rx):
    rows = list(rows)
    return (sum(bool(rx.search(r["message"] or "")) for r in rows) / len(rows), len(rows)) if rows else (None, 0)


def codings(pk=None, net=None):
    """Simple keyword codings of the messages, per model: {name: {model: (rate, n)}}."""
    pk = pk if pk is not None else packet_rows()
    net = net if net is not None else network_rows()
    out = defaultdict(dict)
    for m, rows in pk.items():
        one_side = [r for r in rows if r["config"]["experiment"] == "P2" and r["config"]["signing"] in ("truth", "false")]
        # share of messages that cite missing signatures, when only one side signs, without and with the sentence
        # saying every report is correctly attributed (after which a missing signature is no reason to discount)
        out["unsigned"][m] = _rate((r for r in one_side if not r["config"].get("attribution_note")), UNSIGNED_WORDS)
        out["unsignedattr"][m] = _rate((r for r in one_side if r["config"].get("attribution_note")), UNSIGNED_WORDS)
        forged = [r for r in rows if r["config"]["experiment"] == "P2" and r["config"]["packet"] == "forged"
                  and r["config"]["signing"] == "all" and not r["config"].get("attribution_note")]
        out["invalidmention"][m] = _rate(forged, INVALID_WORDS)
        p3 = [r for r in rows if r["config"]["experiment"] == "P3"]
        out["checkuse"][m] = ((sum(bool(r["tool_log"]) for r in p3) / len(p3)), len(p3)) if p3 else (None, 0)
    for m, runs in net.items():
        for sig in ("all", "puppets"):
            msgs = []
            for r in runs:
                c = r["config"]
                if c["experiment"] == "P2" and c["adversary"] == "sockpuppet" and c["signing"] == sig:
                    skip = set(r.get("observers", {})) | set(r.get("puppets") or [])
                    msgs += [e for e in r["log"] if e["round"] == c["rounds"] - 1 and e["agent"] not in skip]
            out[f"netunsigned_{sig}"][m] = _rate(msgs, UNSIGNED_WORDS)
    return dict(out)
