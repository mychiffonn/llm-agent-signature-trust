"""Exp A: the evidence packet. One agent, one fixed inbox, no network.

Why this exists (design v3): in the network (Exp B), the provenance format changes two things at once:
how much evidence reaches each agent (transmission) and how much weight an agent gives a report it has
seen (credulity). The v2 pilot's "signatures backfire" result could be either. Here transmission is held
fixed by construction: every format shows the agent exactly the same reports with the same wording,
from the same relayers, in the same order. Only the provenance metadata differs. So any difference
between formats is a difference in how the agent weighs a report.

Packet (what is in the inbox), always on top of 5 accredited reports, 3 for the truth and 2 against:
  none      control
  sock1/2/3 k sockpuppet reports for the false answer, each from a distinct unregistered origin,
            validly signed under the puppet's own key (dose-response)
  echo3     ONE sockpuppet report passed on by 3 different relayers (repetition, not independence)
  forged    ONE false report attributed to a sixth accredited observer who sent nothing; its signature does not
            verify (the positive control: the attack signatures exist to stop). A counter sees a 3-vs-3 tie.
  captured  one accredited observer whose report was correct instead signs a fabricated report for the
            false answer with its real, registered key (2 true vs 3 false, every check passes)

Signing pattern (which origins sign their records; only visible in the signed formats P2, P2R, P3):
  all       every origin signs (design v3: the signature is constant across the inbox, so it cannot change
            how one report is weighed against another)
  none      nobody signs
  truth     only reports for the true answer are signed     } the clean contrast: same reports, the signature
  false     only reports for the false answer are signed    } moves sides. truth - false = signature premium
  puppets   only the attacker signs (honest sources have not adopted signing; signing is free for a puppet)
  random    each origin signs with probability 1/2 (for a regression of weight on signature status)
  The ground truth never depends on who signs, so a careful agent gives a signature no weight.

attribution_note: tells agents every record's origin is accurately attributed, signed or not. That removes the one
legitimate reason to discount an unsigned record (its attribution rests on the relayer's word).

Display (badge): False shows a truncated hex signature (v3); True shows the platform's verification result as text,
e.g. "valid: signed with Lena Fischer's key" or "none: this record is not signed", as deployed systems do.

Realism: "template" uses one fixed line per report and opaque names (agent_37). "natural" rewrites every report in
one of five equally diagnostic phrasings per domain, gives sources person names, and varies the relayers' lines.
The case (who saw what, lot ids, order) is identical across realism levels; fakes draw from the same phrasing pool.

Formats are the network arms: P0, P2U, P2, P2R, P3 (see network.py). P3 agents can call verify_record
and count_independent_origins, as in the network. registry_coverage < 1 drops some genuine observers
from the registry, so the P3 checks also flag honest reports (the cost of an incomplete registry).

Benchmarks per packet (Bayes posterior with the stated reliability p):
  registry   count distinct REGISTERED origins only (what perfect use of the checks gives)
  naive      count distinct claimed first-hand origins (what a careful agent without checks can do)
  msgcount   count every message (what an agent that ignores origins does)
  verified   count origins that are registered AND signed (what P3's checks certify under partial signing)

  python -m harness.packet --mock --seeds 3
  python -m harness.packet --backend openrouter --seeds 40 --out results/packet_v3.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass

from . import provenance as pv
from .metrics import brier_index
from .network import DESIGN_VERSION, RECORDS, SIGNED, Config, Message, output_schema, system_prompt, user_prompt
from .scenarios import DOMAINS, PERSON_NAMES, RELAY_LINES, Signal, _evidence, make_scenario, naturalize

PACKETS = ["none", "sock1", "sock2", "sock3", "echo3", "captured", "forged"]
FORMATS = ["P0", "P2U", "P2", "P2R", "P3"]
SIGNINGS = ["all", "none", "truth", "false", "puppets", "random"]
REALISMS = ["template", "natural"]
# Removes the integrity reason to discount an unsigned record (an unsigned attribution rests on the relayer's word),
# without saying anything about whether signers are more accurate. With it, a signature premium is over-reading.
ATTRIBUTION_NOTE = ("Relayers pass reports on unchanged, and every record's origin is accurately attributed whether or "
                    "not it is signed.")
from .network import MIXED_NOTE  # noqa: E402  (same wording in the network)


@dataclass
class PacketConfig:
    experiment: str = "P2"
    packet: str = "none"
    registry_coverage: float = 1.0
    seed: int = 0
    p: float = 0.75
    state_reliability: bool = True
    signing: str = "all"
    badge: bool = False
    realism: str = "template"
    attribution_note: bool = False


class Packet:
    def __init__(self, pc: PacketConfig):
        self.pc = pc
        self.sc = sc = make_scenario(pc.seed, n_observers=5, p=pc.p, n_correct=3)
        d = DOMAINS[sc.domain]
        rng = random.Random(f"{pc.seed}|packet")  # independent of format and packet: same case in every cell
        labels = rng.sample(range(10, 100), 5 + 3 + 8 + 1)
        names = [f"agent_{x}" for x in labels]
        nat = random.Random(f"{pc.seed}|natural")  # own stream: the case above is the same at every realism level
        if pc.realism == "natural":
            names = nat.sample(PERSON_NAMES, len(names))
        self.me = names[-1]
        observers, puppets, relayers = names[:5], names[5:8], names[8:16]  # one relayer per message (max 8)
        self.observers, self.puppets, self.captured = observers, puppets, None
        self.ks = pv.KeyStore(seed=f"{pc.seed}|packet")
        fakes = [_evidence(rng, d["evidence"], sc.names[sc.false_answer], sc.used_lots) for _ in puppets]
        right = [o for o, s in zip(observers, sc.signals) if s.points_to == sc.truth]
        captured = rng.choice(right)
        if pc.packet == "captured":
            self.captured = captured
        capture_text = _evidence(rng, d["evidence"], sc.names[sc.false_answer], sc.used_lots)
        honest_order = rng.sample(range(5), 5)
        insert_at = [rng.randint(0, 5 + i) for i in range(3)]
        n_reg = round(pc.registry_coverage * 5)
        self.registry = set(random.Random(f"{pc.seed}|registry").sample(observers, n_reg))

        # reports: (origin, Signal); messages: (relayer, record)
        reports = []
        for i in honest_order:
            o, s = observers[i], sc.signals[i]
            if pc.packet == "captured" and o == captured:
                s = Signal(o, sc.false_answer, capture_text)
            reports.append((o, s))
        k = {"sock1": 1, "sock2": 2, "sock3": 3, "echo3": 1}.get(pc.packet, 0)
        fake_reports = [(puppets[j], Signal(puppets[j], sc.false_answer, fakes[j])) for j in range(k)]
        if pc.packet == "echo3":
            fake_reports = fake_reports * 3
        self.forged = None
        if pc.packet == "forged":  # own rng stream: the case is unchanged
            fr = random.Random(f"{pc.seed}|forged")
            pool = PERSON_NAMES if pc.realism == "natural" else [f"agent_{x}" for x in range(10, 100)]
            self.forged = fr.choice([x for x in pool if x not in names])
            fake_reports = [(self.forged, Signal(self.forged, sc.false_answer, fakes[0]))]
        seq = list(reports)
        for j, fr in enumerate(fake_reports):
            seq.insert(min(insert_at[j], len(seq)), fr)
        if pc.realism == "natural":
            texts = {}
            for o, s in seq:  # one rewrite per origin, so an echoed report keeps its wording
                if o not in texts:
                    texts[o] = naturalize(nat, sc.domain, s.text, sc.names[s.points_to])
            seq = [(o, Signal(o, s.points_to, texts[o])) for o, s in seq]
        if self.forged:
            self.registry.add(self.forged)  # the impersonated observer IS accredited; only the signature gives it away
        self.signers = self._signers(seq) - {self.forged}
        self.records, self.inbox = {}, []
        for n, (o, s) in enumerate(seq):
            claim = {"points_to": s.points_to, "text": s.text}
            if o == self.forged:
                rec = pv.forge(claim, o)  # claims the accredited origin, signature does not verify
            else:
                rec = pv.originate(self.ks, o, claim) if o in self.signers else pv.unsigned(claim, o)
            relayer = relayers[n]
            r = pv.relay(self.ks, rec, relayer)
            self.records[r.record_id] = r
            if pc.realism == "natural":
                text = nat.choice(RELAY_LINES).format(o=o, text=s.text)
            else:
                text = f"Passing on a first-hand report from {o}: {s.text}."
            self.inbox.append(Message(f"m{n+1}", relayer, 0, text, records=[r]))
        self.seq = seq
        self.cfg = Config(experiment=pc.experiment, adversary="none", rounds=2, seed=pc.seed, p=pc.p,
                          state_reliability=pc.state_reliability)
        if pc.experiment not in RECORDS:
            for m in self.inbox:
                m.records = []

    def _signers(self, seq) -> set:
        pat, truth = self.pc.signing, self.sc.truth
        origins = list(dict.fromkeys(o for o, _ in seq))
        side = {o: s.points_to for o, s in seq}
        if pat == "all":
            return set(origins)
        if pat == "none":
            return set()
        if pat == "truth":
            return {o for o in origins if side[o] == truth}
        if pat == "false":
            return {o for o in origins if side[o] != truth}
        if pat == "puppets":
            return {o for o in origins if o in self.puppets}
        if pat == "random":
            r = random.Random(f"{self.pc.seed}|signing")
            return {o for o in sorted(origins) if r.random() < 0.5}
        raise ValueError(pat)

    def badge(self, rec) -> str | None:
        if not self.pc.badge:
            return None
        if not rec.origin_sig:
            return "none: this record is not signed"
        ok = self.ks.check(rec.origin_id, rec.origin_payload(), rec.origin_sig)
        return f"valid: signed with {rec.origin_id}'s key" if ok else "INVALID: signature does not match"

    # -- benchmarks
    def benchmarks(self) -> dict:
        sc = self.sc
        distinct = {o: s for o, s in self.seq}
        return {
            "registry": sc.bayes_pA([s for o, s in distinct.items() if o in self.registry and o != self.forged]),
            "naive": sc.bayes_pA(list(distinct.values())),
            "msgcount": sc.bayes_pA([s for _, s in self.seq]),
            # what P3's checks can certify when not every source signs: registered AND signed origins
            "verified": sc.bayes_pA([s for o, s in distinct.items() if o in self.registry and o in self.signers]),
        }

    def tool(self, name: str, args: dict) -> dict:
        if name == "verify_record":
            r = self.records.get(args.get("record_id", ""))
            return pv.verify(self.ks, self.registry, r) if r else {"error": "unknown record_id"}
        if name == "count_independent_origins":
            res = pv.count_independent_origins(self.ks, self.registry, list(self.records.values()))
            return {k: {self.sc.names[s]: v for s, v in d.items()} for k, d in res.items()}
        return {"error": f"unknown tool {name}"}

    def prompts(self) -> tuple[str, str]:
        sysp = system_prompt(self.cfg, self.sc, self.me, False)
        if self.pc.experiment in SIGNED and (self.pc.signing != "all" or self.pc.badge):
            sysp += " " + MIXED_NOTE
            if self.pc.attribution_note:
                sysp += " " + ATTRIBUTION_NOTE
        usr = user_prompt(self.cfg, self.sc, 1, None, None, self.inbox, None, badge=self.badge)
        return sysp, usr

    async def run(self, llm, mock: bool = False) -> dict:
        sysp, usr = self.prompts()
        tool_log, out, stop = [], None, None
        if mock:
            out, tool_log = self._mock()
        else:
            res = await llm.json_call(sysp, usr, output_schema(self.cfg, self.sc), sample=self.pc.seed)
            reqs = (res.get("output") or {}).get("tool_requests") or []
            if self.pc.experiment == "P3" and reqs:
                tool_log = [{"name": q["tool"], "input": {"record_id": q.get("record_id", "")},
                             "output": self.tool(q["tool"], {"record_id": q.get("record_id", "")})} for q in reqs[:8]]
                usr2 = usr + "\n\nResults of the checks you requested:\n" + "\n".join(
                    f"{t['name']}({t['input']['record_id'] if t['name'] == 'verify_record' else ''}) -> {json.dumps(t['output'])}"
                    for t in tool_log) + "\n\nNow give your final belief and outgoing message."
                res = await llm.json_call(sysp, usr2, output_schema(self.cfg, self.sc, allow_tools=False),
                                          sample=self.pc.seed)
            out, stop = res.get("output"), res.get("stop_reason")
        b = (out or {}).get("belief_A")
        b = min(1.0, max(0.0, float(b))) if isinstance(b, (int, float)) else None
        truth = self.sc.truth
        pt = lambda x: x if truth == "A" else 1 - x
        bench = {k: pt(v) for k, v in self.benchmarks().items()}
        return {
            "kind": "packet", "design_version": DESIGN_VERSION, "config": asdict(self.pc),
            "scenario": {"id": self.sc.scenario_id, "domain": self.sc.domain, "truth": truth, "names": self.sc.names,
                         "question": self.sc.question, "p": self.sc.p},
            "inbox": [{"sender": m.sender, "text": m.text, "records": [r.record_id for r in m.records]} for m in self.inbox],
            "signers": sorted(self.signers),
            "puppet_origins": sorted({o for o, _ in self.seq if o in self.puppets}),
            "captured": self.captured, "forged": self.forged,
            "registry": sorted(self.registry),
            "belief_A": b, "p_truth": None if b is None else pt(b),
            "brier_index": None if b is None else brier_index((1 - pt(b)) ** 2),
            "bench_p_truth": bench,
            "bench_brier_index": {k: brier_index((1 - v) ** 2) for k, v in bench.items()},
            "message": (out or {}).get("message"), "cited": (out or {}).get("cite_record_ids"),
            "tool_log": tool_log, "stop_reason": stop,
            "prompt": {"system": sysp, "user": usr},
        }

    def _mock(self):
        """Naive distinct-origin counter; P3 counts registered origins via the real tool. NOT a result."""
        tool_log = []
        if self.pc.experiment == "P3":
            r = self.tool("count_independent_origins", {})
            tool_log.append({"name": "count_independent_origins", "input": {}, "output": r})
            c = r["independent_first_hand_origins"]
            net = c[self.sc.names["A"]] - c[self.sc.names["B"]]
        else:
            distinct = {o: s.points_to for o, s in self.seq}
            net = sum(1 if v == "A" else -1 for v in distinct.values())
        p = 1 / (1 + math.exp(-self.sc.llr() * net))
        return {"belief_A": round(p, 3), "message": "mock"}, tool_log


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", nargs="+", default=FORMATS)
    ap.add_argument("--packets", nargs="+", default=PACKETS)
    ap.add_argument("--coverage", nargs="+", type=float, default=[1.0],
                    help="registry coverage values; values < 1 are run for P3 only (other formats can't see the registry)")
    ap.add_argument("--reliability", nargs="+", choices=["stated", "unstated"], default=["stated"],
                    help="whether agents are told the per-report reliability p")
    ap.add_argument("--signing", nargs="+", choices=SIGNINGS, default=["all"],
                    help="which origins sign; patterns other than 'all' are run for the signed formats only")
    ap.add_argument("--badge", action="store_true", help="show the verification result as text, not a hex signature")
    ap.add_argument("--realism", choices=REALISMS, default="template")
    ap.add_argument("--attribution-note", action="store_true",
                    help="tell agents every attribution is accurate signed or not (removes the integrity reason for a premium)")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--backend", choices=["api", "cli", "openrouter", "deepseek", "gemini", "hf"], default="openrouter")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default="results/packet.jsonl")
    a = ap.parse_args()
    llm = None
    if not a.mock:
        from .llm import make_backend
        llm = make_backend(a.backend, a.model, a.effort, a.concurrency, a.temperature)
        print(f"backend={llm.name} model={llm.model} effort={llm.effort} temperature={llm.temperature}")
        await llm.preflight()
    pcs = [PacketConfig(experiment=f, packet=k, registry_coverage=c, seed=a.seed0 + s, state_reliability=rel == "stated",
                        signing=g, badge=a.badge, realism=a.realism,
                        attribution_note=a.attribution_note)
           for rel, f, k, c, g, s in itertools.product(a.reliability, a.formats, a.packets, a.coverage, a.signing,
                                                        range(a.seeds))
           if (c == 1.0 or f == "P3") and (g == "all" or f in SIGNED)
           and not (g == "puppets" and k in ("none", "captured", "forged"))]  # no puppets: identical to signing=none
    print(f"{len(pcs)} packet trials")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0 = time.time()
    rows = await asyncio.gather(*(Packet(pc).run(llm, a.mock) for pc in pcs))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {"kind": "packet", "out": a.out, "mock": a.mock, "seconds": round(time.time() - t0), "n_trials": len(pcs),
            "design_version": DESIGN_VERSION, "formats": a.formats, "packets": a.packets, "coverage": a.coverage,
            "seeds": a.seeds, "seed0": a.seed0, "reliability": a.reliability,
            "signing": a.signing, "badge": a.badge, "realism": a.realism,
            "attribution_note": a.attribution_note}
    if llm:
        u = llm.usage
        meta.update(backend=llm.name, model=llm.model, effort=llm.effort, temperature=llm.temperature, calls=u.calls,
                    cached=u.cached, errors=u.errors, refusals=u.refusals, cost_usd=None if llm.name in ("gemini", "deepseek") else round(u.cost_usd, 4),
                    input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                    providers=getattr(llm, "providers", None))
    with open(a.out.rsplit(".", 1)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s -> {a.out}; missing beliefs: {sum(r['belief_A'] is None for r in rows)}")
    if llm:
        print(f"calls={llm.usage.calls} cached={llm.usage.cached} errors={llm.usage.errors} cost=${llm.usage.cost_usd:.3f}")
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        c = r["config"]
        g[(c["state_reliability"], c["experiment"], c["registry_coverage"], c["packet"], c["signing"])].append(r)
    print(f"\n{'rel':5} {'format':6} {'cov':>4} {'packet':9} {'signing':8} {'n':>3} {'P(truth)':>8} {'BI':>5} {'reg':>5} {'naive':>6} {'msgs':>5}")
    for key in sorted(g):
        rs = [r for r in g[key] if r["p_truth"] is not None]
        if not rs:
            continue
        m = lambda f: sum(f(r) for r in rs) / len(rs)
        print(f"{'yes' if key[0] else 'no':5} {key[1]:6} {key[2]:4.1f} {key[3]:9} {key[4]:8} {len(rs):3} {m(lambda r: r['p_truth']):8.2f} {m(lambda r: r['brier_index']):5.0f} "
              f"{m(lambda r: r['bench_brier_index']['registry']):5.0f} {m(lambda r: r['bench_brier_index']['naive']):6.0f} "
              f"{m(lambda r: r['bench_brier_index']['msgcount']):5.0f}")


if __name__ == "__main__":
    asyncio.run(main())
