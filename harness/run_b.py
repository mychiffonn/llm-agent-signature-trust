"""Sweep Exp B: topology x experiment x adversary x seed.

  python -m harness.run_b --mock --seeds 3                       # offline smoke test, no API calls
  python -m harness.run_b --seeds 2 --experiments P0 P3 --adversaries sockpuppet --effort low   # small pilot
  python -m harness.run_b --backend cli --model claude-haiku-5-5 --seeds 20   # via your Claude Code login
  python -m harness.run_b --backend openrouter --seeds 15 --experiments P0 P2U P2 P3 --adversaries none sockpuppet
      # DeepSeek V4.1 Flash via OpenRouter (default model for that backend); seeds are independent samples
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import statistics as st
import time
from collections import defaultdict

from .metrics import run_metrics
from .network import ADVERSARIES, DESIGN_VERSION, EXPERIMENTS, Config, Sim


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", nargs="+", default=EXPERIMENTS)
    ap.add_argument("--adversaries", nargs="+", default=ADVERSARIES)
    ap.add_argument("--topologies", nargs="+", default=["small_world", "hub"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--agents", type=int, default=15)
    ap.add_argument("--n-correct", type=int, default=3, help="exactly k of the observers are correct (-1 = random with p)")
    ap.add_argument("--observers", type=int, default=5)
    ap.add_argument("--puppets", type=int, default=3)
    ap.add_argument("--puppet-degree", type=int, default=2)
    ap.add_argument("--insiders", type=int, default=1, help="accredited observers turned adversary (insider / captured)")
    ap.add_argument("--puppet-once", action="store_true", help="adversaries speak only in round 0")
    ap.add_argument("--puppet-style", choices=["llm", "template"], default="llm")
    ap.add_argument("--reflect", action="store_true", help="extra re-read call for honest turns without a tool round")
    ap.add_argument("--registry-coverage", type=float, default=1.0, help="share of genuine observers in the registry")
    ap.add_argument("--unstated-reliability", action="store_true", help="don't tell agents the per-report reliability")
    ap.add_argument("--signing", nargs="+", choices=["all", "puppets", "none"], default=["all"],
                    help="who signs origin records (a grid factor; non-'all' values only for signed arms)")
    ap.add_argument("--badge", action="store_true", help="show signature verification results as text")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--backend", choices=["api", "cli", "openrouter", "deepseek", "gemini", "hf"], default="cli")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-sims", type=int, default=8, help="simulations running at once")
    ap.add_argument("--out", default="results/b.jsonl")
    a = ap.parse_args()

    llm = None
    if not a.mock:
        from .llm import make_backend
        llm = make_backend(a.backend, a.model, a.effort, a.concurrency, a.temperature)
        print(f"backend={llm.name} model={llm.model} effort={llm.effort}")
        await llm.preflight()

    common = dict(rounds=a.rounds, n_agents=a.agents, n_observers=a.observers,
                  n_correct=None if a.n_correct < 0 else a.n_correct, n_puppets=a.puppets,
                  puppet_degree=a.puppet_degree, n_insiders=a.insiders, puppet_once=a.puppet_once, puppet_style=a.puppet_style,
                  reflect=a.reflect, registry_coverage=a.registry_coverage,
                  state_reliability=not a.unstated_reliability, badge=a.badge)
    from .network import SIGNED
    cfgs = [Config(topology=t, experiment=experiment, adversary=adv, seed=a.seed0 + s, signing=g, **common)
            for t, experiment, adv, g, s in itertools.product(a.topologies, a.experiments, a.adversaries, a.signing,
                                                              range(a.seeds))
            if g == "all" or experiment in SIGNED]
    turns = sum(c.rounds * c.n_agents for c in cfgs)
    import sys
    print("params:", common, file=sys.stderr)
    print(f"{len(cfgs)} simulations, {turns} agent turns (P3 turns may add tool round-trips)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    gate = asyncio.Semaphore(a.max_sims)
    t0 = time.time()

    async def one(cfg):
        async with gate:
            res = await Sim(cfg, llm, mock=a.mock).run()
            res["metrics"] = run_metrics(res)
            return res

    groups = defaultdict(list)
    with open(a.out, "w") as f:  # one file per sweep; the cache makes reruns free
        for coro in asyncio.as_completed([one(c) for c in cfgs]):
            res = await coro
            f.write(json.dumps(res) + "\n")
            f.flush()
            c = res["config"]
            groups[(c["topology"], c["experiment"] + ("" if c["signing"] == "all" else f"/{c['signing']}"),
                    c["adversary"])].append(res["metrics"])

    print(f"\ndone in {time.time() - t0:.0f}s -> {a.out}")
    meta = {"out": a.out, "mock": a.mock, "seconds": round(time.time() - t0), "n_sims": len(cfgs),
            "design_version": DESIGN_VERSION, "params": common, "experiments": a.experiments,
            "adversaries": a.adversaries, "topologies": a.topologies, "seeds": a.seeds,
            "signing": a.signing, "seed0": a.seed0}
    if llm:
        u = llm.usage
        meta.update(backend=llm.name, model=llm.model, effort=llm.effort, temperature=llm.temperature,
                    providers=getattr(llm, "providers", None), calls=u.calls, cached=u.cached,
                    errors=u.errors, refusals=u.refusals, cost_usd=None if llm.name in ("gemini", "deepseek") else round(u.cost_usd, 4),
                    input_tokens=u.input_tokens, output_tokens=u.output_tokens)
    with open(a.out.rsplit(".", 1)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    if llm:
        u = llm.usage
        print(f"model calls={u.calls} cached={u.cached} in_tok={u.input_tokens} out_tok={u.output_tokens} "
              f"refusals={u.refusals} errors={u.errors} cost_usd={u.cost_usd:.3f}")

    def avg(ms, k):
        v = [m[k] for m in ms if m.get(k) is not None]
        return f"{st.fmean(v):.2f}" if v and k != "brier_index" and k != "bayes_brier_index" else (f"{st.fmean(v):.0f}" if v else "  - ")

    print(f"\n{'topology':12} {'experiment':10} {'adversary':11} {'n':>3} {'acc':>5} {'BI':>6} {'BIbayes':>7} {'reach':>6} "
          f"{'cite':>5} {'advfwd':>6} {'verify':>6}")
    for k in sorted(groups):
        ms = groups[k]
        print(f"{k[0]:12} {k[1]:10} {k[2]:11} {len(ms):3} {avg(ms, 'accuracy'):>5} {avg(ms, 'brier_index'):>6} {avg(ms, 'bayes_brier_index'):>7} "
              f"{avg(ms, 'falsehood_reach'):>6} {avg(ms, 'cite_rate'):>5} {avg(ms, 'forwarded_adversarial_share'):>6} "
              f"{avg(ms, 'verify_turn_rate'):>6}")


if __name__ == "__main__":
    asyncio.run(main())
