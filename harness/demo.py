"""Readable trace of one simulation.

  python -m harness.demo --mock                                   # offline heuristic stand-in
  python -m harness.demo --backend cli --model claude-haiku-5-5   # real model via your Claude Code login
  python -m harness.demo --experiment P0 --adversary sockpuppet --rounds 3 --seed 4
"""
from __future__ import annotations

import argparse
import asyncio
import textwrap

from .metrics import run_metrics
from .network import Config, Sim

W = 100


def wrap(s: str, ind: str = "    ") -> str:
    return textwrap.fill(s, W, initial_indent=ind, subsequent_indent=ind)


def hr(title: str = "") -> None:
    print("\n" + ("─── " + title + " " if title else "").ljust(W, "─"))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="P3")
    ap.add_argument("--adversary", default="sockpuppet")
    ap.add_argument("--topology", default="small_world")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--observers", type=int, default=3)
    ap.add_argument("--puppets", type=int, default=2)
    ap.add_argument("--insiders", type=int, default=1, help="accredited observers turned adversary (insider / captured)")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n-correct", type=int, default=2, help="exactly k observers correct (-1 = random)")
    ap.add_argument("--puppet-once", action="store_true")
    ap.add_argument("--puppet-style", choices=["llm", "template"], default="llm")
    ap.add_argument("--reflect", action="store_true")
    ap.add_argument("--show-prompt", default=None, help="agent name whose full prompt to print (default: a non-observer)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--save", default=None, help="write the full result JSON here (e.g. results/demo_P3.json)")
    ap.add_argument("--backend", choices=["api", "cli", "openrouter", "deepseek", "gemini", "hf"], default="cli")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    a = ap.parse_args()

    cfg = Config(topology=a.topology, experiment=a.experiment, adversary=a.adversary, rounds=a.rounds,
                 n_agents=a.agents, n_observers=a.observers, n_puppets=a.puppets, n_insiders=a.insiders, seed=a.seed,
                 n_correct=None if a.n_correct < 0 else a.n_correct, puppet_once=a.puppet_once,
                 puppet_style=a.puppet_style, reflect=a.reflect)
    llm = None
    if not a.mock:
        from .llm import make_backend
        llm = make_backend(a.backend, a.model, a.effort)
        await llm.preflight()
    sim = Sim(cfg, llm, mock=a.mock)
    sc = sim.sc

    hr("THE INVESTIGATION (generated)")
    print(wrap(f"Domain: {sc.domain}.  {sc.question}"))
    print(wrap(f"Hidden truth: {sc.names[sc.truth]}   (option {sc.truth}; agents never see this)"))
    hs = [s for o, s in sim.observer_of.items() if o not in sim.insiders]
    bp = sc.bayes_pA(hs)
    print(wrap(f"{sum(s.points_to == sc.truth for s in hs)} of {len(hs)} honest observers hold correct evidence "
               f"(fixed margin: n_correct={cfg.n_correct}). Bayes-optimal P(truth) pooling them = "
               f"{bp if sc.truth == 'A' else 1 - bp:.2f}"))
    print(wrap(f"Experiment: {cfg.experiment}    Adversary: {cfg.adversary}    Topology: {cfg.topology}    Rounds: {cfg.rounds}"))

    hr("WHO KNOWS WHAT")
    for ag, s in sim.observer_of.items():
        if ag in sim.insiders:
            continue
        tag = "correct" if s.points_to == sc.truth else "MISLEADING (honest noise)"
        print(f"  {ag:9} observer   [{tag}]")
        print(wrap(f"\"{s.text}\"", "      "))
    for p in sim.adversaries:
        info = sim.adv_info[p]
        rec = info["record"]
        kind = {"sockpuppet": "signs an invented observation with its OWN valid key (not in registry)",
                "forged": f"claims origin {rec.origin_id if rec else '?'} with a bogus signature",
                "unsourced": "pushes the claim with no record",
                "fake_verify": "own valid key (not in registry) + claims its record verified",
                "insider": "REAL accredited observer with wrong evidence, turned adversarial",
                "captured": "REAL accredited observer, captured: signs a FABRICATED record with its registered key"}[info["kind"]]
        print(f"  {p:9} ADVERSARY  [{kind}]")
        print(wrap(f"\"{info['claim']['text']}\"", "      "))
    others = [h for h in sim.honest if h not in sim.observer_of]
    print(f"  no first-hand evidence: {', '.join(others)}")
    print(f"  registry of accredited observers: {sorted(sim.registry)}")

    hr("NETWORK (who hears whom)")
    for n in sim.honest + sim.adversaries:
        mark = "*" if n in sim.adversaries else ("o" if n in sim.observer_of else " ")
        print(f"  {mark} {n:9} -> {', '.join(sorted(sim.nbrs[n]))}")
    print("  (o = observer, * = adversary)")

    show = a.show_prompt or next((n for n in others if any(p in sim.nbrs[n] for p in sim.adversaries)), others[0])
    captured = {}

    orig_turn = sim._honest_turn

    async def turn(me, rnd):
        r = await orig_turn(me, rnd)
        captured[(me, rnd)] = r[1]
        return r
    sim._honest_turn = turn

    # run round by round so we can print as it unfolds
    full_rounds = cfg.rounds
    res = await sim.run()

    for rnd in range(full_rounds):
        hr(f"ROUND {rnd}")
        for e in [e for e in sim.log if e["round"] == rnd]:
            ag = e["agent"]
            b = e["belief_A"]
            pt = None if b is None else (b if sc.truth == "A" else 1 - b)
            tag = "o" if e["observer"] else " "
            print(f"  {tag} {ag:9} P(truth)={'n/a' if pt is None else f'{pt:.2f}':5}  "
                  f"{'✓' if pt is not None and pt > .5 else '✗'}")
            print(wrap(f"says: \"{e['message']}\"", "      "))
            if e.get("source_note"):
                print(wrap(f"source line: {e['source_note']}", "      "))
            if e["cited_valid"]:
                print(wrap(f"forwards records: {', '.join(e['cited_valid'])}", "      "))
            for t in e["tool_log"]:
                print(wrap(f"tool {t['name']}({t['input'] or ''}) -> {t['output']}", "      "))
        for e in [e for e in sim.adv_log if e["round"] == rnd]:
            print(f"  * {e['agent']:9} (adversary: {e['kind']}{', TEMPLATE FALLBACK' if e['fallback'] else ''})")
            print(wrap(f"says: \"{e['message']}\"", "      "))
            if e.get("source_note") and cfg.experiment == "P1":
                print(wrap(f"source line: {e['source_note']}", "      "))
            if e["records"]:
                print(wrap(f"attaches records: {', '.join(e['records'])}", "      "))

    hr(f"WHAT {show} SEES IN ROUND {min(2, full_rounds - 1)} (exact prompt)")
    c = captured.get((show, min(2, full_rounds - 1)))
    if c:
        print(textwrap.indent("[system]\n" + c["system"] + "\n\n[user]\n" + c["user"], "  "))

    if a.save:
        import json, os
        os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
        res["metrics"] = run_metrics(res)
        res["prompt_example"] = {"agent": show, "round": min(2, full_rounds - 1), **(captured.get((show, min(2, full_rounds - 1))) or {})}
        res["prompt_example"].pop("res", None)
        res["mock"] = a.mock
        res["backend"] = None if a.mock else {"name": llm.name, "model": llm.model, "effort": llm.effort}
        with open(a.save, "w") as f:
            json.dump(res, f)

    hr("OUTCOME")
    m = run_metrics(res)
    print(f"  accuracy={m['accuracy']:.2f}   Brier Index={m['brier_index']:.0f}   "
          f"(coin-flip = 0, perfect = 100, Bayes ceiling for this evidence = {m['bayes_brier_index']:.0f})")
    print(f"  falsehood reach per round={[round(x, 2) for x in m['reach_curve']]}")
    if llm:
        u = llm.usage
        print(f"  model calls={u.calls} cached={u.cached} errors={u.errors} refusals={u.refusals} cost=${u.cost_usd:.3f}")
    if a.mock:
        print("\n  NOTE: --mock: agent beliefs/messages above come from a heuristic stand-in, not an LLM.\n"
              "  Scenario, evidence, puppets, signatures, records, tool outputs and prompts are real harness output.")


if __name__ == "__main__":
    asyncio.run(main())
