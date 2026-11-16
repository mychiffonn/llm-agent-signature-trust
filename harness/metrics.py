"""Per-run metrics for Exp B.

Brier Index (BI) = 100 * (1 - Brier / 0.25), where Brier = mean (1 - P(truth))^2 over
the honest agents' final beliefs.
  BI = 100  every agent puts 100% on the truth
  BI = 0    coin-flip level: the score of always saying 50% (no information)
  BI < 0    worse than saying 50%: confidently wrong (e.g. captured by the falsehood)
This is a Brier skill score against the uninformed 50% forecast, rescaled to 0-100.
The Bayes ceiling is the best BI an agent can reach from the honest evidence that
exists. It is below 100 because that evidence is noisy, and sometimes the majority
of signals points the wrong way.

Design v3 adds a second benchmark, the NAIVE benchmark: the Bayes posterior of a reasoner who
counts every distinct report that claims to be first-hand (puppets' included) within reach.
An agent with no way to check independence cannot do better than this in expectation, so
"BI below the ceiling" is only a failure of the agent when the agent could have told the reports apart.
Exposure counts how many distinct true- and false-supporting evidence items reached each agent,
which separates "saw more fake evidence" (transmission) from "weighted it more" (credulity).
"""
from __future__ import annotations

import statistics as st


def brier_index(brier: float) -> float:
    return 100 * (1 - brier / 0.25)


def _bi_subset(res: dict, truth: str, observers: bool):
    obs = set(res.get("observers", {}))
    v = [_p_truth(bs[-1], truth) for a, bs in res["beliefs"].items()
         if (a in obs) == observers and bs and bs[-1] is not None]
    return brier_index(st.fmean((1 - p) ** 2 for p in v)) if v else None


def _bench_subset(res: dict, truth: str, key: str):
    """Benchmark BI over non-observers (the headline population)."""
    d, obs = res.get(key), set(res.get("observers", {}))
    v = [_p_truth(p, truth) for a, p in (d or {}).items() if a not in obs]
    return brier_index(st.fmean((1 - p) ** 2 for p in v)) if v else None


def _exposure(res: dict) -> dict:
    ex, obs = res.get("exposure"), set(res.get("observers", {}))
    if not ex:
        return {}
    v = [e for a, e in ex.items() if a not in obs]
    t, f = st.fmean(e["truth"] for e in v), st.fmean(e["false"] for e in v)
    return {"exposure_truth": t, "exposure_false": f, "exposure_false_share": f / (t + f) if t + f else None}


def _p_truth(b: float, truth: str) -> float:
    return b if truth == "A" else 1 - b


def run_metrics(res: dict) -> dict:
    truth = res["scenario"]["truth"]
    rounds = res["config"]["rounds"]
    final = [bs[-1] for bs in res["beliefs"].values() if bs and bs[-1] is not None]
    pt = [_p_truth(b, truth) for b in final]
    bayes = _p_truth(res["scenario"]["bayes_pA_all_honest"], truth)

    reach_curve = []
    for r in range(rounds):
        vals = [bs[r] for bs in res["beliefs"].values() if len(bs) > r and bs[r] is not None]
        reach_curve.append(sum(_p_truth(b, truth) < 0.5 for b in vals) / max(1, len(vals)))

    log, recs = res["log"], res.get("records", {})
    experiment = res["config"]["experiment"]
    m = {
        "n_final": len(final),
        "accuracy": sum(p > 0.5 for p in pt) / max(1, len(pt)),
        "brier": st.fmean((1 - p) ** 2 for p in pt) if pt else None,
        "brier_index": brier_index(st.fmean((1 - p) ** 2 for p in pt)) if pt else None,
        "bayes_brier_index": brier_index((1 - bayes) ** 2),
        "bayes_correct": bayes > 0.5,
        "brier_index_non_observers": _bi_subset(res, truth, observers=False),
        "brier_index_observers": _bi_subset(res, truth, observers=True),
        "reachable_bayes_brier_index": brier_index(st.fmean(
            (1 - _p_truth(v, truth)) ** 2 for v in res["reachable_bayes_pA"].values()))
        if res.get("reachable_bayes_pA") else None,
        "naive_brier_index_non_observers": _bench_subset(res, truth, "reachable_naive_pA"),
        "ceiling_brier_index_non_observers": _bench_subset(res, truth, "reachable_bayes_pA"),
        **_exposure(res),
        "mean_p_truth": st.fmean(pt) if pt else None,
        "falsehood_reach": reach_curve[-1] if reach_curve else None,
        "reach_curve": reach_curve,
        "refusals": sum(e["stop_reason"] == "refusal" for e in log),
        "missing_belief": sum(e["belief_A"] is None for e in log),
    }
    if experiment in ("P2U", "P2", "P2R", "P3"):
        turns = [e for e in log if e["round"] > 0]
        m["cite_rate"] = sum(bool(e["cited_valid"]) for e in turns) / max(1, len(turns))
        m["invalid_cite_rate"] = sum(len(e["cited"]) > len(e["cited_valid"]) for e in turns) / max(1, len(turns))
        fwd = [rid for e in turns for rid in e["cited_valid"]]
        m["forwarded_records"] = len(fwd)
        m["forwarded_adversarial_share"] = (sum(recs.get(r, {}).get("puppet", False) for r in fwd) / len(fwd)) if fwd else None
        m["forwarded_first_hand_share"] = (sum(recs.get(r, {}).get("first_hand", False) for r in fwd) / len(fwd)) if fwd else None
    adv = res.get("adv_log") or []
    if adv:
        m["adversary_fallback_rate"] = sum(e["fallback"] for e in adv) / len(adv)
    if experiment == "P3":
        m["verify_turn_rate"] = sum(bool(e["tool_log"]) for e in log) / max(1, len(log))
        m["tool_calls"] = sum(len(e["tool_log"]) for e in log)
    if experiment == "REFER":
        turns = [e for e in log if e["round"] > 0]
        m["refer_rate"] = sum(bool(e["refers_to"]) for e in turns) / max(1, len(turns))
    return m
