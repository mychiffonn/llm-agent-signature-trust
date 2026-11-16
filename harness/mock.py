"""Deterministic offline stand-in for the LLM, for testing the pipeline at zero cost.

Policy: naive vote-counting in P0/P1, counting distinct record ids in REFER/P2,
counting only verified first-hand origins (via the real tool handler) in P3.
Its numbers are NOT results; it only exercises every code path.
"""
from __future__ import annotations

import math


def _side(sim, text: str) -> str | None:
    a, b = sim.sc.names["A"], sim.sc.names["B"]
    ia, ib = text.find(a), text.find(b)
    if ia < 0 and ib < 0:
        return None
    if ib < 0 or (ia >= 0 and ia < ib):
        return "A"
    return "B"


def mock_agent(sim, me: str, rnd: int) -> dict:
    cfg, llr = sim.cfg, sim.sc.llr()
    net = 0.0
    tool_log = []
    sig = sim.observer_of.get(me)
    if sig:
        net += llr if sig.points_to == "A" else -llr
    if cfg.experiment == "P3":
        handler = sim._tool_handler(me)
        res = handler("count_independent_origins", {})
        tool_log.append({"name": "count_independent_origins", "input": {}, "output": res})
        counts = res["independent_first_hand_origins"]
        own = sig is not None
        na = counts[sim.sc.names["A"]] - (own and sig.points_to == "A")
        nb = counts[sim.sc.names["B"]] - (own and sig.points_to == "B")
        net += llr * (na - nb)
    else:
        seen = set()
        for m in sim.inbox[me]:
            key = m.records[0].record_id if (cfg.experiment in ("REFER", "P2U", "P2", "P2R") and m.records) else m.msg_id
            if cfg.experiment == "REFER" and m.cites:
                continue  # rule: don't count references as new readings
            if key in seen:
                continue
            seen.add(key)
            s = _side(sim, m.text)
            if s:
                net += 0.5 * llr * (1 if s == "A" else -1)
    p = 1 / (1 + math.exp(-max(-30, min(30, net))))
    fav = "A" if p >= 0.5 else "B"
    msg = f"My current view: it is {sim.sc.names[fav]} (P={max(p, 1-p):.2f})."
    if sig and rnd == 0:
        msg = f"First-hand: {sig.text}. " + msg
    out = {"belief_A": round(p, 3), "message": msg}
    if cfg.experiment == "P1":
        out["source_note"] = "first-hand" if sig else "colleagues' messages"
    if cfg.experiment in ("P2U", "P2", "P2R", "P3"):
        ids = list(sim.received_records[me])
        if me in sim.own_record:
            ids.append(sim.own_record[me].record_id)
        out["cite_record_ids"] = ids[:4]
    if cfg.experiment == "REFER":
        out["refers_to"] = [m.msg_id for m in sim.inbox[me][:2]]
    if cfg.experiment == "P3" and ids:
        r = ids[0]
        tool_log.append({"name": "verify_record", "input": {"record_id": r},
                         "output": sim._tool_handler(me)("verify_record", {"record_id": r})})
    return {"output": out, "tool_log": tool_log, "stop_reason": "mock"}
