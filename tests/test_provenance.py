from harness import provenance as pv


def test_integrity_vs_independence():
    ks, reg = pv.KeyStore(), {"obs1", "obs2"}
    real = pv.originate(ks, "obs1", {"points_to": "A", "text": "x"})
    puppet = pv.originate(ks, "puppet9", {"points_to": "B", "text": "y"})
    forged = pv.forge({"points_to": "B", "text": "z"}, "obs2")
    relayed = pv.relay(ks, pv.relay(ks, real, "agent_3"), "agent_4")

    assert pv.verify(ks, reg, real)["first_hand"]
    assert pv.verify(ks, reg, relayed)["first_hand"]
    v = pv.verify(ks, reg, puppet)
    assert v["signature_valid"] and not v["origin_registered"]  # signed but not independent
    assert not pv.verify(ks, reg, forged)["signature_valid"]

    tampered = pv.relay(ks, real, "agent_3")
    tampered.claim = {"points_to": "B", "text": "x"}
    assert not pv.verify(ks, reg, tampered)["signature_valid"]

    c = pv.count_independent_origins(ks, reg, [real, relayed, puppet, forged])
    assert c["independent_first_hand_origins"] == {"A": 1, "B": 0}  # relayed copy is not a 2nd origin
    assert c["unverified_or_unregistered_records"] == {"A": 0, "B": 2}
