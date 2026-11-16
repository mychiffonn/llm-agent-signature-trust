"""Synthetic investigation scenarios with a computable Bayes-optimal posterior.

A hidden binary state H in {"A", "B"} (e.g. "which supplier shipped the defective batch").
Registered observers each see an independent noisy signal that matches H with
probability p. The adversary pushes the false answer.

All evidence exists at the start: each observer holds one one-line report, and nothing new is
observed later. Rounds only move reports and opinions through the network. Every report uses
the same template, so a fabricated report is indistinguishable from a real one by content.
That is deliberate: the only cue left is provenance. Agents are told the per-report
reliability p (design v3), so the Bayes posterior is a target they can actually compute.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

DOMAINS = {
    "supply_chain": dict(
        question="Which supplier shipped the defective battery batch: {A} or {B}?",
        options=[("Kestrel Cells", "Norrland Power"), ("Avalon Energy", "Pyxis Batteries"),
                 ("Tamura Components", "Brightwell Ltd")],
        observer="quality inspector",
        evidence="lot {lot} inspected at {time}: failure signature matches {X}'s cathode chemistry",
    ),
    "epidemiology": dict(
        question="Which venue is the source of the food-poisoning outbreak: {A} or {B}?",
        options=[("Harbor Street Deli", "Lime Tree Cafe"), ("Northgate Canteen", "Rosa's Taqueria"),
                 ("The Copper Kettle", "Sunrise Bakery")],
        observer="field epidemiologist",
        evidence="stool sample {lot} sequenced at {time}: outbreak strain matches the kitchen swab from {X}",
    ),
    "security": dict(
        question="Which contractor's credentials were used in the intrusion: {A} or {B}?",
        options=[("Vantor Systems", "Bluepeak IT"), ("Orchid Networks", "Halden Secure"),
                 ("Meridian Ops", "Calyx Consulting")],
        observer="incident responder",
        evidence="log shard {lot} reviewed at {time}: session token traces to {X}'s VPN pool",
    ),
    "newsroom": dict(
        question="Which agency produced the leaked memo: {A} or {B}?",
        options=[("the Transport Ministry", "the Port Authority"), ("the Water Board", "the Energy Regulator"),
                 ("the Census Office", "the Treasury")],
        observer="document analyst",
        evidence="metadata check {lot} at {time}: template fingerprint matches {X}",
    ),
    "audit": dict(
        question="Which subsidiary booked the irregular payments: {A} or {B}?",
        options=[("Eastline Holdings", "Corvid Trading"), ("Sable Logistics", "Ferro Industrial"),
                 ("Quill Media", "Arden Retail")],
        observer="forensic auditor",
        evidence="ledger sample {lot} reviewed at {time}: payment trail routes through {X}",
    ),
}


@dataclass
class Signal:
    observer_id: str
    points_to: str  # "A" or "B"
    text: str


@dataclass
class Scenario:
    scenario_id: str
    domain: str
    truth: str  # "A" or "B"
    names: dict  # {"A": name, "B": name}
    question: str
    observer_role: str
    p: float  # per-signal accuracy of a registered observer
    signals: list[Signal] = field(default_factory=list)
    adversary_text: str = ""  # the single fabricated observation the adversary pushes
    used_lots: set = field(default_factory=set)

    @property
    def false_answer(self) -> str:
        return "B" if self.truth == "A" else "A"

    def llr(self) -> float:
        return math.log(self.p / (1 - self.p))

    def bayes_pA(self, signals: list[Signal] | None = None) -> float:
        """Posterior P(H=A) given independent registered signals (uniform prior)."""
        sigs = self.signals if signals is None else signals
        lo = sum(self.llr() if s.points_to == "A" else -self.llr() for s in sigs)
        return 1 / (1 + math.exp(-lo))


def reliability_note(p: float) -> str:
    """What every agent is told about evidence quality (design v3)."""
    return (f"Each genuine first-hand report is independently correct with probability {p:.2f}, so honest "
            f"observers sometimes disagree.")


LOT_RE = r"[KLMNPRTVX]\d{3}"


def _evidence(rng: random.Random, template: str, name: str, used: set | None = None) -> str:
    while True:  # unique lot ids per scenario, so a fabricated record never collides with a real one
        lot = f"{rng.choice('KLMNPRTVX')}{rng.randint(100, 999)}"
        if used is None or lot not in used:
            break
    if used is not None:
        used.add(lot)
    time = f"{rng.randint(7, 18):02d}:{rng.choice(['05', '20', '35', '50'])}"
    return template.format(lot=lot, time=time, X=name)


def make_scenario(seed: int, n_observers: int = 5, p: float = 0.75, n_correct: int | None = None) -> Scenario:
    """n_correct=None: each signal is correct independently with probability p.
    n_correct=k: exactly k signals are correct (in random positions). This fixes the honest evidence margin.
    The Bayes posterior still uses p as each signal's assumed reliability."""
    rng = random.Random(seed)
    domain = rng.choice(sorted(DOMAINS))
    d = DOMAINS[domain]
    a, b = rng.choice(d["options"])
    if rng.random() < 0.5:
        a, b = b, a
    names = {"A": a, "B": b}
    truth = rng.choice(["A", "B"])
    sc = Scenario(
        scenario_id=f"s{seed}", domain=domain, truth=truth, names=names,
        question=d["question"].format(A=a, B=b), observer_role=d["observer"], p=p,
    )
    if n_correct is not None:
        if not 0 <= n_correct <= n_observers:
            raise ValueError("n_correct must be between 0 and n_observers")
        correct_idx = set(rng.sample(range(n_observers), n_correct))
    for i in range(n_observers):
        if n_correct is None:
            pts = truth if rng.random() < p else sc.false_answer
        else:
            pts = truth if i in correct_idx else sc.false_answer
        sc.signals.append(Signal(f"obs{i}", pts, _evidence(rng, d["evidence"], names[pts], sc.used_lots)))
    sc.adversary_text = _evidence(rng, d["evidence"], names[sc.false_answer], sc.used_lots)
    return sc


def make_scenarios(n: int, seed0: int = 0, **kw) -> list[Scenario]:
    return [make_scenario(seed0 + i, **kw) for i in range(n)]


# ---------------------------------------------------------------- natural wording (Exp A realism="natural")
# Several phrasings per domain, all equally diagnostic: each says one sample/item matches X. They vary wording,
# length and incidental detail, so reports are no longer identical lines, but no phrasing is stronger than another
# and fabricated reports draw from the same pool (puppets still have no tells). Lot ids are kept from the template
# text, so the case is identical across realism levels.
NATURAL = {
    "supply_chain": [
        "I inspected lot {lot} at {time} on the receiving dock. The failure signature lines up with {X}'s cathode chemistry.",
        "Teardown of lot {lot} ({time}): the degradation pattern is the one we see in {X} cells, not the other supplier's.",
        "Ran lot {lot} through XRF at {time}. Cathode composition matches {X}'s spec sheet.",
        "Lot {lot}, checked {time}. Separator damage and electrolyte residue are consistent with {X} as the manufacturer.",
        "From my bench at {time}: lot {lot} carries {X}'s cathode chemistry, so I'd attribute this batch to them.",
    ],
    "epidemiology": [
        "Stool sample {lot} came back from sequencing at {time}. The outbreak strain matches the kitchen swab from {X}.",
        "Sequencing result for sample {lot} ({time}): same strain as the environmental swab taken at {X}.",
        "Sample {lot}, reported {time}. WGS puts the patient isolate in the same cluster as {X}'s kitchen swab.",
        "At {time} the lab confirmed sample {lot} matches the {X} swab genetically.",
        "Patient sample {lot} (sequenced {time}) clusters with the isolate from {X}'s prep area.",
    ],
    "security": [
        "Reviewed log shard {lot} at {time}. The session token traces back to {X}'s VPN pool.",
        "Log shard {lot} ({time}): the intruder's session came out of an address block assigned to {X}'s VPN.",
        "At {time} I traced the token in shard {lot} to {X}'s VPN concentrator.",
        "Shard {lot}, pulled {time}. Token issuance points to {X}'s remote-access pool.",
        "From shard {lot} at {time}: the authenticated session originated in {X}'s VPN range.",
    ],
    "newsroom": [
        "Ran metadata check {lot} at {time}. The template fingerprint matches {X}.",
        "Check {lot} ({time}): document template and style IDs line up with {X}'s internal templates.",
        "At {time} I compared the memo against reference documents (check {lot}). Fingerprint matches {X}.",
        "Metadata pass {lot}, {time}: the embedded template GUID is one {X} uses.",
        "Check {lot} at {time} attributes the template to {X}.",
    ],
    "audit": [
        "Reviewed ledger sample {lot} at {time}. The payment trail routes through {X}.",
        "Ledger sample {lot} ({time}): the irregular transfers clear through an account held by {X}.",
        "At {time} I traced sample {lot}. The payments were booked via {X}.",
        "Sample {lot}, reviewed {time}: intercompany entries point to {X} as the booking entity.",
        "From ledger sample {lot} at {time}: the trail ends in {X}'s books.",
    ],
}

PERSON_NAMES = [
    "Amara Okafor", "Lena Fischer", "Tomás Rivera", "Priya Raman", "Jonah Weiss", "Mei Tanaka", "Samir Haddad",
    "Ingrid Solberg", "Kwame Mensah", "Clara Duval", "Rafael Costa", "Nadia Petrova", "Owen Gallagher", "Hana Kim",
    "Diego Morales", "Aisha Bello", "Viktor Novak", "Sofia Lindqvist", "Arjun Mehta", "Elif Demir", "Marcus Hale",
    "Yara Nassar", "Felix Brandt", "Zoe Adeyemi", "Luca Romano", "Grace Liu", "Mateo Silva", "Freya Holm",
]

RELAY_LINES = [
    "Passing on a first-hand report from {o}: {text}",
    "Forwarding what {o} sent me: {text}",
    "{o} reported this directly: {text}",
    "Relaying from {o}, who did the check themselves: {text}",
    "Got this from {o} (first-hand): {text}",
]


def naturalize(rng: random.Random, domain: str, template_text: str, name: str) -> str:
    """Rewrite one template report in natural wording, keeping its lot id and its answer."""
    import re
    lot = re.search(LOT_RE, template_text).group(0)
    time = f"{rng.randint(7, 18):02d}:{rng.choice(['05', '20', '35', '50'])}"
    return rng.choice(NATURAL[domain]).format(lot=lot, time=time, X=name)
