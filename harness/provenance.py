"""Signed, hash-linked provenance records (a minimal C2PA-like scheme).

Integrity vs independence is the key distinction this module makes explicit:
  * a valid signature proves WHO emitted / relayed a record (integrity);
  * only the registry says whether an origin is an accredited first-hand observer,
    and only counting distinct registered origins says how many INDEPENDENT readings
    a pile of messages contains.
Sockpuppets can sign perfectly valid records under their own keys; they fail only
the registry / independence check.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


class KeyStore:
    """Holds every identity's keypair. Public keys are 'published'; private keys are
    only used by the harness on behalf of the identity that owns them."""

    def __init__(self, seed: int | str | None = None):
        self.seed = seed  # deterministic keys per seed -> identical prompts -> reproducible, cacheable runs
        self._priv: dict[str, Ed25519PrivateKey] = {}
        self.pub: dict[str, Ed25519PublicKey] = {}

    def ensure(self, ident: str) -> None:
        if ident not in self._priv:
            if self.seed is None:
                k = Ed25519PrivateKey.generate()
            else:
                k = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(f"{self.seed}|{ident}".encode()).digest())
            self._priv[ident] = k
            self.pub[ident] = k.public_key()

    def sign(self, ident: str, payload: bytes) -> str:
        self.ensure(ident)
        return self._priv[ident].sign(payload).hex()

    def check(self, ident: str, payload: bytes, sig_hex: str) -> bool:
        pk = self.pub.get(ident)
        if pk is None:
            return False
        try:
            pk.verify(bytes.fromhex(sig_hex), payload)
            return True
        except (InvalidSignature, ValueError):
            return False


@dataclass
class Record:
    record_id: str
    claim: dict  # {"points_to": "A"|"B", "text": str}
    origin_id: str
    origin_sig: str
    chain: list[dict] = field(default_factory=list)  # [{"agent_id", "sig"}]

    def origin_payload(self) -> bytes:
        return _canon({"record_id": self.record_id, "claim": self.claim, "origin_id": self.origin_id})

    def hop_payload(self, i: int) -> bytes:
        prev = self.origin_sig if i == 0 else self.chain[i - 1]["sig"]
        return _canon({"record_id": self.record_id, "prev": prev, "agent_id": self.chain[i]["agent_id"]})

    def to_dict(self) -> dict:
        return asdict(self)

    def short(self, signed: bool = True, badge: str | None = None) -> dict:
        """What an agent sees in its prompt. signed=False is the P2U view: the same record structure with no
        signature field. By default the signature is shown as a truncated hex string the agent cannot check
        (design v3); an unsigned record then has no signature field. badge= instead shows the platform's
        verification result as text (e.g. "valid: signed with Lena Fischer's key"), the way deployed provenance
        systems display it."""
        d = {"record_id": self.record_id, "claim": self.claim["text"], "origin": self.origin_id}
        if signed and badge is not None:
            d["signature"] = badge
        elif signed and self.origin_sig:
            d["origin_sig"] = self.origin_sig[:16] + "…"
        d["relayed_via"] = [h["agent_id"] for h in self.chain]
        return d


def content_id(claim: dict, origin_id: str) -> str:
    return "r_" + hashlib.sha256(_canon({"c": claim, "o": origin_id})).hexdigest()[:10]


def originate(ks: KeyStore, origin_id: str, claim: dict) -> Record:
    rid = content_id(claim, origin_id)
    rec = Record(rid, claim, origin_id, "")
    rec.origin_sig = ks.sign(origin_id, rec.origin_payload())
    return rec


def forge(claim: dict, claimed_origin: str) -> Record:
    """Record that claims a registered origin but carries a bogus signature."""
    rid = content_id(claim, claimed_origin)
    return Record(rid, claim, claimed_origin, hashlib.sha256(rid.encode()).hexdigest() * 2)


def unsigned(claim: dict, origin_id: str) -> Record:
    """A record whose origin did not sign it (partial adoption). It fails signature verification."""
    return Record(content_id(claim, origin_id), claim, origin_id, "")


def relay(ks: KeyStore, rec: Record, agent_id: str) -> Record:
    new = Record(rec.record_id, dict(rec.claim), rec.origin_id, rec.origin_sig, [dict(h) for h in rec.chain])
    new.chain.append({"agent_id": agent_id, "sig": ""})
    new.chain[-1]["sig"] = ks.sign(agent_id, new.hop_payload(len(new.chain) - 1))
    return new


def verify(ks: KeyStore, registry: set[str], rec: Record) -> dict:
    origin_ok = ks.check(rec.origin_id, rec.origin_payload(), rec.origin_sig)
    chain_ok = all(ks.check(h["agent_id"], rec.hop_payload(i), h["sig"]) for i, h in enumerate(rec.chain))
    return {
        "record_id": rec.record_id,
        "signature_valid": origin_ok and chain_ok,
        "origin": rec.origin_id,
        "origin_registered": rec.origin_id in registry,
        "first_hand": origin_ok and chain_ok and rec.origin_id in registry,
    }


def count_independent_origins(ks: KeyStore, registry: set[str], recs: list[Record]) -> dict:
    """The 'independent-origin counter': distinct verified, registered origins per answer."""
    seen: dict[str, set[str]] = {"A": set(), "B": set()}
    unverified: dict[str, int] = {"A": 0, "B": 0}
    for r in recs:
        v = verify(ks, registry, r)
        side = r.claim["points_to"]
        if v["first_hand"]:
            seen[side].add(r.origin_id)
        else:
            unverified[side] += 1
    return {
        "independent_first_hand_origins": {k: len(v) for k, v in seen.items()},
        "unverified_or_unregistered_records": unverified,
    }
