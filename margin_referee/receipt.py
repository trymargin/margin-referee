"""A verdict as a receipt: an in-toto Statement v1 in a DSSE envelope.

    margin-referee replay cases.jsonl ... --receipt receipt.json   # your own verdict, UNSIGNED
    margin-referee verify receipt.json --key <base64 Ed25519 public key>

A receipt you write yourself is unsigned and says ``assuranceLevel: self-run``: it is your own
claim in a format others can read. A receipt Margin issues is signed with Ed25519; ``verify``
checks that signature against a public key you supply. A valid signature shows who issued the
record and that it was not edited since. It is not a certification, and it does not show that the
grader is fit for purpose, that the cases cover production, or what a person decided to ship.

Building and writing a receipt needs only the standard library. Verifying a signature needs the
``cryptography`` package (``pip install 'margin-referee[verify]'``).
"""
from __future__ import annotations

import base64
import hashlib
import json
from typing import Optional

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = "https://trymargin.io/verdict-receipt/v1"
PAYLOAD_TYPE = "application/vnd.in-toto+json"
#: The bar the verdict was judged against: the candidate holds 97% of the baseline's pass rate.
PARITY_BAR = 0.97
NON_CLAIMS = (
    "This receipt does not prove the grader is fit for purpose: each case was graded by the "
    "team's own check.",
    "It does not prove the suite covers the situations the change will meet in production.",
    "It does not record or replace the human decision to ship.",
)


class ReceiptError(ValueError):
    """The receipt cannot be read or does not verify."""


class MissingDependency(ReceiptError):
    """``cryptography`` is not installed, so no signature can be checked."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def statement(v: dict, *, baseline: str, candidate: str, issuer: str, assurance_level: str,
              created_at: str, verdict_id: Optional[str] = None,
              cases_digest: Optional[str] = None) -> dict:
    """The verdict ``v`` as an in-toto Statement. Every figure is read from ``v``; a figure the
    verdict does not carry is null, never a default."""
    gap = v.get("quality_gap") or {}
    return {
        "_type": STATEMENT_TYPE,
        "subject": [{"name": "baseline", "digest": {"sha256": _sha256(baseline)}},
                    {"name": "candidate", "digest": {"sha256": _sha256(candidate)}}],
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "verdictId": verdict_id,
            "decision": v.get("decision"),
            "why": v.get("why"),
            "decisionRule": {
                "metric": "pass rate on the same cases, candidate against baseline",
                "bar": PARITY_BAR,
                "direction": "the candidate must hold at least the bar's share of the "
                             "baseline's pass rate; the bar rises with each candidate tried",
            },
            "setups": {"baseline": baseline, "candidate": candidate},
            "paired": {
                "cases": v.get("cases"), "won": v.get("won"), "lost": v.get("lost"),
                "baselinePassed": v.get("baseline_passed"),
                "candidatePassed": v.get("candidate_passed"),
                "qualityGap": {"estimatePoints": gap.get("estimate_points"),
                               "worstCasePoints": gap.get("worst_case_points"),
                               "barAllowsPoints": gap.get("bar_allows_points")},
            },
            "candidatesTried": v.get("candidates_tried"),
            "costPerCase": {"baseline": v.get("baseline_cost_per_case"),
                            "candidate": v.get("candidate_cost_per_case")},
            "segments": v.get("segments") or [],
            "worseCases": v.get("worse_cases") or [],
            "casesDigest": cases_digest,
            "issuer": issuer,
            "createdAt": created_at,
            "assuranceLevel": assurance_level,
            "nonClaims": list(NON_CLAIMS),
        },
    }


def payload_bytes(st: dict) -> bytes:
    return json.dumps(st, sort_keys=True, separators=(",", ":")).encode()


def pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE pre-authentication encoding: what is actually signed."""
    t = payload_type.encode()
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(payload), payload)


def envelope(st: dict, signatures: tuple = ()) -> dict:
    """A DSSE envelope. With no ``signatures`` it is unsigned, and a reader can see that."""
    return {"payloadType": PAYLOAD_TYPE,
            "payload": base64.b64encode(payload_bytes(st)).decode(),
            "signatures": list(signatures)}


def _parts(doc) -> tuple[dict, bytes]:
    """The envelope and its payload bytes. Accepts the envelope, or a saved Margin response
    that carries it under ``receipt``."""
    env = doc.get("receipt", doc) if isinstance(doc, dict) else None
    if not isinstance(env, dict) or not isinstance(env.get("payload"), str):
        raise ReceiptError("not a receipt: expected a DSSE envelope with a payload")
    if env.get("payloadType") != PAYLOAD_TYPE:
        raise ReceiptError(f"payloadType is {env.get('payloadType')!r}, expected {PAYLOAD_TYPE}")
    try:
        return env, base64.b64decode(env["payload"], validate=True)
    except ValueError:
        raise ReceiptError("the payload is not base64") from None


def decode(doc) -> dict:
    """The statement inside, WITHOUT checking any signature."""
    try:
        return json.loads(_parts(doc)[1])
    except json.JSONDecodeError:
        raise ReceiptError("the payload is not JSON") from None


def is_signed(doc) -> bool:
    return bool(_parts(doc)[0].get("signatures"))


def verify(doc, public_key_b64: str) -> dict:
    """The statement, once one signature over it verifies against the Ed25519 public key.
    ``ReceiptError`` otherwise: unsigned, edited, or signed by another key."""
    env, payload = _parts(doc)
    sigs = env.get("signatures")
    if not isinstance(sigs, list) or not sigs:
        raise ReceiptError("unsigned: this receipt carries no signature")
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        raise MissingDependency("checking a signature needs the cryptography package: "
                                "pip install 'margin-referee[verify]'") from None
    try:
        key = Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64, validate=True))
    except ValueError:
        raise ReceiptError("the key is not a base64 Ed25519 public key (32 bytes)") from None
    signed = pae(PAYLOAD_TYPE, payload)
    for s in sigs:
        try:
            key.verify(base64.b64decode(s["sig"], validate=True), signed)
        except (InvalidSignature, ValueError, KeyError, TypeError):
            continue
        return decode(env)
    raise ReceiptError("the signature does not match: the receipt was edited, or this is not "
                       "the key that signed it")


def _demo() -> None:
    st = statement({"decision": "ship", "cases": 120, "lost": 0, "won": 1}, baseline="a",
                   candidate="b", issuer="self", assurance_level="self-run",
                   created_at="2026-11-26T00:00:00+00:00")
    env = envelope(st)
    assert decode(env) == st and not is_signed(env)
    assert pae("t", b"xy") == b"DSSEv1 1 t 2 xy"
    try:
        verify(env, "")
        raise AssertionError("an unsigned receipt verified")
    except ReceiptError:
        pass


if __name__ == "__main__":
    _demo()
    print("ok")
