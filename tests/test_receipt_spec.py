"""The published JSON Schema and the statement the referee builds agree (the open spec)."""
import json
import pathlib

from margin_referee import receipt

SPEC = pathlib.Path(__file__).resolve().parents[1] / "spec" / "verdict-receipt-v1.schema.json"


def _required(schema: dict, doc: dict, where: str = "") -> list:
    """Every key the schema requires, walked into nested objects; returns what is missing."""
    missing = [where + k for k in schema.get("required", []) if k not in doc]
    for k, sub in (schema.get("properties") or {}).items():
        if isinstance(sub, dict) and sub.get("type") == "object" and isinstance(doc.get(k), dict):
            missing += _required(sub, doc[k], where + k + ".")
    return missing


def test_a_statement_carries_every_field_the_spec_requires():
    schema = json.loads(SPEC.read_text())
    v = {"decision": "ship", "why": "it held", "cases": 120, "won": 1, "lost": 0,
         "quality_gap": {"estimate_points": 0.8, "worst_case_points": -1.2,
                         "bar_allows_points": -2.7}}
    st = receipt.statement(v, baseline="a", candidate="b", issuer="self",
                           assurance_level="self-run", created_at="2026-11-26T00:00:00Z")
    assert _required(schema, st) == []
    pred = schema["properties"]["predicate"]["properties"]
    assert st["predicate"]["decision"] in pred["decision"]["enum"]
    assert st["predicate"]["assuranceLevel"] in pred["assuranceLevel"]["enum"]
    assert st["predicateType"] == schema["properties"]["predicateType"]["const"]


def test_the_spec_names_every_verdict_word_and_no_other():
    pred = json.loads(SPEC.read_text())["properties"]["predicate"]["properties"]
    assert set(pred["decision"]["enum"]) == {"ship", "ship_if_worth_the_cost", "do_not_ship",
                                             "not_enough_evidence"}
