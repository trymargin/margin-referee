"""margin-referee: can you ship this AI change? Ask your own eval cases.

    margin-referee fit results.json
    margin-referee replay cases.jsonl --baseline openai/gpt-4.1 \
        --candidate openai/gpt-6-luna --cap 1

``fit`` is free and offline. ``replay`` calls models through OpenRouter with your own key
(``OPENROUTER_API_KEY``), or a model in your own account with ``--candidate-endpoint`` /
``--baseline-endpoint`` (a JSON object: Bedrock, Vertex AI, Azure OpenAI or any
OpenAI-compatible endpoint, with your per-token price). ``--candidate-system`` tests a prompt
change and ``--candidate-tools FILE.json`` a tool change.

``replay``, ``check``, ``hurdle`` and ``ontology-sql`` take ``--receipt out.json`` to write the
verdict as an UNSIGNED receipt (your own claim, ``assuranceLevel: self-run``). ``verify`` checks
the signature on a receipt Margin issued.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

from margin_referee import cases as cases_mod
from margin_referee import endpoints, replay, stats


def _write_receipt(a, v: dict, baseline: str, candidate: str, cases_digest=None) -> None:
    """``--receipt``: this verdict as an UNSIGNED receipt. Nobody signed it and it says so."""
    if not getattr(a, "receipt", None):
        return
    from datetime import datetime, timezone
    from margin_referee import receipt
    st = receipt.statement(v, baseline=baseline, candidate=candidate, issuer="self",
                           assurance_level="self-run", cases_digest=cases_digest,
                           created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    pathlib.Path(a.receipt).write_text(json.dumps(receipt.envelope(st), indent=1) + "\n")
    print(f"wrote an UNSIGNED receipt to {a.receipt} (self-run: your own claim, signed by "
          "nobody)", file=sys.stderr)


class _Usage(Exception):
    """``verify`` could not check anything: no key, or the key could not be fetched."""


def _verify(a) -> int:
    """0 when the signature verifies, 1 when it does not (or there is none), 2 when it cannot
    be checked."""
    from margin_referee import receipt
    try:
        doc = json.loads(pathlib.Path(a.file).read_text())
        if not receipt.is_signed(doc):
            print(json.dumps({"verified": False, "signed": False,
                              "reason": "UNSIGNED: nobody signed this receipt; it is its "
                                        "producer's own claim",
                              "statement": receipt.decode(doc)}, indent=1))
            return 1
        key = a.key
        if not key and a.key_url:
            if not a.key_url.startswith("https://"):
                raise _Usage("--key-url must be https")
            import urllib.request
            with urllib.request.urlopen(a.key_url, timeout=15) as resp:  # noqa: S310 (https)
                key = (json.loads(resp.read()) or {}).get("public_key")
            if not key:
                raise _Usage(f"{a.key_url} published no public_key")
        if not key:
            raise _Usage("this receipt is signed: pass --key or --key-url to check it")
        st = receipt.verify(doc, key)
    except (receipt.MissingDependency, _Usage, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (receipt.ReceiptError, json.JSONDecodeError) as exc:
        print(json.dumps({"verified": False, "reason": str(exc)}, indent=1))
        return 1
    print(json.dumps({"verified": True, "signed": True,
                      "meaning": "signed by the holder of this key and not edited since; "
                                 "read nonClaims for what it does not show",
                      "statement": st}, indent=1))
    return 0


def _fit(a) -> int:
    parsed = cases_mod.parse(pathlib.Path(a.file).read_text())
    n = len(parsed.cases)
    need = stats.smallest_suite(a.baseline_rate, candidates=a.candidates)
    out = {"graded": n, "ungraded": len(parsed.ungraded), "needed": need,
           "ready": n >= need,
           "losses_absorbed": stats.losses_absorbed(n, a.baseline_rate, a.candidates),
           "ungraded_examples": parsed.ungraded[:5]}
    print(json.dumps(out, indent=1))
    return 0 if out["ready"] else 2


def _endpoint(raw, model: str):
    if raw:
        return endpoints.parse(json.loads(raw))
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY, or pass an endpoint for a model in your account")
    return endpoints.parse({"kind": "openrouter", "model": model, "api_key": key})


def _replay(a) -> int:
    parsed = cases_mod.parse(pathlib.Path(a.file).read_text())
    if not parsed.cases:
        print(json.dumps({"error": "no case can be graded", "ungraded": parsed.ungraded[:5]}))
        return 2
    base = _endpoint(a.baseline_endpoint, a.baseline)
    cand = _endpoint(a.candidate_endpoint, a.candidate)
    prompt = a.candidate_system
    label = cand.label + ("+system:" + hashlib.sha256(prompt.encode()).hexdigest()[:10]
                          if prompt else "")
    tools = None
    if a.candidate_tools:
        tools = cases_mod.tool_list(json.loads(pathlib.Path(a.candidate_tools).read_text()))
        if tools is None:
            raise SystemExit("--candidate-tools is a JSON list of function tool definitions: "
                             '[{"type": "function", "function": {"name": ...}}]')
        label += "+tools:" + hashlib.sha256(
            json.dumps(tools, sort_keys=True).encode()).hexdigest()[:10]
    # counted before the run: an early stop is corrected for every candidate tried
    tried = replay.record_and_count(parsed.cases, base.label, label)
    res = replay.run(parsed.cases, endpoints.chat_for(base), endpoints.chat_for(cand),
                     cap_usd=a.cap, candidate_system=a.candidate_system, repeats=a.repeats,
                     early_stop=not a.no_early_stop, candidates=tried, candidate_tools=tools)
    v = replay.verdict(res, tried, {c.id: c.segment for c in parsed.cases},
                       tuple(a.critical or ()), {c.id: c.weight for c in parsed.cases})
    v.update(baseline=base.label, candidate=label, ungraded=len(parsed.ungraded),
             baseline_endpoint=endpoints.describe(base),
             candidate_endpoint=endpoints.describe(cand))
    print(json.dumps(v, indent=1))
    _write_receipt(a, v, base.label, label, hashlib.sha256(
        " ".join(sorted(replay.case_fps(parsed.cases))).encode()).hexdigest()[:16])
    return 0 if v["promoted"] else 1


def _check(a) -> int:
    from margin_referee import experiments
    v = experiments.compare(experiments.load(a.baseline), experiments.load(a.candidate),
                            scorer=a.scorer, threshold=a.threshold, candidates=a.candidates,
                            critical=tuple(a.critical or ()))
    v.update(baseline=a.baseline, candidate=a.candidate)
    print(json.dumps(v, indent=1))
    _write_receipt(a, v, a.baseline, a.candidate)
    return 0 if v["promoted"] else 1


def _hurdle(a) -> int:
    from margin_referee import hurdle
    runs = hurdle.load_runs(a.root)
    if not runs:
        print(json.dumps({"error": f"no complete Hurdle runs under {a.root}"}))
        return 2
    v = hurdle.compare(runs, by=a.compare, baseline=a.baseline, candidate=a.candidate,
                       candidates=a.candidates)
    print(json.dumps(v, indent=1))
    _write_receipt(a, v, a.baseline, a.candidate)
    return 0 if v["promoted"] else 1


def _ontology_sql(a) -> int:
    from margin_referee import ontology_sql
    try:
        v = ontology_sql.compare(a.run_dir, baseline_arm=a.baseline_arm,
                                 candidate_arm=a.candidate_arm, price_in=a.price_in,
                                 price_out=a.price_out, candidates=a.candidates,
                                 segment_by=a.segment_by, critical=tuple(a.critical or ()))
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not v["files_read"] or not (v["cases"] or v["unanswered"]):
        print(json.dumps({"error": f"no rollout rows for {a.baseline_arm} and {a.candidate_arm} "
                                   f"under {a.run_dir}"}))
        return 2
    if a.json:
        print(json.dumps(v, indent=1))
    else:
        t = v["tokens"]
        print("\n".join([
            f"{v['decision']}: {v['why']}",
            f"  {a.baseline_arm} passed {v['baseline_passed']} of {v['cases']}, "
            f"{a.candidate_arm} passed {v['candidate_passed']}; {v['gain']['line']}",
            f"  unanswered: {v['unanswered']}" + "".join(
                f"\n    {r['cases']} x {r['reason']}" for r in v["unanswered_reasons"]),
            "  tokens per question: " + (
                f"{a.baseline_arm} {t['baseline']['input_per_case']:.0f} in / "
                f"{t['baseline']['output_per_case']:.0f} out, {a.candidate_arm} "
                f"{t['candidate']['input_per_case']:.0f} in / "
                f"{t['candidate']['output_per_case']:.0f} out ({v['token_ratio']}x)"
                if t["baseline"] and t["candidate"] else "not in these files"),
            "  cost per question: " + (
                f"{a.baseline_arm} ${v['baseline_cost_per_case']:.5f}, {a.candidate_arm} "
                f"${v['candidate_cost_per_case']:.5f} ({v['cost_basis']})"
                if v["baseline_cost_per_case"] is not None else v["cost_basis"]),
            *(f"  {s['segment']}: {s['cases']} questions, won {s['won']}, lost {s['lost']}"
              + (", worse" if s["worse"] else "") for s in v["segments"])]))
    _write_receipt(a, v, a.baseline_arm, a.candidate_arm)
    return 0 if v["promoted"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="margin-referee", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="is this eval set big enough to prove a switch? (free)")
    f.add_argument("file")
    f.add_argument("--baseline-rate", type=float, default=stats.DEFAULT_BASELINE_RATE)
    f.add_argument("--candidates", type=int, default=1)
    r = sub.add_parser("replay", help="run both models on the cases and get the verdict")
    r.add_argument("file")
    r.add_argument("--baseline", default="")
    r.add_argument("--candidate", default="")
    r.add_argument("--cap", type=float, required=True, help="the most this replay may spend, USD")
    r.add_argument("--candidate-system", help="test a prompt change: the candidate's system prompt")
    r.add_argument("--candidate-tools", metavar="FILE.json",
                   help="test a tool change: a JSON list of function tools that replaces each "
                        "case's tools for the candidate only")
    r.add_argument("--candidate-endpoint", help="JSON: a model in your own account")
    r.add_argument("--baseline-endpoint", help="JSON: a model in your own account")
    r.add_argument("--critical", action="append", metavar="SEGMENT",
                   help="a segment that is a release criterion (repeatable)")
    r.add_argument("--repeats", type=int, default=1, choices=range(1, 6), metavar="1-5",
                   help="run each case this many times per side; a case passes only if every "
                        "run passed (pass^k). Spend scales with it")
    r.add_argument("--no-early-stop", action="store_true",
                   help="run every case even once the answer is already do_not_ship")
    c = sub.add_parser("check", help="a second opinion on two experiments you already scored "
                       "(Braintrust export or bt:<experiment id>); runs nothing, spends nothing")
    c.add_argument("baseline", help="the current model's experiment")
    c.add_argument("candidate", help="the candidate's experiment")
    c.add_argument("--scorer", required=True, help="the score that decides pass or fail")
    c.add_argument("--threshold", type=float, default=1.0, help="a pass is a score >= this")
    c.add_argument("--candidates", type=int, default=1,
                   help="how many models were tried on these cases (Loop often tries several)")
    c.add_argument("--critical", action="append", metavar="SEGMENT",
                   help="a segment that is a release criterion (repeatable)")
    h = sub.add_parser("hurdle", help="the verdict on a Hurdle suite you already ran "
                       "(baseline vs a condition, or one model vs another); runs nothing")
    h.add_argument("root", help="Hurdle's result store, usually .hurdle")
    h.add_argument("--compare", choices=("condition", "model"), required=True)
    h.add_argument("--baseline", required=True, help="the condition id or model you have now")
    h.add_argument("--candidate", required=True, help="the condition id or model to test")
    h.add_argument("--candidates", type=int, default=1,
                   help="how many conditions or models were tried on these tasks")
    o = sub.add_parser("ontology-sql", help="the verdict on an NVIDIA ontology-sql-eval run you "
                       "already made (schema-only control vs the ontology arm); runs nothing")
    o.add_argument("run_dir", help="the run's output folder, usually runs or your --run-dir")
    o.add_argument("--baseline-arm", default="control",
                   help="the arm you have now: a folder under the run dir, or a rollout file")
    o.add_argument("--candidate-arm", default="treatment",
                   help="the arm to judge (treatment is the ontology arm; ontology also works)")
    o.add_argument("--price-in", type=float, help="USD per million input tokens")
    o.add_argument("--price-out", type=float, help="USD per million output tokens")
    o.add_argument("--segment-by", default="auto",
                   choices=("auto", "difficulty", "db_id", "dataset", "none"),
                   help="the row field to read each segment on (auto: difficulty, else db_id)")
    o.add_argument("--critical", action="append", metavar="SEGMENT",
                   help="a segment that is a release criterion (repeatable)")
    o.add_argument("--candidates", type=int, default=1,
                   help="how many setups were tried on these questions")
    o.add_argument("--json", action="store_true", help="print the whole verdict as JSON")
    for p in (r, c, h, o):
        p.add_argument("--receipt", metavar="OUT.json",
                       help="also write the verdict as an UNSIGNED receipt (in-toto Statement in "
                            "a DSSE envelope, assuranceLevel self-run)")
    vf = sub.add_parser("verify", help="check the signature on a verdict receipt and print the "
                        "statement inside")
    vf.add_argument("file", help="the receipt (a DSSE envelope, or a saved Margin response)")
    key = vf.add_mutually_exclusive_group()
    key.add_argument("--key", help="the issuer's Ed25519 public key, base64")
    key.add_argument("--key-url", help="an https URL publishing it as JSON {public_key}")
    a = ap.parse_args(argv)
    return {"fit": _fit, "replay": _replay, "check": _check, "hurdle": _hurdle,
            "ontology-sql": _ontology_sql, "verify": _verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
