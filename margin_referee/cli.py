"""margin-referee: should you switch models? Ask your own eval set.

    margin-referee fit results.json
    margin-referee replay cases.jsonl --baseline openai/gpt-4.1 \
        --candidate openai/gpt-6-luna --cap 1

``fit`` is free and offline. ``replay`` calls models through OpenRouter with your own key
(``OPENROUTER_API_KEY``), or a model in your own account with ``--candidate-endpoint`` /
``--baseline-endpoint`` (a JSON object: Bedrock, Vertex AI, Azure OpenAI or any
OpenAI-compatible endpoint, with your per-token price).
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
    res = replay.run(parsed.cases, endpoints.chat_for(base), endpoints.chat_for(cand),
                     cap_usd=a.cap, candidate_system=a.candidate_system)
    prompt = a.candidate_system
    label = cand.label + ("+system:" + hashlib.sha256(prompt.encode()).hexdigest()[:10]
                          if prompt else "")
    tried = replay.record_and_count(parsed.cases, base.label, label)
    v = replay.verdict(res, tried)
    v.update(baseline=base.label, candidate=label, ungraded=len(parsed.ungraded),
             baseline_endpoint=endpoints.describe(base),
             candidate_endpoint=endpoints.describe(cand))
    print(json.dumps(v, indent=1))
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
    r.add_argument("--candidate-endpoint", help="JSON: a model in your own account")
    r.add_argument("--baseline-endpoint", help="JSON: a model in your own account")
    a = ap.parse_args(argv)
    return _fit(a) if a.cmd == "fit" else _replay(a)


if __name__ == "__main__":
    sys.exit(main())
