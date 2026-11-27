"""The verdict on a Hurdle suite you already ran: baseline vs a condition, or one model vs another.

Hurdle (Together AI, open source) runs Claude Code or Codex on your tasks under conditions (say
``baseline`` and ``with-skill``) and repeats each, then shows pass rates. It does not say whether
the difference is real. This reads the runs it saved and answers that, without running anything:

    margin-referee hurdle .hurdle --compare condition --baseline baseline --candidate with-skill
    margin-referee hurdle .hurdle --compare model \
        --baseline claude-opus-4-8 --candidate claude-sonnet-4-6

Read from Hurdle 0.5's local result store: ``<root>/runs/run_*/revisions/NNNNNN.json``, the
highest revision being the run's record. Only ``state == "complete"`` records count.

* A case is one task (with its target), compared within the same agent runner and, for a
  condition comparison, the same model; for a model comparison, the same condition.
* Repeats: a case passes only if every repeat passed (the strict pass^k), which is what "it
  works" means for an agent you ship. pass@1 (the share of repeats that passed) is reported too.
* A run whose verdict is ``incomplete`` or ``unknown`` leaves its case unanswered, never failed
  and never passed, and an unanswered case blocks a promotion.
* Cost: read from a numeric ``costUsd`` / ``cost_usd`` in the run's ``metrics.common`` when
  present (ponytail: the key name is unverified against a real Hurdle run). Without it no saving
  is claimed.
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Optional

from margin_referee import replay

_COST_KEYS = ("costUsd", "cost_usd", "totalCostUsd", "agentCostUsd")


def load_runs(root: str) -> list[dict]:
    """The latest complete record of every run under a Hurdle result-store root."""
    base = pathlib.Path(root)
    runs_dir = base / "runs" if (base / "runs").is_dir() else base
    out = []
    for run in sorted(runs_dir.glob("run_*")):
        revs = sorted((run / "revisions").glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"))
        if not revs:
            continue
        try:
            rec = json.loads(revs[-1].read_text())
        except (OSError, ValueError):
            continue
        if (isinstance(rec, dict) and rec.get("state") == "complete"
                and isinstance(rec.get("snapshot"), dict)):
            out.append(rec)
    return out


def _model(rec: dict) -> str:
    agent = (rec.get("snapshot") or {}).get("agent") or {}
    execution = rec.get("execution") or {}
    return str(agent.get("requestedModel") or execution.get("observedModel") or "default")


def _cost(rec: dict) -> Optional[float]:
    common = (rec.get("metrics") or {}).get("common") or {}
    for k in _COST_KEYS:
        v = common.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0:
            return float(v)
    return None


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _arm(rec: dict, compare: str) -> str:
    snap = _d(rec.get("snapshot"))
    return str(_d(snap.get("condition")).get("id")) if compare == "condition" else _model(rec)


def _case(rec: dict, compare: str) -> str:
    snap = _d(rec.get("snapshot"))
    held = _model(rec) if compare == "condition" else str(_d(snap.get("condition")).get("id"))
    return "/".join([str(_d(snap.get("target")).get("id")),
                     str(_d(snap.get("task")).get("id")),
                     str(_d(snap.get("agent")).get("adapterId")), held])


def tally(runs: list[dict], compare: str, arm: str) -> dict:
    """``{case: {"passed": n, "failed": n, "undecided": n, "cost": [..]}}`` for one arm."""
    out: dict = {}
    for rec in runs:
        if _arm(rec, compare) != arm:
            continue
        t = out.setdefault(_case(rec, compare), {"passed": 0, "failed": 0, "undecided": 0,
                                                 "cost": []})
        verdict = rec.get("verdict")
        t["passed" if verdict == "pass" else "failed" if verdict == "fail" else "undecided"] += 1
        t["cost"].append(_cost(rec))
    return out


def compare(runs: list[dict], *, by: str, baseline: str, candidate: str,
            candidates: int = 1) -> dict:
    if by not in ("condition", "model"):
        raise ValueError("compare is condition or model")
    b, c = tally(runs, by, baseline), tally(runs, by, candidate)
    res = replay.Result()
    costed = True
    reliability = {}
    for k in sorted(b.keys() | c.keys()):
        if k not in b or k not in c:
            res.unanswered.append((k, "baseline" if k not in b else "candidate", "no runs"))
            continue
        if b[k]["undecided"] or c[k]["undecided"]:
            res.unanswered.append((k, "both", "a run was incomplete or unknown"))
            continue
        strict = {s: 1.0 if t["failed"] == 0 else 0.0 for s, t in (("baseline", b[k]),
                                                                   ("candidate", c[k]))}
        res.pairs.append((k, strict["baseline"], strict["candidate"]))
        reliability[k] = {s: f"{t['passed']}/{t['passed'] + t['failed']}"
                          for s, t in (("baseline", b[k]), ("candidate", c[k]))}
        costs = b[k]["cost"] + c[k]["cost"]
        if any(x is None for x in costs):
            costed = False
        else:
            res.cost["baseline"] += sum(b[k]["cost"]) / len(b[k]["cost"])
            res.cost["candidate"] += sum(c[k]["cost"]) / len(c[k]["cost"])
    v = replay.verdict(res, candidates)
    if not costed and res.pairs:
        v["baseline_cost_per_case"] = v["candidate_cost_per_case"] = None
        v["promoted"], v["cost"] = False, "unknown"
        if v["decision"] == "ship_if_worth_the_cost":
            v["decision"] = "ship"
    v.update(compare=by, baseline=baseline, candidate=candidate, spend_usd=0.0,
             runs_read=len(runs), repeats_passed=reliability,
             arms_seen=sorted({_arm(r, by) for r in runs}))
    return v


def _demo() -> None:
    def rec(task, cond, model, verdict, i):
        return {"state": "complete", "verdict": verdict, "id": f"run_{task}{cond}{model}{i}",
                "snapshot": {"target": {"id": "api"}, "task": {"id": task},
                             "condition": {"id": cond},
                             "agent": {"adapterId": "claude-code", "requestedModel": model}}}
    runs = [rec(f"t{t}", cond, "m", "pass", i) for t in range(120)
            for cond in ("baseline", "with-skill") for i in range(3)]
    v = compare(runs, by="condition", baseline="baseline", candidate="with-skill")
    assert v["quality"] == "holds" and v["cost"] == "unknown" and not v["promoted"], v
    runs[1]["verdict"] = "fail"  # one repeat of t0 on baseline fails: t0 fails strictly
    v = compare(runs, by="condition", baseline="baseline", candidate="with-skill")
    assert v["repeats_passed"]["api/t0/claude-code/m"]["baseline"] == "2/3"
    runs[2]["verdict"] = "incomplete"
    assert compare(runs, by="condition", baseline="baseline",
                   candidate="with-skill")["unanswered"] == 1


if __name__ == "__main__":
    _demo()
