"""The verdict on an NVIDIA ontology-sql-eval run you already made: should the ontology layer ship?

ontology-sql-eval (NVIDIA, Apache-2.0) runs a text-to-SQL question set through two arms, a
schema-only control and the ontology-grounded agent, scores both with one execution-match
verifier and prints the difference. It does not say whether the difference is real, or what the
second arm cost. This reads the files the run wrote and answers both, without calling a model:

    margin-referee ontology-sql runs
    margin-referee ontology-sql runs/full_20261126 --price-in 5 --price-out 25

File layout and field names were read from NVIDIA/ontology-sql-eval at commit 871bb924
(2026-11-26), through the GitHub API; nothing of theirs was installed or run:

* ``main.py`` writes ``<run-dir>/control/<dataset>.jsonl`` and
  ``<run-dir>/treatment/<dataset>.jsonl``, one row per question per repeat.
* Beside each, NeMo Gym writes ``<dataset>_materialized_inputs.jsonl`` (every question it
  dispatched), ``<dataset>_failures.jsonl`` and ``<dataset>_aggregate_metrics.json``.
* A row carries ``task_id``, ``reward`` (1.0 or 0.0), ``failure_reason``, ``gold_result``,
  ``difficulty``, ``db_id`` and ``dataset`` (``resources_servers/*/app.py``,
  ``ontology_sql_eval/gym/compare.py``).

The rules, which are the referee's and deliberately not NVIDIA's:

* A pass is the verifier's positive verdict only: a reward above zero, with no failure reason
  and a gold query that ran.
* A fail is a wrong answer the verifier determined: ``wrong_result``, ``execution_error``,
  ``execution_timeout`` or ``no_sql_extracted``.
* Everything else is unanswered, with its reason, and never a pass or a fail: a gold query that
  did not run, a broken run (``no_model_output``, ``unknown_error``), a row with no verdict, a
  question one arm has no row for. An unanswered question blocks a promotion. NVIDIA scores
  some of these as misses, so its accuracy and this one differ on the same run.
* With repeats, a question passes only if every repeat passed.
* Tokens come from the ``ng_model_call_capture`` block Gym attaches to each row when the run had
  capture on (every call the arm made), else from the treatment row's ``agent_*`` totals.
  ponytail: the block's shape is from NeMo Gym's own doc page, not from a real run's file; the
  raw ``model_call_capture/<rollout>.capture.jsonl`` files are not read, since the block is
  their summary.
* Cost is tokens times the ``--price-in`` / ``--price-out`` you give, per million tokens, one
  rate for every call, cached input priced as fresh. Without both prices cost is not recorded:
  the files carry no price and none is guessed.
* An added layer that is not shown to be cheaper has to earn its place: when the pass rate
  merely holds, the verdict is ``not_enough_evidence``. A proven gain reads
  ``ship_if_worth_the_cost``. The 0.97 bar and the case floor are the referee's, unchanged.
"""
from __future__ import annotations

import json
import math
import pathlib
from typing import Optional

from margin_referee import replay

# ontology_sql_eval/gym/exec_match.py, FailureCode, at 871bb924
_WRONG = {"wrong_result", "execution_error", "execution_timeout", "no_sql_extracted"}
_PREDICTION_FAILED = _WRONG - {"wrong_result"}
_GOLD = {"gold_execution_error", "gold_execution_timeout", "gold_failure_scored"}
_BROKEN = {"no_model_output", "unknown_error"}
_SIDECARS = ("_failures", "_materialized_inputs")
_ARM_DIRS = {"ontology": "treatment", "schema-only": "control"}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _key(row: dict) -> str:
    return str(row.get("task_id") or row.get("id"))


def _jsonl(path: pathlib.Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line) if line.strip() else None
        except ValueError:
            continue
        if isinstance(row, dict) and (row.get("task_id") or row.get("id") is not None):
            out.append(row)
    return out


def load_arm(run_dir: str, arm: str) -> dict:
    """``{"rows": {question: [row per repeat]}, "dispatched": set, "files": [..]}`` for one arm.
    ``arm`` is a folder under ``run_dir`` (``control``, ``treatment``) or a rollout file."""
    direct = pathlib.Path(arm)
    if direct.is_file():
        files = [direct]
    else:
        base = pathlib.Path(run_dir)
        folder = base / arm if (base / arm).is_dir() else base / _ARM_DIRS.get(arm, arm)
        files = sorted(p for p in folder.glob("*.jsonl") if not p.stem.endswith(_SIDECARS))
    rows: dict = {}
    dispatched: set = set()
    for f in files:
        for row in _jsonl(f):
            rows.setdefault(_key(row), []).append(row)
        sidecar = f.with_name(f.stem + "_materialized_inputs.jsonl")
        if sidecar.is_file():
            dispatched |= {_key(r) for r in _jsonl(sidecar)}
    return {"rows": rows, "dispatched": dispatched, "files": [str(f) for f in files]}


def outcome(row: dict) -> tuple[Optional[float], str]:
    """``(1.0, "")`` a pass, ``(0.0, "")`` a determined wrong answer, ``(None, why)`` unanswered."""
    reason = row.get("failure_reason")
    reason = None if reason in (None, "", "none") else str(reason)
    reward = row.get("reward")
    if reason in _GOLD:
        return None, f"the gold query did not run ({reason})"
    if reason in _BROKEN:
        return None, f"a broken run, not a wrong answer ({reason})"
    if row.get("gold_result") == "failed_gold" and reason not in _PREDICTION_FAILED:
        return None, "the gold query did not run (failed_gold)"
    if not _num(reward):
        return None, "the row has no verdict"
    if reward > 0:
        if reason is None and row.get("execution_match") is not False:
            return 1.0, ""
        return None, "the row contradicts itself: a reward with a failure"
    if reason in _WRONG:
        return 0.0, ""
    return None, ("zero reward with no failure reason" if reason is None
                  else f"an unrecognised failure reason ({reason})")


def tokens(row: dict) -> Optional[tuple[float, float, bool, set]]:
    """``(input, output, is_a_floor, models)`` for one row, or None when it carries no counts.
    Input is total minus output where the provider gave a total: Gym's ``tokens_in`` counts a
    cache write twice on some gateways (``ontology_sql_eval/gym/model_routing.py``)."""
    def inp(c):
        t, o = c.get("tokens_total"), c.get("tokens_out")
        return t - o if _num(t) and _num(o) else c.get("tokens_in")
    cap = row.get("ng_model_call_capture")
    cap = cap if isinstance(cap, dict) else {}
    calls = [c for c in cap.get("calls") or [] if isinstance(c, dict)]
    if not calls and isinstance(cap.get("metrics"), dict):
        calls = [cap["metrics"]]
    if calls:
        pairs = [(inp(c), c.get("tokens_out")) for c in calls]
        if not any(_num(i) or _num(o) for i, o in pairs):
            return None
        return (sum(i for i, _ in pairs if _num(i)), sum(o for _, o in pairs if _num(o)),
                bool(cap.get("gaps")) or any(not (_num(i) and _num(o)) for i, o in pairs),
                {str(c["model"]) for c in calls if c.get("model")})
    i, o = row.get("agent_input_tokens"), row.get("agent_output_tokens")
    if _num(i) and _num(o):
        return i, o, bool(row.get("agent_llm_calls_without_usage")), set()
    return None


def _segment(rows_by_arm: list, by: str) -> dict:
    fields = ("difficulty", "db_id") if by == "auto" else () if by == "none" else (by,)
    for f in fields:
        seg = {k: str(rs[0][f]) for rows in rows_by_arm for k, rs in rows.items() if rs[0].get(f)}
        if seg:
            return {"by": f, "of": seg}
    return {"by": None, "of": {}}


def compare(run_dir: str, *, baseline_arm: str = "control", candidate_arm: str = "treatment",
            price_in: Optional[float] = None, price_out: Optional[float] = None,
            candidates: int = 1, segment_by: str = "auto", critical: tuple = ()) -> dict:
    if (price_in is None) != (price_out is None):
        raise ValueError("give both --price-in and --price-out, or neither")
    arms = {"baseline": load_arm(run_dir, baseline_arm),
            "candidate": load_arm(run_dir, candidate_arm)}
    res = replay.Result()
    used = {s: {"in": 0.0, "out": 0.0, "floor": False, "models": set()} for s in arms}
    counted = True
    for k in sorted(set().union(*(a["rows"].keys() | a["dispatched"] for a in arms.values()))):
        score, why = {}, None
        for side, a in arms.items():
            rows = a["rows"].get(k)
            if not rows:
                why = (side, "no result for this question (it failed to run or never ran)"
                       if k in a["dispatched"] else "no row in this arm")
                break
            results = [outcome(r) for r in rows]
            bad = [w for s, w in results if s is None]
            if bad:
                why = (side, bad[0])
                break
            score[side] = min(s for s, _ in results)
        if why:
            res.unanswered.append((k, *why))
            continue
        res.pairs.append((k, score["baseline"], score["candidate"]))
        for side, a in arms.items():
            got = [tokens(r) for r in a["rows"][k]]
            if any(t is None for t in got):
                counted = False
                continue
            u = used[side]
            u["in"] += sum(t[0] for t in got) / len(got)
            u["out"] += sum(t[1] for t in got) / len(got)
            u["floor"] = u["floor"] or any(t[2] for t in got)
            u["models"] |= set().union(*(t[3] for t in got))
    n = len(res.pairs)
    priced = counted and price_in is not None and n > 0
    if priced:
        for side, u in used.items():
            res.cost[side] = (u["in"] * price_in + u["out"] * price_out) / 1e6
    seg = _segment([a["rows"] for a in arms.values()], segment_by)
    v = replay.verdict(res, candidates, seg["of"], tuple(critical))
    if not priced:
        v["baseline_cost_per_case"] = v["candidate_cost_per_case"] = None
        v["promoted"], v["cost"] = False, "not_recorded"
    per = {s: ({"input_per_case": round(u["in"] / n, 1), "output_per_case": round(u["out"] / n, 1),
                "is_a_floor": u["floor"], "models_seen": sorted(u["models"])}
               if counted and n else None) for s, u in used.items()}
    total = {s: (u["in"] + u["out"]) for s, u in used.items()}
    ratio = (round(total["candidate"] / total["baseline"], 2)
             if counted and n and total["baseline"] > 0 else None)
    gap = v.get("quality_gap") or {}
    low = gap.get("worst_case_points")
    proven = bool(low is not None and low > 0 and not res.unanswered)
    v["gain"] = {"won": v["won"], "lost": v["lost"], "estimate_points": gap.get("estimate_points"),
                 "lower_bound_points": low, "proven": proven,
                 "line": (f"the candidate won {v['won']}, lost {v['lost']}; the 95% bound on the "
                          f"gain is {low:+.2f} points" if low is not None else
                          "no question ran on both arms")}
    fewer = ratio is not None and ratio < 1
    unpriced = v["cost"] == "not_recorded"
    if v["quality"] == "holds" and unpriced and fewer:
        v["decision"] = "ship"  # fewer tokens, no price: quality held, and no saving is claimed
    elif v["quality"] == "holds" and (unpriced or v["cost"] == "dearer"):
        # an added layer that is not shown to be cheaper ships on a proven gain, never on a tie
        spend = (f"uses {ratio}x the tokens per question" if ratio is not None
                 else "has no token counts in these files")
        if proven:
            v["decision"] = "ship_if_worth_the_cost"
            v["why"] = (f"a proven gain: {v['gain']['line']}. The candidate arm "
                        + ("costs more per question" if v["cost"] == "dearer" else spend)
                        + ", so whether the gain is worth it is yours to decide")
        else:
            v["decision"] = "not_enough_evidence"
            v["why"] = (f"the pass rate holds but no gain is proven ({v['gain']['line']}), and "
                        f"the candidate arm {spend}: an added layer that is not cheaper is "
                        "worth shipping only on a proven gain")
    v.update(baseline=baseline_arm, candidate=candidate_arm, spend_usd=0.0, tokens=per,
             token_ratio=ratio, segment_by=seg["by"],
             cost_basis=("tokens times the prices given, one rate for every call, cached input "
                         "priced as fresh" if priced else
                         "not recorded: these files carry no price; pass --price-in and "
                         "--price-out (USD per million tokens) to compute one"
                         if counted and n else
                         "not recorded: a paired question has no token counts in these files"),
             files_read=arms["baseline"]["files"] + arms["candidate"]["files"])
    if len(set().union(*(u["models"] for u in used.values()))) > 1 and priced:
        v["cost_basis"] += "; more than one model was called, so one rate is an approximation"
    return v
