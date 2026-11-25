"""Run both models on every case, grade each answer with the case's own check, and decide.

A case either side could not answer (an error, or an answer cut off by the output limit) is left
out and counted, never graded as a fail, and a replay with any such case promotes nothing: a
verdict on the survivors would grade the candidate only on what it finished.

Every candidate tried against the same baseline on overlapping cases is remembered in a local
ledger (``~/.margin-referee/ledger.json``), so the bar rises with each one and renaming or
trimming the cases does not reset it. A fresh set no candidate ran on is judged as one candidate:
that is how a winner is confirmed on held-out cases.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from margin_referee import stats
from margin_referee.cases import Case, grade

LEDGER = pathlib.Path(os.environ.get("MARGIN_REFEREE_LEDGER",
                                     pathlib.Path.home() / ".margin-referee" / "ledger.json"))
OVERLAP = 0.1


@dataclass
class Result:
    pairs: list = field(default_factory=list)          # (case id, baseline score, candidate score)
    cost: dict = field(default_factory=lambda: {"baseline": 0.0, "candidate": 0.0})
    unanswered: list = field(default_factory=list)     # (case id, side, why)
    stopped: str = ""
    spend_usd: float = 0.0


def with_system(messages: list, system: Optional[str]) -> list:
    if system is None:
        return messages
    rest = messages[1:] if messages and messages[0].get("role") == "system" else messages
    return [{"role": "system", "content": system}, *rest]


def _one(case: Case, side: str, chat: Callable, system: Optional[str], max_tokens: int):
    msgs = with_system(case.messages, system) if side == "candidate" else case.messages
    try:
        r = chat(msgs, max_tokens=case.max_tokens or max_tokens,
                 response_format=case.response_format)
    except Exception as exc:  # noqa: BLE001 - an unanswered call is a missing pair, named
        return None, 0.0, f"{type(exc).__name__}: {exc}"[:200]
    if r.finish_reason == "length":
        return None, r.cost_usd, "cut off by the output limit"
    try:
        return (1.0 if grade(r.text, case) else 0.0), r.cost_usd, None
    except Exception as exc:  # noqa: BLE001 - a check that cannot run grades nothing
        return None, r.cost_usd, f"check failed: {type(exc).__name__}"


def run(cases: list, baseline_chat: Callable, candidate_chat: Callable, *, cap_usd: float,
        candidate_system: Optional[str] = None, max_tokens: int = 8192,
        workers: int = 8) -> Result:
    """Both sides on every case until the next batch could cross ``cap_usd``."""
    out, worst, i = Result(), 0.0, 0

    def pair(case):
        b = _one(case, "baseline", baseline_chat, None, max_tokens)
        c = _one(case, "candidate", candidate_chat, candidate_system, max_tokens)
        return case, b, c
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while i < len(cases):
            room = cap_usd - out.spend_usd
            size = workers if worst == 0 else min(workers, int(room // worst))
            if size < 1 or room <= 0:
                out.stopped = f"stopped at the ${cap_usd:.2f} cap after {i} cases"
                break
            batch = cases[i:i + (1 if worst == 0 else size)]
            for case, (bs, bc, be), (cs, cc, cerr) in pool.map(pair, batch):
                spent = bc + cc
                out.spend_usd += spent
                worst = max(worst, spent)
                if be or cerr:
                    out.unanswered.append((case.id, "baseline" if be else "candidate",
                                           be or cerr))
                    continue
                out.cost["baseline"] += bc
                out.cost["candidate"] += cc
                out.pairs.append((case.id, bs, cs))
            i += len(batch)
    return out


def case_fps(cases: list) -> set:
    return {hashlib.sha256(json.dumps([c.messages, c.check, c.value], sort_keys=True)
                           .encode()).hexdigest()[:12] for c in cases}


def record_and_count(cases: list, baseline: str, candidate: str,
                     ledger: pathlib.Path = LEDGER) -> int:
    """Remember this candidate and return how many distinct candidates were tried against
    ``baseline`` on these cases or a set sharing ``OVERLAP`` of them (this one included)."""
    mine = case_fps(cases)
    try:
        rows = json.loads(ledger.read_text())
    except (OSError, ValueError):
        rows = []
    need = max(1, int(OVERLAP * len(mine)))
    tried = {r["candidate"] for r in rows if r.get("baseline") == baseline
             and len(mine & set(r.get("cases", []))) >= need}
    rows.append({"baseline": baseline, "candidate": candidate, "cases": sorted(mine)})
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps(rows))
    return len(tried | {candidate})


def verdict(res: Result, candidates: int = 1) -> dict:
    """Promoted only when the paired test clears the bar for ``candidates`` tried, the candidate
    costs less per case, and every case got a gradable answer from both sides."""
    n = len(res.pairs)
    k = max(1, candidates)
    pb = stats.paired_bound([(b, c) for _, b, c in res.pairs], z=stats.family_z(k))
    cost = {s: (v / n if n else None) for s, v in res.cost.items()}
    cheaper = n > 0 and cost["candidate"] < cost["baseline"]
    complete = not res.unanswered and not res.stopped
    promoted = bool(pb and pb.supported and cheaper and complete)
    if promoted:
        why = "the candidate held the pass rate at the bar and costs less per case"
    elif pb is None:
        why = "no case ran on both models"
    elif not complete:
        why = (f"{len(res.unanswered)} case(s) got no gradable answer" if res.unanswered
               else res.stopped) + ", so the comparison is incomplete"
    elif not pb.supported:
        need = stats.gate_floor(k)
        if n < need and k == 1:
            why = f"{n} cases are too few to prove a switch: it needs about {need}"
        elif n < need:
            why = (f"with {k} models tried on these cases, the best needs about {need}; "
                   "confirm it alone on fresh cases")
        else:
            why = f"the quality bound is {pb.upper:.1%} behind, the bar allows {pb.margin:.1%}"
    else:
        why = "the candidate costs more per case: an upgrade, not a saving"
    return {"cases": n, "baseline_passed": int(sum(b for _, b, _ in res.pairs)),
            "candidate_passed": int(sum(c for _, _, c in res.pairs)),
            "baseline_cost_per_case": cost["baseline"],
            "candidate_cost_per_case": cost["candidate"],
            "candidates_tried": k, "cases_needed": stats.gate_floor(k),
            "lost": pb.baseline_only if pb else 0, "won": pb.candidate_only if pb else 0,
            "promoted": promoted, "why": why, "unanswered": len(res.unanswered),
            "stopped": res.stopped, "spend_usd": round(res.spend_usd, 6)}
