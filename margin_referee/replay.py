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
from margin_referee.cases import (MIN_EARLY_STOP_CASES, Case, apply_extras, grade,
                                  planned_looks, reliability_terms, stop_early,
                                  weighted_terms)

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
    #: With repeats: ``{case id: {side: [each run's score]}}``; ``pairs`` holds the strict result.
    runs: dict = field(default_factory=dict)
    #: Set when the replay stopped because the answer was already ``do_not_ship``: a deliberate
    #: stop, not a failure, so ``stopped`` stays empty.
    stopped_early: Optional[dict] = None


def with_system(messages: list, system: Optional[str]) -> list:
    if system is None:
        return messages
    rest = messages[1:] if messages and messages[0].get("role") == "system" else messages
    return [{"role": "system", "content": system}, *rest]


def _one(case: Case, side: str, chat: Callable, system: Optional[str], max_tokens: int,
         tools: Optional[list] = None):
    """One call for one side. ``tools`` replaces the case's own tool list when given (``[]``
    offers none); ``run`` gives it to the candidate only. ``tool_choice`` is sent only
    together with tools."""
    msgs = with_system(case.messages, system) if side == "candidate" else case.messages
    tools = case.tools if tools is None else tools
    extra = {"tools": tools} if tools else {}
    if tools and case.tool_choice is not None:
        extra["tool_choice"] = case.tool_choice
    try:
        r = chat(msgs, max_tokens=case.max_tokens or max_tokens,
                 response_format=case.response_format, **extra)
    except Exception as exc:  # noqa: BLE001 - an unanswered call is a missing pair, named
        return None, 0.0, f"{type(exc).__name__}: {exc}"[:200]
    if r.finish_reason == "length":
        return None, r.cost_usd, "cut off by the output limit"
    try:
        passed = grade(r.text, case, getattr(r, "tool_calls", None))
        return (1.0 if passed else 0.0), r.cost_usd, None
    except Exception as exc:  # noqa: BLE001 - a check that cannot run grades nothing
        return None, r.cost_usd, f"check failed: {type(exc).__name__}: {exc}"[:200]


def _side(case: Case, side: str, chat: Callable, system: Optional[str], max_tokens: int,
          repeats: int, tools: Optional[list] = None):
    """``(strict score, mean cost of one run, total cost, error, each run's score)``: the side
    passes only when all ``repeats`` runs passed (pass^k). One run with no answer drops it."""
    scores, total = [], 0.0
    for _ in range(max(1, repeats)):
        score, cost, err = _one(case, side, chat, system, max_tokens, tools)
        total += cost
        if err:
            return None, 0.0, total, err, []
        scores.append(score)
    return min(scores), total / len(scores), total, None, scores


def run(cases: list, baseline_chat: Callable, candidate_chat: Callable, *, cap_usd: float,
        candidate_system: Optional[str] = None, max_tokens: int = 8192,
        workers: int = 8, repeats: int = 1, early_stop: bool = True,
        candidates: int = 1, candidate_tools: Optional[list] = None) -> Result:
    """Both sides on every case until the next batch could cross ``cap_usd``.

    ``repeats`` runs each case that many times per side (spend scales with it; the cap prices a
    whole case, every repeat included). With ``early_stop``, once ``MIN_EARLY_STOP_CASES`` cases
    have both sides the replay stops as soon as the answer is already ``do_not_ship``, at a z
    corrected for the planned looks and the ``candidates`` tried; it never stops early for
    ship. ``candidate_tools`` is a tool change: the candidate is offered that tool list in
    place of each case's own, and the baseline keeps the case's tools."""
    out, worst, i = Result(), 0.0, 0
    looks, looked = planned_looks(len(cases), workers), 0

    def bound(ps, family):
        return stats.paired_bound(ps, z=stats.family_z(family))

    def pair(case):
        b = _side(case, "baseline", baseline_chat, None, max_tokens, repeats)
        c = _side(case, "candidate", candidate_chat, candidate_system, max_tokens, repeats,
                  candidate_tools)
        return case, b, c
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while i < len(cases):
            room = cap_usd - out.spend_usd
            size = workers if worst == 0 else min(workers, int(room // worst))
            if size < 1 or room <= 0:
                out.stopped = f"stopped at the ${cap_usd:.2f} cap after {i} cases"
                break
            batch = cases[i:i + (1 if worst == 0 else size)]
            for case, (bs, bc, bt, be, br), (cs, cc, ct, cerr, cr) in pool.map(pair, batch):
                spent = bt + ct
                out.spend_usd += spent
                worst = max(worst, spent)
                if be or cerr:
                    out.unanswered.append((case.id, "baseline" if be else "candidate",
                                           be or cerr))
                    continue
                out.cost["baseline"] += bc
                out.cost["candidate"] += cc
                out.pairs.append((case.id, bs, cs))
                if repeats > 1:
                    out.runs[case.id] = {"baseline": br, "candidate": cr}
            i += len(batch)
            # never more looks than the test was corrected for (a tight cap shrinks batches)
            if early_stop and looked < looks and len(out.pairs) >= MIN_EARLY_STOP_CASES:
                looked += 1
                out.stopped_early = stop_early(out.pairs, len(cases), looks, bound, candidates,
                                               len(out.unanswered))
                if out.stopped_early:
                    break
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


#: A segment is judged on its own only with at least this many paired cases.
MIN_SEGMENT_CASES = 10


def segment_terms(pairs, seg_of: dict, bound, k: int, complete: bool, critical=(),
                  floor=None) -> list:
    """Each labelled segment's own reading: cases, lost, won, and ``worse`` when even the
    favourable bound misses the bar there. The z is corrected for every segment looked at, so
    slicing finely cannot manufacture a loss. A ``critical`` segment (a release criterion the team
    names) also gets ``holds``: proven not worse on its own, on enough of its own cases
    (``floor(looks)``). Empty when fewer than two segments are labelled and none is critical."""
    groups: dict = {}
    for cid, b, c in pairs:
        groups.setdefault(seg_of.get(cid, ""), []).append((b, c))
    crit = {str(c) for c in critical if str(c)}
    named = sorted({g for g in groups if g} | crit)
    if len(named) < 2 and not crit:
        return []
    looks = max(1, k) * len(named)
    out = []
    for g in named:
        ps = groups.get(g, [])
        pb = bound(ps, looks) if len(ps) >= MIN_SEGMENT_CASES else None
        row = {"segment": g, "cases": len(ps), "lost": sum(1 for b, c in ps if b > c),
               "won": sum(1 for b, c in ps if c > b),
               "worse": bool(pb is not None and complete
                             and (2 * pb.shortfall - pb.upper) > pb.margin)}
        if g in crit:
            need = floor(looks) if floor else None
            row.update(critical=True, cases_needed=need,
                       holds=bool(pb is not None and complete and pb.upper <= pb.margin
                                  and need is not None and len(ps) >= need))
        out.append(row)
    return out


def apply_segments(v: dict, segs: list) -> dict:
    """A change proven worse in a segment does not ship, whatever the overall reading; a change
    not proven to hold in a release-critical segment is not enough evidence."""
    v["segments"] = segs
    shippable = v["decision"] in ("ship", "ship_if_worth_the_cost")
    worse = [s["segment"] for s in segs if s["worse"]]
    unproven = [s for s in segs if s.get("critical") and not s.get("holds") and not s["worse"]]
    if worse and v["decision"] != "do_not_ship":
        v["decision"], v["quality"], v["promoted"] = "do_not_ship", "worse_in_segment", False
        v["why"] = ((f"it holds overall but is worse than the bar allows in {', '.join(worse)}; "
                     if shippable else
                     f"it is proven worse than the bar allows in {', '.join(worse)}; ")
                    + "ship it only where it holds, or not at all")
    elif unproven and shippable:
        s = unproven[0]
        v["decision"], v["quality"] = "not_enough_evidence", "unproven_in_critical_segment"
        v["promoted"] = False
        v["why"] = (f"it holds overall, but the release-critical segment {s['segment']} has "
                    f"{s['cases']} case(s) and is not proven to hold on its own; it needs about "
                    f"{s['cases_needed']}")
    return v


def decision_terms(pb, complete: bool, cost: dict, n: int, floor: int, pairs: list,
                   filled=None) -> dict:
    """ship / ship_if_worth_the_cost / do_not_ship / not_enough_evidence, the quality gap and
    the cases that got worse. The same rule as Margin's hosted verdict (a parity test pins it).

    ``filled`` is the bound recomputed as if every unanswered case had gone the candidate's way
    (:func:`filled_bound`). It can only turn ``unclear`` into ``worse``: a regression that
    survives the most favourable fill is proven whatever the missing answers were. It never
    reaches ``holds``: an unanswered case never helps a change ship."""
    if pb is not None and complete and pb.supported and n >= floor:
        quality = "holds"
    elif pb is not None and complete and (2 * pb.shortfall - pb.upper) > pb.margin:
        quality = "worse"
    elif filled is not None and (2 * filled.shortfall - filled.upper) > filled.margin:
        quality = "worse"  # ... and still is with every unanswered case given to the candidate
    else:
        quality = "unclear"
    b, c = cost.get("baseline"), cost.get("candidate")
    if b is None or c is None or (b == 0 and c == 0):
        cost_word = "unknown"
    else:
        cost_word = "cheaper" if c < b else "dearer" if c > b else "same"
    decision = ("do_not_ship" if quality == "worse" else "not_enough_evidence"
                if quality == "unclear" else "ship_if_worth_the_cost" if cost_word == "dearer"
                else "ship")
    worse = [k for k, bs, cs in pairs if bs > cs][:10]
    gap = None if pb is None else {"estimate_points": round(-100 * pb.shortfall, 2) + 0.0,
                                   "worst_case_points": round(-100 * pb.upper, 2) + 0.0,
                                   "bar_allows_points": round(-100 * pb.margin, 2) + 0.0}
    return {"decision": decision, "quality": quality, "cost": cost_word,
            "quality_gap": gap, "worse_cases": worse}


def filled_bound(pairs: list, unanswered: int, stopped: str, bound):
    """The paired bound as if each unanswered case were one the baseline failed and the candidate
    passed: the reading most favourable to the candidate. None unless unanswered cases are the
    only thing missing: a replay stopped by its cap never ran the rest of the set, and nobody
    knows how many of those would have gone unanswered."""
    if not unanswered or stopped or not pairs:
        return None
    return bound([(b, c) for _, b, c in pairs] + [(0.0, 1.0)] * unanswered)


def worse_despite_missing(unanswered: list) -> str:
    return ("worse than the bar allows even if every one of the "
            f"{len(unanswered)} unanswered case(s) had gone the candidate's way"
            f"{_top_reason(unanswered)}")


def unanswered_reasons(unanswered: list, top: int = 5) -> list:
    """Why cases got no gradable answer, most common first: ``[{reason, cases}]``. A reader seeing
    "79 cases got no answer" needs to know whether that is a cap, a timeout or a broken check
    (R11, 2026-11-27: 79 IFEval answers were cut off at a 2,048-token cap and the verdict did not
    say so)."""
    from collections import Counter
    c = Counter(str(u[-1])[:120] for u in unanswered)
    return [{"reason": r, "cases": n} for r, n in c.most_common(top)]


def _top_reason(unanswered: list) -> str:
    r = unanswered_reasons(unanswered, 1)
    return f" (most: {r[0]['reason']}, {r[0]['cases']} case(s))" if r else ""


def verdict(res: Result, candidates: int = 1, seg_of: Optional[dict] = None,
            critical: tuple = (), weight_of: Optional[dict] = None) -> dict:
    """Promoted only when the paired test clears the bar for ``candidates`` tried, the candidate
    costs less per case, and every case got a gradable answer from both sides."""
    n = len(res.pairs)
    k = max(1, candidates)
    # an early stop looked at the result ``looks`` times, so its bound is held to that family
    early = res.stopped_early
    looks = max(1, int(early["looks"])) if early else 1
    pb = stats.paired_bound([(b, c) for _, b, c in res.pairs], z=stats.family_z(k * looks))
    cost = {s: (v / n if n else None) for s, v in res.cost.items()}
    cheaper = n > 0 and cost["candidate"] < cost["baseline"]
    complete = not res.unanswered and not res.stopped
    promoted = bool(pb and pb.supported and cheaper and complete)
    if promoted:
        why = "the candidate held the pass rate at the bar and costs less per case"
    elif pb is None:
        why = "no case ran on both models"
    elif not complete:
        why = (f"{len(res.unanswered)} case(s) got no gradable answer{_top_reason(res.unanswered)}"
               if res.unanswered
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
    filled = filled_bound(res.pairs, len(res.unanswered), res.stopped,
                          lambda ps: stats.paired_bound(ps, z=stats.family_z(k * looks)))
    terms = decision_terms(pb, complete, cost, n, stats.gate_floor(k), res.pairs, filled)
    if terms["quality"] == "worse" and not complete:
        why = worse_despite_missing(res.unanswered)
    outv = {**terms, "cases": n, "baseline_passed": int(sum(b for _, b, _ in res.pairs)),
            "candidate_passed": int(sum(c for _, _, c in res.pairs)),
            "baseline_cost_per_case": cost["baseline"],
            "candidate_cost_per_case": cost["candidate"],
            "candidates_tried": k, "cases_needed": stats.gate_floor(k),
            "lost": pb.baseline_only if pb else 0, "won": pb.candidate_only if pb else 0,
            "promoted": promoted, "why": why, "unanswered": len(res.unanswered),
            "unanswered_reasons": unanswered_reasons(res.unanswered),
            "stopped": res.stopped, "spend_usd": round(res.spend_usd, 6)}
    repeats = max((len(x) for s in res.runs.values() for x in s.values()), default=1)
    outv["repeats"] = repeats
    outv = apply_segments(outv, segment_terms(
        res.pairs, seg_of or {},
        lambda ps, kk: stats.paired_bound(ps, z=stats.family_z(kk * looks)),
        k, complete, critical, stats.gate_floor))
    return apply_extras(outv, early, k, reliability_terms(res.runs, repeats),
                        weighted_terms(res.pairs, weight_of or {}))
