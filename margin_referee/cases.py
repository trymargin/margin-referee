"""Read an eval export into cases, and grade an answer with the case's own checks.

Copied verbatim from Margin's hosted product (src/margin/dataset_replay.py); a test in the
product repo fails if the two ever differ.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

#: A free-text expected answer longer than this is a judgement, not a label: never compared exactly.
LABEL_MAX_CHARS = 80
CHECKS = ("equals", "icontains", "contains", "regex", "is-json", "json-field", "json-match",
          "tool-call")
#: A case carries at most this many tool definitions: they are sent with every call.
MAX_TOOLS = 64


@dataclass(frozen=True)
class Case:
    id: str
    messages: list
    check: str
    value: str = ""
    #: The production call's structured-output request, replayed as it was (OpenAI shape).
    response_format: Optional[dict] = None
    #: The production call's own output limit, when the export carries one.
    max_tokens: Optional[int] = None
    #: Further ``(check, value)`` pairs the case must also pass (a promptfoo case's other asserts).
    also: tuple = ()
    #: The slice of traffic the case belongs to (``segment``, or ``metadata.segment`` /
    #: ``metadata.category``), so a change that ties overall but loses one slice is caught.
    segment: str = ""
    #: The case's share of real traffic (a share or a count; ``weight`` or ``metadata.weight``).
    #: It weights the report only: the decision stays on the unweighted paired test.
    weight: Optional[float] = None
    #: The tool definitions the production call offered (OpenAI function tools), sent as they
    #: were. A replay's ``candidate_tools`` replaces them for the candidate side only.
    tools: Optional[list] = None
    #: The production call's ``tool_choice``, sent only together with tools.
    tool_choice: Optional[object] = None


@dataclass
class Parsed:
    cases: list = field(default_factory=list)
    #: ``(case id, why it cannot be graded here)`` for every case left out.
    ungraded: list = field(default_factory=list)


def _messages(x) -> Optional[list]:
    """The OpenAI chat messages a dataset row means, or None when it names none."""
    if isinstance(x, list) and x and all(isinstance(m, dict) and "role" in m for m in x):
        return x
    if isinstance(x, str) and x.strip():
        try:
            return _messages(json.loads(x))
        except ValueError:
            return [{"role": "user", "content": x}]
    if isinstance(x, dict):
        for k in ("messages", "input", "question", "prompt", "query"):
            if k in x and (m := _messages(x[k])) is not None:
                return m
    return None


def _label(expected) -> Optional[str]:
    """A dataset's expected answer, when it is short enough to be a label rather than prose."""
    if isinstance(expected, dict):
        vals = [v for v in expected.values() if isinstance(v, (str, int, float, bool))]
        expected = vals[0] if len(vals) == 1 else None
    if isinstance(expected, (int, float, bool)):
        expected = str(expected)
    if isinstance(expected, str) and 0 < len(expected.strip()) <= LABEL_MAX_CHARS:
        return expected.strip()
    return None


def tool_list(x) -> Optional[list]:
    """``x`` when it is an OpenAI-style function tool list (``[]`` means no tools), else None.
    A list that is not one is never repaired or trimmed."""
    ok = isinstance(x, list) and len(x) <= MAX_TOOLS and all(
        isinstance(t, dict) and t.get("type") == "function"
        and isinstance(t.get("function"), dict)
        and isinstance(t["function"].get("name"), str) and t["function"]["name"].strip()
        for t in x)
    return x if ok else None


def _tool_want(want: str) -> Optional[tuple]:
    """A ``tool-call`` check's value as ``(tool name, {argument: wanted value})``. The value is a
    bare tool name, or ``;``-separated pairs: ``name=<tool>`` and ``args.<field>=<value>``.
    None when it names no tool or has a part that is neither."""
    name, args = "", {}
    for part in (p.strip() for p in want.split(";")):
        key, eq, val = (x.strip() for x in part.partition("="))
        if not part:
            continue
        if not eq and not name:
            name = part
        elif key == "name" and val and not name:
            name = val
        elif eq and key.startswith("args.") and key[5:]:
            args[key[5:]] = val
        else:
            return None
    return (name, args) if name else None


def _segment(row) -> str:
    """The case's segment label, from the row or its metadata; empty when it has none."""
    if not isinstance(row, dict):
        return ""
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for v in (row.get("segment"), meta.get("segment"), meta.get("category")):
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()[:64]
    return ""


def _weight(row) -> Optional[float]:
    """The case's traffic weight, from the row or its metadata; None when it has none or it is
    not a positive, finite number (a weight is never guessed)."""
    if not isinstance(row, dict):
        return None
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for v in (row.get("weight"), meta.get("weight")):
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < float("inf"):
            return float(v)
    return None


def _row(i: int, row: dict, out: Parsed) -> None:
    cid = str(row.get("id") or row.get("case_id") or f"case-{i}")
    msgs = _messages(row.get("messages") or row.get("input") or row.get("inputs"))
    if msgs is None:
        out.ungraded.append((cid, "no input messages"))
        return
    tools, choice = row.get("tools"), row.get("tool_choice")
    if tools is not None and tool_list(tools) is None:
        out.ungraded.append((cid, "its tools are not a list of function tools"))
        return
    if choice is not None and not isinstance(choice, (str, dict)):
        out.ungraded.append((cid, "its tool_choice is not a string or an object"))
        return
    call = {"tools": tools or None, "tool_choice": choice if tools else None}
    check = row.get("check")
    if check is not None:
        if check not in CHECKS:
            out.ungraded.append((cid, f"check {check!r} is not deterministic"))
            return
        if check == "tool-call" and _tool_want(
                str(row.get("expected", row.get("value", "")))) is None:
            out.ungraded.append((cid, "its tool-call check names no tool"))
            return
        if check == "regex" and _bad_regex(str(row.get("expected", row.get("value", "")))):
            out.ungraded.append((cid, "its regex does not compile"))
            return
        out.cases.append(Case(cid, msgs, check, str(row.get("expected", row.get("value", ""))),
                              row.get("response_format"), row.get("max_tokens"),
                              segment=_segment(row), weight=_weight(row), **call))
        return
    keys = ("expected", "expectedOutput", "expected_output", "outputs", "output")
    expected = next((row[k] for k in keys if row.get(k) is not None), None)
    if isinstance(expected, dict) and len(expected) > 1 and all(
            isinstance(v, (str, int, float, bool)) for v in expected.values()):
        # MAR-2313: a structured answer (several fields) is graded on every field, not left out
        out.cases.append(Case(cid, msgs, "json-match", json.dumps(expected),
                              row.get("response_format"), row.get("max_tokens"),
                              segment=_segment(row), weight=_weight(row), **call))
        return
    label = _label(expected)
    if label is None:
        out.ungraded.append((cid, "expected answer is free text, which needs a judge"))
        return
    out.cases.append(Case(cid, msgs, "equals", label, segment=_segment(row),
                          weight=_weight(row), **call))


def _promptfoo(doc: dict, out: Parsed) -> None:
    rows = (doc.get("results") or {}).get("results") or []
    for i, r in enumerate(rows):
        cid = str((r.get("testCase") or {}).get("description") or r.get("id") or f"case-{i}")
        msgs = _messages((r.get("prompt") or {}).get("raw"))
        every = (r.get("testCase") or {}).get("assert") or []
        asserts = [a for a in every if a.get("type") in CHECKS]
        judged = sorted({str(a.get("type")) for a in every if a.get("type") not in CHECKS})
        if msgs is None:
            out.ungraded.append((cid, "no rendered prompt"))
        elif judged and asserts:
            # MAR-2313: grading only the deterministic half would pass answers the suite fails
            out.ungraded.append((cid, f"it also asserts {', '.join(judged)}, which this replay "
                                      "cannot grade, so passing the rest would not be a pass"))
        elif not asserts:
            out.ungraded.append((cid, "no deterministic assertion (equals, contains, regex...)"))
        elif any(a["type"] == "regex" and _bad_regex(str(a.get("value", ""))) for a in asserts):
            out.ungraded.append((cid, "its regex does not compile"))
        else:
            a, rest = asserts[0], asserts[1:]
            tc = r.get("testCase") or {}
            out.cases.append(Case(cid, msgs, a["type"], str(a.get("value", "")),
                                  also=tuple((b["type"], str(b.get("value", ""))) for b in rest),
                                  segment=_segment({**(tc.get("vars") or {}),
                                                    "metadata": tc.get("metadata")}),
                                  # only metadata: a prompt variable named weight is not traffic
                                  weight=_weight({"metadata": tc.get("metadata")})))


def _bad_regex(pattern: str) -> bool:
    try:
        re.compile(pattern)
        return False
    except re.error:
        return True


def parse(text: str) -> Parsed:
    """Cases from a promptfoo results file, a JSON array, or JSONL (Braintrust, LangSmith, ours)."""
    out = Parsed()
    text = text.strip()
    try:
        doc = json.loads(text)
    except ValueError:
        doc = None
    if isinstance(doc, dict) and "results" in doc:
        _promptfoo(doc, out)
        return _unique_ids(out)
    rows: Iterable = doc if isinstance(doc, list) else (
        [doc] if isinstance(doc, dict) else _lines(text))
    for i, row in enumerate(rows):
        if isinstance(row, dict):
            _row(i, row, out)
        else:
            out.ungraded.append((f"case-{i}", "not a JSON object"))
    return _unique_ids(out)


def _unique_ids(out: Parsed) -> Parsed:
    """Cases pair by id, so a repeated id (a promptfoo description used twice, say) would pair
    one case's answers with another's. Repeats get ``#2``, ``#3``; the first keeps its name."""
    taken, cases = {c.id for c in out.cases}, []
    used: set[str] = set()
    for c in out.cases:
        name, n = c.id, 1
        # a rename never takes a name already used, or one a later case carries as its own
        while name in used or (name != c.id and name in taken):
            n += 1
            name = f"{c.id}#{n}"
        used.add(name)
        cases.append(c if name == c.id else replace(c, id=name))
    out.cases = cases
    return out


def _lines(text: str):
    for line in text.splitlines():
        if line.strip():
            try:
                yield json.loads(line)
            except ValueError:
                yield None


def grade(output: str, case: Case, tool_calls: Optional[list] = None) -> bool:
    """Pass only when the answer passes the case's check and every one in ``also``.
    ``tool_calls`` is the response's tool calls (OpenAI shape), read by a ``tool-call`` check:
    ``[]`` when the model made none, None when the response does not report them."""
    return all(_check(output, c, v, tool_calls)
               for c, v in ((case.check, case.value), *case.also))


def _tool_call_ok(tool_calls: Optional[list], want: str) -> bool:
    """Whether the model's FIRST tool call is the wanted tool with the wanted arguments. No tool
    call is a fail. ⛔ ``ValueError`` (no answer: neither a pass nor a fail) when the response
    does not report tool calls, or the call or the arguments being checked cannot be read."""
    wanted = _tool_want(want)
    if wanted is None:
        raise ValueError("the tool-call check names no tool")
    if tool_calls is None:
        raise ValueError("the response does not report tool calls")
    if not tool_calls:
        return False
    fn = tool_calls[0].get("function") if isinstance(tool_calls[0], dict) else None
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
        raise ValueError("the first tool call cannot be read")
    if fn["name"] != wanted[0]:
        return False
    if not wanted[1]:
        return True
    got = fn.get("arguments")
    try:
        got = json.loads(got) if isinstance(got, str) else got
    except ValueError:
        got = None
    if not isinstance(got, dict):
        raise ValueError("the tool call's arguments are not a JSON object")
    return all(k in got and str(got[k]).strip().lower() == v.lower()
               for k, v in wanted[1].items())


def _check(output: str, check: str, want: str, tool_calls: Optional[list] = None) -> bool:
    out = (output or "").strip()
    if check == "tool-call":  # value names the tool (and arguments) the first tool call must be
        return _tool_call_ok(tool_calls, want)
    if check == "equals":
        return out.lower().strip(" .\"'") == want.lower().strip(" .\"'")
    if check == "contains":
        return want in out
    if check == "icontains":
        return want.lower() in out.lower()
    if check == "regex":
        return re.search(want, out) is not None
    if check == "json-field":  # value is "field=expected": one field of a JSON answer
        name, _, want = want.partition("=")
        try:
            got = json.loads(out).get(name)
        except (ValueError, AttributeError):
            return False
        return str(got).strip().lower() == want.strip().lower()
    if check == "is-json":
        try:
            json.loads(out)
            return True
        except ValueError:
            return False
    if check == "json-match":  # value is a JSON object: every field must match
        try:
            got, wanted = json.loads(out), json.loads(want)
            return all(str(got.get(k)).strip().lower() == str(v).strip().lower()
                       for k, v in wanted.items())
        except (ValueError, AttributeError):
            return False
    raise ValueError(check)


#: Early stopping looks only once this many cases have an answer from both sides.
MIN_EARLY_STOP_CASES = 40
#: One-sided 95%, the confidence the ``runs_needed`` estimate is stated at.
RUNS_NEEDED_Z = 1.645


def planned_looks(total: int, batch: int) -> int:
    """How many times a replay of ``total`` cases, run ``batch`` at a time, may look at its
    own result: the family the early-stop test is corrected for."""
    return max(1, math.ceil(total / max(1, batch)))


def stop_early(pairs: list, total: int, looks: int, bound, candidates: int = 1,
               unanswered: int = 0) -> Optional[dict]:
    """Stop spending when the answer is already ``do_not_ship``: even the bound most favourable
    to the candidate misses the bar on the cases so far (borrowed from VERDICT, ``verdict-gate``).

    ``bound(ps, family)`` is the paired test at a z corrected for ``family`` looks; the family is
    every planned look times every candidate tried (Bonferroni), so looking often cannot
    manufacture a loss. ⛔ It never stops for ship: a pass needs every case. It also never stops
    with a case unanswered, because that case may be one the candidate would have passed."""
    n = len(pairs)
    if n < MIN_EARLY_STOP_CASES or n >= total or unanswered:
        return None
    pb = bound([(b, c) for _, b, c in pairs], max(1, candidates) * max(1, looks))
    if pb is None or (2 * pb.shortfall - pb.upper) <= pb.margin:
        return None
    return {"after_cases": n, "of_cases": total, "looks": looks,
            "reason": f"after {n} of {total} cases even the reading most favourable to the "
                      "candidate was past what the bar allows"}


def runs_needed(flip_baseline: float, flip_candidate: float, cases: int,
                gap: float = 0.01, z: float = RUNS_NEEDED_Z) -> dict:
    """An ESTIMATE of how many repeats per case would show a ``gap`` (1 point) between the two
    pass rates through run-to-run noise alone.

    A flip rate ``q`` (two runs of one case disagree) means a per-run variance of ``q / 2`` for
    that case. The gap between the two sides' mean pass rates over ``cases`` cases and ``r``
    repeats then has a run-noise standard error of ``sqrt((q_b + q_c) / 2 / (cases * r))``;
    ``r`` is the smallest count that puts ``z`` of those inside ``gap``:

        r = ceil(z^2 * (q_b + q_c) / 2 / (cases * gap^2))

    It covers run-to-run noise only, not which cases were sampled, so it is a floor."""
    r = z * z * (flip_baseline + flip_candidate) / 2 / (max(1, cases) * gap * gap)
    return {"repeats": max(1, math.ceil(r)), "for_gap_points": round(100 * gap, 2),
            "estimate": True,
            "note": "an estimate from the measured flip rates: repeats per case needed for "
                    "run-to-run noise alone to be smaller than the gap; it does not cover which "
                    "cases were sampled"}


def reliability_terms(runs: dict, k: int) -> Optional[dict]:
    """With ``k`` repeats: per side ``pass_at_1`` (the mean over every run) and ``pass_pow_k``
    (the share of cases that passed ALL k runs, the rule the verdict is judged on), and the cases
    whose result flipped between repeats. ``runs`` is ``{case id: {side: [scores]}}``. None with
    one run: one run cannot show a flip."""
    if k < 2 or not runs:
        return None
    out: dict = {"repeats": k, "rule": f"a case passes only if all {k} runs passed (pass^{k}); "
                                       "the paired test uses that strict result"}
    rate = {}
    for side in ("baseline", "candidate"):
        per = {cid: [float(x) for x in r[side]] for cid, r in runs.items()}
        flipped = sorted(c for c, s in per.items() if 0 < sum(s) < len(s))
        # the share of repeat pairs of one case that disagree, averaged over cases
        rate[side] = sum(sum(s) * (len(s) - sum(s)) / (len(s) * (len(s) - 1) / 2)
                         for s in per.values() if len(s) > 1) / len(per)
        out[side] = {"pass_at_1": round(sum(sum(s) / len(s) for s in per.values()) / len(per), 4),
                     "pass_pow_k": round(sum(1 for s in per.values() if min(s) >= 1) / len(per),
                                         4),
                     "flipped": len(flipped), "flipped_cases": flipped[:10],
                     "flip_rate": round(rate[side], 4)}
    out["runs_needed"] = runs_needed(rate["baseline"], rate["candidate"], len(runs))
    return out


def weighted_terms(pairs: list, weight_of: dict) -> Optional[dict]:
    """The same paired cases, weighted by the traffic each carries: the weighted pass rate on
    each side and the share of traffic on cases that got worse. ⛔ A weighting of the same cases,
    not a new test: the decision stays on the unweighted paired test. None when no case has a
    weight; a case without one is left out of the weighting and counted."""
    ws = [(weight_of[cid], b, c) for cid, b, c in pairs if weight_of.get(cid)]
    total = sum(w for w, _, _ in ws)
    if not ws or total <= 0:
        return None
    worse = sum(w for w, b, c in ws if b > c) / total
    return {"cases": len(ws), "unweighted_cases": len(pairs) - len(ws),
            "baseline_pass_rate": round(sum(w * b for w, b, _ in ws) / total, 4),
            "candidate_pass_rate": round(sum(w * c for w, _, c in ws) / total, 4),
            "worse_traffic_share": round(worse, 4),
            "better_traffic_share": round(sum(w for w, b, c in ws if c > b) / total, 4),
            "summary": (f"the change breaks cases covering {worse:.1%} of your traffic" if worse
                        else "no weighted case got worse"),
            "note": "the same cases weighted by the traffic share you gave each one, not a new "
                    "test: the decision is made on the unweighted paired test"}


def apply_extras(v: dict, early: Optional[dict], k: int, reliability: Optional[dict],
                 weighted: Optional[dict]) -> dict:
    """Put the early stop, the reliability block and the traffic weighting on a verdict.

    ⛔ A replay stopped early is a deliberate stop on ``do_not_ship`` and nothing else: if the
    verdict it reaches is anything else (more candidates were tried than the stop was corrected
    for), it is ``not_enough_evidence``. It never ships and never promotes."""
    v.update(stopped_early=early, reliability=reliability, weighted=weighted)
    if early:
        left = early["of_cases"] - early["after_cases"]
        v["promoted"] = False
        if v["decision"] != "do_not_ship":
            v["decision"], v["quality"] = "not_enough_evidence", "unclear"
            v["why"] = (f"the replay stopped early after {early['after_cases']} of "
                        f"{early['of_cases']} cases, but with {k} models tried on these cases "
                        "that stop is not proven; run it again with early stopping off")
        elif v["quality"] == "worse":
            v["why"] = (f"stopped after {early['after_cases']} of {early['of_cases']} cases: even "
                        "the reading most favourable to the candidate is past what the bar "
                        f"allows, so the remaining {left} cases were not run because the answer "
                        "was already clear")
        else:  # a loss proven in one segment (``apply_segments``): its reason stands
            v["why"] += (f"; the replay stopped early after {early['after_cases']} of "
                         f"{early['of_cases']} cases")
    if weighted and weighted["worse_traffic_share"] > 0:
        v["why"] += (f"; {weighted['summary']} (a weighting of the same cases; the decision is "
                     "on the unweighted paired test)")
    return v
