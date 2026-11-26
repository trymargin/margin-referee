"""Read an eval export into cases, and grade an answer with the case's own checks.

Copied verbatim from Margin's hosted product (src/margin/dataset_replay.py); a test in the
product repo fails if the two ever differ.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

#: A free-text expected answer longer than this is a judgement, not a label: never compared exactly.
LABEL_MAX_CHARS = 80
CHECKS = ("equals", "icontains", "contains", "regex", "is-json", "json-field", "json-match")


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


def _row(i: int, row: dict, out: Parsed) -> None:
    cid = str(row.get("id") or row.get("case_id") or f"case-{i}")
    msgs = _messages(row.get("messages") or row.get("input") or row.get("inputs"))
    if msgs is None:
        out.ungraded.append((cid, "no input messages"))
        return
    check = row.get("check")
    if check is not None:
        if check not in CHECKS:
            out.ungraded.append((cid, f"check {check!r} is not deterministic"))
            return
        if check == "regex" and _bad_regex(str(row.get("expected", row.get("value", "")))):
            out.ungraded.append((cid, "its regex does not compile"))
            return
        out.cases.append(Case(cid, msgs, check, str(row.get("expected", row.get("value", ""))),
                              row.get("response_format"), row.get("max_tokens")))
        return
    keys = ("expected", "expectedOutput", "expected_output", "outputs", "output")
    expected = next((row[k] for k in keys if row.get(k) is not None), None)
    if isinstance(expected, dict) and len(expected) > 1 and all(
            isinstance(v, (str, int, float, bool)) for v in expected.values()):
        # MAR-2313: a structured answer (several fields) is graded on every field, not left out
        out.cases.append(Case(cid, msgs, "json-match", json.dumps(expected),
                              row.get("response_format"), row.get("max_tokens")))
        return
    label = _label(expected)
    if label is None:
        out.ungraded.append((cid, "expected answer is free text, which needs a judge"))
        return
    out.cases.append(Case(cid, msgs, "equals", label))


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
            out.cases.append(Case(cid, msgs, a["type"], str(a.get("value", "")),
                                  also=tuple((b["type"], str(b.get("value", ""))) for b in rest)))


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
    from dataclasses import replace
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


def grade(output: str, case: Case) -> bool:
    """Pass only when the answer passes the case's check and every one in ``also``."""
    return all(_check(output, c, v) for c, v in ((case.check, case.value), *case.also))


def _check(output: str, check: str, want: str) -> bool:
    out = (output or "").strip()
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
