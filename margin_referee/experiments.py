"""A second opinion on a comparison you already ran: two scored Braintrust experiments in, the
switch verdict out. Nothing is re-run and nothing is spent.

An experiment is read as Braintrust stores it: a list of events (a JSON array, JSONL, or the
``{"events": [...]}`` page that ``/v1/experiment/<id>/fetch`` returns). One case is one root span
(``span_id == root_span_id``; every event when there are no span ids). Cases are paired across the
two experiments by dataset record (``origin.id``, else ``dataset_record_id``), else by ``input``.

A case passes when the named score is at least ``threshold`` (default 1.0: full credit only). A
case with no such score is left out and counted. A case one experiment has and the other lacks is
unanswered, and an unanswered case blocks a promotion. Cost is ``metrics.estimated_cost`` summed
per case when both experiments logged it; without it the verdict says the saving is unknown.
"""
from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
from typing import Iterable, Optional

from margin_referee import replay

API = "https://api.braintrust.dev"


def parse_events(text: str) -> list[dict]:
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data.get("events", [])
    return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []


def fetch(experiment_id: str, api_key: str, base: str = API, page: int = 1000) -> list[dict]:
    """Every event of one experiment, through Braintrust's REST API, cursor-paged.

    ponytail: never run against a live key yet; the shape is Braintrust's published OpenAPI
    (``/v1/experiment/{id}/fetch`` -> ``{"events", "cursor"}``). Export to a file if it differs."""
    out, cursor = [], None
    while True:
        q = {"limit": page, **({"cursor": cursor} if cursor else {})}
        req = urllib.request.Request(
            f"{base}/v1/experiment/{urllib.parse.quote(experiment_id)}/fetch?"
            + urllib.parse.urlencode(q), headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.load(r)
        events = body.get("events") or []
        out.extend(e for e in events if isinstance(e, dict))
        cursor = body.get("cursor")
        if not events or not cursor:
            return out


def _key(e: dict) -> str:
    origin = e.get("origin") if isinstance(e.get("origin"), dict) else {}
    rid = origin.get("id") or e.get("dataset_record_id")
    if rid:
        return f"record:{rid}"
    return "input:" + json.dumps(e.get("input"), sort_keys=True, default=str)


def _roots(events: Iterable[dict]) -> list[dict]:
    events = list(events)
    with_ids = [e for e in events if e.get("span_id") and e.get("root_span_id")]
    if not with_ids:
        return events
    return [e for e in with_ids if e["span_id"] == e["root_span_id"]]


def segment_of(e: dict) -> str:
    """An event's segment label (``metadata.segment``, else ``metadata.category``)."""
    meta = e.get("metadata") if isinstance(e.get("metadata"), dict) else {}
    for v in (meta.get("segment"), meta.get("category")):
        if isinstance(v, (str, int)) and str(v).strip():
            return str(v).strip()[:64]
    return ""


def segments(events: Iterable[dict]) -> dict:
    """``{case key: segment}`` for the root spans that carry one."""
    return {_key(e): segment_of(e) for e in _roots(events) if segment_of(e)}


def cases(events: Iterable[dict], scorer: str, threshold: float) -> tuple[dict, int]:
    """``{case key: (passed 0/1, cost or None)}`` and how many cases had no usable score."""
    out, unscored = {}, 0
    for e in _roots(events):
        scores = e.get("scores") if isinstance(e.get("scores"), dict) else {}
        s = scores.get(scorer)
        if isinstance(s, bool):
            s = float(s)
        if not isinstance(s, (int, float)) or not 0.0 <= s <= 1.0:
            unscored += 1
            continue
        metrics = e.get("metrics") if isinstance(e.get("metrics"), dict) else {}
        cost = metrics.get("estimated_cost")
        numeric = (isinstance(cost, (int, float)) and not isinstance(cost, bool)
                   and math.isfinite(cost) and cost >= 0)
        cost = float(cost) if numeric else None
        k = _key(e)
        if k in out:  # the same case twice in one experiment (trials): keep the worst
            p, c = out[k]
            out[k] = (min(p, 1.0 if s >= threshold else 0.0),
                      None if c is None or cost is None else c + cost)
        else:
            out[k] = (1.0 if s >= threshold else 0.0, cost)
    return out, unscored


def compare(baseline: list[dict], candidate: list[dict], *, scorer: str, threshold: float = 1.0,
            candidates: int = 1, critical: tuple = ()) -> dict:
    b, b_unscored = cases(baseline, scorer, threshold)
    c, c_unscored = cases(candidate, scorer, threshold)
    res = replay.Result()
    costed = True
    for k in sorted(b.keys() | c.keys()):
        if k not in b or k not in c:
            res.unanswered.append((k, "baseline" if k not in b else "candidate", "not in export"))
            continue
        (bp, bc), (cp, cc) = b[k], c[k]
        res.pairs.append((k, bp, cp))
        if bc is None or cc is None:
            costed = False
        else:
            res.cost["baseline"] += bc
            res.cost["candidate"] += cc
    v = replay.verdict(res, candidates, {**segments(baseline), **segments(candidate)}, critical)
    if not costed and res.pairs:
        v["baseline_cost_per_case"] = v["candidate_cost_per_case"] = None
        v["cost"] = "unknown"
        if v["decision"] == "ship_if_worth_the_cost":
            v["decision"] = "ship"
        v["promoted"] = False  # a partial cost sum must never stand in for the saving
        # with no cost logged, verdict() saw 0 vs 0 and called it "costs more"; that line is
        # only reached when the quality bound held and every case was answered
        if v["why"] == "the candidate costs more per case: an upgrade, not a saving":
            v["why"] = ("the candidate held the pass rate at the bar, but the exports carry no "
                        "estimated_cost, so the saving is not shown")
    v.update(scorer=scorer, threshold=threshold, unscored={"baseline": b_unscored,
             "candidate": c_unscored}, spend_usd=0.0)
    return v


def load(source: str, api_key: Optional[str] = None) -> list[dict]:
    """A file path, or ``bt:<experiment id>`` read with ``BRAINTRUST_API_KEY``."""
    if source.startswith("bt:"):
        key = api_key or os.environ.get("BRAINTRUST_API_KEY", "")
        if not key:
            raise SystemExit("set BRAINTRUST_API_KEY (a read-only key) to read bt:<experiment id>")
        return fetch(source[3:], key)
    with open(source) as f:
        return parse_events(f.read())


def _demo() -> None:
    ev = lambda rid, s, cost: {"span_id": rid, "root_span_id": rid, "origin": {"id": rid},  # noqa
                               "scores": {"correct": s}, "metrics": {"estimated_cost": cost}}
    base = [ev(f"r{i}", 1, 0.01) for i in range(120)]
    cand = [ev(f"r{i}", 1, 0.002) for i in range(120)]
    assert compare(base, cand, scorer="correct")["promoted"]
    # 118 of 120 against 120 of 120 reads as a tie on a dashboard; the bound says not proven
    lost2 = [ev(f"r{i}", 1 if i >= 2 else 0, 0.002) for i in range(120)]
    v = compare(base, lost2, scorer="correct")
    assert v["lost"] == 2 and not v["promoted"] and "bound" in v["why"], v
    assert not compare(base[:30], cand[:30], scorer="correct")["promoted"]  # too few cases
    v = compare(base, cand[:-1], scorer="correct")
    assert v["unanswered"] == 1 and not v["promoted"]
    bare = [dict(e, metrics={}) for e in cand]
    v = compare(base, bare, scorer="correct")
    assert not v["promoted"] and "no estimated_cost" in v["why"], v


if __name__ == "__main__":
    _demo()
