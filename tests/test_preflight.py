"""Runs that cannot reach a verdict are refused before they spend (2026-11-27).

One test per cause measured that week: an output limit that cut answers off, a 402 mid-run, a
candidate that 404s on every call, a cap smaller than the run, and a suite smaller than the
gate's floor. No network: every model is a fake.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from margin_referee import cases as cases_mod
from margin_referee import cli, endpoints, replay, stats
from margin_referee.cases import Case

FLOOR = stats.gate_floor(1)


def _cases(n):
    return [Case(f"c{i}", [{"role": "user", "content": f"q{i}"}], "equals", "yes")
            for i in range(n)]


class Fake:
    """A model that answers "yes" at ``cost`` unless ``behave(i, max_tokens, nth_call)`` returns
    an answer, a finish reason, or raises. Counts its calls and the limits it was sent."""

    def __init__(self, cost, behave=None):
        self.cost, self.behave = cost, behave or (lambda i, limit, nth: None)
        self.calls, self.limits, self.seen, self.lock = 0, [], {}, threading.Lock()

    def __call__(self, messages, *, max_tokens, **_):
        i = int(messages[-1]["content"][1:])
        with self.lock:
            self.calls += 1
            self.limits.append(max_tokens)
            self.seen[i] = nth = self.seen.get(i, 0) + 1
        got = self.behave(i, max_tokens, nth)
        if got == "length":
            return SimpleNamespace(text="", cost_usd=self.cost, finish_reason="length")
        return SimpleNamespace(text=got or "yes", cost_usd=self.cost, finish_reason="stop")


def _run(n, base, cand, cap=10.0, **kw):
    cs = _cases(n)
    run_kw = {k: kw.pop(k) for k in ("workers",) if k in kw}
    res = cases_mod.checked_run(
        lambda c, **k: replay.run(c, base, cand, **run_kw, **k), cs, cap_usd=cap,
        floor=FLOOR, **kw)
    return res, replay.verdict(res)


def _raises(text, when=lambda i, nth: True):
    def behave(i, limit, nth):
        if when(i, nth):
            raise RuntimeError(text)
    return behave


def test_cause_5_a_suite_below_the_gate_floor_is_refused_before_any_call():
    base, cand = Fake(0.01), Fake(0.001)
    res, v = _run(41, base, cand)
    assert (base.calls, cand.calls, res.spend_usd) == (0, 0, 0.0)
    assert v["decision"] == "refused_before_spend" and not v["promoted"]
    assert v["cases_needed"] == 100 and "the gate needs 100" in v["why"]
    assert "Add 59 more" in v["why"]
    assert v["preflight"] == {"size": "refused", "smoke": "skipped", "budget": "unchecked",
                              "estimate_usd": None, "balance_usd": None, "max_tokens": 8192}


def test_anyway_runs_a_small_suite_as_a_check_that_can_only_find_a_loss():
    res, v = _run(41, Fake(0.01), Fake(0.001), anyway=True)
    assert v["preflight"]["size"] == "overridden"
    assert "can only find a loss" in v["preflight"]["note"]
    assert v["decision"] == "not_enough_evidence" and v["cases"] == 41 and not v["promoted"]
    lost = Fake(0.001, lambda i, limit, nth: "no")
    _, v = _run(41, Fake(0.01), lost, anyway=True)
    assert v["decision"] == "do_not_ship"  # a loss it can still find


def test_cause_3_a_candidate_that_404s_is_refused_on_the_smoke_cases():
    base = Fake(0.01)
    cand = Fake(0.0, _raises("HTTP 404 from the endpoint: b'no chat/completions for :batch'"))
    res, v = _run(314, base, cand)
    assert v["decision"] == "refused_before_spend" and v["preflight"]["smoke"] == "refused"
    assert "HTTP 404 from the endpoint: b'no chat/completions for :batch'" in v["why"]
    assert "c0 (candidate)" in v["why"] and "$0.02 spent" in v["why"]
    assert base.calls == 2 and cand.calls == 2      # the two smoke cases, not 314
    assert res.spend_usd == pytest.approx(0.02)


def test_cause_1_an_answer_still_cut_off_after_one_raise_is_refused_naming_the_limit():
    cand = Fake(0.001, lambda i, limit, nth: "length" if limit < 8192 else None)
    res, v = _run(120, Fake(0.01), cand, max_tokens=2048)
    assert v["decision"] == "refused_before_spend" and v["preflight"]["smoke"] == "refused"
    assert "cut off by the output limit" in v["why"] and "4096 output tokens" in v["why"]
    assert v["preflight"]["max_tokens"] == 4096
    assert sorted(set(cand.limits)) == [2048, 4096] and cand.calls == 4
    assert res.spend_usd <= 4 * (0.01 + 0.001) + 1e-9  # the smoke cases and their one retry


def test_cause_1_a_cut_off_smoke_answer_raises_the_limit_for_the_whole_run():
    cand = Fake(0.001, lambda i, limit, nth: "length" if limit < 4096 else None)
    res, v = _run(120, Fake(0.01), cand, max_tokens=2048)
    assert v["decision"] == "ship" and v["promoted"] and v["unanswered"] == 0
    assert v["preflight"]["max_tokens"] == 4096 and v["cases"] == 120
    # only the two smoke cases were ever sent the low limit
    assert cand.limits.count(2048) == 2 and cand.limits.count(4096) == 120


def test_a_case_cut_off_mid_run_is_retried_once_at_double_the_limit():
    cand = Fake(0.001, lambda i, limit, nth: "length" if i == 50 and limit < 16384 else None)
    res, v = _run(120, Fake(0.01), cand)
    assert v["unanswered"] == 0 and v["cases"] == 120 and v["decision"] == "ship"
    assert v["preflight"]["max_tokens"] == 16384 and cand.seen[50] == 2


def test_cause_2_a_402_stops_the_run_at_once_and_is_not_retried():
    cand = Fake(0.001, _raises("HTTP 402 from the endpoint: b'Insufficient credits'",
                               lambda i, nth: i >= 40))
    base = Fake(0.01)
    res, v = _run(200, base, cand, workers=8)
    assert res.stopped.startswith("stopped after 41 of 200 cases: out of credit")
    assert v["decision"] == "not_enough_evidence" and v["why"] == res.stopped
    assert cand.calls == 41 and base.calls == 41      # the rest was never paid for
    assert cand.seen[40] == 1                         # a 402 is not worth a retry


def test_cause_2_a_key_that_cannot_cover_the_estimate_is_refused_before_the_full_run():
    base, cand = Fake(0.01), Fake(0.001)
    res, v = _run(200, base, cand, balance=lambda: 0.5)
    assert v["decision"] == "refused_before_spend" and v["preflight"]["budget"] == "refused"
    assert v["preflight"]["balance_usd"] == 0.5
    assert v["preflight"]["estimate_usd"] == pytest.approx(0.011 * 200 * 1.2)
    assert "the key has $0.50 left" in v["why"] and base.calls == 2 and cand.calls == 2


def test_a_balance_that_cannot_be_read_never_blocks_the_run():
    def down():
        raise OSError("no route")
    _, v = _run(120, Fake(0.01), Fake(0.001), balance=down)
    assert v["decision"] == "ship" and v["preflight"]["balance_note"] == "balance not checked"
    assert v["preflight"]["budget"] == "ok" and v["preflight"]["balance_usd"] is None
    _, v = _run(120, Fake(0.01), Fake(0.001), balance=lambda: None)
    assert v["preflight"]["balance_note"] == "no limit to read on this key"


def test_cause_4_a_cap_smaller_than_the_estimate_is_refused_with_the_cap_needed():
    base, cand = Fake(0.02), Fake(0.02)
    res, v = _run(120, base, cand, cap=1.0)
    assert v["decision"] == "refused_before_spend" and v["preflight"]["budget"] == "refused"
    assert v["preflight"]["estimate_usd"] == pytest.approx(0.04 * 120 * 1.2)   # 5.76
    assert "estimated at $5.76" in v["why"] and "the cap is $1.00" in v["why"]
    assert "raise the cap to about $5.76" in v["why"]
    assert base.calls == 2 and res.spend_usd == pytest.approx(0.08)


def test_a_transient_error_is_retried_once_when_the_run_ends():
    cand = Fake(0.001, _raises("HTTP 429 from the endpoint: b'slow down'",
                               lambda i, nth: i == 7 and nth == 1))
    res, v = _run(120, Fake(0.01), cand)
    assert v["unanswered"] == 0 and v["cases"] == 120 and v["decision"] == "ship"
    assert cand.seen[7] == 2
    always = Fake(0.001, _raises("HTTP 503 from the endpoint: b''", lambda i, nth: i == 7))
    res, v = _run(120, Fake(0.01), always)
    assert always.seen[7] == 2 and v["unanswered"] == 1    # once more, then it stands
    assert v["unanswered_reasons"][0]["reason"].startswith("RuntimeError: HTTP 503")


def test_a_run_that_can_no_longer_reach_the_floor_stops_spending():
    cand = Fake(0.001, _raises("HTTP 400 from the endpoint: b'schema rejected'",
                               lambda i, nth: i >= 10))
    base = Fake(0.01)
    res, v = _run(120, base, cand, workers=8)
    # 120 cases, floor 100: the 21st unanswered case makes ship impossible
    assert res.stopped.startswith("stopped after 33 of 120 cases: 23 got no gradable answer")
    assert "at most 97 can still be judged and the gate needs 100" in res.stopped
    assert v["decision"] == "not_enough_evidence" and v["why"] == res.stopped
    assert base.calls == 33 and cand.seen.get(119) is None


def test_an_ordinary_passing_suite_gets_the_same_verdict_and_numbers_as_before():
    cs = _cases(120)
    plain = replay.verdict(replay.run(cs, Fake(0.01), Fake(0.001), cap_usd=10))
    base, cand = Fake(0.01), Fake(0.001)
    res, checked = _run(120, base, cand)
    assert checked.pop("preflight") == {"size": "ok", "smoke": "ok", "budget": "ok",
                                        "estimate_usd": pytest.approx(0.011 * 120 * 1.2),
                                        "balance_usd": None, "max_tokens": 8192}
    assert plain.pop("preflight") is None
    assert checked == plain and checked["decision"] == "ship" and checked["promoted"]
    assert base.calls == 120 and cand.calls == 120      # the smoke cases are not paid twice


@pytest.mark.parametrize("why,kind", [
    ("RuntimeError: HTTP 402 from the endpoint: b''", "credit"),
    ("HTTPError: HTTP Error 402: Payment Required", "credit"),
    ("OpenRouterUnavailable: refused with HTTP 403: this account or key has hit its limit",
     "credit"),
    ("RuntimeError: HTTP 429 from the endpoint: b''", "transient"),
    ("RuntimeError: HTTP 500 from the endpoint: b''", "transient"),
    ("TimeoutError: The read operation timed out", "transient"),
    ("ConnectionResetError: [Errno 54] Connection reset by peer", "transient"),
    ("RuntimeError: HTTP 400 from the endpoint: b''", "fatal"),
    ("RuntimeError: HTTP 401 from the endpoint: b''", "fatal"),
    ("RuntimeError: HTTP 403 from the endpoint: b''", "fatal"),
    ("RuntimeError: HTTP 404 from the endpoint: b''", "fatal"),
    ("RuntimeError: HTTP 422 from the endpoint: b''", "fatal"),
    ("check failed: ValueError: the first tool call cannot be read", "fatal"),
    ("cut off by the output limit", "cut_off"),
    ("truncated at 2048 output tokens (reasoning counts toward it)", "cut_off"),
])
def test_each_failure_is_sorted_into_the_action_the_run_takes(why, kind):
    assert cases_mod.error_kind(why) == kind


def test_the_cli_exits_3_with_the_reason_first_and_records_nothing(
        tmp_path, monkeypatch, capsys):
    f = tmp_path / "cases.jsonl"
    f.write_text("\n".join(json.dumps({"id": f"c{i}", "input": f"q{i}", "expected": "yes"})
                           for i in range(41)))
    ledger = tmp_path / "ledger.json"
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(replay.record_and_count, "__defaults__", (ledger,))
    base, cand = Fake(0.01), Fake(0.001)
    monkeypatch.setattr(endpoints, "chat_for",
                        lambda ep: base if ep.model == "base/m" else cand)
    argv = ["replay", str(f), "--baseline", "base/m", "--candidate", "cand/m", "--cap", "5",
            "--receipt", str(tmp_path / "r.json")]
    assert cli.main(argv) == 3
    got = capsys.readouterr()
    assert got.err.startswith("refused before spend: 41 gradable case(s) cannot prove")
    assert json.loads(got.out)["decision"] == "refused_before_spend"
    assert base.calls == 0 and not ledger.exists() and not (tmp_path / "r.json").exists()
    assert cli.main(argv + ["--anyway"]) == 1      # judged, and too small to promote
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "not_enough_evidence" and out["preflight"]["size"] == "overridden"
    assert base.calls == 41 and ledger.exists()


def test_the_key_balance_is_the_lower_of_the_key_limit_and_the_account_credit(monkeypatch):
    ep = endpoints.parse({"kind": "openrouter", "model": "m/x", "api_key": "k"})
    data = {"key": {"limit_remaining": 4.0}, "credits": {"total_credits": 10, "total_usage": 9}}
    monkeypatch.setattr(endpoints, "_get", lambda url, headers: data[url.rsplit("/", 1)[1]])
    assert endpoints.credit_left(ep, ep) == pytest.approx(1.0)
    data["key"] = {"limit_remaining": None}
    data["credits"] = {}                                   # a key that cannot read the account
    assert endpoints.credit_left(ep) is None
    own = endpoints.parse({"kind": "bedrock", "region": "us-east-1", "model": "m", "api_key": "k",
                           "price": {"input_per_mtok": 1, "output_per_mtok": 2}})
    assert endpoints.credit_left(own) is None              # not an OpenRouter key: nothing read
