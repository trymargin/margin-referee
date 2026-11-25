"""margin-referee, offline: the floors, the paired test, best-of-N, the ledger and the CLI."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from margin_referee import cli, endpoints, replay, stats
from margin_referee.cases import Case


def test_the_floors_are_the_gates():
    assert stats.trials_needed() == 88
    assert [stats.gate_floor(k) for k in (1, 2, 3, 4)] == [100, 125, 147, 163]
    assert stats.smallest_suite(1.0) == 100 and stats.smallest_suite(0.9) == 111


def test_a_small_clean_sweep_does_not_promote():
    assert not stats.paired_bound([(1.0, 1.0)] * 20).supported
    assert stats.paired_bound([(1.0, 1.0)] * 120).supported
    assert stats.paired_bound([]) is None


def _fake(scores, cost):
    it = iter(scores)

    def chat(messages, **_):
        return SimpleNamespace(text="yes" if next(it) else "no", cost_usd=cost,
                               finish_reason="stop")
    return chat


def _cases(n, start=0):
    return [Case(f"c{i}", [{"role": "user", "content": f"q{i}"}], "equals", "yes")
            for i in range(start, start + n)]


def test_the_best_of_several_must_clear_a_higher_bar(tmp_path):
    cases = _cases(120)
    res = replay.run(cases, _fake([1] * 120, 0.01), _fake([1] * 120, 0.001), cap_usd=10,
                     workers=1)
    assert replay.verdict(res, 1)["promoted"]
    assert not replay.verdict(res, 4)["promoted"]          # 120 < 163 for best of 4


def test_an_unanswered_case_promotes_nothing():
    def broken(messages, **_):
        raise TimeoutError("slow")
    res = replay.run(_cases(120), _fake([1] * 120, 0.01), broken, cap_usd=10, workers=1)
    v = replay.verdict(res)
    assert not v["promoted"] and v["unanswered"] == 120


def test_renaming_cases_does_not_reset_the_count_but_fresh_cases_do(tmp_path):
    ledger = tmp_path / "ledger.json"
    assert replay.record_and_count(_cases(150), "b", "m1", ledger) == 1
    renamed = [Case("x" + c.id, c.messages, c.check, c.value) for c in _cases(150)]
    assert replay.record_and_count(renamed, "b", "m2", ledger) == 2
    assert replay.record_and_count(_cases(200, start=1000), "b", "m3", ledger) == 1


def test_an_unpriced_endpoint_is_refused():
    with pytest.raises(endpoints.EndpointError, match="price"):
        endpoints.parse({"kind": "bedrock", "region": "us-east-1", "model": "m", "api_key": "k"})


def test_fit_reports_ready_or_not(tmp_path, capsys):
    f = tmp_path / "cases.jsonl"
    f.write_text("\n".join(json.dumps({"input": f"q{i}", "expected": "yes"}) for i in range(50)))
    assert cli.main(["fit", str(f)]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["graded"] == 50 and out["needed"] == 111 and not out["ready"]


def test_the_reason_fits_the_number_of_candidates():
    res = replay.run(_cases(5), _fake([1] * 5, 0.01), _fake([1] * 5, 0.001), cap_usd=1, workers=1)
    why = replay.verdict(res, 1)["why"]
    assert why == "5 cases are too few to prove a switch: it needs about 100"
    assert "confirm it alone on fresh cases" in replay.verdict(res, 3)["why"]
