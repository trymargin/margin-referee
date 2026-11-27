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


def _candidate(fails: int, errors_at: set):
    """Fails the first ``fails`` cases, raises on the cases numbered in ``errors_at``."""
    def chat(messages, **_):
        i = int(messages[-1]["content"][1:])
        if i in errors_at:
            raise RuntimeError("HTTP Error 402: Payment Required")
        return SimpleNamespace(text="no" if i < fails else "yes", cost_usd=0.001,
                               finish_reason="stop")
    return chat


def test_a_failed_call_does_not_hide_a_loss_it_could_not_have_rescued():
    """20 lost and none won on 199 answered cases: still worse with the one unanswered case
    counted as a win for the candidate, so it is do_not_ship, not not_enough_evidence."""
    res = replay.run(_cases(200), _fake([1] * 200, 0.01), _candidate(20, {199}), cap_usd=10,
                     workers=1, early_stop=False)
    v = replay.verdict(res)
    assert (v["cases"], v["lost"], v["won"], v["unanswered"]) == (199, 20, 0, 1)
    assert (v["decision"], v["quality"], v["promoted"]) == ("do_not_ship", "worse", False)
    assert v["why"].startswith("worse than the bar allows even if every one of the 1 unanswered "
                               "case(s) had gone the candidate's way (most: RuntimeError")


def test_a_failed_call_that_could_have_rescued_it_is_still_not_enough_evidence():
    """12 lost on 199: worse on the answered cases alone, but one more candidate win would put
    it back inside the bar, so the missing answer still matters."""
    res = replay.run(_cases(200), _fake([1] * 200, 0.01), _candidate(12, {199}), cap_usd=10,
                     workers=1, early_stop=False)
    v = replay.verdict(res)
    assert (v["cases"], v["lost"], v["unanswered"]) == (199, 12, 1)
    assert (v["decision"], v["quality"]) == ("not_enough_evidence", "unclear")


def test_unanswered_cases_never_ship_and_a_capped_replay_is_judged_as_before():
    clean = replay.run(_cases(200), _fake([1] * 200, 0.01), _candidate(0, {7}), cap_usd=10,
                       workers=1, early_stop=False)
    v = replay.verdict(clean)
    assert v["decision"] == "not_enough_evidence" and not v["promoted"]
    capped = replay.run(_cases(200), _fake([1] * 200, 0.01), _candidate(20, {50}), cap_usd=1.1,
                        workers=1, early_stop=False)
    v = replay.verdict(capped)
    assert capped.stopped and v["lost"] == 20 and v["unanswered"] == 1
    assert v["decision"] == "not_enough_evidence" and "even if every one" not in v["why"]


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


def test_a_clearly_bad_candidate_stops_early_and_a_stop_never_ships():
    cases = _cases(300)
    res = replay.run(cases, _fake([1] * 300, 0.01), _fake([0] * 300, 0.001), cap_usd=10)
    assert res.stopped_early["after_cases"] == len(res.pairs) == 41 and res.stopped == ""
    v = replay.verdict(res)
    assert v["decision"] == "do_not_ship" and not v["promoted"]
    assert "the remaining 259 cases were not run because the answer was already clear" in v["why"]
    full = replay.run(cases, _fake([1] * 300, 0.01), _fake([0] * 300, 0.001), cap_usd=10,
                      early_stop=False)
    assert full.stopped_early is None and len(full.pairs) == 300
    assert res.spend_usd < full.spend_usd / 7
    good = replay.run(_cases(120), _fake([1] * 120, 0.01), _fake([1] * 120, 0.001), cap_usd=10)
    assert good.stopped_early is None and replay.verdict(good)["decision"] == "ship"
    good.stopped_early = {"after_cases": 120, "of_cases": 300, "looks": 1, "reason": "x"}
    assert replay.verdict(good)["decision"] == "not_enough_evidence"


def test_with_repeats_a_case_passes_only_if_every_run_passed():
    runs: dict = {}

    def flaky(messages, **_):  # fails its second run of the first three cases
        q = messages[-1]["content"]
        runs[q] = runs.get(q, 0) + 1
        bad = q in ("q0", "q1", "q2") and runs[q] == 2
        return SimpleNamespace(text="no" if bad else "yes", cost_usd=0.001, finish_reason="stop")
    res = replay.run(_cases(120), _fake([1] * 360, 0.01), flaky, cap_usd=10, repeats=3)
    assert [c for _, _, c in res.pairs[:4]] == [0.0, 0.0, 0.0, 1.0]
    assert res.spend_usd == pytest.approx(120 * 3 * 0.011)
    v = replay.verdict(res)
    r = v["reliability"]
    assert v["repeats"] == 3 and v["candidate_passed"] == 117
    assert v["candidate_cost_per_case"] == pytest.approx(0.001)  # one run's cost, not three
    assert r["candidate"]["flipped_cases"] == ["c0", "c1", "c2"]
    assert r["candidate"]["pass_pow_k"] == 0.975 and r["candidate"]["pass_at_1"] == 0.9917
    assert r["baseline"]["flipped"] == 0 and r["runs_needed"]["estimate"] is True
    one = replay.run(_cases(120), _fake([1] * 120, 0.01), _fake([1] * 120, 0.001), cap_usd=10)
    assert replay.verdict(one)["reliability"] is None


def test_weights_report_the_traffic_on_worse_cases_and_leave_the_decision():
    res = replay.Result()
    res.pairs = [(f"c{i}", 1.0, 0.0 if i == 0 else 1.0) for i in range(400)]
    res.cost = {"baseline": 4.0, "candidate": 0.4}
    plain = replay.verdict(res)
    v = replay.verdict(res, weight_of={f"c{i}": (600 if i == 0 else 1) for i in range(400)})
    assert plain["weighted"] is None and v["decision"] == plain["decision"] == "ship"
    assert v["weighted"]["worse_traffic_share"] == round(600 / 999, 4)
    assert "breaks cases covering 60.1% of your traffic" in v["why"]
    assert "not a new test" in v["weighted"]["note"]
