"""`margin-referee check`: the verdict on two experiments a team already scored."""
import json

from margin_referee import cli, experiments


def ev(rid, score, cost=0.01, **extra):
    e = {"span_id": rid, "root_span_id": rid, "origin": {"id": rid},
         "scores": {"correct": score}, "metrics": {"estimated_cost": cost}}
    e.update(extra)
    return e


BASE = [ev(f"r{i}", 1) for i in range(120)]
CHEAP = [ev(f"r{i}", 1, 0.002) for i in range(120)]


def check(b, c, **kw):
    return experiments.compare(b, c, scorer="correct", **kw)


def test_a_held_cheaper_candidate_is_promoted():
    v = check(BASE, CHEAP)
    assert v["promoted"] and v["cases"] == 120 and v["spend_usd"] == 0.0


def test_two_losses_in_120_look_like_a_tie_and_are_not_proven():
    v = check(BASE, [ev(f"r{i}", 1 if i >= 2 else 0, 0.002) for i in range(120)])
    assert v["lost"] == 2 and not v["promoted"] and "bound" in v["why"]


def test_too_few_cases_cannot_promote():
    v = check(BASE[:30], CHEAP[:30])
    assert not v["promoted"] and "too few" in v["why"]


def test_several_models_tried_raise_the_bar():
    assert check(BASE, CHEAP)["promoted"]
    v = check(BASE, CHEAP, candidates=8)
    assert not v["promoted"] and v["cases_needed"] > 120


def test_a_case_missing_from_one_side_blocks_promotion():
    v = check(BASE, CHEAP[:-1])
    assert v["unanswered"] == 1 and not v["promoted"]


def test_no_logged_cost_means_no_saving_is_claimed():
    v = check(BASE, [dict(e, metrics={}) for e in CHEAP])
    assert not v["promoted"] and v["candidate_cost_per_case"] is None
    assert "no estimated_cost" in v["why"]


def test_a_partial_cost_sum_never_promotes():
    cand = [dict(e, metrics={}) if i == 0 else e for i, e in enumerate(CHEAP)]
    assert not check(BASE, cand)["promoted"]


def test_only_root_spans_are_cases():
    child = {"span_id": "c1", "root_span_id": "r0", "scores": {"correct": 0}}
    assert check(BASE, CHEAP + [child])["promoted"]


def test_repeated_trials_keep_the_worst():
    again = [ev("r0", 0, 0.002)]
    v = check(BASE, CHEAP + again)
    assert v["lost"] == 1


def test_unscored_cases_are_counted_not_guessed():
    cand = [ev(f"r{i}", None if i == 0 else 1, 0.002) for i in range(120)]
    v = check(BASE, cand)
    assert v["unscored"]["candidate"] == 1 and v["unanswered"] == 1 and not v["promoted"]


def test_a_partial_score_is_a_fail_at_the_default_threshold():
    cand = [ev(f"r{i}", 0.9, 0.002) for i in range(120)]
    assert check(BASE, cand)["candidate_passed"] == 0
    assert check(BASE, cand, threshold=0.9)["candidate_passed"] == 120


def test_pairs_by_input_when_there_is_no_dataset_record():
    b = [{"input": {"q": i}, "scores": {"correct": 1}, "metrics": {"estimated_cost": 0.01}}
         for i in range(120)]
    c = [{"input": {"q": i}, "scores": {"correct": 1}, "metrics": {"estimated_cost": 0.001}}
         for i in reversed(range(120))]
    assert check(b, c)["promoted"]


def test_reads_array_jsonl_and_the_fetch_page():
    rows = BASE[:3]
    assert len(experiments.parse_events(json.dumps(rows))) == 3
    assert len(experiments.parse_events("\n".join(json.dumps(r) for r in rows))) == 3
    assert len(experiments.parse_events(json.dumps({"events": rows, "cursor": None}))) == 3


def test_cli_exit_code_is_the_verdict(tmp_path, capsys):
    b, c, bad = tmp_path / "b.jsonl", tmp_path / "c.jsonl", tmp_path / "bad.jsonl"
    b.write_text("\n".join(json.dumps(e) for e in BASE))
    c.write_text("\n".join(json.dumps(e) for e in CHEAP))
    bad.write_text("\n".join(json.dumps(e) for e in CHEAP[:40]))
    assert cli.main(["check", str(b), str(c), "--scorer", "correct"]) == 0
    assert json.loads(capsys.readouterr().out)["promoted"]
    assert cli.main(["check", str(b), str(bad), "--scorer", "correct"]) == 1
