"""`margin-referee ontology-sql`: the verdict on an NVIDIA ontology-sql-eval run's saved files.

The fixtures below are shaped like the real files, read from NVIDIA/ontology-sql-eval at commit
871bb9244ae8efc57a950721e4496dbb4c3adbb5 (Apache-2.0) through the GitHub API. Nothing of theirs
was installed or run, and no row here is copied from a real run: the values are made up.

* layout ``<run-dir>/{control,treatment}/<dataset>.jsonl`` with ``<dataset>_failures.jsonl``,
  ``<dataset>_materialized_inputs.jsonl`` and ``<dataset>_aggregate_metrics.json`` beside it:
  ``main.py`` (RUN_LABEL, ``base / label / f"{database_name}.jsonl"``), ``README.md`` "Reading
  the results", ``ontology_sql_eval/gym/compare.py`` (``unfinished``)
* control row fields: ``resources_servers/schema_only_sql/app.py``, SchemaOnlySqlVerifyResponse
* treatment row fields: ``resources_servers/auto_ontology_sql/app.py``,
  AutoOntologySqlVerifyResponse (the ``agent_*`` token totals)
* ``failure_reason`` codes: ``ontology_sql_eval/gym/exec_match.py``, FailureCode
* ``ng_model_call_capture``: ``scripts/run_gym_arm.sh`` says Gym attaches it to each rollout row;
  its shape is from NVIDIA-NeMo/Gym at 5d6429f5 (the commit ontology-sql-eval pins),
  ``fern/versions/latest/pages/model-server/model-call-capture.mdx``
* cached input: the field names are from a real run's files (871bb924 on WideWorldImporters,
  2026-11-27, ``deepseek/deepseek-v4.1-flash``): each call in
  ``treatment/wideworldimporters.jsonl``'s ``ng_model_call_capture.calls`` carries
  ``cached_tokens`` (a part of ``tokens_in``, beside ``cache_hit`` and
  ``cache_creation_tokens``), the block's ``metrics`` carries their sum, and the treatment row
  carries ``agent_cached_input_tokens``. The raw
  ``treatment/model_call_capture/<rollout>.capture.jsonl`` lines hold the same count at
  ``response.usage.input_tokens_details.cached_tokens``. The values below are still made up.
"""
import json

import pytest

from margin_referee import cli, ontology_sql


def capture(calls):
    """Gym's per-row block: ``calls`` is ``[(tokens_in, tokens_out, tokens_total)]``, with an
    optional fourth ``cached_tokens``; a call given without one carries no cached count."""
    return {"rollout_id": "0-0",
            "metrics": {"tokens_in": sum(c[0] or 0 for c in calls),
                        "tokens_out": sum(c[1] or 0 for c in calls), "num_calls": len(calls)},
            "calls": [{"model_call_id": f"c{n}", "call_index": n,
                       "model_ref": {"type": "responses_api_models", "name": "policy_model"},
                       "model": "azure/anthropic/claude-opus-5-5", "dialect": "responses",
                       "status_code": 200, "tokens_in": c[0], "tokens_out": c[1],
                       "tokens_total": c[2],
                       **({"cache_hit": c[3] > 0, "cached_tokens": c[3],
                           "cache_creation_tokens": 0} if len(c) > 3 else {})}
                      for n, c in enumerate(calls)]}


def control_row(i, ok, reason=None, calls=((2700, 500, 3200),), **extra):
    row = {"reward": 1.0 if ok else 0.0, "task_id": f"bird-{i}", "question": f"q{i}",
           "gt_sql": "SELECT 1", "db_id": "california_schools" if i % 2 else "financial",
           "dataset": "bird", "dialect": "sqlite",
           "difficulty": ("simple", "moderate", "challenging")[i % 3],
           "instance_config": {"mask_sample": False}, "model_output": "```sql\nSELECT 1\n```",
           "extracted_sql": "SELECT 1", "had_codeblock": True, "execution_match": bool(ok),
           "failure_reason": reason or ("none" if ok else "wrong_result"),
           "gold_result": "nonempty_gold"}
    if calls:
        row["ng_model_call_capture"] = capture(calls)
    return {**row, **extra}


def treatment_row(i, ok, reason=None, calls=None, **extra):
    row = control_row(i, ok, reason, calls=calls)
    for k in ("model_output", "had_codeblock"):
        del row[k]
    row.update(agent_answer="SELECT 1", agent_error="", agent_nodes=["retrieve", "generate"],
               policy_output_ignored=True, sql_examples_used=0, value_anchors_used=0,
               agent_llm_calls=19, agent_llm_calls_without_usage=0, agent_input_tokens=38000,
               agent_cached_input_tokens=9000, agent_output_tokens=5700, agent_total_tokens=43700)
    return {**row, **extra}


def write_run(root, control, treatment, dispatched=None):
    for arm, rows in (("control", control), ("treatment", treatment)):
        d = root / arm
        (d / "model_call_capture").mkdir(parents=True)
        (d / "bird.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        ids = dispatched if dispatched is not None else [r["task_id"] for r in rows]
        (d / "bird_materialized_inputs.jsonl").write_text(
            "".join(json.dumps({"task_id": t}) + "\n" for t in ids))
        (d / "bird_failures.jsonl").write_text(json.dumps({"task_id": "bird-x"}) + "\n")
        (d / "bird_aggregate_metrics.json").write_text(json.dumps({"pass@1/accuracy": 0.5}))
    return str(root)


def run(tmp_path, control_ok, treatment_ok, n=160, **kw):
    return ontology_sql.compare(write_run(
        tmp_path, [control_row(i, control_ok(i)) for i in range(n)],
        [treatment_row(i, treatment_ok(i)) for i in range(n)]), **kw)


def test_pairs_the_two_arms_by_question_and_ignores_the_sidecar_files(tmp_path):
    v = run(tmp_path, lambda i: True, lambda i: True, n=12)
    assert v["cases"] == 12 and v["unanswered"] == 0
    assert v["baseline_passed"] == 12 and v["candidate_passed"] == 12
    assert sorted(f.rsplit("/", 2)[-2] for f in v["files_read"]) == ["control", "treatment"]
    assert v["spend_usd"] == 0.0 and v["segment_by"] == "difficulty"


def test_a_pass_is_the_verifiers_positive_verdict_only():
    assert ontology_sql.outcome(control_row(0, True)) == (1.0, "")
    assert ontology_sql.outcome(control_row(0, False)) == (0.0, "")
    for reason in ("execution_error", "execution_timeout", "no_sql_extracted"):
        assert ontology_sql.outcome(control_row(0, False, reason)) == (0.0, "")
    undecided = [control_row(0, False, "none"), control_row(0, True, reward=None),
                 control_row(0, True, reward="1.0"), control_row(0, True, reward=True),
                 control_row(0, True, "wrong_result"), control_row(0, True, execution_match=False),
                 control_row(0, False, "something_new"), {"task_id": "bird-0"}]
    for row in undecided:
        score, why = ontology_sql.outcome(row)
        assert score is None and why, row


@pytest.mark.parametrize("reason", ["no_model_output", "unknown_error", "gold_execution_error",
                                    "gold_execution_timeout", "gold_failure_scored"])
def test_an_errored_verdict_is_unanswered_with_its_reason_never_a_fail(tmp_path, reason):
    control = [control_row(i, True) for i in range(160)]
    treatment = [treatment_row(i, True) for i in range(160)]
    treatment[7] = treatment_row(7, False, reason)
    v = ontology_sql.compare(write_run(tmp_path, control, treatment), price_in=5, price_out=25)
    assert v["unanswered"] == 1 and v["lost"] == 0 and v["cases"] == 159
    assert reason in v["unanswered_reasons"][0]["reason"]
    assert not v["promoted"] and v["decision"] == "not_enough_evidence"


def test_a_match_against_a_gold_query_that_never_ran_is_not_a_pass():
    beaver = control_row(0, True, gold_result="failed_gold")
    assert ontology_sql.outcome(beaver)[0] is None
    # BIRD: the prediction timed out and used the shared time limit, so gold was not run
    assert ontology_sql.outcome(control_row(0, False, "execution_timeout",
                                            gold_result="failed_gold")) == (0.0, "")


def test_a_question_one_arm_has_no_row_for_is_unanswered(tmp_path):
    control = [control_row(i, True) for i in range(6)]
    treatment = [treatment_row(i, True) for i in range(5)]
    v = ontology_sql.compare(write_run(tmp_path, control, treatment,
                                       dispatched=[f"bird-{i}" for i in range(7)]))
    assert v["cases"] == 5 and v["unanswered"] == 2 and not v["promoted"]
    reasons = {r["reason"] for r in v["unanswered_reasons"]}
    # bird-5 ran on control only; bird-6 was dispatched and neither arm has a row for it
    assert reasons == {"no result for this question (it failed to run or never ran)"}
    assert ontology_sql.compare(write_run(tmp_path / "b", control, treatment))[
        "unanswered_reasons"][0]["reason"] == "no row in this arm"


def test_tokens_are_summed_from_the_capture_block_when_it_is_there(tmp_path):
    calls = ((2000, 300, 2300), (5940, 28, 3000))  # the second: tokens_in counts a write twice
    v = ontology_sql.compare(write_run(
        tmp_path, [control_row(i, True) for i in range(4)],
        [treatment_row(i, True, calls=calls) for i in range(4)]))
    assert v["tokens"]["baseline"] == {"input_per_case": 2700.0, "output_per_case": 500.0,
                                       "is_a_floor": False,
                                       "models_seen": ["azure/anthropic/claude-opus-5-5"]}
    assert v["tokens"]["candidate"]["input_per_case"] == 2000 + (3000 - 28)
    assert v["tokens"]["candidate"]["output_per_case"] == 328
    assert v["token_ratio"] == round((4972 + 328) / 3200, 2)


def test_without_a_capture_block_the_treatment_rows_own_totals_are_used(tmp_path):
    v = run(tmp_path, lambda i: True, lambda i: True, n=4)
    assert v["tokens"]["candidate"]["input_per_case"] == 38000
    assert v["tokens"]["candidate"]["output_per_case"] == 5700
    assert v["token_ratio"] == round(43700 / 3200, 2)
    rows = [control_row(i, True, calls=None) for i in range(4)]
    v = ontology_sql.compare(write_run(tmp_path / "b", rows,
                                       [treatment_row(i, True) for i in range(4)]),
                             price_in=5, price_out=25)
    assert v["tokens"] == {"baseline": None, "candidate": None} and v["cost"] == "not_recorded"
    assert "no token counts" in v["cost_basis"]


def test_a_call_with_no_usage_marks_the_total_a_floor():
    row = control_row(0, True, calls=((100, 10, 110), (None, None, None)))
    assert ontology_sql.tokens(row)[:3] == (100, 10, True)
    assert ontology_sql.tokens(treatment_row(0, True, agent_llm_calls_without_usage=2))[2] is True


COLD = ((14000, 800, 14800, 0),)                          # one call, nothing cached
WARM = ((100, 600, 700, 0), (20000, 400, 20400, 14000), (30000, 500, 30500, 26000))


def cache_run(tmp_path, control_calls=COLD, treatment_calls=WARM, n=4, **kw):
    return ontology_sql.compare(write_run(
        tmp_path, [control_row(i, True, calls=control_calls) for i in range(n)],
        [treatment_row(i, True, calls=treatment_calls) for i in range(n)]), **kw)


def test_cached_input_is_summed_per_arm_from_each_call(tmp_path):
    assert ontology_sql.tokens(treatment_row(0, True, calls=WARM))[4] == 40000
    assert ontology_sql.tokens(treatment_row(0, True))[4] == 9000  # agent_cached_input_tokens
    v = cache_run(tmp_path)
    assert v["cache"]["baseline"] == {"fresh_in": 56000.0, "cached_in": 0.0, "share": 0.0}
    assert v["cache"]["candidate"] == {"fresh_in": 4 * 10100.0, "cached_in": 4 * 40000.0,
                                       "share": round(40000 / 50100, 4)}
    assert v["tokens"]["candidate"]["input_per_case"] == 50100  # cached is part of input


def test_cached_input_is_priced_at_its_own_rate_only_when_the_rate_is_given(tmp_path):
    fresh = cache_run(tmp_path / "a", price_in=5, price_out=25)
    assert fresh["candidate_cost_per_case"] == pytest.approx((50100 * 5 + 1500 * 25) / 1e6)
    assert fresh["cost_basis"].endswith("cached input priced as fresh: pass --price-cached")
    aware = cache_run(tmp_path / "b", price_in=5, price_out=25, price_cached=0.5)
    assert aware["candidate_cost_per_case"] == pytest.approx(
        (10100 * 5 + 40000 * 0.5 + 1500 * 25) / 1e6)
    assert aware["baseline_cost_per_case"] == fresh["baseline_cost_per_case"]  # nothing cached
    assert aware["cost_basis"].endswith("cached input at the cached rate")
    with pytest.raises(ValueError):
        cache_run(tmp_path / "c", price_cached=0.5)


def test_an_unrecorded_cache_split_is_said_and_never_guessed(tmp_path):
    assert ontology_sql.tokens(control_row(0, True))[4] is None  # the call has no cached count
    one_missing = ((100, 10, 110, 50), (100, 10, 110))
    assert ontology_sql.tokens(control_row(0, True, calls=one_missing))[4] is None
    v = cache_run(tmp_path, control_calls=((14000, 800, 14800),), price_in=5, price_out=25,
                  price_cached=0.5)
    assert v["cache"]["baseline"] is None and v["cache"]["line"] is None
    assert v["cache"]["candidate"]["cached_in"] == 160000.0
    assert "cache split not recorded" in v["cost_basis"]
    # one arm priced with its cache and the other without would bias the comparison
    assert v["candidate_cost_per_case"] == pytest.approx((50100 * 5 + 1500 * 25) / 1e6)


def test_a_ten_point_move_in_the_cache_hit_share_gets_its_own_line(tmp_path):
    rose = cache_run(tmp_path / "a")
    assert rose["cache"]["line"] == "the candidate's cache-hit share rose from 0% to 80%"
    fell = cache_run(tmp_path / "b", control_calls=WARM, treatment_calls=COLD)
    assert fell["cache"]["line"] == "the candidate's cache-hit share fell from 80% to 0%"
    near = ((10000, 100, 10100, 7100),)  # 71% against 80%: under ten points
    assert cache_run(tmp_path / "c", control_calls=WARM,
                     treatment_calls=near)["cache"]["line"] is None


def test_the_cached_rate_moves_cost_and_never_the_quality_verdict(tmp_path):
    quality = ("quality", "won", "lost", "cases", "unanswered", "baseline_passed",
               "candidate_passed", "quality_gap", "gain", "segments")

    def at(sub, **price):
        return ontology_sql.compare(write_run(
            tmp_path / sub, [control_row(i, i >= 2, calls=COLD) for i in range(160)],
            [treatment_row(i, i not in (2, 3), calls=WARM) for i in range(160)]), **price)
    plain = at("a", price_in=5, price_out=5)
    aware = at("b", price_in=5, price_out=5, price_cached=0.05)
    assert plain["cost"] == "dearer" and aware["cost"] == "cheaper"
    assert aware["candidate_cost_per_case"] < plain["candidate_cost_per_case"]
    assert {k: plain[k] for k in quality} == {k: aware[k] for k in quality}
    assert aware["quality"] == "holds" and not aware["gain"]["proven"]
    # the cost column is the only thing that moved: a cheaper arm that holds may ship
    assert (plain["decision"], aware["decision"]) == ("not_enough_evidence", "ship")
    # a losing arm stays do_not_ship however cheap its cached input is
    lost = ontology_sql.compare(write_run(
        tmp_path / "c", [control_row(i, True, calls=COLD) for i in range(160)],
        [treatment_row(i, i >= 30, calls=WARM) for i in range(160)]),
        price_in=5, price_out=25, price_cached=0.0)
    assert lost["decision"] == "do_not_ship" and not lost["promoted"]


def test_without_a_price_cost_is_not_recorded_and_nothing_is_promoted(tmp_path):
    v = run(tmp_path, lambda i: True, lambda i: True)
    assert v["cost"] == "not_recorded" and v["cost_basis"].startswith("not recorded")
    assert v["baseline_cost_per_case"] is None and v["candidate_cost_per_case"] is None
    assert not v["promoted"] and v["token_ratio"] == 13.66
    with pytest.raises(ValueError):
        ontology_sql.compare(str(tmp_path), price_in=5.0)


def test_a_price_turns_tokens_into_cost_per_question(tmp_path):
    v = run(tmp_path, lambda i: True, lambda i: True, price_in=5, price_out=25)
    assert v["baseline_cost_per_case"] == pytest.approx((2700 * 5 + 500 * 25) / 1e6)
    assert v["candidate_cost_per_case"] == pytest.approx((38000 * 5 + 5700 * 25) / 1e6)
    assert v["cost"] == "dearer"


def test_a_proven_gain_on_the_dearer_arm_is_ship_if_worth_the_cost(tmp_path):
    v = run(tmp_path, lambda i: i < 100, lambda i: i < 140, price_in=5, price_out=25)
    assert v["decision"] == "ship_if_worth_the_cost" and v["quality"] == "holds"
    assert (v["won"], v["lost"]) == (40, 0) and v["gain"]["proven"]
    assert v["gain"]["lower_bound_points"] > 0 and not v["promoted"]
    assert "won 40, lost 0" in v["why"] and "costs more per question" in v["why"]
    unpriced = run(tmp_path / "b", lambda i: i < 100, lambda i: i < 140)
    assert unpriced["decision"] == "ship_if_worth_the_cost"
    assert "13.66x the tokens" in unpriced["why"] and unpriced["cost"] == "not_recorded"


def test_a_flat_result_on_a_14x_token_arm_is_not_a_reason_to_ship(tmp_path):
    def control_ok(i):
        return i >= 2

    def treatment_ok(i):
        return i not in (2, 3)  # wins 0 and 1, loses 2 and 3: a tie
    for price in ({}, {"price_in": 5, "price_out": 25}):
        v = run(tmp_path / str(len(price)), control_ok, treatment_ok, **price)
        assert v["quality"] == "holds" and (v["won"], v["lost"]) == (2, 2)
        assert v["decision"] == "not_enough_evidence" and not v["promoted"]
        assert not v["gain"]["proven"] and v["gain"]["lower_bound_points"] < 0
        assert "no gain is proven" in v["why"] and "won 2, lost 2" in v["why"]


def test_the_ontology_arm_losing_questions_is_do_not_ship(tmp_path):
    v = run(tmp_path, lambda i: True, lambda i: i >= 30, price_in=5, price_out=25)
    assert v["decision"] == "do_not_ship" and v["lost"] == 30


def test_dropping_the_layer_when_quality_holds_is_a_plain_ship(tmp_path):
    root = write_run(tmp_path, [control_row(i, True) for i in range(160)],
                     [treatment_row(i, True) for i in range(160)])
    v = ontology_sql.compare(root, baseline_arm="treatment", candidate_arm="control",
                             price_in=5, price_out=25)
    assert v["decision"] == "ship" and v["promoted"] and v["cost"] == "cheaper"
    unpriced = ontology_sql.compare(root, baseline_arm="ontology", candidate_arm="schema-only")
    assert unpriced["decision"] == "ship" and not unpriced["promoted"]


def test_with_repeats_a_question_passes_only_if_every_repeat_passed(tmp_path):
    control = [control_row(i, True) for i in range(4)] + [control_row(0, False)]
    treatment = [treatment_row(i, True) for i in range(4)]
    treatment.append(treatment_row(1, False, "no_model_output"))
    v = ontology_sql.compare(write_run(tmp_path, control, treatment))
    assert v["cases"] == 3 and v["baseline_passed"] == 2 and v["unanswered"] == 1


def test_a_loss_in_one_difficulty_is_read_on_its_own(tmp_path):
    v = run(tmp_path, lambda i: True, lambda i: not (i % 3 == 2 and i < 90),
            n=300, price_in=5, price_out=25)
    worse = [s["segment"] for s in v["segments"] if s["worse"]]
    assert worse == ["challenging"] and v["decision"] == "do_not_ship"
    assert run(tmp_path / "b", lambda i: True, lambda i: True, n=12,
               segment_by="db_id")["segment_by"] == "db_id"


def test_the_command_prints_the_verdict_and_json(tmp_path, capsys):
    root = write_run(tmp_path, [control_row(i, i < 100) for i in range(160)],
                     [treatment_row(i, i < 140) for i in range(160)])
    assert cli.main(["ontology-sql", root, "--price-in", "5", "--price-out", "25"]) == 1
    text = capsys.readouterr().out
    assert text.startswith("ship_if_worth_the_cost: a proven gain: the candidate won 40, lost 0")
    assert "control 2700 in / 500 out, treatment 38000 in / 5700 out (13.66x)" in text
    assert "cache-hit share of input: control cache split not recorded, treatment 24%" in text
    root2 = write_run(tmp_path / "warm", [control_row(i, True, calls=COLD) for i in range(4)],
                      [treatment_row(i, True, calls=WARM) for i in range(4)])
    cli.main(["ontology-sql", root2, "--price-in", "5", "--price-out", "25",
              "--price-cached", "0.5"])
    text = capsys.readouterr().out
    assert "cache-hit share of input: control 0%, treatment 80%" in text
    assert "  the candidate's cache-hit share rose from 0% to 80%" in text
    assert "cached input at the cached rate" in text
    out = tmp_path / "receipt.json"
    assert cli.main(["ontology-sql", root, "--candidate-arm", "ontology", "--json",
                     "--receipt", str(out)]) == 1
    v = json.loads(capsys.readouterr().out)
    assert v["candidate"] == "ontology" and v["cost"] == "not_recorded" and out.exists()
    assert cli.main(["ontology-sql", str(tmp_path / "empty")]) == 2
    with pytest.raises(SystemExit):
        cli.main(["ontology-sql", root, "--price-in", "5"])
