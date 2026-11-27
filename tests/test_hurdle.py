"""`margin-referee hurdle`: the verdict on a Hurdle suite's saved runs."""
import json

from margin_referee import cli, hurdle


def rec(task, cond, model, verdict, i, cost=None):
    r = {"state": "complete", "verdict": verdict, "revision": 2,
         "snapshot": {"target": {"id": "api"}, "task": {"id": task}, "condition": {"id": cond},
                      "agent": {"adapterId": "claude-code", "requestedModel": model}}}
    if cost is not None:
        r["metrics"] = {"common": {"costUsd": cost}}
    return r


def suite(n=120, reps=3, cand_fail=(), cost=False):
    out = []
    for t in range(n):
        for i in range(reps):
            out.append(rec(f"t{t}", "baseline", "m", "pass", i, 0.01 if cost else None))
            v = "fail" if (t in cand_fail and i == 0) else "pass"
            out.append(rec(f"t{t}", "with-skill", "m", v, i, 0.002 if cost else None))
    return out


def by_cond(runs, **kw):
    return hurdle.compare(runs, by="condition", baseline="baseline", candidate="with-skill", **kw)


def test_a_held_condition_with_logged_cost_is_ship():
    v = by_cond(suite(cost=True))
    assert v["decision"] == "ship" and v["promoted"] and v["spend_usd"] == 0.0


def test_without_cost_quality_can_hold_but_no_saving_is_claimed():
    v = by_cond(suite())
    assert v["quality"] == "holds" and v["cost"] == "unknown" and not v["promoted"]


def test_one_failed_repeat_fails_the_case_strictly():
    v = by_cond(suite(cand_fail={0, 1}))
    assert v["lost"] == 2 and v["repeats_passed"]["api/t0/claude-code/m"]["candidate"] == "2/3"


def test_an_incomplete_run_leaves_its_case_unanswered_and_blocks():
    runs = suite(cost=True)
    runs[1]["verdict"] = "incomplete"
    v = by_cond(runs)
    assert v["unanswered"] == 1 and not v["promoted"] and v["decision"] == "not_enough_evidence"


def test_a_task_missing_from_one_condition_blocks():
    runs = [r for r in suite(cost=True)
            if not (r["snapshot"]["task"]["id"] == "t5"
                    and r["snapshot"]["condition"]["id"] == "with-skill")]
    assert not by_cond(runs)["promoted"]


def test_a_handful_of_tasks_is_not_enough_evidence():
    v = by_cond(suite(n=10, cost=True))
    assert v["decision"] == "not_enough_evidence" and not v["promoted"]


def test_a_model_comparison_holds_the_condition_fixed():
    runs = [rec(f"t{t}", "baseline", m, "pass", i, 0.01 if m == "a" else 0.001)
            for t in range(120) for m in ("a", "b") for i in range(2)]
    v = hurdle.compare(runs, by="model", baseline="a", candidate="b")
    assert v["decision"] == "ship" and v["arms_seen"] == ["a", "b"]


def test_reads_the_result_store_layout_and_skips_unfinished_runs(tmp_path, capsys):
    root = tmp_path / ".hurdle" / "runs"
    for n, r in enumerate(suite(cost=True)):
        d = root / f"run_{n:05d}" / "revisions"
        d.mkdir(parents=True)
        (d / "000000.json").write_text(json.dumps({**r, "state": "queued", "verdict": None}))
        (d / "000002.json").write_text(json.dumps(r))
    unfinished = root / "run_x" / "revisions"
    unfinished.mkdir(parents=True)
    (unfinished / "000000.json").write_text(json.dumps({"state": "running"}))
    assert len(hurdle.load_runs(str(tmp_path / ".hurdle"))) == 720
    code = cli.main(["hurdle", str(tmp_path / ".hurdle"), "--compare", "condition",
                     "--baseline", "baseline", "--candidate", "with-skill"])
    assert code == 0 and json.loads(capsys.readouterr().out)["promoted"]
    assert cli.main(["hurdle", str(tmp_path / "empty"), "--compare", "model",
                     "--baseline", "a", "--candidate", "b"]) == 2


def test_a_malformed_record_is_skipped_not_a_crash(tmp_path):
    d = tmp_path / "runs" / "run_bad" / "revisions"
    d.mkdir(parents=True)
    (d / "000002.json").write_text(json.dumps({"state": "complete", "snapshot": "x"}))
    assert hurdle.load_runs(str(tmp_path)) == []
    bad = [{"state": "complete", "verdict": "pass", "snapshot": {"task": "t", "condition": 3}}]
    assert hurdle.compare(bad, by="condition", baseline="a", candidate="b")["cases"] == 0
