"""A tool change: cases carry tools, the candidate gets the new tool list, and ``tool-call``
grades which tool the model called. Offline: no model is called."""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from margin_referee import cases as cases_mod
from margin_referee import cli, endpoints, replay
from margin_referee.cases import Case, grade

EXAMPLE = pathlib.Path(__file__).resolve().parents[1] / "examples" / "tool_change"


def _tool(description):
    return [{"type": "function", "function": {"name": "lookup", "description": description,
                                              "parameters": {"type": "object"}}}]


def _call(name, arguments="{}"):
    return {"function": {"name": name, "arguments": arguments}}


OLD, NEW = _tool("Looks things up."), _tool("Fetch one order by its id.")


def _chat(seen, side, calls):
    def chat(messages, **kw):
        seen.append((side, kw.get("tools"), kw.get("tool_choice")))
        r = SimpleNamespace(text="", cost_usd=0.001, finish_reason="tool_calls")
        if calls is not None:
            r.tool_calls = calls
        return r
    return chat


def test_candidate_tools_reach_only_the_candidate():
    seen = []
    case = Case("c1", [{"role": "user", "content": "q"}], "tool-call", "lookup", tools=OLD,
                tool_choice="required")
    res = replay.run([case], _chat(seen, "baseline", [_call("search")]),
                     _chat(seen, "candidate", [_call("lookup")]), cap_usd=1.0,
                     candidate_tools=NEW)
    assert sorted(seen, key=lambda s: s[0]) == [("baseline", OLD, "required"),
                                                ("candidate", NEW, "required")]
    assert res.pairs == [("c1", 0.0, 1.0)]
    seen.clear()
    replay.run([case], _chat(seen, "baseline", []), _chat(seen, "candidate", []), cap_usd=1.0,
               candidate_tools=[])
    assert ("candidate", None, None) in seen and ("baseline", OLD, "required") in seen
    seen.clear()
    replay.run([case], _chat(seen, "baseline", []), _chat(seen, "candidate", []), cap_usd=1.0)
    assert [s[1:] for s in seen] == [(OLD, "required")] * 2, "unchanged: both get the case's"


def test_the_tool_call_check_passes_fails_and_refuses_to_guess():
    name, args = Case("x", [], "tool-call", "lookup"), Case(
        "x", [], "tool-call", "name=lookup;args.order_id=A1")
    assert grade("", name, [_call("lookup")]) and not grade("", name, [_call("search")])
    assert not grade("it is on its way", name, []), "no tool call is a fail"
    assert grade("", args, [_call("lookup", '{"order_id": "a1"}')])
    assert not grade("", args, [_call("lookup", '{"order_id": "B2"}')])
    for unreadable in (None, [_call("lookup", "{broken")], ["junk"]):
        with pytest.raises(ValueError):
            grade("", args, unreadable)


@pytest.mark.parametrize("calls", [None, [_call("lookup", "{broken")]])
def test_an_unreadable_tool_call_is_unanswered_and_promotes_nothing(calls):
    case = Case("c1", [{"role": "user", "content": "q"}], "tool-call",
                "lookup;args.order_id=A1", tools=OLD)
    res = replay.run([case], _chat([], "baseline", calls), _chat([], "candidate", calls),
                     cap_usd=1.0)
    assert res.pairs == [] and len(res.unanswered) == 1
    assert res.unanswered[0][2].startswith("check failed: ValueError: ")
    assert not replay.verdict(res)["promoted"]


def test_the_cli_replays_the_committed_example_with_its_candidate_tools(monkeypatch, capsys):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr(replay, "record_and_count", lambda cases, base, label, **kw: 1)
    new = json.loads((EXAMPLE / "candidate_tools.json").read_text())
    sides = iter(("baseline", "candidate"))

    def chat_for(ep):
        side = next(sides)

        def chat(messages, tools=None, **kw):
            # the vague descriptions get no tool call; the rewritten ones get the wanted tool
            assert tools == (new if side == "candidate" else cases_mod.parse(
                (EXAMPLE / "cases.jsonl").read_text()).cases[0].tools)
            text = messages[-1]["content"]
            want = "search_kb" if "return" in text else "lookup_order"
            calls = [_call(want, '{"order_id": "A1042"}')] if side == "candidate" else []
            return SimpleNamespace(text="", cost_usd=0.001, finish_reason="stop",
                                   tool_calls=calls)
        return chat
    monkeypatch.setattr(endpoints, "chat_for", chat_for)
    code = cli.main(["replay", str(EXAMPLE / "cases.jsonl"), "--baseline", "m/x",
                     "--candidate", "m/x", "--cap", "1", "--anyway",
                     "--candidate-tools", str(EXAMPLE / "candidate_tools.json")])
    v = json.loads(capsys.readouterr().out)
    assert (v["cases"], v["baseline_passed"], v["candidate_passed"]) == (3, 0, 3)
    assert "+tools:" in v["candidate"] and "+tools:" not in v["baseline"]
    # three cases show the format and prove nothing
    assert v["decision"] == "not_enough_evidence" and code == 1


def test_a_candidate_tools_file_that_is_not_a_tool_list_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr(replay, "record_and_count",
                        lambda *a: pytest.fail("nothing is recorded for a refused replay"))
    bad = tmp_path / "tools.json"
    bad.write_text(json.dumps({"name": "lookup"}))
    with pytest.raises(SystemExit, match="--candidate-tools"):
        cli.main(["replay", str(EXAMPLE / "cases.jsonl"), "--baseline", "m/x",
                  "--candidate", "m/x", "--cap", "1", "--candidate-tools", str(bad)])


def test_an_endpoint_sends_the_tools_and_reports_the_tool_calls(monkeypatch):
    ep = endpoints.parse({"kind": "openrouter", "model": "m/x", "api_key": "k"})
    sent = {}

    def post(url, headers, body):
        sent.update(body)
        return {"choices": [{"finish_reason": "tool_calls",
                             "message": {"content": None, "tool_calls": [_call("lookup")]}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.0001}}
    monkeypatch.setattr(endpoints, "_post", post)
    r = endpoints.chat_for(ep)([{"role": "user", "content": "q"}], tools=OLD, tool_choice="auto")
    assert sent["tools"] == OLD and sent["tool_choice"] == "auto"
    assert r.tool_calls == [_call("lookup")] and r.text == ""
