import json
from pathlib import Path

from agentflow.graph import AgentFlow
from agentflow.llm import MockLLM

Q = "If an automation saves $1250 per month and grows 8% per year, what is the monthly saving after 3 years?"


def test_end_to_end_mock_run(settings):
    res = AgentFlow(settings, llm=MockLLM()).run(Q)
    assert "## Sources" in res.answer and "1574.64" in res.answer
    assert [t["tool"] for t in res.tool_log] == ["web_search", "run_python"]
    assert res.critique["approved"] is True
    kinds = {s["kind"] for s in res.trajectory}
    assert {"run", "node", "llm", "tool"} <= kinds
    saved = json.loads(Path(res.trajectory_path).read_text())
    assert saved["run_id"] == res.run_id and saved["spans"]


def test_spans_are_nested_under_nodes(settings):
    res = AgentFlow(settings, llm=MockLLM()).run(Q)
    by_id = {s["id"]: s for s in res.trajectory}
    tool_spans = [s for s in res.trajectory if s["kind"] == "tool"]
    assert tool_spans and all(by_id[s["parent_id"]]["name"] == "researcher" for s in tool_spans)


def test_llm_transient_failures_are_retried(settings):
    res = AgentFlow(settings, llm=MockLLM(fail_first=2)).run(Q)
    assert res.answer
    assert sum(1 for s in res.trajectory if s["kind"] == "retry") == 2


def test_invalid_json_is_repaired(settings):
    res = AgentFlow(settings, llm=MockLLM(bad_json_first=1)).run(Q)
    assert res.plan and len(res.plan) >= 2
    assert any(s["name"] == "planner_repair" for s in res.trajectory)


def test_planner_falls_back_when_repair_fails(settings):
    res = AgentFlow(settings, llm=MockLLM(bad_json_first=3)).run(Q)
    assert res.plan[0].startswith("Search for:")
    assert any(s["name"] == "planner_fallback" for s in res.trajectory)
    assert res.answer


def test_bad_tool_args_self_correct(settings):
    res = AgentFlow(settings, llm=MockLLM(bad_tool_args_first=1)).run("How does LangGraph work?")
    assert res.tool_log[0]["ok"] is False and res.tool_log[0]["error_type"] == "validation"
    assert any(t["ok"] for t in res.tool_log[1:])
    assert "langgraph" in res.answer.lower()


def test_critic_revision_loop(settings):
    res = AgentFlow(settings, llm=MockLLM(critic_reject_first=1)).run(Q)
    assert res.revisions == 2
    assert "Revised to address reviewer feedback" in res.answer


def test_revision_loop_is_bounded(settings):
    res = AgentFlow(settings, llm=MockLLM(critic_reject_first=99)).run(Q)
    assert res.revisions == settings.max_revisions + 1
    assert res.answer
