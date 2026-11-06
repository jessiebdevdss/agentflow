from pathlib import Path

from agentflow.evaluation import EvalCase, load_dataset, run_eval, score_trajectory
from agentflow.graph import AgentFlow
from agentflow.llm import MockLLM

DATASET = Path(__file__).resolve().parents[1] / "evals" / "dataset.json"


def test_dataset_loads():
    cases = load_dataset(str(DATASET))
    assert len(cases) >= 4


def test_eval_report_on_mock(settings):
    report = run_eval(AgentFlow(settings, llm=MockLLM()), load_dataset(str(DATASET)))
    s = report["summary"]
    assert s["completion_rate"] == 1.0 and s["mean_tool_recall"] == 1.0
    assert s["mean_overall"] > 0.8


def test_missing_expected_tool_lowers_score(settings):
    flow = AgentFlow(settings, llm=MockLLM())
    case = EvalCase(id="x", question="How does LangGraph work?", expected_tools=["web_search", "http_request"])
    score = score_trajectory(case, flow.run(case.question))
    assert score.tool_recall == 0.5 and score.tool_order is False


def test_recovered_error_is_counted(settings):
    flow = AgentFlow(settings, llm=MockLLM(bad_tool_args_first=1))
    case = EvalCase(id="y", question="How does LangGraph work?", expected_tools=["web_search"])
    score = score_trajectory(case, flow.run(case.question))
    assert score.recovered_errors == 1 and score.unrecovered_errors == 0
    assert score.tool_success_rate == 0.5
