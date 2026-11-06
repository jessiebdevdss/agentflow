"""Trajectory evaluation.

Scores *how* the agent got to its answer, not just the answer: did it call the expected
tools in a sensible order, did tool calls succeed, did it recover from errors, did it stay
within budget, was the answer grounded and cited. Optionally adds an LLM-as-judge score.
Results can be pushed to Langfuse as scores on the run's trace.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agentflow.graph import AgentFlow, RunResult
from agentflow.llm import LLMClient, structured_chat
from agentflow.tracing import Tracer


class EvalCase(BaseModel):
    id: str
    question: str
    expected_tools: List[str] = Field(default_factory=list, description="Tools that should appear, in order.")
    must_include: List[str] = Field(default_factory=list, description="Case-insensitive substrings expected in the answer.")
    max_tool_calls: int = 8
    max_llm_calls: int = 15


@dataclass
class TrajectoryScore:
    case_id: str
    completed: bool
    tool_recall: float
    tool_order: bool
    tool_success_rate: float
    recovered_errors: int
    unrecovered_errors: int
    redundant_calls: int
    within_budget: bool
    citations: int
    content_hits: float
    critic_score: float
    judge_score: Optional[float] = None
    overall: float = 0.0
    notes: List[str] = field(default_factory=list)


def _is_subsequence(expected: List[str], actual: List[str]) -> bool:
    it = iter(actual)
    return all(any(a == e for a in it) for e in expected)


def score_trajectory(case: EvalCase, result: RunResult) -> TrajectoryScore:
    spans = result.trajectory
    tool_seq = [t["tool"] for t in result.tool_log]
    ok_calls = [t for t in result.tool_log if t["ok"]]
    llm_calls = [s for s in spans if s["kind"] == "llm"]
    notes: List[str] = []

    expected = case.expected_tools
    recall = (len(set(expected) & set(t["tool"] for t in ok_calls)) / len(set(expected))) if expected else 1.0
    order = _is_subsequence(expected, [t["tool"] for t in ok_calls]) if expected else True

    # An error is "recovered" if the same tool later succeeded, or the run still completed.
    recovered = unrecovered = 0
    for i, t in enumerate(result.tool_log):
        if t["ok"]:
            continue
        if any(later["ok"] and later["tool"] == t["tool"] for later in result.tool_log[i + 1 :]):
            recovered += 1
        else:
            unrecovered += 1
    retries = sum(1 for s in spans if s["kind"] == "retry")
    if retries:
        notes.append(f"{retries} retries with backoff")

    seen = set()
    redundant = 0
    for t in result.tool_log:
        key = (t["tool"], t["arguments"])
        if key in seen and t["ok"]:
            redundant += 1
        seen.add(key)

    within_budget = len(tool_seq) <= case.max_tool_calls and len(llm_calls) <= case.max_llm_calls
    citations = len(set(re.findall(r"https?://\S+", result.answer)))
    hits = [s for s in case.must_include if s.lower() in result.answer.lower()]
    content = len(hits) / len(case.must_include) if case.must_include else 1.0
    critic = float(result.critique.get("score", 0.0))
    completed = bool(result.answer.strip())
    success_rate = len(ok_calls) / len(tool_seq) if tool_seq else 0.0

    overall = (
        0.15 * completed
        + 0.20 * recall
        + 0.10 * order
        + 0.15 * success_rate
        + 0.10 * within_budget
        + 0.10 * min(citations, 3) / 3
        + 0.10 * content
        + 0.10 * critic
        - 0.05 * min(unrecovered, 2)
        - 0.03 * min(redundant, 2)
    )
    if result.degraded:
        notes.append("researcher ran in degraded mode")
    return TrajectoryScore(
        case_id=case.id, completed=completed, tool_recall=round(recall, 3), tool_order=order,
        tool_success_rate=round(success_rate, 3), recovered_errors=recovered, unrecovered_errors=unrecovered,
        redundant_calls=redundant, within_budget=within_budget, citations=citations,
        content_hits=round(content, 3), critic_score=critic, overall=round(max(0.0, min(1.0, overall)), 3), notes=notes,
    )


class JudgeVerdict(BaseModel):
    score: float = Field(..., ge=0, le=1)
    reasoning: str = ""


JUDGE_SYSTEM = """ROLE: judge
You evaluate an AI agent's trajectory. Given the question, the ordered tool calls (with success
flags) and the final answer, rate 0-1 how efficient, well-reasoned and grounded the trajectory was.
Reply JSON: {"score": 0-1, "reasoning": "..."}."""


def llm_judge(llm: LLMClient, case: EvalCase, result: RunResult) -> JudgeVerdict:
    tracer = Tracer(runs_dir=None)
    summary = json.dumps({"question": case.question, "tool_calls": result.tool_log, "answer": result.answer[:3000]})
    return structured_chat(llm, tracer, "judge", [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": summary}], JudgeVerdict)  # type: ignore[return-value]


def load_dataset(path: str) -> List[EvalCase]:
    return [EvalCase.model_validate(x) for x in json.loads(Path(path).read_text())]


def run_eval(flow: AgentFlow, cases: List[EvalCase], *, judge: bool = True) -> Dict[str, Any]:
    rows: List[TrajectoryScore] = []
    for case in cases:
        result = flow.run(case.question)
        score = score_trajectory(case, result)
        if judge:
            try:
                score.judge_score = llm_judge(flow.llm, case, result).score
                score.overall = round(0.8 * score.overall + 0.2 * score.judge_score, 3)
            except Exception as exc:  # judge is best-effort
                score.notes.append(f"judge failed: {exc}")
        tracer = getattr(flow, "last_tracer", None)
        if tracer is not None:
            tracer.score("trajectory_overall", score.overall)
            tracer.flush()
        rows.append(score)
    n = len(rows) or 1
    summary = {
        "cases": len(rows),
        "mean_overall": round(sum(r.overall for r in rows) / n, 3),
        "completion_rate": round(sum(r.completed for r in rows) / n, 3),
        "mean_tool_recall": round(sum(r.tool_recall for r in rows) / n, 3),
        "mean_tool_success": round(sum(r.tool_success_rate for r in rows) / n, 3),
        "recovered_errors": sum(r.recovered_errors for r in rows),
        "unrecovered_errors": sum(r.unrecovered_errors for r in rows),
    }
    return {"summary": summary, "results": [asdict(r) for r in rows]}
