"""LangGraph wiring:

    START -> planner -> researcher -> writer -> critic --approved--> finalize -> END
                                         ^                |
                                         +----revise------+
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from agentflow.agents import AgentState, Agents
from agentflow.config import Settings
from agentflow.errors import TransientError
from agentflow.llm import LLMClient, make_llm
from agentflow.tools import ToolRegistry, default_registry
from agentflow.tracing import Tracer, make_langfuse_client

# Node-level safety net: if a whole node raises a transient error (after the inner
# per-call retries were exhausted), LangGraph re-runs the node with backoff.
NODE_RETRY = RetryPolicy(initial_interval=0.5, backoff_factor=2.0, max_attempts=2, retry_on=TransientError)


@dataclass
class RunResult:
    run_id: str
    question: str
    answer: str
    plan: List[str]
    critique: Dict[str, Any]
    tool_log: List[Dict[str, Any]]
    errors: List[str]
    degraded: bool
    revisions: int
    duration_s: float
    trajectory: List[Dict[str, Any]] = field(repr=False, default_factory=list)
    trajectory_path: Optional[str] = None


def build_graph(agents: Agents):
    g = StateGraph(AgentState)
    g.add_node("planner", agents.planner, retry_policy=NODE_RETRY)
    g.add_node("researcher", agents.researcher, retry_policy=NODE_RETRY)
    g.add_node("writer", agents.writer, retry_policy=NODE_RETRY)
    g.add_node("critic", agents.critic, retry_policy=NODE_RETRY)
    g.add_node("finalize", agents.finalize)
    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", agents.route_after_critic, {"revise": "writer", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


class AgentFlow:
    """High-level entry point. ``AgentFlow().run("question")``."""

    def __init__(self, settings: Optional[Settings] = None, *, llm: Optional[LLMClient] = None, tools: Optional[ToolRegistry] = None, save_runs: bool = True):
        self.settings = settings or Settings.from_env()
        self.llm = llm or make_llm(self.settings)
        self._tools = tools
        self.save_runs = save_runs
        self._langfuse = make_langfuse_client(self.settings)

    def run(self, question: str) -> RunResult:
        tracer = Tracer(langfuse_client=self._langfuse, runs_dir=self.settings.runs_dir if self.save_runs else None)
        tools = self._tools or default_registry(self.settings)
        agents = Agents(self.llm, tools, tracer, max_tool_calls=self.settings.max_tool_calls, max_revisions=self.settings.max_revisions)
        graph = build_graph(agents)
        start = time.time()
        with tracer.span("agentflow_run", "run", input={"question": question}, model=self.llm.model) as root:
            state = graph.invoke({"question": question, "revisions": 0, "errors": [], "tool_log": []}, {"recursion_limit": 40})
            root.output = {"answer": state.get("final_answer")}
        result = RunResult(
            run_id=tracer.run_id,
            question=question,
            answer=state.get("final_answer", ""),
            plan=state.get("plan", []),
            critique=state.get("critique", {}),
            tool_log=state.get("tool_log", []),
            errors=state.get("errors", []),
            degraded=state.get("degraded", False),
            revisions=state.get("revisions", 0),
            duration_s=round(time.time() - start, 3),
            trajectory=tracer.trajectory(),
        )
        path = tracer.save({"question": question, "answer": result.answer, "tool_log": result.tool_log, "critique": result.critique})
        result.trajectory_path = str(path) if path else None
        self.last_tracer = tracer
        tracer.flush()
        return result


def build_agentflow(**kwargs: Any) -> AgentFlow:
    return AgentFlow(**kwargs)
