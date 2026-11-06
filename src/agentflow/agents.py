"""The four agents. Each is a plain function ``(state) -> partial state update`` so it can be
unit-tested on its own and wired into LangGraph as a node."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field

from agentflow.errors import AgentFlowError
from agentflow.llm import LLMClient, structured_chat, traced_chat
from agentflow.tools import ToolRegistry
from agentflow.tracing import Tracer


# ----------------------------------------------------------------------- state
class AgentState(TypedDict, total=False):
    question: str
    plan: List[str]
    notes: str
    tool_log: List[Dict[str, Any]]
    draft: str
    critique: Dict[str, Any]
    revisions: int
    final_answer: str
    errors: List[str]
    degraded: bool


# --------------------------------------------------------------------- schemas
class Plan(BaseModel):
    steps: List[str] = Field(..., min_length=1, max_length=8, description="Ordered research steps.")
    rationale: str = Field("", description="One sentence on why this plan.")


class Critique(BaseModel):
    score: float = Field(..., ge=0, le=1, description="Overall quality, 0-1.")
    approved: bool = Field(..., description="True if the draft is ready to ship.")
    issues: List[str] = Field(default_factory=list)
    feedback: str = Field("", description="Concrete instructions for the writer.")


PLANNER_SYSTEM = """ROLE: planner
You are the planning agent of a research team. Break the user's question into 2-5 concrete
research steps that the researcher can execute with tools (web_search, http_request, run_python).
Reply with JSON: {"steps": [...], "rationale": "..."}."""

RESEARCHER_SYSTEM = """ROLE: researcher
You are the research agent. Execute the plan using the available tools. Prefer authoritative
sources. If a tool returns ok=false, read the error and hint, fix your arguments or try another
approach. When you have enough evidence, reply WITHOUT tool calls with concise bullet notes in
the form "- fact [source: URL]"."""

WRITER_SYSTEM = """ROLE: writer
You are the writing agent. Using ONLY the research notes, write a clear, well-structured markdown
answer with inline numeric citations [n] and a final "## Sources" section listing the URLs."""

CRITIC_SYSTEM = """ROLE: critic
You are the critic agent. Review the draft for factual grounding in the notes, citation coverage,
completeness and clarity. Reply with JSON: {"score": 0-1, "approved": bool, "issues": [...], "feedback": "..."}."""


# ---------------------------------------------------------------------- agents
class Agents:
    def __init__(self, llm: LLMClient, tools: ToolRegistry, tracer: Tracer, *, max_tool_calls: int = 8, max_revisions: int = 2, approve_threshold: float = 0.75):
        self.llm, self.tools, self.tracer = llm, tools, tracer
        self.max_tool_calls = max_tool_calls
        self.max_revisions = max_revisions
        self.approve_threshold = approve_threshold

    # -- planner ----------------------------------------------------------
    def planner(self, state: AgentState) -> AgentState:
        q = state["question"]
        with self.tracer.span("planner", "node", input={"question": q}) as span:
            try:
                plan = structured_chat(
                    self.llm, self.tracer, "planner",
                    [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": f"Question: {q}"}],
                    Plan,
                )
                steps = plan.steps
            except AgentFlowError as exc:
                # Recovery: fall back to a generic plan instead of failing the run.
                self.tracer.event("planner_fallback", kind="recovery", status="error", error=str(exc))
                steps = [f"Search for: {q}", "Summarise the evidence with sources"]
                span.output = {"steps": steps, "fallback": True}
                return {"plan": steps, "errors": state.get("errors", []) + [f"planner: {exc}"]}
            span.output = {"steps": steps}
            return {"plan": steps}

    # -- researcher (ReAct-style tool loop) --------------------------------
    def researcher(self, state: AgentState) -> AgentState:
        q = state["question"]
        plan = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(state.get("plan", [])))
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": RESEARCHER_SYSTEM},
            {"role": "user", "content": f"Question: {q}\n\nPlan:\n{plan}"},
        ]
        tool_log: List[Dict[str, Any]] = list(state.get("tool_log", []))
        errors = list(state.get("errors", []))
        calls_made = 0
        notes: Optional[str] = None
        with self.tracer.span("researcher", "node", input={"plan": state.get("plan")}) as span:
            for step in range(self.max_tool_calls + 1):
                schemas = self.tools.schemas() if calls_made < self.max_tool_calls else None
                try:
                    resp = traced_chat(self.llm, self.tracer, f"researcher_step_{step}", messages, tools=schemas)
                except AgentFlowError as exc:
                    errors.append(f"researcher: {exc}")
                    self.tracer.event("researcher_llm_failed", kind="recovery", status="error", error=str(exc))
                    break
                if not resp.tool_calls:
                    notes = resp.content or ""
                    break
                messages.append({
                    "role": "assistant",
                    "content": resp.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in resp.tool_calls
                    ],
                })
                for tc in resp.tool_calls:
                    calls_made += 1
                    result = self.tools.execute(tc.name, tc.arguments, self.tracer)
                    tool_log.append({
                        "tool": tc.name, "arguments": tc.arguments, "ok": result.ok,
                        "error_type": result.error_type, "attempts": result.attempts,
                        "duration_ms": result.duration_ms,
                    })
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result.to_message()})
            degraded = notes is None
            if degraded:
                # Recovery: budget exhausted or LLM failed -> salvage raw tool outputs as notes.
                notes = _salvage_notes(messages)
                self.tracer.event("researcher_degraded", kind="recovery", status="error", tool_calls=calls_made)
            span.output = {"notes": notes, "tool_calls": calls_made, "degraded": degraded}
        return {"notes": notes, "tool_log": tool_log, "errors": errors, "degraded": degraded or state.get("degraded", False)}

    # -- writer -----------------------------------------------------------
    def writer(self, state: AgentState) -> AgentState:
        q = state["question"]
        prompt = f"Question: {q}\n\nResearch notes:\n{state.get('notes', '')}"
        crit = state.get("critique")
        if crit and not crit.get("approved"):
            prompt += f"\n\nPrevious draft:\n{state.get('draft', '')}\n\nReviewer feedback: {crit.get('feedback')}\nIssues: {crit.get('issues')}"
        with self.tracer.span("writer", "node", input={"revision": state.get("revisions", 0)}) as span:
            resp = traced_chat(self.llm, self.tracer, "writer", [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": prompt}], temperature=0.4)
            span.output = {"draft": resp.content}
        return {"draft": resp.content or ""}

    # -- critic -----------------------------------------------------------
    def critic(self, state: AgentState) -> AgentState:
        revisions = state.get("revisions", 0)
        with self.tracer.span("critic", "node", input={"revision": revisions}) as span:
            try:
                c = structured_chat(
                    self.llm, self.tracer, "critic",
                    [
                        {"role": "system", "content": CRITIC_SYSTEM},
                        {"role": "user", "content": f"Question: {state['question']}\n\nNotes:\n{state.get('notes', '')}\n\nDraft:\n{state.get('draft', '')}"},
                    ],
                    Critique,
                )
                critique = c.model_dump()
                critique["approved"] = c.approved and c.score >= self.approve_threshold
            except AgentFlowError as exc:
                # Recovery: a broken critic should not block delivery.
                self.tracer.event("critic_fallback", kind="recovery", status="error", error=str(exc))
                critique = {"score": 0.0, "approved": True, "issues": ["critic unavailable"], "feedback": ""}
            span.output = critique
            self.tracer.score("critic_score", float(critique["score"]), critique.get("feedback"))
        return {"critique": critique, "revisions": revisions + 1}

    def route_after_critic(self, state: AgentState) -> str:
        crit = state.get("critique", {})
        if crit.get("approved") or state.get("revisions", 0) > self.max_revisions:
            return "finalize"
        return "revise"

    def finalize(self, state: AgentState) -> AgentState:
        return {"final_answer": state.get("draft", "")}


def _salvage_notes(messages: List[Dict[str, Any]]) -> str:
    lines = ["Partial findings (research loop ended early):"]
    for m in messages:
        if m["role"] != "tool":
            continue
        try:
            payload = json.loads(m["content"])
        except json.JSONDecodeError:
            continue
        out = payload.get("output")
        if isinstance(out, dict):
            for r in out.get("results", []):
                lines.append(f"- {r.get('snippet')} [source: {r.get('url')}]")
            if "stdout" in out:
                lines.append(f"- Calculation output: {out['stdout'].strip()}")
    return "\n".join(lines)
