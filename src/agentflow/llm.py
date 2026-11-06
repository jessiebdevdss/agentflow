"""LLM clients.

* ``DeepSeekClient`` talks to DeepSeek through its OpenAI-compatible API (native tool calling
  and JSON mode), with retry + exponential backoff on transient failures.
* ``MockLLM`` is a deterministic, role-aware fake used for tests, demos and offline evals,
  so the whole graph runs without an API key. It can inject failures to exercise recovery.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, ValidationError

from agentflow.errors import LLMOutputError, PermanentError, TransientError
from agentflow.retry import call_with_retry
from agentflow.tracing import Tracer


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)


class LLMClient(Protocol):
    model: str

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------- helpers
def traced_chat(
    llm: LLMClient,
    tracer: Tracer,
    name: str,
    messages: List[Dict[str, Any]],
    *,
    attempts: int = 3,
    **kwargs: Any,
) -> LLMResponse:
    """Call the LLM inside a tracing span, retrying transient errors with backoff."""
    with tracer.span(name, "llm", input=messages[-1:], model=llm.model) as span:

        def on_retry(n: int, exc: BaseException) -> None:
            tracer.event("llm_retry", kind="retry", status="retry", call=name, attempt=n, error=str(exc))

        resp, used = call_with_retry(lambda: llm.chat(messages, **kwargs), attempts=attempts, on_retry=on_retry)
        span.output = {
            "content": resp.content,
            "tool_calls": [{"name": t.name, "arguments": t.arguments} for t in resp.tool_calls],
        }
        span.metadata.update({"attempts": used, "usage": resp.usage})
        return resp


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise


def structured_chat(
    llm: LLMClient,
    tracer: Tracer,
    name: str,
    messages: List[Dict[str, Any]],
    schema: type[BaseModel],
    *,
    repair_attempts: int = 2,
) -> BaseModel:
    """Ask for JSON matching ``schema``. On invalid output, feed the validation error back
    to the model and ask it to repair (self-correction) before giving up."""
    convo = list(messages)
    last_err: Optional[str] = None
    for attempt in range(repair_attempts + 1):
        resp = traced_chat(llm, tracer, name if attempt == 0 else f"{name}_repair", convo, json_mode=True)
        try:
            return schema.model_validate(_extract_json(resp.content or ""))
        except (json.JSONDecodeError, ValidationError) as exc:
            last_err = str(exc)[:800]
            tracer.event("llm_output_invalid", kind="error", status="error", call=name, attempt=attempt + 1, error=last_err)
            convo = convo + [
                {"role": "assistant", "content": resp.content or ""},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply was not valid JSON for the required schema.\n"
                        f"Error: {last_err}\nReturn ONLY a JSON object matching: "
                        f"{json.dumps(schema.model_json_schema())}"
                    ),
                },
            ]
    raise LLMOutputError(f"{name}: could not obtain valid {schema.__name__} after repairs: {last_err}")


# --------------------------------------------------------------------------- DeepSeek
class DeepSeekClient:
    """DeepSeek via the OpenAI-compatible Chat Completions API."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", model: str = "deepseek-chat", timeout: float = 60.0):
        import openai

        self._openai = openai
        self.model = model
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    def chat(self, messages, *, tools=None, json_mode=False, temperature=0.2) -> LLMResponse:
        o = self._openai
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            r = self._client.chat.completions.create(**kwargs)
        except (o.APITimeoutError, o.APIConnectionError, o.RateLimitError, o.InternalServerError) as exc:
            raise TransientError(str(exc)) from exc
        except o.APIStatusError as exc:
            if exc.status_code in (408, 409, 429) or exc.status_code >= 500:
                raise TransientError(str(exc)) from exc
            raise PermanentError(str(exc)) from exc
        msg = r.choices[0].message
        calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}") for tc in (msg.tool_calls or [])]
        usage = {}
        if r.usage:
            usage = {"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens}
        return LLMResponse(content=msg.content, tool_calls=calls, usage=usage)


# --------------------------------------------------------------------------- Mock
class MockLLM:
    """Deterministic stand-in for a real model.

    It recognises the agent role from the system prompt (``ROLE: planner`` etc.) and
    produces plausible, schema-valid output. Options:

    * ``fail_first`` - raise ``TransientError`` on the first N calls (tests retry/backoff)
    * ``bad_json_first`` - return malformed JSON on the first N structured calls (tests repair)
    * ``bad_tool_args_first`` - emit invalid tool arguments first (tests tool self-correction)
    * ``critic_reject_first`` - critic rejects the first N drafts (tests the revision loop)
    """

    model = "mock-llm"

    def __init__(self, *, fail_first: int = 0, bad_json_first: int = 0, bad_tool_args_first: int = 0, critic_reject_first: int = 0):
        self.fail_first = fail_first
        self.bad_json_first = bad_json_first
        self.bad_tool_args_first = bad_tool_args_first
        self.critic_reject_first = critic_reject_first
        self.calls = 0
        self._json_calls = 0
        self._critic_calls = 0
        self._bad_args_emitted = 0

    def chat(self, messages, *, tools=None, json_mode=False, temperature=0.2) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.fail_first:
            raise TransientError(f"mock transient failure #{self.calls}")
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        role = (re.search(r"ROLE:\s*(\w+)", system) or [None, "unknown"])[1]
        question = _find_question(messages)
        if json_mode:
            self._json_calls += 1
            if self._json_calls <= self.bad_json_first:
                return LLMResponse(content="{not valid json", usage={"input": 10, "output": 3})
        handler = getattr(self, f"_{role}", self._unknown)
        return handler(messages, question, tools)

    # -- roles -------------------------------------------------------------
    def _planner(self, messages, q, tools):
        steps = [
            f"Search the web for recent, authoritative information on: {q}",
            "Collect key facts, numbers and trade-offs with their sources",
        ]
        if _needs_math(q):
            steps.append("Run a quick calculation to verify the numbers")
        steps.append("Synthesise findings into a cited answer")
        return _json({"steps": steps, "rationale": "Gather evidence first, verify, then write."})

    def _researcher(self, messages, q, tools):
        tool_msgs = [m for m in messages if m["role"] == "tool"]
        names = {t["function"]["name"] for t in (tools or [])}
        if not tool_msgs:
            if "web_search" in names and self._bad_args_emitted < self.bad_tool_args_first:
                self._bad_args_emitted += 1
                return LLMResponse(None, [ToolCall("call_bad", "web_search", json.dumps({"max_results": 99}))])
            calls = []
            if "web_search" in names:
                calls.append(ToolCall("call_search", "web_search", json.dumps({"query": q, "max_results": 3})))
            if _needs_math(q) and "run_python" in names:
                calls.append(ToolCall("call_py", "run_python", json.dumps({"code": "print(round(1250 * 1.08 ** 3, 2))"})))
            return LLMResponse(None, calls, usage={"input": 50, "output": 20})
        last_tool = tool_msgs[-1]["content"]
        if '"ok": false' in last_tool and len(tool_msgs) < 3:
            # self-correct: retry the search with valid arguments
            return LLMResponse(None, [ToolCall("call_fix", "web_search", json.dumps({"query": q, "max_results": 3}))])
        notes = ["Findings:"]
        for m in tool_msgs:
            try:
                payload = json.loads(m["content"])
            except json.JSONDecodeError:
                continue
            out = payload.get("output")
            if isinstance(out, dict) and "results" in out:
                for r in out["results"]:
                    notes.append(f"- {r['snippet']} [source: {r['url']}]")
            elif isinstance(out, dict) and "stdout" in out:
                notes.append(f"- Calculation result: {out['stdout'].strip()}")
        return LLMResponse("\n".join(notes), usage={"input": 120, "output": 60})

    def _writer(self, messages, q, tools):
        user = messages[-1]["content"]
        bullets = [ln for ln in user.splitlines() if ln.startswith("- ")]
        feedback = "Revised to address reviewer feedback. " if "Reviewer feedback" in user else ""
        urls: List[str] = []
        lines = []
        for b in bullets:
            m = re.search(r"\[source: (\S+?)\]", b)
            text = re.sub(r"\s*\[source: \S+?\]", "", b)
            if m:
                if m.group(1) not in urls:
                    urls.append(m.group(1))
                text += f" [{urls.index(m.group(1)) + 1}]"
            lines.append(text)
        body = "\n".join(lines)
        sources = "\n".join(f"[{i + 1}] {u}" for i, u in enumerate(urls))
        return LLMResponse(
            f"# {q}\n\n{feedback}Summary based on the research notes.\n\n{body}\n\n## Sources\n{sources or '- none found'}\n",
            usage={"input": 200, "output": 150},
        )

    def _critic(self, messages, q, tools):
        self._critic_calls += 1
        draft = messages[-1]["content"]
        has_sources = "## Sources" in draft and "http" in draft
        if self._critic_calls <= self.critic_reject_first or not has_sources:
            return _json({"score": 0.55, "approved": False, "issues": ["Needs clearer structure and citations"], "feedback": "Add explicit citations for each claim."})
        return _json({"score": 0.88, "approved": True, "issues": [], "feedback": "Well cited and on-topic."})

    def _judge(self, messages, q, tools):
        return _json({"score": 0.8, "reasoning": "Trajectory used tools sensibly and the answer is cited."})

    def _unknown(self, messages, q, tools):
        return LLMResponse("OK")


def _json(obj: Any) -> LLMResponse:
    return LLMResponse(json.dumps(obj), usage={"input": 40, "output": 30})


def _find_question(messages: List[Dict[str, Any]]) -> str:
    for m in messages:
        if m["role"] == "user" and m.get("content"):
            match = re.search(r"Question:\s*(.+)", m["content"])
            if match:
                return match.group(1).strip()
    return "the question"


def _needs_math(q: str) -> bool:
    return bool(re.search(r"\d|calculate|growth|cost|price|roi", q, re.I))


def make_llm(settings: Any) -> LLMClient:
    if settings.use_mock_llm:
        return MockLLM()
    return DeepSeekClient(settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model)
