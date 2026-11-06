"""Tool framework: Pydantic input schemas -> OpenAI/DeepSeek function-calling JSON schemas,
argument validation, retries for transient failures, and a per-tool circuit breaker."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError

from agentflow.errors import CircuitOpenError, PermanentError, TransientError
from agentflow.retry import call_with_retry
from agentflow.tracing import Tracer


@dataclass
class ToolResult:
    tool: str
    ok: bool
    output: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None  # validation | transient | permanent | circuit_open | unknown_tool
    attempts: int = 1
    duration_ms: float = 0.0

    def to_message(self) -> str:
        """Serialised form handed back to the LLM as the tool message."""
        payload: Dict[str, Any] = {"ok": self.ok}
        if self.ok:
            payload["output"] = self.output
        else:
            payload["error"] = self.error
            payload["error_type"] = self.error_type
            payload["hint"] = _HINTS.get(self.error_type or "", "")
        return json.dumps(payload, default=str)


_HINTS = {
    "validation": "Fix the arguments so they match the tool's JSON schema and call it again.",
    "transient": "The tool failed temporarily after retries; try a different query or continue without it.",
    "permanent": "This call cannot succeed as written; change the request or use another tool.",
    "circuit_open": "This tool is disabled for the rest of the run; continue with other tools.",
    "unknown_tool": "Only call tools from the provided list.",
}


class Tool:
    """A callable tool with a typed, documented input schema."""

    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel]

    def run(self, args: BaseModel) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def json_schema(self) -> Dict[str, Any]:
        schema = self.args_schema.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": schema},
        }


class FunctionTool(Tool):
    def __init__(self, name: str, description: str, args_schema: Type[BaseModel], fn: Callable[[BaseModel], Any]):
        self.name, self.description, self.args_schema, self._fn = name, description, args_schema, fn

    def run(self, args: BaseModel) -> Any:
        return self._fn(args)


class ToolRegistry:
    def __init__(self, tools: List[Tool], *, max_attempts: int = 3, breaker_threshold: int = 3, backoff_initial: float = 0.3):
        self._tools = {t.name: t for t in tools}
        self.max_attempts = max_attempts
        self.breaker_threshold = breaker_threshold
        self.backoff_initial = backoff_initial
        self._consecutive_failures: Dict[str, int] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.json_schema() for t in self._tools.values() if not self.is_open(t.name)]

    def is_open(self, name: str) -> bool:
        return self._consecutive_failures.get(name, 0) >= self.breaker_threshold

    def execute(self, name: str, raw_args: Any, tracer: Tracer) -> ToolResult:
        start = time.time()
        with tracer.span(f"tool:{name}", "tool", input=raw_args) as span:
            result = self._execute(name, raw_args, tracer)
            result.duration_ms = round((time.time() - start) * 1000, 2)
            span.output = result.output if result.ok else {"error": result.error}
            span.metadata.update({"attempts": result.attempts, "ok": result.ok, "error_type": result.error_type})
            if not result.ok:
                span.status = "error"
                span.error = result.error
            return result

    def _execute(self, name: str, raw_args: Any, tracer: Tracer) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, False, error=f"Unknown tool '{name}'. Available: {self.names}", error_type="unknown_tool")
        if self.is_open(name):
            return ToolResult(name, False, error=str(CircuitOpenError(f"{name} disabled after repeated failures")), error_type="circuit_open")

        try:
            data = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            args = tool.args_schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            # Not counted towards the breaker: the model can fix its own arguments.
            return ToolResult(name, False, error=_short_validation(exc), error_type="validation")

        def on_retry(n: int, exc: BaseException) -> None:
            tracer.event("tool_retry", kind="retry", status="retry", tool=name, attempt=n, error=str(exc))

        try:
            output, attempts = call_with_retry(
                lambda: tool.run(args), attempts=self.max_attempts, initial=self.backoff_initial, on_retry=on_retry
            )
        except TransientError as exc:
            self._fail(name)
            return ToolResult(name, False, error=str(exc), error_type="transient", attempts=self.max_attempts)
        except PermanentError as exc:
            self._fail(name)
            return ToolResult(name, False, error=str(exc), error_type="permanent")
        except Exception as exc:  # unexpected bug inside a tool: contain it
            self._fail(name)
            return ToolResult(name, False, error=f"{type(exc).__name__}: {exc}", error_type="permanent")
        self._consecutive_failures[name] = 0
        return ToolResult(name, True, output=output, attempts=attempts)

    def _fail(self, name: str) -> None:
        self._consecutive_failures[name] = self._consecutive_failures.get(name, 0) + 1


def _short_validation(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(str(p) for p in e['loc']) or 'args'}: {e['msg']}" for e in exc.errors()]
        return "Invalid arguments - " + "; ".join(parts)
    return f"Arguments are not valid JSON: {exc}"
