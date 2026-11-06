"""Tracing: every node, LLM call, tool call, retry and error is recorded as a span.

Spans are always kept locally (the *trajectory*, saved to ``runs/<run_id>.json``) so runs
can be evaluated offline. When Langfuse keys are configured, the same spans are mirrored
to Langfuse as nested observations (agent / generation / tool / evaluator).
"""
from __future__ import annotations

import json
import logging
import contextvars
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

log = logging.getLogger(__name__)

# Our span kinds -> Langfuse observation types
_LANGFUSE_TYPES = {
    "run": "agent",
    "node": "agent",
    "llm": "generation",
    "tool": "tool",
    "eval": "evaluator",
}


def _safe(value: Any, limit: int = 4000) -> Any:
    """Make a value JSON-friendly and bounded in size."""
    try:
        text = json.dumps(value, default=str)
    except Exception:  # pragma: no cover - defensive
        text = json.dumps(str(value))
    if len(text) <= limit:
        return json.loads(text)
    return text[:limit] + "...<truncated>"


@dataclass
class Span:
    id: str
    name: str
    kind: str
    parent_id: Optional[str]
    start: float
    input: Any = None
    output: Any = None
    end: Optional[float] = None
    status: str = "ok"  # ok | error | retry
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return round(((self.end or time.time()) - self.start) * 1000, 2)


class Tracer:
    def __init__(
        self,
        run_id: Optional[str] = None,
        *,
        langfuse_client: Any = None,
        runs_dir: Optional[str] = "runs",
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.spans: List[Span] = []
        self.runs_dir = runs_dir
        self._langfuse = langfuse_client
        self._stack_var: contextvars.ContextVar[tuple] = contextvars.ContextVar(f"agentflow_stack_{self.run_id}", default=())

    # ---------------------------------------------------------------- spans
    def _parent_id(self) -> Optional[str]:
        stack = self._stack_var.get()
        return stack[-1] if stack else None

    @contextmanager
    def span(self, name: str, kind: str, input: Any = None, **metadata: Any) -> Iterator[Span]:
        span = Span(
            id=uuid.uuid4().hex[:12],
            name=name,
            kind=kind,
            parent_id=self._parent_id(),
            start=time.time(),
            input=_safe(input),
            metadata=_safe(metadata) if metadata else {},
        )
        self.spans.append(span)
        token = self._stack_var.set(self._stack_var.get() + (span.id,))
        lf_cm = self._langfuse_start(span)
        lf_obs = None
        if lf_cm is not None:
            try:
                lf_obs = lf_cm.__enter__()
            except Exception as exc:  # tracing must never break the app
                log.debug("langfuse span start failed: %s", exc)
                lf_cm = None
        try:
            yield span
        except BaseException as exc:
            span.status = "error"
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            span.end = time.time()
            self._stack_var.reset(token)
            if lf_cm is not None:
                try:
                    if lf_obs is not None:
                        update: Dict[str, Any] = {"output": _safe(span.output), "metadata": span.metadata}
                        if span.status == "error":
                            update.update(level="ERROR", status_message=span.error)
                        lf_obs.update(**update)
                    lf_cm.__exit__(None, None, None)
                except Exception as exc:
                    log.debug("langfuse span end failed: %s", exc)

    def event(self, name: str, kind: str = "event", status: str = "ok", **data: Any) -> Span:
        """Record an instantaneous event (e.g. a retry or a recovered error)."""
        now = time.time()
        span = Span(
            id=uuid.uuid4().hex[:12],
            name=name,
            kind=kind,
            parent_id=self._parent_id(),
            start=now,
            end=now,
            status=status,
            metadata=_safe(data),
        )
        self.spans.append(span)
        if self._langfuse is not None:
            try:
                self._langfuse.create_event(
                    name=name,
                    metadata=span.metadata,
                    level="WARNING" if status != "ok" else "DEFAULT",
                )
            except Exception as exc:
                log.debug("langfuse event failed: %s", exc)
        return span

    def score(self, name: str, value: float, comment: Optional[str] = None) -> None:
        self.event(f"score:{name}", kind="score", value=value, comment=comment)
        if self._langfuse is not None:
            try:
                self._langfuse.score_current_trace(name=name, value=value, comment=comment)
            except Exception as exc:
                log.debug("langfuse score failed: %s", exc)

    # ------------------------------------------------------------ langfuse
    def _langfuse_start(self, span: Span) -> Any:
        if self._langfuse is None:
            return None
        kwargs: Dict[str, Any] = {
            "name": span.name,
            "as_type": _LANGFUSE_TYPES.get(span.kind, "span"),
            "input": span.input,
            "metadata": {**span.metadata, "agentflow_run_id": self.run_id},
        }
        if span.kind == "llm" and "model" in span.metadata:
            kwargs["model"] = span.metadata["model"]
        try:
            return self._langfuse.start_as_current_observation(**kwargs)
        except Exception as exc:
            log.debug("langfuse unavailable: %s", exc)
            return None

    def flush(self) -> None:
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception as exc:
                log.debug("langfuse flush failed: %s", exc)

    # ---------------------------------------------------------- trajectory
    def trajectory(self) -> List[Dict[str, Any]]:
        out = []
        for s in self.spans:
            d = asdict(s)
            d["duration_ms"] = s.duration_ms
            out.append(d)
        return out

    def save(self, extra: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        if not self.runs_dir:
            return None
        path = Path(self.runs_dir)
        path.mkdir(parents=True, exist_ok=True)
        file = path / f"{self.run_id}.json"
        payload = {"run_id": self.run_id, "spans": self.trajectory(), **(extra or {})}
        file.write_text(json.dumps(payload, indent=2, default=str))
        return file


def make_langfuse_client(settings: Any) -> Any:
    """Return a Langfuse client if configured and installed, else ``None``."""
    if not getattr(settings, "langfuse_enabled", False):
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        log.warning("LANGFUSE_* keys set but `langfuse` is not installed: pip install 'agentflow[tracing]'")
        return None
    try:
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:
        log.warning("Could not initialise Langfuse, continuing with local tracing only: %s", exc)
        return None
