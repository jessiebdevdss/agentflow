"""Web search tool. Backends: Tavily (if TAVILY_API_KEY), DuckDuckGo, or an offline corpus."""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from agentflow.errors import PermanentError, TransientError
from agentflow.tools.base import Tool


class WebSearchArgs(BaseModel):
    """Search the public web."""

    query: str = Field(..., min_length=2, max_length=300, description="Search query in natural language.")
    max_results: int = Field(5, ge=1, le=10, description="Number of results to return (1-10).")


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for up-to-date information. Returns a list of {title, url, snippet}."
    args_schema = WebSearchArgs

    def __init__(self, *, tavily_api_key: Optional[str] = None, offline: bool = False, timeout: float = 15.0):
        self.tavily_api_key = tavily_api_key
        self.offline = offline
        self.timeout = timeout

    def run(self, args: WebSearchArgs) -> Dict[str, Any]:  # type: ignore[override]
        if self.offline:
            results = offline_search(args.query, args.max_results)
            backend = "offline"
        elif self.tavily_api_key:
            results, backend = self._tavily(args), "tavily"
        else:
            results, backend = self._ddg(args), "duckduckgo"
        return {"backend": backend, "query": args.query, "results": results}

    def _tavily(self, args: WebSearchArgs) -> List[Dict[str, str]]:
        try:
            r = httpx.post(
                "https://api.tavily.com/search",
                json={"api_key": self.tavily_api_key, "query": args.query, "max_results": args.max_results},
                timeout=self.timeout,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientError(f"tavily network error: {exc}") from exc
        if r.status_code == 429 or r.status_code >= 500:
            raise TransientError(f"tavily HTTP {r.status_code}")
        if r.status_code >= 400:
            raise PermanentError(f"tavily HTTP {r.status_code}: {r.text[:200]}")
        return [
            {"title": i.get("title", ""), "url": i.get("url", ""), "snippet": (i.get("content") or "")[:500]}
            for i in r.json().get("results", [])
        ]

    def _ddg(self, args: WebSearchArgs) -> List[Dict[str, str]]:
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            raise PermanentError("DuckDuckGo backend not installed: pip install 'agentflow[search]' or set TAVILY_API_KEY") from exc
        try:
            with DDGS() as ddgs:
                items = list(ddgs.text(args.query, max_results=args.max_results))
        except Exception as exc:  # ddg raises rate-limit / network errors with various types
            raise TransientError(f"duckduckgo error: {exc}") from exc
        return [{"title": i.get("title", ""), "url": i.get("href", ""), "snippet": (i.get("body") or "")[:500]} for i in items]


_CORPUS = [
    ("LangGraph documentation", "https://langchain-ai.github.io/langgraph/",
     "LangGraph models agents as stateful graphs with explicit nodes, edges, checkpoints and human-in-the-loop, suited to controllable multi-agent workflows."),
    ("Langfuse tracing overview", "https://langfuse.com/docs/tracing",
     "Langfuse records nested traces of LLM calls, tools and agents, with scores and datasets for evaluating trajectories in production."),
    ("DeepSeek API docs", "https://api-docs.deepseek.com/",
     "DeepSeek exposes an OpenAI-compatible chat completions API with function calling and JSON output at low cost per token."),
    ("Retry patterns for distributed systems", "https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/",
     "Exponential backoff with jitter prevents retry storms; only idempotent, transient failures should be retried."),
    ("Agent evaluation guide", "https://www.anthropic.com/research/building-effective-agents",
     "Effective agents favour simple, composable patterns; evaluate both final answers and the intermediate tool-use trajectory."),
]


def offline_search(query: str, k: int) -> List[Dict[str, str]]:
    """Deterministic offline results so demos and tests need no network."""
    words = {w.lower().strip("?,.") for w in query.split() if len(w) > 3}
    scored = []
    for i, (title, url, snippet) in enumerate(_CORPUS):
        text = f"{title} {snippet}".lower()
        score = sum(w in text for w in words)
        tie = int(hashlib.md5(f"{query}{i}".encode()).hexdigest(), 16) % 100 / 1000
        scored.append((score + tie, {"title": title, "url": url, "snippet": snippet}))
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:k]]
