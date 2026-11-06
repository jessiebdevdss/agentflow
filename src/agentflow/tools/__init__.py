from __future__ import annotations

from typing import Any

from agentflow.tools.base import FunctionTool, Tool, ToolRegistry, ToolResult
from agentflow.tools.code_exec import CodeExecTool, RunPythonArgs
from agentflow.tools.http_api import HttpApiTool, HttpRequestArgs
from agentflow.tools.web_search import WebSearchArgs, WebSearchTool


def default_registry(settings: Any) -> ToolRegistry:
    offline = settings.use_offline_tools
    return ToolRegistry(
        [
            WebSearchTool(tavily_api_key=settings.tavily_api_key, offline=offline),
            HttpApiTool(offline=offline),
            CodeExecTool(),
        ]
    )


__all__ = [
    "CodeExecTool", "FunctionTool", "HttpApiTool", "HttpRequestArgs", "RunPythonArgs",
    "Tool", "ToolRegistry", "ToolResult", "WebSearchArgs", "WebSearchTool", "default_registry",
]
