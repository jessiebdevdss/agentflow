"""Runtime configuration loaded from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


class Settings(BaseModel):
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    tavily_api_key: Optional[str] = None

    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    max_tool_calls: int = Field(default=8, ge=1, le=50)
    max_revisions: int = Field(default=2, ge=0, le=5)
    runs_dir: str = "runs"
    offline: bool = False
    mock: bool = False

    @property
    def use_mock_llm(self) -> bool:
        return self.mock or not self.deepseek_api_key

    @property
    def use_offline_tools(self) -> bool:
        return self.offline or self.use_mock_llm

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @classmethod
    def from_env(cls, *, mock: bool = False, dotenv: bool = True) -> "Settings":
        if dotenv:
            load_dotenv(override=False)
        return cls(
            deepseek_api_key=_env("DEEPSEEK_API_KEY"),
            deepseek_base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=_env("DEEPSEEK_MODEL", "deepseek-chat"),
            tavily_api_key=_env("TAVILY_API_KEY"),
            langfuse_public_key=_env("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=_env("LANGFUSE_SECRET_KEY"),
            langfuse_host=_env("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            max_tool_calls=int(_env("AGENTFLOW_MAX_TOOL_CALLS", "8")),
            max_revisions=int(_env("AGENTFLOW_MAX_REVISIONS", "2")),
            runs_dir=_env("AGENTFLOW_RUNS_DIR", "runs"),
            offline=_env("AGENTFLOW_OFFLINE", "0") in ("1", "true", "yes"),
            mock=mock,
        )
