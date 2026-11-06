import pytest

from agentflow.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(mock=True, runs_dir=str(tmp_path / "runs"), max_tool_calls=6, max_revisions=2)
