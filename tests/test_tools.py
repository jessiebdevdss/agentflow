import json

import httpx
import pytest

from agentflow.errors import PermanentError, TransientError
from agentflow.tools import CodeExecTool, FunctionTool, HttpApiTool, ToolRegistry, WebSearchTool
from agentflow.tools.base import ToolResult
from agentflow.tools.code_exec import RunPythonArgs, check_policy
from agentflow.tracing import Tracer
from pydantic import BaseModel


def tracer():
    return Tracer(runs_dir=None)


def test_schemas_are_openai_function_format():
    reg = ToolRegistry([WebSearchTool(offline=True), HttpApiTool(offline=True), CodeExecTool()])
    schemas = reg.schemas()
    assert {s["function"]["name"] for s in schemas} == {"web_search", "http_request", "run_python"}
    ws = next(s for s in schemas if s["function"]["name"] == "web_search")["function"]["parameters"]
    assert ws["type"] == "object"
    assert ws["required"] == ["query"]
    assert ws["properties"]["max_results"]["maximum"] == 10


def test_validation_error_is_returned_to_model_not_raised():
    reg = ToolRegistry([WebSearchTool(offline=True)])
    res = reg.execute("web_search", json.dumps({"max_results": 99}), tracer())
    assert not res.ok and res.error_type == "validation"
    msg = json.loads(res.to_message())
    assert "query" in msg["error"] and msg["hint"]


def test_malformed_json_arguments():
    reg = ToolRegistry([WebSearchTool(offline=True)])
    res = reg.execute("web_search", "{oops", tracer())
    assert res.error_type == "validation"


def test_unknown_tool():
    res = ToolRegistry([]).execute("nope", "{}", tracer())
    assert res.error_type == "unknown_tool"


class Echo(BaseModel):
    x: int


def flaky(fail_times, exc=TransientError):
    state = {"n": 0}

    def fn(args):
        state["n"] += 1
        if state["n"] <= fail_times:
            raise exc("boom")
        return {"x": args.x}

    return fn, state


def test_transient_errors_are_retried_with_backoff():
    fn, state = flaky(2)
    t = tracer()
    reg = ToolRegistry([FunctionTool("echo", "echo", Echo, fn)], backoff_initial=0.01)
    res = reg.execute("echo", {"x": 1}, t)
    assert res.ok and res.attempts == 3 and state["n"] == 3
    assert sum(1 for s in t.spans if s.kind == "retry") == 2


def test_permanent_errors_are_not_retried():
    fn, state = flaky(5, PermanentError)
    reg = ToolRegistry([FunctionTool("echo", "echo", Echo, fn)], backoff_initial=0.01)
    res = reg.execute("echo", {"x": 1}, tracer())
    assert not res.ok and res.error_type == "permanent" and state["n"] == 1


def test_circuit_breaker_opens_after_repeated_failures():
    fn, _ = flaky(100, PermanentError)
    reg = ToolRegistry([FunctionTool("echo", "echo", Echo, fn)], breaker_threshold=2)
    for _ in range(2):
        reg.execute("echo", {"x": 1}, tracer())
    res = reg.execute("echo", {"x": 1}, tracer())
    assert res.error_type == "circuit_open"
    assert reg.schemas() == []  # disabled tools are hidden from the model


def test_code_exec_runs_and_captures_stdout():
    out = CodeExecTool().run(RunPythonArgs(code="import math\nprint(math.sqrt(16))"))
    assert out["exit_code"] == 0 and out["stdout"].strip() == "4.0"


@pytest.mark.parametrize("code", ["import os", "open('x','w')", "().__class__.__bases__", "__import__('socket')", "from subprocess import run"])
def test_code_exec_policy_blocks_dangerous_code(code):
    with pytest.raises(PermanentError):
        check_policy(code)


def test_code_exec_timeout():
    with pytest.raises(PermanentError, match="timeout"):
        CodeExecTool().run(RunPythonArgs(code="while True: pass", timeout_s=1))


def _http(handler):
    return HttpApiTool(transport=httpx.MockTransport(handler))


def test_http_tool_success_and_classification():
    from agentflow.tools.http_api import HttpRequestArgs

    ok = _http(lambda r: httpx.Response(200, json={"hello": "world"}))
    assert ok.run(HttpRequestArgs(url="https://api.example.com/x"))["body"] == {"hello": "world"}
    with pytest.raises(TransientError):
        _http(lambda r: httpx.Response(503)).run(HttpRequestArgs(url="https://api.example.com/x"))
    with pytest.raises(TransientError):
        _http(lambda r: httpx.Response(429)).run(HttpRequestArgs(url="https://api.example.com/x"))
    with pytest.raises(PermanentError):
        _http(lambda r: httpx.Response(404, text="nope")).run(HttpRequestArgs(url="https://api.example.com/x"))


def test_http_tool_blocks_private_addresses():
    from agentflow.tools.http_api import HttpRequestArgs

    with pytest.raises(PermanentError, match="non-public"):
        HttpApiTool().run(HttpRequestArgs(url="http://127.0.0.1:8080/admin"))


def test_http_allowlist():
    from agentflow.tools.http_api import HttpRequestArgs

    tool = HttpApiTool(allowed_hosts=["api.github.com"], transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    with pytest.raises(PermanentError, match="allowlist"):
        tool.run(HttpRequestArgs(url="https://evil.example.com"))
