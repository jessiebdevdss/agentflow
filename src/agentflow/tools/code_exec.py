"""Restricted Python execution.

Defence in depth for *model-written* snippets: an AST policy check (no imports outside an
allowlist, no file/process/network builtins, no dunder access), then execution in a separate
``python -I`` process with a timeout, a scratch working directory, an empty environment and
CPU/memory rlimits where the OS supports them.

This is NOT a security boundary against a determined attacker. For untrusted multi-tenant
workloads run code in a container / microVM sandbox (Docker, gVisor, E2B, Firecracker).
"""
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from typing import Any, Dict

from pydantic import BaseModel, Field

from agentflow.errors import PermanentError, TransientError
from agentflow.tools.base import Tool

ALLOWED_IMPORTS = {
    "math", "statistics", "json", "re", "datetime", "decimal", "fractions",
    "itertools", "functools", "collections", "random", "string", "textwrap",
}
BLOCKED_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars", "breakpoint", "help", "exit", "quit"}


class RunPythonArgs(BaseModel):
    """Execute a short Python snippet and capture stdout."""

    code: str = Field(..., min_length=1, max_length=8000, description="Python 3 code. Use print() to return results. Only stdlib math/data modules may be imported.")
    timeout_s: int = Field(5, ge=1, le=20, description="Wall-clock timeout in seconds.")


def check_policy(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PermanentError(f"SyntaxError: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS:
                    raise PermanentError(f"import of '{m}' is not allowed; allowed: {sorted(ALLOWED_IMPORTS)}")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise PermanentError(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PermanentError(f"dunder attribute access ('{node.attr}') is not allowed")


def _limits() -> None:  # pragma: no cover - runs in the child process
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    except Exception:
        pass  # macOS does not support RLIMIT_AS; the wall-clock timeout still applies


class CodeExecTool(Tool):
    name = "run_python"
    description = "Run a short, self-contained Python 3 snippet (stdlib math/data only) and return stdout/stderr. Use for calculations and data checks."
    args_schema = RunPythonArgs

    def __init__(self, max_output: int = 4000):
        self.max_output = max_output

    def run(self, args: RunPythonArgs) -> Dict[str, Any]:  # type: ignore[override]
        check_policy(args.code)
        with tempfile.TemporaryDirectory(prefix="agentflow_exec_") as tmp:
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-S", "-c", args.code],
                    cwd=tmp,
                    env={"PYTHONHASHSEED": "0"},
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_s,
                    preexec_fn=_limits if sys.platform != "win32" else None,
                )
            except subprocess.TimeoutExpired as exc:
                raise PermanentError(f"execution exceeded {args.timeout_s}s timeout") from exc
            except OSError as exc:
                raise TransientError(f"could not start interpreter: {exc}") from exc
        return {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[: self.max_output],
            "stderr": proc.stderr[-self.max_output :],
        }
