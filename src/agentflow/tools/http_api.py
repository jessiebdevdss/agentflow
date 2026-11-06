"""Generic HTTP API tool with SSRF protection, retry classification and response truncation."""
from __future__ import annotations

import ipaddress
import json
import socket
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, HttpUrl

from agentflow.errors import PermanentError, TransientError
from agentflow.tools.base import Tool


class HttpRequestArgs(BaseModel):
    """Call a public JSON/HTTP API."""

    url: HttpUrl = Field(..., description="Absolute http(s) URL of a public API endpoint.")
    method: Literal["GET", "POST"] = Field("GET", description="HTTP method.")
    params: Dict[str, str] = Field(default_factory=dict, description="Query-string parameters.")
    json_body: Optional[Dict[str, Any]] = Field(None, description="JSON body for POST requests.")
    timeout_s: float = Field(10.0, gt=0, le=30, description="Request timeout in seconds.")


class HttpApiTool(Tool):
    name = "http_request"
    description = "Make a GET/POST request to a public HTTP API and return status, headers and (truncated) body."
    args_schema = HttpRequestArgs

    def __init__(self, *, allowed_hosts: Optional[List[str]] = None, max_chars: int = 4000, offline: bool = False, transport: Any = None):
        self.allowed_hosts = allowed_hosts
        self.max_chars = max_chars
        self.offline = offline
        self._transport = transport  # injectable for tests (httpx.MockTransport)

    def run(self, args: HttpRequestArgs) -> Dict[str, Any]:  # type: ignore[override]
        url = str(args.url)
        host = urlparse(url).hostname or ""
        if self.allowed_hosts is not None and host not in self.allowed_hosts:
            raise PermanentError(f"host '{host}' is not in the allowlist {self.allowed_hosts}")
        if self.offline and self._transport is None:
            return {"status": 200, "url": url, "body": {"offline": True, "note": "network disabled in offline mode"}}
        if self._transport is None:
            _block_private(host)
        try:
            with httpx.Client(transport=self._transport, timeout=args.timeout_s, follow_redirects=False) as client:
                r = client.request(args.method, url, params=args.params or None, json=args.json_body)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientError(f"network error calling {host}: {exc}") from exc
        if r.status_code == 429 or r.status_code >= 500:
            raise TransientError(f"HTTP {r.status_code} from {host}")
        if r.status_code >= 400:
            raise PermanentError(f"HTTP {r.status_code} from {host}: {r.text[:300]}")
        try:
            body: Any = r.json()
            text = json.dumps(body)
            if len(text) > self.max_chars:
                body = text[: self.max_chars] + "...<truncated>"
        except ValueError:
            body = r.text[: self.max_chars]
        return {"status": r.status_code, "url": str(r.url), "content_type": r.headers.get("content-type"), "body": body}


def _block_private(host: str) -> None:
    """Refuse loopback/private/link-local targets (basic SSRF guard)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise PermanentError(f"cannot resolve host '{host}'") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise PermanentError(f"refusing to call non-public address {ip} for '{host}'")
