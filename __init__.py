"""GBrain memory provider plugin — hardcoded tool schemas.

Connects to GBrain via MCP-over-HTTP (StreamableHTTP) and exposes
83 hardcoded tool schemas through the MemoryProvider interface.
No dynamic discovery — schemas are generated from the live server
and stored in _schemas.py. Regenerate with:

    python3 -c "from _schemas import *; ..."

Config source (priority):
  1. Env vars: MCP_GBRAIN_URL, MCP_GBRAIN_API_KEY
  2. Env vars: GBRAIN_MCP_URL, GBRAIN_API_TOKEN, GBRAIN_MCP_TOKEN
  3. $HERMES_HOME/gbrain_memory.json
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import httpx

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home

from ._schemas import HARDCODED_SCHEMAS, HARDCODED_SCHEMA_DICTS

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://gbrain.plainrandom.com/mcp"
_DEFAULT_TIMEOUT = 30


class _McpError(RuntimeError):
    pass


class _McpClient:
    """Thread-safe MCP-over-HTTP client. Connects on-demand, no tool discovery."""

    def __init__(self, url: str, api_token: str, timeout: float = _DEFAULT_TIMEOUT):
        self._base_url = url.rstrip("/")
        self._api_token = api_token
        self._timeout = timeout
        self._session_id: str | None = None
        self._lock = threading.RLock()
        self._closed = False
        self._req_id = 0

    def call(self, tool_name: str, arguments: dict) -> dict:
        if self._closed:
            raise _McpError("Client is closed")
        with self._lock:
            self._ensure_session()
            body = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments or {}},
            }
            headers = self._auth_headers()
        try:
            resp = httpx.post(self._base_url, json=body, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            result = self._parse_sse(resp.text)
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
            logger.debug("MCP call failed, reconnecting: %s", exc)
            with self._lock:
                self._session_id = None
                self._ensure_session()
                body["id"] = self._next_id()
                headers = self._auth_headers()
            retry_resp = httpx.post(self._base_url, json=body, headers=headers, timeout=self._timeout)
            retry_resp.raise_for_status()
            result = self._parse_sse(retry_resp.text)
        error = result.get("error")
        if error:
            raise _McpError(error.get("message", str(error)))
        r = result.get("result", {})
        if r.get("isError"):
            content = r.get("content", [])
            msg = content[0].get("text", "Unknown error") if content else "Unknown error"
            raise _McpError(msg)
        return r

    def close(self):
        with self._lock:
            self._closed = True
            self._session_id = None

    def _auth_headers(self) -> dict:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self._api_token:
            h["Authorization"] = f"Bearer {self._api_token}"
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        return h

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    @staticmethod
    def _parse_sse(text: str) -> dict:
        last_result = None
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("data: "):
                try:
                    last_result = json.loads(stripped[6:])
                except json.JSONDecodeError:
                    continue
        if last_result is not None:
            return last_result
        return json.loads(text)

    def _ensure_session(self):
        if self._session_id is not None:
            return
        with self._lock:
            if self._session_id is not None:
                return
            resp = httpx.post(
                self._base_url,
                json={
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "hermes-gbrain", "version": "1.0.0"},
                    },
                },
                headers=self._auth_headers(),
                timeout=self._timeout,
            )
            resp.raise_for_status()
            sid = resp.headers.get("mcp-session-id") or resp.headers.get("Mcp-Session-Id")
            if sid:
                self._session_id = str(sid)
            try:
                httpx.post(
                    self._base_url,
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                    headers=self._auth_headers(),
                    timeout=5,
                )
            except Exception:
                pass


def _load_config() -> dict:
    hermes_home = get_hermes_home()
    config_path = Path(hermes_home) / "gbrain_memory.json"
    config = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read %s: %s", config_path, e)
    config["url"] = (
        os.environ.get("MCP_GBRAIN_URL")
        or os.environ.get("GBRAIN_MCP_URL")
        or config.get("url", _DEFAULT_URL)
    )
    config["api_token"] = (
        os.environ.get("MCP_GBRAIN_API_KEY")
        or os.environ.get("GBRAIN_API_TOKEN")
        or os.environ.get("GBRAIN_MCP_TOKEN")
        or config.get("api_token", "")
    )
    config["timeout"] = float(
        os.environ.get("GBRAIN_MCP_TIMEOUT") or config.get("timeout", _DEFAULT_TIMEOUT)
    )
    return config


class GBrainMemoryProvider(MemoryProvider):

    def __init__(self):
        self._config: dict = {}
        self._mcp: _McpClient | None = None

    @property
    def name(self) -> str:
        return "gbrain"

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "url",
                "description": "GBrain MCP endpoint URL",
                "default": _DEFAULT_URL,
            },
            {
                "key": "api_token",
                "description": "GBrain MCP API bearer token",
                "secret": True,
                "env_var": "MCP_GBRAIN_API_KEY",
                "required": True,
            },
            {
                "key": "timeout",
                "description": "Request timeout in seconds",
                "default": _DEFAULT_TIMEOUT,
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / "gbrain_memory.json"
        existing = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        existing["url"] = values.get("url", existing.get("url", _DEFAULT_URL))
        existing["timeout"] = int(values.get("timeout", existing.get("timeout", _DEFAULT_TIMEOUT)))
        path.write_text(json.dumps(existing, indent=2) + "\n")

    def is_available(self) -> bool:
        """Check config only — NO network calls (per MemoryProvider contract)."""
        cfg = _load_config()
        return bool(cfg.get("api_token"))

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        url = self._config["url"]
        token = self._config["api_token"]
        timeout = self._config["timeout"]
        if not token:
            logger.warning("GBrain not configured: missing api_token")
            return
        self._mcp = _McpClient(url, token, timeout)
        logger.info("GBrain initialized — %d hardcoded tool schemas", len(HARDCODED_SCHEMA_DICTS))

    def shutdown(self) -> None:
        if self._mcp:
            self._mcp.close()
            self._mcp = None

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return hardcoded tool schemas — no dynamic discovery needed."""
        return list(HARDCODED_SCHEMA_DICTS)

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if not self._mcp:
            return json.dumps({"error": "GBrain not initialized"})
        try:
            schema = HARDCODED_SCHEMAS.get(tool_name)
            if not schema:
                return json.dumps({"error": f"Unknown gbrain tool: {tool_name}"})
            mcp_name = schema["mcp_name"]
            result = self._mcp.call(mcp_name, args)
            content_list = result.get("content", [])
            texts = [c["text"] for c in content_list if c.get("type") == "text"]
            return "\n".join(texts) if texts else json.dumps(result)
        except _McpError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            logger.exception("GBrain tool call failed: %s", tool_name)
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def system_prompt_block(self) -> str:
        names = [s["name"] for s in HARDCODED_SCHEMA_DICTS[:8]]
        return (
            "## GBrain Persistent Memory\n"
            "You have persistent memory via GBrain. The gbrain_* tools let you "
            "search, save, and reason across your knowledge base. Available tools: "
            f"{', '.join(names)}, and more."
        )


def register(ctx) -> None:
    ctx.register_memory_provider(GBrainMemoryProvider())
