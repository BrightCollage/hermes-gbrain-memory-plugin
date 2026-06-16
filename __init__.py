"""GBrain memory provider plugin — dynamic MCP server wrapper.

Connects to GBrain via MCP-over-HTTP (StreamableHTTP) and dynamically
discovers all tools from the server via tools/list at init time.
Exposes them through the MemoryProvider interface so the GBrain toolset
is gated by the `memory` toolset instead of loading 83 tools in every
session via mcp_servers.

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

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://gbrain.plainrandom.com/mcp"
_DEFAULT_TIMEOUT = 30


class _McpError(RuntimeError):
    pass


class _McpClient:
    """Thread-safe MCP-over-HTTP client with auto-reconnect and tool discovery."""

    def __init__(self, url: str, api_token: str, timeout: float = _DEFAULT_TIMEOUT):
        self._base_url = url.rstrip("/")
        self._api_token = api_token
        self._timeout = timeout
        self._session_id: str | None = None
        self._lock = threading.RLock()
        self._closed = False
        self._req_id = 0
        self._tools: list[dict] | None = None

    @property
    def tools(self) -> list[dict]:
        if self._tools is None:
            self._ensure_session()
        return self._tools or []

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
                self._tools = None
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

    def health(self) -> bool:
        try:
            base = self._base_url.rstrip("/mcp").rstrip("/")
            headers = {"Accept": "application/json"}
            if self._api_token:
                headers["Authorization"] = f"Bearer {self._api_token}"
            resp = httpx.get(f"{base}/health", headers=headers, timeout=5.0)
            return resp.status_code < 500
        except Exception:
            return False

    def close(self):
        with self._lock:
            self._closed = True
            self._session_id = None
            self._tools = None

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
        """Parse StreamableHTTP SSE response into a JSON-RPC dict.

        Handles both single-event and multi-event SSE payloads. For
        multi-event streams (e.g. streaming progress + final result),
        returns the LAST valid JSON-RPC message (typically the final
        result). Ignores non-JSON data lines and event metadata.
        """
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
        # Fallback: try raw JSON
        return json.loads(text)

    def _ensure_session(self):
        if self._session_id is not None and self._tools is not None:
            return
        with self._lock:
            if self._session_id is not None and self._tools is not None:
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
            try:
                tools_resp = httpx.post(
                    self._base_url,
                    json={"jsonrpc": "2.0", "id": self._next_id(), "method": "tools/list", "params": {}},
                    headers=self._auth_headers(),
                    timeout=self._timeout,
                )
                tools_resp.raise_for_status()
                tools_result = self._parse_sse(tools_resp.text)
                self._tools = tools_result.get("result", {}).get("tools", [])
                logger.info("Discovered %d GBrain MCP tools", len(self._tools or []))
            except Exception as exc:
                logger.warning("Failed to discover GBrain tools: %s", exc)
                self._tools = []


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


def _mcp_tool_to_schema(mcp_tool: dict) -> dict:
    name = mcp_tool.get("name", "")
    desc = mcp_tool.get("description", "")
    input_schema = mcp_tool.get("inputSchema", mcp_tool.get("input_schema", {}))
    return {"name": name, "description": desc, "parameters": input_schema}


class GBrainMemoryProvider(MemoryProvider):

    def __init__(self):
        self._config: dict = {}
        self._mcp: _McpClient | None = None
        self._tool_cache: list[dict] = []

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
        try:
            tools = self._mcp.tools
            self._tool_cache = [_mcp_tool_to_schema(t) for t in tools]
            logger.info("GBrain loaded %d tools from server", len(self._tool_cache))
        except Exception as exc:
            logger.warning("GBrain tool discovery failed: %s", exc)
            self._tool_cache = []

    def shutdown(self) -> None:
        if self._mcp:
            self._mcp.close()
            self._mcp = None

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return list(self._tool_cache)

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if not self._mcp:
            return json.dumps({"error": "GBrain not initialized"})
        try:
            result = self._mcp.call(tool_name, args)
            content_list = result.get("content", [])
            texts = [c["text"] for c in content_list if c.get("type") == "text"]
            return "\n".join(texts) if texts else json.dumps(result)
        except _McpError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            logger.exception("GBrain tool call failed: %s", tool_name)
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    def system_prompt_block(self) -> str:
        return (
            "## GBrain Persistent Memory\n"
            "You have persistent memory via GBrain. The gbrain_* tools let you "
            "search, save, and reason across your knowledge base. For most recall "
            "use gbrain_search. For recent activity use gbrain_salience. "
            "For entity facts use gbrain_recall. Save important information "
            "with gbrain_put."
        )


def register(ctx) -> None:
    ctx.register_memory_provider(GBrainMemoryProvider())
