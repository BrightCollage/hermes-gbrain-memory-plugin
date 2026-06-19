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
import re
import threading
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home

from ._schemas import HARDCODED_SCHEMAS, HARDCODED_SCHEMA_DICTS
from ._stop_words import _STOP_WORDS

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
        or config.get("url", _DEFAULT_URL)
    )
    config["api_token"] = (
        os.environ.get("MCP_GBRAIN_API_KEY")
        or config.get("api_token", "")
    )
    config["timeout"] = float(
        os.environ.get("MCP_GBRAIN_TIMEOUT") or config.get("timeout", _DEFAULT_TIMEOUT)
    )
    return config


class GBrainMemoryProvider(MemoryProvider):

    def __init__(self):
        self._config: dict = {}
        self._mcp: _McpClient | None = None

    # ── brain-agent-loop helpers ──────────────────────────────────────

    @staticmethod
    def _extract_candidates(text: str) -> list[str]:
        """Extract potential entity names using a mask-based approach.

        Assumes ALL words are potential entities, then strips out words
        that are provably NOT entities (articles, pronouns, prepositions,
        conjunctions, verbs, adjectives, adverbs, discourse markers, etc.)
        using a comprehensive mask of 2,226 English words.

        Consecutive non-masked words are grouped into multi-word phrases.
        Returns at most 10 candidates, deduplicated.
        """
        if not text:
            return []

        seen: set[str] = set()
        candidates: list[str] = []
        current_phrase: list[str] = []
        stop = _STOP_WORDS

        for token in text.split():
            # Strip leading/trailing punctuation — keep internal hyphens/apostrophes
            clean = token.strip(".,!?;:()[]{}«»""`´~@#$%^&*=+<>/\\|¬·–—")
            if not clean:
                continue

            if clean.lower() in stop:
                # Flush accumulated phrase
                if current_phrase:
                    phrase = " ".join(current_phrase)
                    if phrase.lower() not in seen:
                        seen.add(phrase.lower())
                        candidates.append(phrase)
                    current_phrase = []
            else:
                current_phrase.append(clean)

        # Flush final phrase
        if current_phrase:
            phrase = " ".join(current_phrase)
            if phrase.lower() not in seen:
                candidates.append(phrase)

        return candidates[:10]

    def _search_entity(self, name: str) -> str:
        """Search GBrain for an entity name. Return formatted context or ''."""
        if not self._mcp:
            return ""
        try:
            result = self._mcp.call("search", {"query": name, "limit": 3})
            content_list = result.get("content", [])
            for c in content_list:
                text = c.get("text", "")
                if not text:
                    continue
                # Try to parse as structured JSON
                try:
                    data = json.loads(text)
                    if isinstance(data, list):
                        if data:
                            # Return compiled_truth from top 2 matches
                            lines = [f"## Brain context for \"{name}\"", ""]
                            for item in data[:2]:
                                slug = item.get("slug", "")
                                title = item.get("title", slug)
                                chunk = item.get("chunk_text", "")
                                page_type = item.get("type", "")
                                # Truncate long chunks
                                if len(chunk) > 600:
                                    chunk = chunk[:600] + "..."
                                lines.append(
                                    f"**{title}** (/{slug}, {page_type})"
                                )
                                lines.append(chunk)
                                lines.append("")
                            return "\n".join(lines)
                        # Empty list — no results, try next content item
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
                # Fallback: return raw text (capped)
                if len(text) > 1200:
                    text = text[:1200] + "..."
                return f"## Brain context for \"{name}\"\n\n{text}"
        except Exception as e:
            logger.debug("_search_entity failed for %r: %s", name, e)
        return ""

    def _resolve_entities(self, text: str) -> list[dict]:
        """Extract entity names and resolve to brain slugs + types.

        Returns list of {name, slug, type} dicts.
        """
        candidates = self._extract_candidates(text)
        if not candidates:
            return []

        resolved: list[dict] = []
        seen_slugs: set[str] = set()

        for name in candidates[:5]:
            try:
                result = self._mcp.call("search", {"query": name, "limit": 3})
                content_texts = [
                    c.get("text", "")
                    for c in result.get("content", [])
                    if c.get("text")
                ]
                for raw in content_texts:
                    try:
                        data = json.loads(raw)
                        if isinstance(data, list):
                            for item in data:
                                page_type = item.get("type", "")
                                if page_type not in ("person", "company"):
                                    continue
                                slug = item.get("slug", "")
                                title = item.get("title", "")
                                if not slug or slug in seen_slugs:
                                    continue
                                # Verify the candidate name matches the page title
                                name_lower = name.lower()
                                title_lower = title.lower()
                                if not (name_lower in title_lower
                                        or title_lower.startswith(name_lower)
                                        or name_lower.replace(" ", "-") in slug.lower()):
                                    continue
                                resolved.append({
                                    "name": name,
                                    "slug": slug,
                                    "type": page_type,
                                })
                                seen_slugs.add(slug)
                    except (json.JSONDecodeError, TypeError):
                        # Fallback: extract slugs from wiki-link markdown
                        for match in re.finditer(
                            r'\[([^\]]+)\]\(/([^)]+)\)', raw
                        ):
                            slug = match.group(2)
                            if slug not in seen_slugs:
                                resolved.append({
                                    "name": name,
                                    "slug": slug,
                                    "type": "unknown",
                                })
                                seen_slugs.add(slug)
            except Exception as e:
                logger.debug("_resolve_entities search failed for %r: %s", name, e)

        return resolved[:5]

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

    # ── brain-agent-loop: READ step (per guide — read brain BEFORE responding) ──

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Search GBrain for entities in the current message SYNCHRONOUSLY.

        Per the brain-agent-loop guide: "Read BEFORE responding, not after."
        Extracts entity names from the current message, searches GBrain for
        each, and returns formatted context for injection into the model's
        system prompt — all within the same turn.

        The search is capped at ~1s total (up to 3 entity lookups) so the
        latency hit is minimal. Providers with no entities to look up return
        immediately.
        """
        if not self._mcp or not query:
            return ""
        candidates = self._extract_candidates(query)
        if not candidates:
            return ""

        context_parts = []
        for name in candidates[:3]:  # cap at 3 to keep latency low
            ctx = self._search_entity(name)
            if ctx:
                context_parts.append(ctx)

        if not context_parts:
            return ""

        logger.info(
            "brain-loop prefetch: %d entities detected, %d context blocks returned "
            "(candidates=%r)", len(candidates), len(context_parts), candidates[:5]
        )

        return (
            "## GBrain — automatically detected context\n\n"
            + "\n---\n\n".join(context_parts)
        )

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Warm the brain cache for the NEXT turn (optional — runs in background).

        Called after each turn completes. Searches the current message's
        entities so the next turn's prefetch() may benefit from cached
        results, but the primary read path is prefetch() which runs
        synchronously on the current turn.
        """
        # No-op: the primary read happens synchronously in prefetch().
        # This hook is left available for future cache-warming use.
        pass

    def _write_timeline(self, slug: str, today_str: str) -> None:
        """Add a timeline entry for a resolved entity slug."""
        if not self._mcp:
            return
        try:
            self._mcp.call("add_timeline_entry", {
                "slug": slug,
                "date": today_str,
                "summary": f"Mentioned in conversation on {today_str}",
                "source": "hermes-brain-loop",
            })
            logger.info("brain-loop: added timeline to /%s", slug)
        except Exception as e:
            logger.debug("brain-loop: timeline write failed for /%s: %s", slug, e)

    # ── brain-agent-loop: WRITE step ──────────────────────────────────

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages=None,
    ) -> None:
        """Write timeline entries for entities mentioned in the conversation.

        Called after each turn completes. Already runs on a background
        worker thread via memory_manager's ThreadPoolExecutor — no
        additional daemon thread needed.
        """
        if not self._mcp or not user_content:
            return

        today_str = date.today().isoformat()
        try:
            entities = self._resolve_entities(user_content)
            for e in entities:
                self._write_timeline(e["slug"], today_str)
        except Exception as e:
            logger.debug("brain-loop sync_turn background failed: %s", e)


def register(ctx) -> None:
    ctx.register_memory_provider(GBrainMemoryProvider())
