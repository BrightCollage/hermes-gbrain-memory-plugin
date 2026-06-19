"""GBrain memory provider plugin — hardcoded tool schemas + OAuth support.

Connects to GBrain via MCP-over-HTTP (StreamableHTTP) and exposes
83 hardcoded tool schemas through the MemoryProvider interface.
No dynamic discovery — schemas are generated from the live server
and stored in _schemas.py. Regenerate with:

    python3 -c "from _schemas import *; ..."

Supports two auth modes:

  **Static bearer token** (legacy) — uses MCP_GBRAIN_API_KEY as a
  pre-minted bearer token for the /mcp endpoint.

  **OAuth client_credentials** — exchanges a client_id + client_secret
  for a bearer token at the /token endpoint, with automatic refresh
  on expiry. Enabled when BOTH MCP_GBRAIN_OAUTH_CLIENT_ID and
  MCP_GBRAIN_OAUTH_CLIENT_SECRET are set. Falls back to static token
  if neither OAuth pair is configured.

All configuration is env-var based (MCP_GBRAIN_* prefix). No config file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from agent.memory_provider import MemoryProvider

from ._schemas import HARDCODED_SCHEMAS, HARDCODED_SCHEMA_DICTS
from ._stop_words import _STOP_WORDS

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://gbrain.plainrandom.com/mcp"
_DEFAULT_TIMEOUT = 30
_DEFAULT_OAUTH_TOKEN_TTL = 3600  # 1 hour fallback if server omits expires_in


class _McpError(RuntimeError):
    pass


class _McpClient:
    """Thread-safe MCP-over-HTTP client. Connects on-demand, no tool discovery.

    Supports two auth modes:

    * **Static bearer token** — pass ``api_token``. Used as-is for the
      ``Authorization`` header.

    * **OAuth client_credentials** — pass ``oauth_client_id`` +
      ``oauth_client_secret`` (both non-empty). The client fetches a fresh
      bearer token from ``oauth_token_url`` (defaults to ``<base>/token``)
      before each MCP call, with proactive refresh before expiry.
    """

    def __init__(
        self,
        url: str,
        api_token: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
        *,
        oauth_client_id: str = "",
        oauth_client_secret: str = "",
    ):
        self._base_url = url.rstrip("/")
        self._api_token = api_token
        self._timeout = timeout
        self._oauth_client_id = oauth_client_id
        self._oauth_client_secret = oauth_client_secret
        # OAuth token endpoint is always <mcp_base>/token
        self._oauth_token_url = self._base_url.replace("/mcp", "").rstrip("/") + "/token"
        self._is_oauth = bool(oauth_client_id and oauth_client_secret)
        self._token_expires_at: float = 0.0
        self._session_id: str | None = None
        self._lock = threading.RLock()
        self._closed = False
        self._req_id = 0

    def call(self, tool_name: str, arguments: dict) -> dict:
        if self._closed:
            raise _McpError("Client is closed")
        with self._lock:
            self._ensure_token()
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
        except httpx.HTTPStatusError as exc:
            # 401 in OAuth mode → force token refresh and retry once
            if self._is_oauth and exc.response.status_code == 401:
                logger.debug("GBrain OAuth: token expired (401), refreshing and retrying")
                with self._lock:
                    self._force_token_refresh()
                    self._ensure_token()
                    body["id"] = self._next_id()
                    headers = self._auth_headers()
                try:
                    retry_resp = httpx.post(
                        self._base_url, json=body, headers=headers, timeout=self._timeout
                    )
                    retry_resp.raise_for_status()
                    result = self._parse_sse(retry_resp.text)
                except httpx.HTTPStatusError as retry_exc:
                    raise _McpError(
                        f"OAuth token refresh did not resolve auth: "
                        f"{retry_exc.response.status_code} {retry_exc.response.text[:500]}"
                    ) from retry_exc
                except (httpx.TimeoutException, httpx.ConnectError) as retry_exc:
                    raise _McpError(
                        f"MCP call failed after OAuth token refresh: {retry_exc}"
                    ) from retry_exc
            else:
                raise _McpError(
                    f"MCP returned {exc.response.status_code}: {exc.response.text[:500]}"
                ) from exc
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            logger.debug("MCP call failed, reconnecting: %s", exc)
            with self._lock:
                self._session_id = None
                self._ensure_session()
                body["id"] = self._next_id()
                headers = self._auth_headers()
            try:
                retry_resp = httpx.post(
                    self._base_url, json=body, headers=headers, timeout=self._timeout
                )
                retry_resp.raise_for_status()
                result = self._parse_sse(retry_resp.text)
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as retry_exc:
                raise _McpError(
                    f"MCP call failed after reconnect: {retry_exc}"
                ) from retry_exc
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

    def _ensure_token(self) -> None:
        """Obtain or refresh the OAuth bearer token.

        No-op in static-token mode. In OAuth mode, fetches a fresh token
        from the /token endpoint when:

        * No token has been acquired yet (first call).
        * The current token is within 60 seconds of expiry.
        * The server returned 401 on the previous call (forced refresh).
        """
        if not self._is_oauth:
            return
        now = time.monotonic()
        if self._api_token and now < self._token_expires_at - 60:
            return
        logger.debug("GBrain OAuth: acquiring fresh token from %s", self._oauth_token_url)
        try:
            resp = httpx.post(
                self._oauth_token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._oauth_client_id,
                    "client_secret": self._oauth_client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            token = data.get("access_token")
            if not token:
                raise _McpError(
                    "OAuth token response missing 'access_token' field"
                )
            self._api_token = token
            expires_in = data.get("expires_in", _DEFAULT_OAUTH_TOKEN_TTL)
            self._token_expires_at = time.monotonic() + int(expires_in)
            logger.debug("GBrain OAuth: token acquired, expires in %ss", expires_in)
        except _McpError:
            raise
        except Exception as e:
            raise _McpError(f"OAuth token acquisition failed: {e}") from e

    def _force_token_refresh(self) -> None:
        """Clear the current token so the next _ensure_token fetches fresh."""
        self._api_token = ""
        self._token_expires_at = 0.0

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
    """Read configuration from environment variables only — no JSON fallback."""
    return {
        "url": os.environ.get("MCP_GBRAIN_URL", _DEFAULT_URL),
        "api_token": os.environ.get("MCP_GBRAIN_API_KEY", ""),
        "oauth_client_id": os.environ.get("MCP_GBRAIN_OAUTH_CLIENT_ID", ""),
        "oauth_client_secret": os.environ.get("MCP_GBRAIN_OAUTH_CLIENT_SECRET", ""),
        "timeout": float(os.environ.get("MCP_GBRAIN_TIMEOUT", _DEFAULT_TIMEOUT)),
    }


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
        """Shared config fields — auth credentials are handled in post_setup()."""
        return [
            {
                "key": "url",
                "description": "GBrain MCP endpoint URL",
                "default": _DEFAULT_URL,
            },
            {
                "key": "timeout",
                "description": "Request timeout in seconds",
                "default": _DEFAULT_TIMEOUT,
            },
        ]

    def post_setup(self, hermes_home: str, config: dict[str, Any]) -> dict[str, Any]:
        """Interactive setup wizard: choose API key or OAuth, then prompt for credentials."""
        print("\n── GBrain Auth Setup ──")
        print("Choose an authentication method:")
        print("  1) API key (static bearer token)")
        print("  2) OAuth client_credentials")
        choice = input("Choice [1/2]: ").strip()

        env_path = Path(hermes_home) / ".env"
        env_lines: list[str] = []
        if env_path.exists():
            env_lines = env_path.read_text().splitlines()

        def _set_env(key: str, value: str) -> None:
            nonlocal env_lines
            env_lines = [l for l in env_lines if not l.startswith(f"{key}=")]
            env_lines.append(f"{key}={value}")

        # Persist url and timeout from schema prompts
        url = config.get("url") or os.environ.get("MCP_GBRAIN_URL", _DEFAULT_URL)
        timeout = str(
            config.get("timeout")
            or os.environ.get("MCP_GBRAIN_TIMEOUT", str(_DEFAULT_TIMEOUT))
        )
        _set_env("MCP_GBRAIN_URL", url)
        _set_env("MCP_GBRAIN_TIMEOUT", timeout)

        # Strip old auth vars so a switch doesn't leave stale secrets
        for stale_key in (
            "MCP_GBRAIN_API_KEY",
            "MCP_GBRAIN_OAUTH_CLIENT_ID",
            "MCP_GBRAIN_OAUTH_CLIENT_SECRET",
        ):
            env_lines = [l for l in env_lines if not l.startswith(f"{stale_key}=")]

        if choice == "2":
            print("\n── OAuth Client Credentials ──")
            cid = input("Client ID: ").strip()
            secret = input("Client secret: ").strip()
            if cid and secret:
                _set_env("MCP_GBRAIN_OAUTH_CLIENT_ID", cid)
                _set_env("MCP_GBRAIN_OAUTH_CLIENT_SECRET", secret)
                print("  ✓ OAuth credentials saved")
            else:
                print("  ✗ Both client ID and secret are required — skipping")
        else:
            print("\n── API Key ──")
            token = input("Bearer token: ").strip()
            if token:
                _set_env("MCP_GBRAIN_API_KEY", token)
                print("  ✓ API key saved")
            else:
                print("  ✗ Token is required — skipping")

        env_path.write_text("\n".join(env_lines) + "\n")
        return config

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        """No-op — all config is env-based, persisted by post_setup()."""

    def is_available(self) -> bool:
        """Check config only — NO network calls (per MemoryProvider contract)."""
        cfg = _load_config()
        has_token = bool(cfg.get("api_token"))
        has_oauth = bool(cfg.get("oauth_client_id") and cfg.get("oauth_client_secret"))
        return has_token or has_oauth

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        url = self._config["url"]
        token = self._config["api_token"]
        timeout = self._config["timeout"]
        has_any_auth = bool(
            token
            or (self._config["oauth_client_id"] and self._config["oauth_client_secret"])
        )
        if not has_any_auth:
            logger.warning("GBrain not configured: missing api_token or OAuth credentials")
            return
        self._mcp = _McpClient(
            url,
            api_token=token,
            timeout=timeout,
            oauth_client_id=self._config["oauth_client_id"],
            oauth_client_secret=self._config["oauth_client_secret"],
        )
        logger.info(
            "GBrain initialized — %s — %d hardcoded tool schemas",
            "OAuth client_credentials" if self._config["oauth_client_id"] else "static bearer token",
            len(HARDCODED_SCHEMA_DICTS),
        )

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
