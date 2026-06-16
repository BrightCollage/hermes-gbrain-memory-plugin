# Architecture Decisions

## 1. Plugin Location: `plugins/gbrain/` not `plugins/memory/gbrain/`

**Decision:** User-installed memory provider plugins go in `$HERMES_HOME/plugins/<name>/`, NOT `$HERMES_HOME/plugins/memory/<name>/`.

**Rationale:** The Hermes memory provider discovery system (`plugins/memory/__init__.py`) scans two locations:
1. **Bundled providers**: `plugins/memory/<name>/` (shipped with hermes-agent)
2. **User-installed providers**: `$HERMES_HOME/plugins/<name>/` (root plugins directory)

The `memory/` subdirectory is only for bundled plugins. User plugins go in the root.

## 2. Config Source Priority

**Decision:** Read environment variables first, then `gbrain_memory.json`.

**Order (highest priority first):**
1. `MCP_GBRAIN_URL`, `MCP_GBRAIN_API_KEY` — same names as the old MCP server config
2. `GBRAIN_MCP_URL`, `GBRAIN_API_TOKEN`, `GBRAIN_MCP_TOKEN` — legacy fallbacks
3. `$HERMES_HOME/gbrain_memory.json` — file-based config

## 3. `is_available()` Must Not Make Network Calls

**Decision:** Check config only — return `bool(cfg.get("api_token"))`.

**Rationale:** The `MemoryProvider` ABC contract explicitly states "NO network calls" in `is_available()`. Network connectivity is verified lazily when tools are first used.

## 4. Dynamic Tool Discovery via `tools/list`

**Decision:** Don't hardcode 6 curated tools — discover all tools from the MCP server via `tools/list` during `initialize()`.

**Rationale:** The user wanted this to be a "wrapper for the MCP server" that dynamically exposes whatever tools the server provides. This approach:
- Automatically picks up new tools when the server is updated
- Avoids maintaining a curated list
- Mirrors what the MCP server connection did (83 tools) but gated by the `memory` toolset

## 5. SSE Parser: Collect All Events, Return Last

**Decision:** Parse all `data:` lines from the SSE response, return the last valid JSON-RPC message.

**Rationale:** The GBrain MCP server uses StreamableHTTP which can send:
- Single-event responses (most tools)
- Multi-event streams (progress + final result for streaming tools)

Returning the last valid message handles both cases — it's the final result.

## 6. Auto-Reconnect on Transport Error

**Decision:** Retry once on transport error by closing the session, re-initializing, and re-sending the request.

**Rationale:** MCP sessions expire (idle timeout, server restart). A single retry handles transient failures without infinite retry loops.

## 7. Fire-and-Forget for Writes

**Decision:** `gbrain_put` spawns a daemon thread and returns immediately.

**Rationale:** Mirror the `mcp_gb_save` pattern — writes are fire-and-forget, reads are synchronous. The daemon thread avoids blocking the agent loop.

## 8. Two-Step Delete

**Decision:** `gbrain_delete` previews the page on first call, requires `confirm=true` to execute.

**Rationale:** Prevent accidental data loss. The preview shows page content before deletion.
