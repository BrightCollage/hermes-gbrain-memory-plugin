# TODO: hermes-gbrain-memory-plugin

## Context

This is a Hermes Agent memory provider plugin that wraps the GBrain MCP server. It lives at `~/.hermes/plugins/gbrain/` and connects to `https://gbrain.plainrandom.com/mcp` via MCP-over-HTTP (StreamableHTTP). It dynamically discovers all tools from the server via `tools/list` at init time.

The plugin is installed at `~/.hermes/plugins/gbrain/` with `memory.provider: gbrain` set in config.yaml. The old MCP server entry (`mcp_servers.gbrain`) has been removed. The old git-installed plugin at `plugins/gbrain-memory/` has been moved to `~/gbrain-memory.backup`.

## Known Issues

### 1. gbrain_* tools not showing up in Hermes sessions

**Status:** UNRESOLVED — this is the main blocker.

When starting a new Hermes session, the agent reports no `gbrain_*` tools in its available tool list. The plugin is registered and enabled (`hermes plugins list` shows it), but tools are not being surfaced.

**Hypothesis:** The `register()` function calls `ctx.register_memory_provider(GBrainMemoryProvider())`, but maybe the MemoryManager isn't checking the `memory` toolset for the provider's tools, or the plugin's `get_tool_schemas()` isn't being called during prompt assembly.

**Investigation needed:**
- Check if `agent.memory_manager.MemoryManager` is actually activating the `gbrain` provider
- Check if the `memory` toolset is enabled for the CLI platform (check `platform_toolsets.cli` in config.yaml — should include `memory`)
- Trace why the MCP-replaced gbrain plugin tools aren't registering as tool schemas
- Compare with the old working plugin at `~/gbrain-memory.backup` — it worked and showed 6 tools
- Add debug logging to `initialize()`, `get_tool_schemas()`, and `handle_tool_call()` to trace the issue

**Files to check:**
- `agent/memory_manager.py` — how providers are activated and tools surfaced
- `agent/prompt_builder.py` — how memory provider tools are injected into the system prompt
- `toolsets.py` — how the `memory` toolset is defined

### 2. Plugin `kind: memory` not recognized

**Status:** FIXED — removed `kind: memory` from plugin.yaml.

The Hermes plugin system accepts: `backend`, `exclusive`, `model-provider`, `platform`, `standalone`. Error was:

```
Plugin gbrain: unknown kind 'memory'; treating as 'standalone'
```

The `register()` function handles memory provider registration via `ctx.register_memory_provider()` — the plugin.yaml doesn't need a special kind.

### 3. Plugin location confusion

**Status:** FIXED — plugin is at `~/.hermes/plugins/gbrain/`.

User-installed memory providers go in `$HERMES_HOME/plugins/<name>/` (root), NOT `plugins/memory/<name>/`. The `memory/` subdirectory is only for bundled (shipped-with-Hermes) plugins. The memory discovery loader (`plugins/memory/__init__.py`) scans:
- Bundled: `plugins/memory/<name>/`
- User: `$HERMES_HOME/plugins/<name>/`

### 4. `is_available()` made network calls

**Status:** FIXED — now checks config only.

The `MemoryProvider` ABC contract states "NO network calls" in `is_available()`. Removed the HTTP health check. The call is now `return bool(cfg.get("api_token"))`.

### 5. SSE parser returned first event only

**Status:** FIXED — now collects all `data:` lines, returns last valid JSON-RPC.

StreamableHTTP can send multiple events (progress + final result). The parser was returning the first matching `data:` line. Now it iterates all lines and returns the last valid JSON-RPC message.

### 6. Name collision with old git-installed plugin

**Status:** FIXED — old plugin moved to `~/gbrain-memory.backup`.

The old `plugins/gbrain-memory/` directory had `name: gbrain` in its plugin.yaml and was installed via git (`source: git`). This caused a name collision with the new plugin at `plugins/gbrain/`. The old plugin was moved out of the plugins directory.

## Remaining Work

### High Priority

- [ ] **Debug why gbrain_* tools don't appear in sessions**
  - Add logging to initialize(), get_tool_schemas() to trace registration
  - Check MemoryManager activation path
  - Compare with the working backup plugin
  - Verify `memory` toolset is in platform_toolsets.cli

- [ ] **Test MCP connectivity from the plugin**
  - Verify the MCP handshake works (initialize + initialized notification)
  - Verify tools/list returns the expected tool definitions
  - Test with a known-working tool like get_brain_identity

- [ ] **Test with a real Hermes session**
  - Start a session and confirm `gbrain_search`, `gbrain_put` etc. are invocable
  - Test putting a page and retrieving it
  - Test salience and think

### Medium Priority

- [ ] **Handle `***` credential substitution in gbrain_memory.json**
  - The config file at `$HERMES_HOME/gbrain_memory.json` may contain the literal string `***` which gets substituted by Hermes' credential provider
  - The plugin reads this file directly — if the token is `***`, it will fail
  - Fix: prefer env vars (MCP_GBRAIN_API_KEY) over the JSON file, since env vars bypass credential substitution

- [ ] **Add proper error handling for missing config**
  - If `initialize()` is called without a token, it logs a warning but doesn't surface it to the user
  - Consider raising a clear error or using `tool_error()` in handler

- [ ] **Write unit tests for _McpClient**
  - Test SSE parsing with single-event and multi-event payloads
  - Test auto-reconnect on transport errors
  - Test tool discovery

- [ ] **Add GitHub Actions CI**
  - Lint Python files
  - Run unit tests

### Low Priority

- [ ] **Support for MCP streaming tools**
  - Some GBrain tools (dream, think) may stream progress events
  - Current SSE parser returns the last message, which is correct for tools/call
  - Verify streaming works for dream/think specifically

- [ ] **Cache tool schemas across sessions**
  - Currently tools are re-discovered on every session start
  - Cache the tool list with a TTL to reduce latency

- [ ] **Add prefetch support**
  - Implement `queue_prefetch()` and `prefetch()` for automatic context injection before each turn
  - Query GBrain for relevant pages based on the user's message

- [ ] **Add sync_turn support**
  - Optionally save conversation turns as GBrain pages
  - Use a writer thread with a queue (like the old plugin did)

- [ ] **Publish to Hermes skills hub**
  - Make it installable via `hermes skills install`

## Architecture Reference

### File Structure

```
~/.hermes/plugins/gbrain/
├── plugin.yaml        # Metadata
└── __init__.py        # _McpClient + GBrainMemoryProvider + register()
```

### Key Classes

- **`_McpClient`**: Thread-safe MCP-over-HTTP client with auto-reconnect and dynamic tool discovery via `tools/list`
- **`GBrainMemoryProvider`**: Implements `MemoryProvider` ABC from `agent/memory_provider.py`
- **`_McpError`**: RuntimeError subclass for MCP protocol errors

### Config Resolution Order

1. `MCP_GBRAIN_URL` env var
2. `GBRAIN_MCP_URL` env var
3. `$HERMES_HOME/gbrain_memory.json` → `url` field

Same order for API token with `MCP_GBRAIN_API_KEY` → `GBRAIN_API_TOKEN` → `GBRAIN_MCP_TOKEN` → JSON file.

### MCP Protocol Flow

1. `POST /mcp` → `{"method": "initialize", ...}` → receive `Mcp-Session-Id` header
2. `POST /mcp` → `{"method": "notifications/initialized"}` (fire-and-forget)
3. `POST /mcp` → `{"method": "tools/list"}` → discover tools
4. `POST /mcp` → `{"method": "tools/call", "params": {"name": "...", "arguments": {...}}}` → call tool

All responses are StreamableHTTP SSE: `event: message\ndata: {json-rpc}\n\n`

### Header Requirements

```http
Content-Type: application/json
Accept: application/json, text/event-stream
Authorization: Bearer <token>
```

The GBrain MCP server rejects requests that only send `application/json` with a 406 Not Acceptable. Both `application/json` AND `text/event-stream` are required in the Accept header.

## Logs

Check `~/.hermes/logs/agent.log` for plugin registration and memory provider activation.
