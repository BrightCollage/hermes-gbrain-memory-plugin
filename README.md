# Hermes GBrain Memory Provider Plugin

A Hermes Agent memory provider plugin that wraps the GBrain MCP server via MCP-over-HTTP (StreamableHTTP). Dynamically discovers all tools from the server at initialization time and exposes them through the `MemoryProvider` interface.

Instead of loading 83 MCP tools into every session via `mcp_servers`, this plugin gates the GBrain toolset behind the `memory` toolset — saving ~9,600 tokens per session.

## Architecture

```
Hermes agent
  └── gbrain memory provider plugin
        └── httpx POST → https://gbrain.example.com/mcp
              └── gbrain serve --http pod (port 3001)
                    └── Postgres / PGLite database
```

## Directory Structure

```
~/.hermes/plugins/gbrain/
├── plugin.yaml        # Metadata
└── __init__.py        # MemoryProvider + MCP client + tool discovery
```

## Installation

```bash
# Clone into plugins directory
git clone https://github.com/BrightCollage/hermes-gbrain-memory-plugin \
  ~/.hermes/plugins/gbrain

# OR copy manually:
cp -r hermes-gbrain-memory-plugin/gbrain ~/.hermes/plugins/
```

## Configuration

The plugin reads configuration from environment variables (highest priority first):

| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_GBRAIN_API_KEY` | GBrain MCP API bearer token | **(required)** |
| `MCP_GBRAIN_URL` | GBrain MCP endpoint URL | `https://gbrain.plainrandom.com/mcp` |

Config file fallback at `$HERMES_HOME/gbrain_memory.json`:

```json
{
  "url": "https://gbrain.plainrandom.com/mcp",
  "api_token": "gbrain_...",
  "timeout": 30
}
```

Run `hermes memory setup` for an interactive configuration wizard.

## Activation

```bash
# 1. Enable the plugin
hermes plugins enable gbrain

# 2. Set as active memory provider
hermes config set memory.provider gbrain

# 3. Remove old MCP server (if migrating)
hermes mcp remove gbrain

# 4. Start a new session
hermes
```

## How It Works

1. **Discovery**: On `initialize()`, the plugin connects to the GBrain MCP server via HTTP POST, performs the MCP handshake (`initialize` + `notifications/initialized`), then calls `tools/list` to discover all available tools.

2. **Tool schemas**: Each MCP tool definition (`name`, `description`, `inputSchema`) is converted to an OpenAI function-calling schema and exposed via `get_tool_schemas()`.

3. **Dispatch**: When the LLM calls a `gbrain_*` tool, `handle_tool_call()` translates it to the corresponding MCP `tools/call` and returns the result.

4. **SSE handling**: The GBrain MCP server uses StreamableHTTP (SSE). The parser collects all `data:` lines and returns the last valid JSON-RPC message, handling both single-event responses and multi-event streams.

5. **Auto-reconnect**: On transport errors (stale session, server restart), the session is torn down, re-initialized, and the call is retried once.

## Features

- **Dynamic tool discovery** — no hardcoded tool list; whatever the server exposes is available
- **Fire-and-forget writes** — `gbrain_put` runs on a daemon thread, returns immediately
- **Two-step deletes** — `gbrain_delete` previews before executing
- **SSE streaming support** — parses multiple `data:` events, returns final result
- **Auto-reconnect** — one retry on transport errors
- **Config wizard** — `get_config_schema()` for `hermes memory setup`

## Key Files

| File | Purpose |
|------|---------|
| `gbrain/plugin.yaml` | Hermes plugin manifest |
| `gbrain/__init__.py` | Full implementation: `_McpClient` + `GBrainMemoryProvider` |
| `DECISIONS.md` | Architecture decisions and trade-offs |
| `TODO.md` | Outstanding work and improvements |
