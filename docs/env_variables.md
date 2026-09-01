# Environment Variables

All environment variables are optional. Defaults are tuned for typical usage.

A malformed value falls back to the default rather than failing startup, and
since 0.5.3 logs a warning naming the variable — a silent fallback leaves you
believing a setting took effect when the symptom surfaces somewhere else.

## Cache & Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_CACHE_DIR` | Platform-specific\* | Override cache/database directory path. All data (database, models, metrics) lives under this directory. |
| `MAX_CACHE_ENTRIES` | `10000` | Maximum cached file entries before W-TinyLFU eviction kicks in. Higher values use more memory and disk. |
| `MAX_CONTENT_SIZE` | `100000` | Maximum bytes returned by a single read operation. Files larger than this are truncated with a hint to use `offset`/`limit`. |

\* Linux: `~/.cache/semantic-cache-mcp/`, macOS: `~/Library/Caches/semantic-cache-mcp/`, Windows: `%LOCALAPPDATA%\semantic-cache-mcp\`


## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Set to `DEBUG` for troubleshooting storage issues. An unrecognized value falls back to `INFO` instead of failing startup. |

## Tool Response

| Variable | Default | Description |
|----------|---------|-------------|
| `TOOL_OUTPUT_MODE` | `compact` | Response detail level. Options: `compact` (minimal metadata, best for token savings), `normal` (includes context lines in grep, extra diagnostics), `debug` (full diagnostics including timing and internal state). |
| `TOOL_MAX_RESPONSE_TOKENS` | `0` | Global cap on response tokens per tool call. `0` disables the cap. Useful for constraining token budget on large operations. |
| `TOOL_TIMEOUT` | `30` | Seconds before a tool call times out and returns an error. On timeout, the executor is automatically reset so subsequent calls work without restarting. Lower for fast machines, raise for slow I/O or large files. |

## Wire Shape

Two costs are paid on every turn no matter what any tool does: the advertised
tool list sits in the prompt prefix of every request, and a tool result may go
out twice — once as a text block, once as `structuredContent`. Both defaults
here are chosen for the cheaper shape. Turn them on only for a client that
actually consumes structured output.

| Variable | Default | Description |
|----------|---------|-------------|
| `SCMCP_STRUCTURED_CONTENT` | `false` | Also send each result as MCP `structuredContent`. It duplicates the text block byte for byte (measured at 8.8k + 8.9k tokens for one 584-line file), and clients disagree about which they forward, so a client that forwards both charges the model twice for every file. Accepts `1/true/yes/on` and `0/false/no/off`; anything else logs a warning and keeps the default. |
| `SCMCP_PUBLISH_OUTPUT_SCHEMA` | `false` | Advertise per-tool output schemas in `tools/list`. They were 11.5k of this server's 19.8k advertised tokens, and the Anthropic Messages API has no field to receive them. The Pydantic response models stay either way — they remain the declared contract, enforced by `tests/test_response_contract.py`. Turning this on forces `SCMCP_STRUCTURED_CONTENT` on as well, because MCP requires structured content from any tool that declares a schema; the server logs a warning when it does so. |

## Example: MCP Server Config with Custom Env

```json
{
  "mcpServers": {
    "semantic-cache": {
      "command": "uvx",
      "args": ["semantic-cache-mcp"],
      "env": {
        "LOG_LEVEL": "DEBUG",
        "MAX_CACHE_ENTRIES": "20000"
      }
    }
  }
}
```
