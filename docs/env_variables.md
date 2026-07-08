# Environment Variables

All environment variables are optional. Defaults are tuned for typical usage.

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
