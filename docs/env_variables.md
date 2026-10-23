# Environment Variables

All environment variables are optional. Defaults are tuned for typical usage.

A value that does not parse (`MAX_CACHE_ENTRIES=lots`) falls back to the
default rather than failing startup, and since 0.5.3 logs a warning naming the
variable — a silent fallback leaves you believing a setting took effect when
the symptom surfaces somewhere else. `SCMCP_IDLE_TIMEOUT` is the exception: it
falls back silently.

A value that parses but is out of range fails startup with `Configuration
errors`: `MAX_CACHE_ENTRIES` or `MAX_CONTENT_SIZE` ≤ 0, `TOOL_TIMEOUT` ≤ 0,
`TOOL_MAX_RESPONSE_TOKENS` < 0, or a `TOOL_OUTPUT_MODE` other than the three
listed below.

## Cache & Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_CACHE_DIR` | Platform-specific\* | Override cache/database directory path (`~` is expanded). The database, metrics, tokenizer file and logs all live under this directory. |
| `MAX_CACHE_ENTRIES` | `10000` | Maximum cached file entries before W-TinyLFU eviction kicks in. Higher values use more memory and disk. |
| `MAX_CONTENT_SIZE` | `100000` | Default `max_size` of `read`. A file larger than this returns a map instead of its text: the outline, or for a file with no definitions its first and last lines. Use `offset`/`limit` to read the lines the map names. Must be > 0. |

\* Linux: `$XDG_CACHE_HOME/semantic-cache-mcp/` if set, else `~/.cache/semantic-cache-mcp/`, macOS: `~/Library/Caches/semantic-cache-mcp/`, Windows: `%LOCALAPPDATA%\semantic-cache-mcp\`


## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Set to `DEBUG` for troubleshooting storage issues. Case-insensitive. An unrecognized value logs a warning and falls back to `INFO`. |
| `LOG_DIR` | `<cache dir>/logs` | Directory for the daily log file `semantic-cache-mcp-YYYY-MM-DD.log`. The shared daemon runs detached with no stderr, so this file is where its messages go. |

## Tool Response

| Variable | Default | Description |
|----------|---------|-------------|
| `TOOL_OUTPUT_MODE` | `compact` | Response detail level. Options: `compact` (minimal metadata, best for token savings), `normal` (includes context lines in grep, extra diagnostics), `debug` (full diagnostics including timing and internal state). Case-insensitive; any other value fails startup. |
| `TOOL_MAX_RESPONSE_TOKENS` | `0` | Global cap on response tokens per tool call. `0` disables the cap. Useful for constraining token budget on large operations. |
| `TOOL_TIMEOUT` | `30` | Seconds before a tool call times out and returns an error. On timeout, the executor is automatically reset so subsequent calls work without restarting. Accepts fractions; must be > 0. Lower for fast machines, raise for slow I/O or large files. |

## Tool Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `SCMCP_FORMAT_TIMEOUT_S` | `15` | Seconds an `auto_format` formatter subprocess may run. Values below `1` are raised to `1`; a malformed value logs a warning and keeps the default. Raise for slow formatters such as eslint on a monorepo. |
| `SCMCP_MAX_IMAGE_BYTES` | `5242880` (5 MiB) | Largest image file `read_image` accepts. Values below `1024` are raised to `1024`; a malformed value logs a warning and keeps the default. |
| `SCMCP_MAX_ENCODED_IMAGE_BYTES` | `5000000` | Largest base64-encoded image payload `read_image` returns. Same floor and malformed-value handling. |

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

## Shared Server

| Variable | Default | Description |
|----------|---------|-------------|
| `SCMCP_SHARED_SERVER` | `true` | The command relays to one shared daemon per user and configuration, starting it when needed. `0/false/no/off` serves in-process, one server per client; any other value, including a typo, leaves it on. If the daemon cannot be reached or started, the command serves in-process and warns on stderr. See [architecture.md](architecture.md#process-model). |
| `SCMCP_SHARED_TRANSPORT` | `unix` | `unix`: a Unix socket in a `0700` directory. `tcp`: a random loopback port, with a token from a `0600` port file required on every connection. Windows always uses `tcp`; any other value means `unix`. |
| `SCMCP_IDLE_TIMEOUT` | `600` | Seconds a shared daemon stays up with no client connected. A malformed value silently keeps the default; a negative value means `0`, which exits as soon as the last client disconnects. A new daemon always waits at least 10 s for its first client. |
| `XDG_RUNTIME_DIR` | *(system temp dir)* | POSIX: parent of the private `semantic-cache-mcp-<uid>/` directory holding daemon endpoints. Windows uses `%LOCALAPPDATA%\semantic-cache-mcp\run\`. |

A daemon is keyed by version, install path, and the values of `SEMANTIC_CACHE_DIR`, `LOG_DIR`, `LOG_LEVEL`, `MAX_CACHE_ENTRIES`, `MAX_CONTENT_SIZE`, `TOOL_MAX_RESPONSE_TOKENS`, `TOOL_OUTPUT_MODE`, `TOOL_TIMEOUT`, `XDG_CACHE_HOME`, `LOCALAPPDATA`, `PATH`, `HOME`, `USERPROFILE` and every `SCMCP_` variable. Clients that differ in any of them get separate daemons, so a setting never leaks between clients, and a new build gets a fresh daemon.

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
