# Claude Code Hooks

Semantic Cache MCP includes a PreToolUse hook that intercepts the built-in Read tool and returns cached content when available, saving 80%+ tokens automatically.

## How It Works

1. Claude tries to use the `Read` tool to read a file
2. The hook intercepts the call and checks the semantic cache
3. If the file is cached and unchanged → returns cache hit message, blocks Read
4. If not cached or file changed → allows Read to proceed normally

## Installation

### Quick Install (Recommended)

```bash
./hooks/install.sh
```

This automatically:
- Checks for required dependencies (sqlite3, jq, bc)
- Copies the hook script to `~/.claude/hooks/`
- Updates `~/.claude/settings.json` with proper configuration
- Creates backups before modifying settings

### Manual Installation

1. Copy the hook script:

```bash
mkdir -p ~/.claude/hooks
cp hooks/pre-tool-use.sh ~/.claude/hooks/
chmod +x ~/.claude/hooks/pre-tool-use.sh
```

2. Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read",
        "hooks": [
          {
            "type": "command",
            "command": "~/.claude/hooks/pre-tool-use.sh"
          }
        ]
      }
    ]
  }
}
```

3. Restart Claude Code

### Uninstall

```bash
./hooks/install.sh --uninstall
```

## Requirements

- `sqlite3` command-line tool (for cache queries)
- `jq` for JSON parsing
- `bc` for floating-point comparison

On Ubuntu/Debian:
```bash
sudo apt install sqlite3 jq bc
```

On macOS:
```bash
brew install sqlite jq bc
```

## What You'll See

When Claude tries to read a cached file:

```
[Semantic Cache] File unchanged: /path/to/file.py (1234 tokens cached)
Use 'read' tool from semantic-cache MCP for full cache benefits including diffs.
```

The Read is blocked, saving those 1234 tokens.

## Cache Location

- Database: `~/.cache/semantic-cache-mcp/cache.db`
- Hook log: `~/.cache/semantic-cache-mcp/hook.log`

## Troubleshooting

### Hook not intercepting reads

1. Verify matcher is `"Read"` (case-sensitive)
2. Check script is executable: `chmod +x ~/.claude/hooks/pre-tool-use.sh`
3. Ensure sqlite3/jq/bc are installed

### Always allowing reads through

The hook allows Read to proceed when:
- File not in cache (first read)
- File has been modified since caching
- Cache database doesn't exist
- Any error occurs (fails open)

Check the log at `~/.cache/semantic-cache-mcp/hook.log` for cache hits.

### Debugging

Add debug output:

```bash
# At top of pre-tool-use.sh
exec 2>> /tmp/hook-debug.log
set -x
```
