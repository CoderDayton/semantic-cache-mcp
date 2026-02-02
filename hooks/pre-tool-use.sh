#!/bin/bash
# Semantic Cache MCP - PreToolUse Hook
# Intercepts Read tool calls and uses semantic cache for 80%+ token savings
#
# Install:
#   cp hooks/pre-tool-use.sh ~/.claude/hooks/
#   chmod +x ~/.claude/hooks/pre-tool-use.sh
#
# Add to ~/.claude/settings.json:
#   {
#     "hooks": {
#       "PreToolUse": [
#         {
#           "matcher": "Read",
#           "hooks": [
#             {
#               "type": "command",
#               "command": "~/.claude/hooks/pre-tool-use.sh"
#             }
#           ]
#         }
#       ]
#     }
#   }

set -e

CACHE_DB="${HOME}/.cache/semantic-cache-mcp/cache.db"
CACHE_LOG="${HOME}/.cache/semantic-cache-mcp/hook.log"

# Read hook input from stdin
INPUT=$(cat)

# Extract file path from tool input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .input.file_path // empty' 2>/dev/null)

# If no file path or not a Read tool, allow through
if [ -z "$FILE_PATH" ]; then
    exit 0
fi

# Resolve path
RESOLVED_PATH=$(realpath "$FILE_PATH" 2>/dev/null || echo "$FILE_PATH")

# Check if cache database exists
if [ ! -f "$CACHE_DB" ]; then
    exit 0  # No cache, allow Read
fi

# Get file mtime
if [ ! -f "$RESOLVED_PATH" ]; then
    exit 0  # File doesn't exist, let Read handle the error
fi

CURRENT_MTIME=$(stat -c %Y "$RESOLVED_PATH" 2>/dev/null || stat -f %m "$RESOLVED_PATH" 2>/dev/null)

# Query cache for this file
CACHED=$(sqlite3 "$CACHE_DB" "SELECT mtime, tokens, content_hash FROM files WHERE path = '$RESOLVED_PATH' LIMIT 1" 2>/dev/null || echo "")

if [ -z "$CACHED" ]; then
    # Not in cache, allow Read to proceed (it will be cached by MCP server)
    exit 0
fi

# Parse cached data
CACHED_MTIME=$(echo "$CACHED" | cut -d'|' -f1)
CACHED_TOKENS=$(echo "$CACHED" | cut -d'|' -f2)

# Compare mtimes (cached mtime should be >= current mtime for unchanged file)
if [ "$(echo "$CACHED_MTIME >= $CURRENT_MTIME" | bc -l 2>/dev/null || echo "0")" = "1" ]; then
    # File unchanged - return cache hit message and BLOCK the Read
    echo "[Semantic Cache] File unchanged: $FILE_PATH ($CACHED_TOKENS tokens cached)"
    echo "Use 'read' tool from semantic-cache MCP for full cache benefits including diffs."

    # Log the cache hit
    echo "$(date -Iseconds) CACHE_HIT $RESOLVED_PATH ($CACHED_TOKENS tokens)" >> "$CACHE_LOG" 2>/dev/null || true

    # Exit non-zero to block the Read tool
    # The message above will be shown to Claude as the "result"
    exit 1
fi

# File changed, allow Read to proceed
exit 0
