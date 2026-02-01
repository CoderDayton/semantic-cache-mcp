# Semantic Cache MCP

Lightweight MCP server for semantic file caching with 80%+ token reduction.

## Features

- **Diff-based updates**: Returns only changes for modified files
- **Smart truncation**: Preserves file structure when truncating large files
- **Brotli compression**: Efficient storage with 4-10x compression
- **Minimal dependencies**: Just `mcp`, `httpx`, and `brotli`

## Installation

```bash
uv tool install .
```

## Usage

Add to Claude Code settings:

```json
{
  "mcpServers": {
    "semantic-cache": {
      "command": "semantic-cache-mcp"
    }
  }
}
```

Then use `smart_read` instead of `Read` for cached file access.

## Tools

- `smart_read` - Read files with caching and diffs
- `cache_stats` - View cache statistics
- `cache_clear` - Clear the cache
