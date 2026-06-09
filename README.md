<p align="center">
  <img
    src="https://cdn.jsdelivr.net/gh/CoderDayton/semantic-cache-mcp@f8af5804ddc7c3fed62d6901c0c7df098a76164e/assets/logo.svg"
    width="128"
    height="128"
    alt="Semantic Cache MCP Logo"
  />
</p>

<h1 align="center">Semantic Cache MCP</h1>

<p align="center">
  <a href="https://ko-fi.com/U7U01WTJF9">
    <img
      src="https://ko-fi.com/img/githubbutton_sm.svg"
      alt="Support on Ko-fi"
      height="36"
    />
  </a>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/" >
    <img src="https://img.shields.io/badge/Python-3.12%2B-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python 3.12+" />
  </a>
  <a href="https://github.com/modelcontextprotocol/python-sdk">
    <img src="https://img.shields.io/badge/FastMCP-3.2%2B-00A67E?style=for-the-badge" alt="FastMCP 3.2+" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-D4A017?style=for-the-badge" alt="License: MIT" />
  </a>
</p>

---

**Cut your MCP client's token usage by about 98% on cached reads, and get answers back in milliseconds.**

Semantic Cache MCP is a [Model Context Protocol](https://modelcontextprotocol.io) server that puts every file operation behind one cache. The first read of a file seeds the cache and returns a content hash. After that, an unchanged file comes back as a short `unchanged` reply instead of the whole file, a changed file comes back as a unified diff, and a file that is too large is summarized down to its structure. Search and grep run over the same cached files, so the agent searches what it already read instead of going back to disk. Thirteen tools (`read`, `read_image`, `batch_read`, `write`, `edit`, `edit_preview`, `batch_edit`, `search`, `grep`, `glob`, `delete`, `clear`, `stats`) share that one cache-aware layer.

---

## Why this exists

**1. Reads stop costing tokens.** The first read seeds the cache and hands back a `content_hash`. Send that hash back on the next read (as `known_hash`) and the server replies `unchanged` without resending the file. A modified file returns a unified diff with the changed line numbers. A file larger than the budget collapses to a structure-preserving summary instead of a blind cut at a byte offset.

**2. Search and grep run on the cache, not the disk.** Keyword search (BM25), glob, and grep all read from the same indexed corpus that `read`/`batch_read` populate. An in-session result LRU collapses repeated queries to sub-millisecond hits.

**3. Mutations are bounded by default.** `write`, `edit`, and `batch_edit` enforce size and match limits, support `dry_run`, can run formatters, and refresh the cache atomically.

---

## Installation

Add to Claude Code settings (`~/.claude.json`):

**Option 1**: `uvx` (always runs the latest version):

```json
{
  "mcpServers": {
    "semantic-cache": {
      "command": "uvx",
      "args": ["semantic-cache-mcp"]
    }
  }
}
```

**Option 2**: `uv tool install`:

```bash
uv tool install semantic-cache-mcp
```

```json
{
  "mcpServers": {
    "semantic-cache": {
      "command": "semantic-cache-mcp"
    }
  }
}
```

Restart Claude Code.

### Block Native File Tools (Recommended)

Disable the client's built-in file tools so all file I/O routes through semantic-cache.

**Claude Code**: add to `~/.claude/settings.json`:

```json
{
  "permissions": {
    "deny": ["Read", "Edit", "Write"]
  }
}
```

**OpenCode**: add to `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "read": "deny",
    "edit": "deny",
    "write": "deny"
  }
}
```

### CLAUDE.md Configuration

Add to `~/.claude/CLAUDE.md` to enforce semantic-cache globally:

```markdown
## Tools

- MUST use `semantic-cache-mcp` instead of native I/O tools (98% token savings on cached reads)
```
---

## Tools

### Core

| Tool | Description |
|------|-------------|
| `read` | Single-file cache-aware read. Returns full content plus a `content_hash` on the first read, a short `unchanged` reply when you pass back a matching `known_hash`, and a diff when the file changed. Supports `offset`/`limit` for targeted line recovery. |
| `read_image` | Pass-through for image files. Returns an MCP image content block (base64 + mime) so vision models can see the pixels; sidecar metadata holds size and mime. Format verified by magic bytes (PNG, JPEG, GIF, TIFF, BMP, WebP), not by extension. Bypasses the semantic cache. Capped at 5 MiB (`SCMCP_MAX_IMAGE_BYTES`). |
| `delete` | Single-path delete for one file or symlink, with cache eviction and `dry_run=true`. Intentionally does not support globs, recursive delete, or real-directory delete. |
| `write` | Full-file create or replace with cache refresh. Returns creation status or an overwrite diff, supports `append=true`, and can run formatters. |
| `edit` | Single-file exact edit using cached content. Supports scoped and line-range replacement plus `dry_run=true`. For multiple edits to the same file, prefer `batch_edit`. |
| `batch_edit` | Multiple exact edits in one file with partial success reporting. Preferred over repeated `edit` calls on the same file: single response, atomic, faster on large files. |
| `edit_preview` | Read-only probe that returns match count, line numbers, and small context snippets for a candidate `old_string`. Use before a costly `edit` to confirm anchor uniqueness. |

### Discovery

| Tool | Description |
|------|-------------|
| `search` | Cache-only BM25 keyword search that ranks cached files by relevance to a query. Seed likely files first with `batch_read`. |
| `glob` | File discovery plus cache coverage. Use it to find candidates, then pass those paths into `batch_read`. |
| `batch_read` | Multi-file cache-aware read for seeding and retrieval. Handles globs, priorities, token budgets, unchanged suppression, and diff/full routing. |
| `grep` | Cache-only exact search with regex or literal matching, line numbers, and optional context. Best for symbols and exact strings. |

### Management

| Tool | Description |
|------|-------------|
| `stats` | Cache metrics, session usage (tokens saved, tool calls), and lifetime aggregates. |
| `clear` | Reset all cache entries. |

---

## Tool Reference

The table above is the authoritative tool map. This section only shows the common call shapes.

<details>
<summary><strong>read</strong>: single file, automatic caching</summary>

```
read path="/src/app.py"                        # automatic: full, unchanged, or diff
read path="/src/app.py" offset=120 limit=80    # lines 120 to 199 only
```

**Three states, picked for you:**

| State | Response | Token cost |
|-------|----------|------------|
| First read | Full content plus a `content_hash` | Normal |
| Unchanged | `unchanged: true`, returned when you pass back a matching `known_hash` | A few tokens |
| Modified | Unified diff only | 5 to 20% of original |

</details>

<details>
<summary><strong>write</strong>: create or overwrite files</summary>

```
write path="/src/new.py" content="..."
write path="/src/new.py" content="..." auto_format=true
write path="/src/large.py" content="...chunk1..." append=false   # first chunk
write path="/src/large.py" content="...chunk2..." append=true    # subsequent chunks
```

</details>

<details>
<summary><strong>edit</strong>: find/replace with three modes</summary>

```
# Mode A: find/replace, searches the entire file
edit path="/src/app.py" old_string="def foo():" new_string="def foo(x: int):"
edit path="/src/app.py" old_string="..." new_string="..." replace_all=true auto_format=true

# Mode B: scoped find/replace, searches only within the line range (a shorter old_string works)
edit path="/src/app.py" old_string="pass" new_string="return x" start_line=42 end_line=42

# Mode C: line replace, swaps the whole range with no old_string needed (most token savings)
edit path="/src/app.py" new_string="    return result\n" start_line=80 end_line=83
```

**Mode selection:**

| Mode | Parameters | Best for |
|------|-----------|----------|
| Find/replace | `old_string` + `new_string` | Unique strings, no line numbers known |
| Scoped | `old_string` + `new_string` + `start_line`/`end_line` | Shorter context when `read` gave you line numbers |
| Line replace | `new_string` + `start_line`/`end_line` (no `old_string`) | Maximum token savings when line numbers are known |

</details>

<details>
<summary><strong>batch_edit</strong>: multiple edits in one call</summary>

```
# Mode A: find/replace, [old, new]
batch_edit path="/src/app.py" edits='[["old1","new1"],["old2","new2"]]'

# Mode B: scoped, [old, new, start_line, end_line]
batch_edit path="/src/app.py" edits='[["pass","return x",42,42]]'

# Mode C: line replace, [null, new, start_line, end_line]
batch_edit path="/src/app.py" edits='[[null,"    return result\n",80,83]]'

# Mixed modes in one call (object syntax also supported)
batch_edit path="/src/app.py" edits='[
  ["old1", "new1"],
  {"old": "pass", "new": "return x", "start_line": 42, "end_line": 42},
  {"old": null, "new": "    return result\n", "start_line": 80, "end_line": 83}
]' auto_format=true
```

</details>

<details>
<summary><strong>batch_read</strong>: multiple files with a token budget</summary>

```
batch_read paths="/src/a.py,/src/b.py" max_total_tokens=50000
batch_read paths='["/src/a.py","/src/b.py"]' priority="/src/main.py"
batch_read paths="/src/*.py" max_total_tokens=30000
```

- Expands simple globs, honors `priority`, enforces `max_total_tokens`, and reports skipped paths with recovery hints.
- Unchanged files are collapsed into the summary instead of repeating content.

</details>

<details>
<summary><strong>discovery</strong>: search, glob, grep</summary>

```
search query="authentication middleware logic" k=5
glob pattern="**/*.py" directory="./src" cached_only=true
grep pattern="class Cache" path="src/**/*.py"
```

</details>

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `TOOL_OUTPUT_MODE` | `compact` | Response detail (`compact`, `normal`, `debug`) |
| `TOOL_MAX_RESPONSE_TOKENS` | `0` | Global response token cap (`0` = disabled) |
| `TOOL_TIMEOUT` | `30` | Seconds before tool call times out (auto-resets executor) |
| `MAX_CONTENT_SIZE` | `100000` | Max bytes returned by read operations |
| `MAX_CACHE_ENTRIES` | `10000` | Max cache entries before W-TinyLFU eviction |
| `SEMANTIC_CACHE_DIR` | *(platform)* | Override cache/database directory path |

See [docs/env_variables.md](docs/env_variables.md) for detailed descriptions, model selection guidance, and examples.

### Safety Limits

| Limit | Value | Protects Against |
|-------|-------|-----------------|
| `MAX_WRITE_SIZE` | 10 MB | Memory exhaustion via large writes |
| `MAX_EDIT_SIZE` | 10 MB | Memory exhaustion via large file edits |
| `MAX_MATCHES` | 10,000 | CPU exhaustion via unbounded `replace_all` |

### MCP Server Config

```json
{
  "mcpServers": {
    "semantic-cache": {
      "command": "uvx",
      "args": ["semantic-cache-mcp"],
      "env": {
        "LOG_LEVEL": "INFO",
        "TOOL_OUTPUT_MODE": "compact",
        "MAX_CONTENT_SIZE": "100000"
      }
    }
  }
}
```

**Cache location:** `~/.cache/semantic-cache-mcp/` (Linux), `~/Library/Caches/semantic-cache-mcp/` (macOS), `%LOCALAPPDATA%\semantic-cache-mcp\` (Windows). Override with `SEMANTIC_CACHE_DIR`.

---

## How It Works

```
┌──────────┐     ┌────────────┐     ┌──────────────────────────┐
│  Claude  │────▶│ smart_read │────▶│ stat() + cache lookup    │
│   Code   │     │            │     │ (BEFORE any disk read)   │
└──────────┘     └────────────┘     └──────────────────────────┘
                        │
       ┌────────────────┼─────────────────┬──────────────────┐
       ▼                ▼                 ▼                  ▼
 ┌──────────┐    ┌──────────┐      ┌──────────┐      ┌────────────┐
 │ mtime    │    │ mtime    │      │ Changed  │      │ New /      │
 │ match    │    │ drift,   │      │ content  │      │ Large      │
 │ FAST     │    │ hash     │      │ → diff   │      │ → summary  │
 │ PATH     │    │ match    │      │ (80-95%) │      │  or full   │
 │ ~5 tok   │    │ ~5 tok   │      └──────────┘      └────────────┘
 │ (99%)    │    │ (99%)    │
 │ ~1 ms    │    │ ~1 ms    │
 │ no I/O   │    │ +update  │
 └──────────┘    └──────────┘
```

Every read also returns a `content_hash`. Hand it back as `known_hash` on your
next read and the server answers `unchanged` from that fact alone, with no guess
about what it already sent you.

`search` works the same way. An in-session LRU keyed on `(query, k, directory)`
returns warm hits in ~10 µs; misses fall through to BM25 keyword search. Every
cache mutation (`put`, `clear`, `delete_path`, `update_mtime`) bumps the LRU, so
callers never see a result that predates a write.

---

## Performance

Measured on this project's 40 source files (**177,509 tokens**), i9-13900K, with the corpus held fixed across all phases. Reproducible via `--json` output for CI diffing.

### Token savings: **98.9%** overall (phases 2 to 6)

| Phase | Scenario | Savings |
|-------|----------|--------:|
| **Overall (cached, phases 2 to 6)** | **Aggregate token reduction** | **98.9%** |
| Unchanged re-read | mtime match, fast path skips disk I/O | 99.1% |
| Content hash | mtime drifted, BLAKE3 still matches | 99.1% |
| Batch read | All files via `batch_read`, 200K budget | 99.1% |
| Search previews | 5 queries × k=5, previews vs full reads | 99.7% |
| Small edits | Real ~5% line changes in 30% of files | 97.8% |
| Cold read | First read, no cache (baseline) | 0% |

### Latency: **unchanged reads ~0.9 ms; repeat searches < 0.01 ms**

| Operation | p50 | Notes |
|-----------|----:|-------|
| Single unchanged read (fast path) | **0.9 ms** | mtime + cache hit; no disk I/O |
| Single diff read (changed file) | 0.7 ms | hash check + unified diff |
| Search k=5 (cache **hit**) | **< 0.01 ms** | in-session LRU; hundreds× vs cold |
| Search k=5 (cache **miss**) | 1.5 ms | BM25 keyword search |
| Edit (scoped find/replace) | 2.4 ms | uses cached content |
| Grep (literal `def `) | 1.3 ms | FTS5 over cached corpus |
| Grep (regex) | 3.7 ms | regex compiled once |
| Batch read (40 files, diff mode) | 26.0 ms | chunk + tokenize new/changed files |
| Unchanged re-read (40 files) | 18 ms | whole-corpus pass |
| Cold read (40 files, total) | 125 ms | no embedding model, pure disk I/O plus tokenisation |
| Write (200-line file) | 1.8 ms | creates + caches (no embed) |

Run benchmarks yourself:

```bash
uv run python benchmarks/benchmark_token_savings.py    # token savings
uv run python benchmarks/benchmark_performance.py      # operation latency
```

See [docs/performance.md](docs/performance.md) for full benchmarks and methodology.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Architecture](docs/architecture.md) | Component design, algorithms, data flow |
| [Performance](docs/performance.md) | Optimization techniques, benchmarks |
| [Security](docs/security.md) | Threat model, input validation, size limits |
| [Advanced Usage](docs/advanced-usage.md) | Programmatic API, custom storage backends |
| [Troubleshooting](docs/troubleshooting.md) | Common issues, debug logging |
| [Environment Variables](docs/env_variables.md) | All configurable env vars with defaults and examples |

---

## Contributing

```bash
git clone https://github.com/CoderDayton/semantic-cache-mcp.git
cd semantic-cache-mcp
uv sync
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for commit conventions, pre-commit hooks, and code standards.

---

## License

[MIT License](LICENSE). Use it freely in personal and commercial projects.

---

## Credits

Built with [FastMCP 3.2+](https://github.com/jlowin/fastmcp) and:

- SQLite with FTS5 for keyword (BM25) full-text search, vendored as a small built-in store
- Semantic summarization based on TCRA-LLM ([arXiv:2310.15556](https://arxiv.org/abs/2310.15556))
- BLAKE3 cryptographic hashing for content freshness
- W-TinyLFU frequency-aware cache eviction
