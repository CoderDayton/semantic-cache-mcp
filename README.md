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

**Cut your MCP client's token usage by ~98% on cached reads, with millisecond responses.**

Semantic Cache MCP is a [Model Context Protocol](https://modelcontextprotocol.io) server that puts every file operation behind one cache. Re-reading a file you already hold costs a few tokens instead of the whole file, and search and grep run over that same corpus rather than the disk.

Thirteen tools share the layer: `read`, `read_image`, `batch_read`, `write`, `edit`, `edit_preview`, `batch_edit`, `search`, `grep`, `glob`, `delete`, `clear`, `stats`.

---

## Why this exists

**Reads stop costing tokens.** The first read hands back a `content_hash`. Send it back — `known_hash` on `read`, a `known_hashes` entry on `batch_read` — and the server replies `unchanged` without resending. A modified file returns a diff with changed line numbers; an oversized one collapses to a structure-preserving summary rather than a blind cut at a byte offset.

That echoed hash is the whole contract, and it is the only evidence the server has that a file is still in your context. A warm cache proves the *server* holds the file, never that you do — the store is on disk and outlives the process, the session, and your context window. A read without a matching hash always sends the file, so forgetting is safe: after a compaction, omit the hashes and get your files back in full.

**Search and grep run on the cache, not the disk.** BM25 keyword search, glob, and grep all read the corpus that `read` and `batch_read` populate. An in-session result LRU collapses repeated queries to sub-millisecond hits.

**Mutations are bounded by default.** `write`, `edit`, and `batch_edit` enforce size and match limits, can run formatters, and refresh the cache atomically. A `dry_run` writes nothing and says so — the status becomes `would_create` / `would_update` / `would_edit` — so a preview is never mistaken for a completed write.

---

## Installation

Add to Claude Code settings (`~/.claude.json`).

**Option 1**: `uvx`, always runs the latest version:

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

**Claude Code** — `~/.claude/settings.json`:

```json
{
  "permissions": {
    "deny": ["Read", "Edit", "Write"]
  }
}
```

**OpenCode** — `~/.config/opencode/opencode.json`:

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
| `read` | Cache-aware single-file read: full content plus a `content_hash` on the first read, `unchanged` for a matching `known_hash`, a diff for a changed file. `offset`/`limit` recover exact line ranges. A partial or summarized read reports `file_hash` (prefixed `partial:`) — it identifies the file but is never proof you hold it. A ranged read also returns a signed `coverage_token` for the lines delivered: echo it back and a window you hold answers `unchanged`; windows covering the whole file mint a claimable `content_hash`. |
| `read_image` | Image pass-through. Returns an MCP image content block (base64 + mime) so vision models see the pixels; sidecar metadata carries size and mime. Format verified by magic bytes (PNG, JPEG, GIF, TIFF, BMP, WebP), not extension. Bypasses the cache. Capped at 5 MiB (`SCMCP_MAX_IMAGE_BYTES`). |
| `write` | Full-file create or replace with cache refresh. Returns creation status or an overwrite diff; supports `append=true` and formatters. A full write hands back a claimable `content_hash`; an append needs `known_hash` to earn one. |
| `edit` | Exact edit against cached content, with scoped and line-range modes plus `dry_run=true`. Pass `known_hash` to get a claimable `content_hash` back and skip the read afterwards. For several edits to one file, use `batch_edit`. |
| `batch_edit` | Many exact edits to one file, applied atomically, with per-edit success reporting. Takes `known_hash` on the same terms as `edit`. An ambiguous anchor, an anchor inside another edit's line range, and two overlapping ranges are each rejected rather than silently resolved; every reported success is verified against the text it produced. |
| `edit_preview` | Read-only probe returning match count, line numbers, and context snippets for a candidate `old_string`. Confirms anchor uniqueness before a costly `edit`. |
| `delete` | Single-path delete for a file or symlink, with cache eviction and `dry_run=true`. No globs, no recursion, no directory delete. |

### Discovery

| Tool | Description |
|------|-------------|
| `batch_read` | Multi-file cache-aware read. Handles globs, priorities, token budgets, and diff/full routing. Returns each file's `content_hash`; pass them back as `known_hashes` to suppress the ones you still hold. |
| `search` | Cache-only BM25 ranking of cached files. Terms join with `OR`, so a word your corpus lacks narrows the ranking instead of emptying the results. Seed likely files with `batch_read` first. |
| `grep` | Cache-only exact search — regex or literal, with line numbers and optional context. Best for symbols and exact strings. An invalid, over-long, or catastrophically backtracking pattern is an error, never an empty result; use `fixed_string=true` for literal text. Responses state whether the scan completed, so a capped result is never read as a total. |
| `glob` | File discovery plus cache coverage. Find candidates, then pass the paths to `batch_read`. |

### Management

| Tool | Description |
|------|-------------|
| `stats` | Cache metrics, session usage (tokens saved, tool calls), and lifetime aggregates. |
| `clear` | Reset all cache entries. |

---

## Tool Reference

The table above is the authoritative map; these are the common call shapes.

<details>
<summary><strong>read</strong>: single file, automatic caching</summary>

```
read path="/src/app.py"                        # automatic: full, unchanged, or diff
read path="/src/app.py" offset=120 limit=80    # lines 120 to 199 only
```

| State | Response | Token cost |
|-------|----------|------------|
| First read | Full content plus a `content_hash` | Normal |
| Unchanged | `unchanged: true`, when you pass back a matching `known_hash` | A few tokens |
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

| Mode | Parameters | Best for |
|------|-----------|----------|
| Find/replace | `old_string` + `new_string` | Unique strings, no line numbers known |
| Scoped | `old_string` + `new_string` + `start_line`/`end_line` | Shorter context when `read` gave you line numbers |
| Line replace | `new_string` + `start_line`/`end_line` | Maximum token savings when line numbers are known |

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
batch_read paths="/src/a.py,/src/b.py" known_hashes='{"/src/a.py":"8f3c..."}'
```

Expands simple globs, honors `priority`, enforces `max_total_tokens`, and reports skipped paths with recovery hints. Every file is returned in full unless you prove you still hold it: echo the delivered `content_hash` values back as `known_hashes` and the ones you hold collapse into an `unchanged` count.

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
| `TOOL_TIMEOUT` | `30` | Seconds before a tool call times out (auto-resets executor) |
| `MAX_CONTENT_SIZE` | `100000` | Max bytes returned by read operations |
| `MAX_CACHE_ENTRIES` | `10000` | Max cache entries before W-TinyLFU eviction |
| `SEMANTIC_CACHE_DIR` | *(platform)* | Override cache/database directory path |

A malformed value falls back to the default and logs a warning naming the variable. See [docs/env_variables.md](docs/env_variables.md) for detail.

### Safety Limits

| Limit | Value | Protects against |
|-------|-------|-----------------|
| `MAX_WRITE_SIZE` | 10 MB | Memory exhaustion via large writes |
| `MAX_EDIT_SIZE` | 10 MB | Memory exhaustion via large file edits, in `edit` and `batch_edit` alike |
| `MAX_MATCHES` | 10,000 | CPU exhaustion via unbounded `replace_all` |
| `GREP_MAX_PATTERN_LEN` | 1,000 chars | Oversized `grep` regex source |
| Regex shape check | — | Catastrophic backtracking ([details](docs/security.md#regular-expression-safety)) |

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

`search` is cached on the same principle. An in-session LRU keyed on `(query, k, directory)` returns warm hits in ~10 µs, and misses fall through to BM25. Every cache mutation (`put`, `clear`, `delete_path`, `update_mtime`) bumps the LRU, so callers never see a result that predates a write.

---

## Performance

Measured on this project's 41 source files (**212,499 tokens**), i9-13900K, ext4 on NVMe, corpus held fixed across phases. Every phase models a caller that keeps its hashes and echoes them back — that is what earns the savings.

### Token savings: **98.9%** overall (phases 2 to 6)

| Phase | Scenario | Savings |
|-------|----------|--------:|
| **Overall (cached, phases 2 to 6)** | **Aggregate token reduction** | **98.9%** |
| Unchanged re-read | mtime match, fast path skips disk I/O | 99.3% |
| Content hash | mtime drifted, BLAKE3 still matches | 99.3% |
| Batch read | All files via `batch_read`, 200K budget | 99.3% |
| Search previews | 5 queries × k=5, previews vs full reads | 98.6% |
| Small edits | Real ~5% line changes in 30% of files | 98.1% |
| Cold read | First read, no cache; one file exceeds `MAX_CONTENT_SIZE` and returns summarised, which is not a cache saving | 5.9% |

### Latency: **unchanged reads ~1 ms; repeat searches < 0.01 ms**

| Operation | p50 | Notes |
|-----------|----:|-------|
| Single unchanged read (fast path) | **1.1 ms** | mtime + cache hit, no disk I/O |
| Single diff read (changed file) | 0.7 ms | hash check + unified diff |
| Search k=5 (cache **hit**) | **< 0.01 ms** | in-session LRU |
| Search k=5 (cache **miss**) | 1.4 ms | BM25 keyword search |
| Edit (scoped find/replace) | 3.1 ms | cached content, plus the atomic write's fsync |
| Grep (literal `def `) | 1.5 ms | FTS5 over cached corpus |
| Grep (regex) | 3.4 ms | compiled once |
| Batch read (41 files, diff mode) | 45.6 ms | chunk + tokenize changed files; one summarises each full pass |
| Unchanged re-read (41 files) | 19.5 ms | whole-corpus pass |
| Cold read (41 files, total) | 100 ms | single unrepeated pass: I/O, tokenisation, one summarisation |
| Write (200-line file) | 2.7 ms | creates + caches, durable before it returns |

Run them yourself. Pin `TMPDIR` to a real disk — the default `/tmp` is usually tmpfs, which discards `fsync` and reports write latency ~40% low:

```bash
TMPDIR="$HOME/.cache/scmcp-bench" \
  uv run python benchmarks/benchmark_performance.py    # operation latency
uv run python benchmarks/benchmark_token_savings.py    # token savings
```

See [docs/performance.md](docs/performance.md) for full methodology.

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Architecture](docs/architecture.md) | Component design, algorithms, data flow |
| [Performance](docs/performance.md) | Benchmarks, methodology, cache footprint |
| [Security](docs/security.md) | Threat model, input validation, size limits |
| [Advanced Usage](docs/advanced-usage.md) | Programmatic API, custom storage backends |
| [Troubleshooting](docs/troubleshooting.md) | Common issues, debug logging |
| [Environment Variables](docs/env_variables.md) | All env vars with defaults and examples |

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
