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
    <img src="https://img.shields.io/badge/FastMCP-3.0-00A67E?style=for-the-badge" alt="FastMCP 3.0" />
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-D4A017?style=for-the-badge" alt="License: MIT" />
  </a>
</p>

---

**Reduce Claude Code token usage by 80%+ with intelligent file caching.**

Semantic Cache MCP is a [Model Context Protocol](https://modelcontextprotocol.io) server that eliminates redundant token consumption when Claude reads files. Instead of sending full file contents on every request, it returns diffs for changed files, suppresses unchanged files entirely, and intelligently summarizes large files — all transparently through 13 purpose-built MCP tools.

---

## Features

- **80%+ Token Reduction** — Unchanged files cost ~0 tokens; changed files return diffs only
- **Automatic Three-State Reads** — First read (full + cache), unchanged (`"unchanged":true`, 99% savings), modified (diff, 80–95% savings) — fully automatic, no configuration
- **Semantic Search** — Hybrid BM25 + HNSW vector search via local ONNX embeddings (configurable model, default BAAI/bge-small-en-v1.5), no API keys, works offline
- **Batch Embedding** — `batch_smart_read` pre-scans all new/changed files and embeds them in a single model call (N calls → 1)
- **Content Hash Freshness** — BLAKE3 hash detects when mtime changes but content is identical (touch, git checkout) — returns cached instead of re-reading
- **Grep** — Regex/literal pattern search across cached files with line numbers and context
- **Semantic Summarization** — 50–80% token savings on large files, structure preserved
- **DoS Protection** — Write size, edit size, and match count limits enforced at every boundary

---

## Installation

Add to Claude Code settings (`~/.claude/settings.json`):

**Option 1** — `uvx` (always runs latest version):

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

**Option 2** — `uv tool install`:

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

### GPU Acceleration (Optional)

For NVIDIA GPU acceleration, install with the `gpu` extra:

```bash
uv tool install "semantic-cache-mcp[gpu]"
# or with uvx: uvx "semantic-cache-mcp[gpu]"
```

Then set `EMBEDDING_DEVICE=gpu` in your MCP config env block. Falls back to CPU automatically if CUDA is unavailable.

### Custom Embedding Models

Any HuggingFace model with an ONNX export works — set `EMBEDDING_MODEL` in your env config:

```json
"env": {
  "EMBEDDING_MODEL": "Snowflake/snowflake-arctic-embed-m-v2.0"
}
```

If the model isn't in fastembed's built-in list, it's automatically downloaded and registered from HuggingFace Hub on first startup (ONNX file integrity is verified via SHA256). See [env_variables.md](docs/env_variables.md) for model recommendations.

### Block Native File Tools (Recommended)

Disable the client's built-in file tools so all file I/O routes through semantic-cache.

**Claude Code** — add to `~/.claude/settings.json`:

```json
{
  "permissions": {
    "deny": ["Read", "Edit", "Write"]
  }
}
```

**OpenCode** — add to `~/.config/opencode/opencode.json`:

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

- MUST use `semantic-cache-mcp` instead of native I/O tools (80%+ token savings)
```
---

## Tools

### Core

| Tool | Description |
|------|-------------|
| `read` | Single-file cache-aware read. Returns full content on first read, unchanged markers on cache hits, diffs on modifications, and supports `offset`/`limit` for targeted recovery. |
| `delete` | Single-path delete for one file or symlink, with cache eviction and `dry_run=true`. Intentionally does not support globs, recursive delete, or real-directory delete. |
| `write` | Full-file create or replace with cache refresh. Returns creation status or an overwrite diff, supports `append=true`, and can run formatters. |
| `edit` | Single-file exact edit using cached content. Best for one localized change; supports scoped and line-range replacement plus `dry_run=true`. |
| `batch_edit` | Multiple exact edits in one file with partial success reporting. Best when several localized changes belong in the same file. |

### Discovery

| Tool | Description |
|------|-------------|
| `search` | Cache-only semantic search for meaning or mixed keyword intent. Seed likely files first with `batch_read`; use `grep` for exact text. |
| `similar` | Cache-only nearest-neighbor lookup for one source file. Best after seeding a directory with `batch_read`. |
| `glob` | File discovery plus cache coverage. Use it to find candidates, then pass those paths into `batch_read`. |
| `batch_read` | Multi-file cache-aware read for seeding and retrieval. Handles globs, priorities, token budgets, unchanged suppression, and diff/full routing. |
| `grep` | Cache-only exact search with regex or literal matching, line numbers, and optional context. Best for symbols and exact strings. |
| `diff` | Explicit side-by-side file comparison with unified diff and semantic similarity. Use `read` instead for “what changed since last read?”. |

### Management

| Tool | Description |
|------|-------------|
| `stats` | Cache metrics, session usage (tokens saved, tool calls), and lifetime aggregates. |
| `clear` | Reset all cache entries. |

---

## Tool Reference

<details>
<summary><strong>read</strong> — Single file, automatic caching</summary>

```
read path="/src/app.py"                        # automatic: full, unchanged, or diff
read path="/src/app.py" offset=120 limit=80    # lines 120–199 only
```

**Automatic three states:**

| State | Response | Token cost |
|-------|----------|------------|
| First read | Full content + cached | Normal |
| Unchanged | `"File unchanged (1,234 tokens cached)"` | ~5 tokens |
| Modified | Unified diff only | 5–20% of original |

</details>

<details>
<summary><strong>write</strong> — Create or overwrite files</summary>

```
write path="/src/new.py" content="..."
write path="/src/new.py" content="..." auto_format=true
write path="/src/large.py" content="...chunk1..." append=false   # first chunk
write path="/src/large.py" content="...chunk2..." append=true    # subsequent chunks
```

</details>

<details>
<summary><strong>edit</strong> — Find/replace with three modes</summary>

```
# Mode A — find/replace: searches entire file
edit path="/src/app.py" old_string="def foo():" new_string="def foo(x: int):"
edit path="/src/app.py" old_string="..." new_string="..." replace_all=true auto_format=true

# Mode B — scoped find/replace: search only within line range (shorter old_string suffices)
edit path="/src/app.py" old_string="pass" new_string="return x" start_line=42 end_line=42

# Mode C — line replace: replace entire range, no old_string needed (maximum token savings)
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
<summary><strong>batch_edit</strong> — Multiple edits in one call</summary>

```
# Mode A — find/replace: [old, new]
batch_edit path="/src/app.py" edits='[["old1","new1"],["old2","new2"]]'

# Mode B — scoped: [old, new, start_line, end_line]
batch_edit path="/src/app.py" edits='[["pass","return x",42,42]]'

# Mode C — line replace: [null, new, start_line, end_line]
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
<summary><strong>search</strong> — Semantic search across cached files</summary>

```
search query="authentication middleware logic" k=5
search query="database connection pooling" k=3
```

</details>

<details>
<summary><strong>similar</strong> — Find semantically related files</summary>

```
similar path="/src/auth.py" k=3
similar path="/tests/test_auth.py" k=5
```

</details>

<details>
<summary><strong>glob</strong> — Pattern matching with cache awareness</summary>

```
glob pattern="**/*.py" directory="./src"
glob pattern="**/*.py" directory="./src" cached_only=true
```

</details>

<details>
<summary><strong>batch_read</strong> — Multiple files with token budget</summary>

```
batch_read paths="/src/a.py,/src/b.py" max_total_tokens=50000
batch_read paths='["/src/a.py","/src/b.py"]' priority="/src/main.py"
batch_read paths="/src/*.py" max_total_tokens=30000
```

- **Glob expansion**: `src/*.py` expanded inline (max 50 files per glob)
- **Priority ordering**: `priority` paths read first, remainder sorted smallest-first
- **Token budget**: stops reading new files once `max_total_tokens` reached; skipped files include `est_tokens` hint
- **Unchanged suppression**: unchanged files appear in `summary.unchanged` with no content (zero tokens)
- **Batch embedding**: pre-scans all new/changed files and embeds them in a single model call before reading — N model calls reduced to 1
- **Recovery**: use `read` with `offset`/`limit` for targeted line-range recovery after truncation or context loss

</details>

<details>
<summary><strong>diff</strong> — Compare two files</summary>

```
diff path1="/src/v1.py" path2="/src/v2.py"
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
| `MAX_CACHE_ENTRIES` | `10000` | Max cache entries before LRU-K eviction |
| `EMBEDDING_DEVICE` | `cpu` | Embedding hardware: `cpu`, `cuda` (GPU), `auto` (detect) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model for search/similarity ([options](https://qdrant.github.io/fastembed/examples/Supported_Models/)) |
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
        "MAX_CONTENT_SIZE": "100000",
        "EMBEDDING_DEVICE": "cpu",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5"
      }
    }
  }
}
```

**Cache location:** `~/.cache/semantic-cache-mcp/` (Linux), `~/Library/Caches/semantic-cache-mcp/` (macOS), `%LOCALAPPDATA%\semantic-cache-mcp\` (Windows). Override with `SEMANTIC_CACHE_DIR`.

---

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Claude     │────▶│  smart_read  │────▶│  Cache Lookup    │
│  Code       │     │              │     │  (VectorStorage) │
└─────────────┘     └──────────────┘     └──────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐     ┌──────────────┐
   │Unchanged │     │ Changed  │     │  New / Large │
   │  ~0 tok  │     │  diff    │     │ summarize or │
   │  (99%)   │     │ (80-95%) │     │ full content │
   └──────────┘     └──────────┘     └──────────────┘
```

---

## Performance

Measured on this project's 30 source files (~136K tokens). Benchmarks run on a standard dev machine (CPU embeddings).

### Token Savings

| Phase | Scenario | Savings |
|-------|----------|--------:|
| Cold read | First read, no cache | 0% (baseline) |
| Unchanged re-read | Same files, no modifications | **99.1%** |
| Content hash | Touch files (mtime changed, content identical) | **99.1%** |
| Small edits | ~5% of lines changed in 30% of files | **98.1%** |
| Batch read | All files via `batch_read` | **99.1%** |
| Search | 5 queries × k=5, previews vs full reads | **98.4%** |
| **Overall (cached)** | **Phases 2–6 combined** | **98.8%** |

### Operation Latency

| Operation | Time |
|-----------|-----:|
| Unchanged read (single file) | 2 ms |
| Unchanged re-read (29 files) | 25 ms |
| Batch read (29 files, diff mode) | 35 ms |
| Cold read (29 files, incl. embed) | 2,554 ms |
| Write (200-line file) | 47 ms |
| Edit (scoped find/replace) | 48 ms |
| Semantic search (k=5) | 4 ms |
| Semantic search (k=10) | 5 ms |
| Find similar (k=3) | 49 ms |
| Grep (literal) | 1 ms |
| Grep (regex) | 2 ms |
| Embedding model warmup | 206 ms |
| Single embedding (largest file) | 47 ms |
| Batch embedding (10 files) | 469 ms |

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

[MIT License](LICENSE) — use freely in personal and commercial projects.

---

## Credits

Built with [FastMCP 3.0](https://github.com/modelcontextprotocol/python-sdk) and:

- [FastEmbed](https://github.com/qdrant/fastembed) — local ONNX embeddings (configurable, default BAAI/bge-small-en-v1.5)
- [SimpleVecDB](https://github.com/CoderDayton/SimpleVecDB) — HNSW vector storage with FTS5 keyword search
- Semantic summarization based on TCRA-LLM ([arXiv:2310.15556](https://arxiv.org/abs/2310.15556))
- BLAKE3 cryptographic hashing for content freshness
- LRU-K frequency-aware cache eviction
