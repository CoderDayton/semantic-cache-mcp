<p align="center">
  <img src="assets/logo.svg" width="128" height="128" alt="Semantic Cache MCP Logo">
</p>

<h1 align="center">Semantic Cache MCP</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"></a>
  <a href="https://github.com/modelcontextprotocol/python-sdk"><img src="https://img.shields.io/badge/FastMCP-3.0-green.svg" alt="FastMCP 3.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

**Reduce Claude Code token usage by 80%+ with intelligent file caching.**

Semantic Cache MCP is a [Model Context Protocol](https://modelcontextprotocol.io) server that eliminates redundant token consumption when Claude reads files. Instead of sending full file contents on every request, it returns diffs for changed files, suppresses unchanged files entirely, and intelligently summarizes large files — all transparently through 11 purpose-built MCP tools.

---

## Features

- **80%+ Token Reduction** — Unchanged files cost ~0 tokens; changed files return diffs only
- **Three-State Read Model** — First read (full + cache), unchanged (message only, 99% savings), modified (diff, 80–95% savings)
- **Semantic Search** — Local embeddings via FastEmbed, no API keys, works offline
- **LSH Acceleration** — O(1) candidate retrieval for caches ≥ 100 files (vs O(N) linear scan)
- **int8 Quantization** — 22x smaller embeddings (772 bytes vs 17 KB/vector)
- **SIMD-Parallel Chunking** — 5–7x faster content-defined deduplication (~70–95 MB/s)
- **Adaptive Compression** — ZSTD primary (6.9 GB/s for text), LZ4 and Brotli fallbacks
- **Content-Addressable Storage** — BLAKE3-hashed chunks, 3.8x faster than BLAKE2b
- **Semantic Summarization** — 50–80% token savings on large files, structure preserved
- **DoS Protection** — Write size, edit size, and match count limits enforced at every boundary

---

## Installation

```bash
uv tool install git+https://github.com/CoderDayton/semantic-cache-mcp.git
```

Add to Claude Code settings (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "semantic-cache": {
      "command": "semantic-cache-mcp"
    }
  }
}
```

Restart Claude Code. Done.

### Block Native File Tools (Recommended)

Force Claude to use semantic-cache for all file I/O:

```json
{
  "permissions": {
    "deny": ["Read", "Write", "Edit"]
  }
}
```

### CLAUDE.md Configuration

Add to `~/.claude/CLAUDE.md` to enforce semantic-cache globally:

```markdown
## Tools

- semantic-cache: MUST use instead of native file tools (80%+ token savings)
  - `read` → single-file; diff_mode=true by default (set false after context compression)
    - `offset`/`limit` → read specific line ranges
  - `batch_read` → 2+ files; supports glob patterns; set diff_mode=false after context compression
  - `write` → new files or full rewrites; append=true for chunked writes of large files
  - `edit` → targeted find/replace; returns diff
  - `batch_edit` → 2+ edits in one file
  - `search`/`similar` → semantic search; seed cache first with read/batch_read
  - `glob` → find files by pattern; cached_only=true to see what's already cached
```

---

## Tools

### Core

| Tool | Description |
|------|-------------|
| `read` | Smart file reading with diff-mode. Three states: first read (full + cache), unchanged (99% savings), modified (diff, 80–95% savings). Use `offset`/`limit` for line ranges. |
| `write` | Write files with cache integration. `auto_format=true` runs formatter. `append=true` enables chunked writes for large files. Returns diff on overwrite. |
| `edit` | Targeted find/replace using cached reads. `dry_run=true` previews changes. `replace_all=true` handles multiple matches. `auto_format=true` runs formatter. Returns unified diff. |
| `batch_edit` | Up to 50 edits per call with partial success — some can fail while others succeed. `auto_format=true` and `dry_run=true` supported. |

### Discovery

| Tool | Description |
|------|-------------|
| `search` | Semantic/embedding search across cached files by meaning — not keywords. Seed cache first with `read` or `batch_read`. |
| `similar` | Finds semantically similar cached files to a given path. Start with `k=3–5`. Only searches cached files. |
| `glob` | Pattern matching with cache status per file. `cached_only=true` filters to already-cached files. Max 1000 matches, 5s timeout. |
| `batch_read` | Read 2+ files in one call. Supports glob expansion in paths, priority ordering, token budget, and per-file diff suppression for unchanged files. Set `diff_mode=false` after context compression. |
| `diff` | Compare two files. Returns unified diff plus semantic similarity score. Large diffs are auto-summarized to stay within token budget. |

### Management

| Tool | Description |
|------|-------------|
| `stats` | Cache metrics: file count, token counts, hit rate, compression ratio. |
| `clear` | Reset all cache entries. |

---

## Tool Reference

<details>
<summary><strong>read</strong> — Single file with diff-mode</summary>

```
read path="/src/app.py"
read path="/src/app.py" diff_mode=true         # default
read path="/src/app.py" diff_mode=false        # full content (use after context compression)
read path="/src/app.py" offset=120 limit=80    # lines 120–199 only
```

**Three states:**

| State | Response | Token cost |
|-------|----------|------------|
| First read | Full content + cached | Normal |
| Unchanged | `"File unchanged (1,234 tokens cached)"` | ~5 tokens |
| Modified | Unified diff only | 5–20% of original |

Set `diff_mode=false` after context compression — Claude has lost its cached copy and needs full content.

</details>

<details>
<summary><strong>write</strong> — Create or overwrite files</summary>

```
write path="/src/new.py" content="..."
write path="/src/new.py" content="..." auto_format=true
write path="/src/large.py" content="...chunk1..." append=false   # first chunk
write path="/src/large.py" content="...chunk2..." append=true    # subsequent chunks
```

- Returns diff on overwrite, confirms creation on new files
- `append=true` appends content rather than replacing — use for writing large files in chunks
- Cache is updated immediately after write

</details>

<details>
<summary><strong>edit</strong> — Targeted find/replace</summary>

```
edit path="/src/app.py" old_string="def foo():" new_string="def foo(x: int):"
edit path="/src/app.py" old_string="..." new_string="..." dry_run=true
edit path="/src/app.py" old_string="..." new_string="..." replace_all=true auto_format=true
```

- Uses cached content (no token cost for the read)
- Returns unified diff of the change
- Use `batch_edit` when applying 2+ independent changes to the same file

</details>

<details>
<summary><strong>batch_edit</strong> — Multiple edits in one call</summary>

```
batch_edit path="/src/app.py" edits='[["old1","new1"],["old2","new2"]]'
batch_edit path="/src/app.py" edits='[["old1","new1"],["old2","new2"]]' dry_run=true auto_format=true
```

- Up to 50 edits per call
- Partial success: individual edit failures don't block others
- Single round-trip, single cache update

</details>

<details>
<summary><strong>search</strong> — Semantic search across cached files</summary>

```
search query="authentication middleware logic" k=5
search query="database connection pooling" k=3
```

- Embedding-based semantic search — finds meaning, not keywords
- Only searches files that have been previously cached via `read` or `batch_read`
- Seed the cache first, then search

</details>

<details>
<summary><strong>similar</strong> — Find semantically related files</summary>

```
similar path="/src/auth.py" k=3
similar path="/tests/test_auth.py" k=5
```

- Finds cached files most similar to the given file
- Useful for discovering related tests, implementations, or documentation
- Only considers cached files; start with `k=3–5`

</details>

<details>
<summary><strong>glob</strong> — Pattern matching with cache awareness</summary>

```
glob pattern="**/*.py" directory="./src"
glob pattern="**/*.py" directory="./src" cached_only=true
```

- Shows cache status (cached/uncached) for each matched file
- `cached_only=true` returns only files already in cache — useful for scoping searches
- Max 1000 matches, 5-second timeout

</details>

<details>
<summary><strong>batch_read</strong> — Multiple files with token budget</summary>

```
batch_read paths="/src/a.py,/src/b.py" max_total_tokens=50000
batch_read paths='["/src/a.py","/src/b.py"]' diff_mode=true priority="/src/main.py"
batch_read paths="/src/*.py" max_total_tokens=30000 diff_mode=false
```

- **Glob expansion**: `src/*.py` expanded inline (max 50 files per glob)
- **Priority ordering**: `priority` paths read first, remainder sorted smallest-first
- **Token budget**: stops reading new files once `max_total_tokens` reached; skipped files include `est_tokens` hint
- **Unchanged suppression**: unchanged files appear in `summary.unchanged` with no content (zero tokens)
- **Context compression recovery**: set `diff_mode=false` when Claude needs full content after losing context

</details>

<details>
<summary><strong>diff</strong> — Compare two files</summary>

```
diff path1="/src/v1.py" path2="/src/v2.py"
```

- Returns unified diff between two files
- Includes semantic similarity score (cosine distance of embeddings)
- Large diffs auto-summarized to stay within token budget

</details>

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `TOOL_OUTPUT_MODE` | `compact` | Response detail (`compact`, `normal`, `debug`) |
| `TOOL_MAX_RESPONSE_TOKENS` | `0` | Global response token cap (`0` = disabled) |
| `MAX_CONTENT_SIZE` | `100000` | Max bytes returned by read operations |
| `MAX_CACHE_ENTRIES` | `10000` | Max cache entries before LRU-K eviction |

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
      "command": "semantic-cache-mcp",
      "env": {
        "LOG_LEVEL": "INFO",
        "TOOL_OUTPUT_MODE": "compact",
        "MAX_CONTENT_SIZE": "100000"
      }
    }
  }
}
```

**Embeddings:** Uses [FastEmbed](https://github.com/qdrant/fastembed) with `nomic-ai/nomic-embed-text-v1.5`. Runs entirely locally — no API keys, no network calls during search.

**Cache location:** `~/.cache/semantic-cache-mcp/`

---

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Claude     │────▶│  smart_read  │────▶│  Cache Lookup    │
│  Code       │     │              │     │  (SQLite + LSH)  │
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

**Read pipeline (in priority order):**

1. **File unchanged** — mtime matches cache entry → return "no changes" message (~5 tokens)
2. **File changed** — compute unified diff → return diff only (80–95% savings)
3. **Semantically similar cached file** — return diff from nearest neighbor (LSH O(1) lookup ≥ 100 files)
4. **Large file** — semantic summarization preserving docstrings and key function signatures
5. **New file** — full content returned, stored via SIMD-accelerated HyperCDC chunking

---

## Performance

| Component | Improvement | Details |
|-----------|-------------|---------|
| Embeddings | 22x smaller | int8 quantization: 772 bytes vs 17 KB/vector |
| Similarity search | O(1) lookup | LSH acceleration for caches ≥ 100 files |
| Array conversion | ~100x faster | `frombytes()` memcpy replaces `tolist()` iteration |
| Chunking | 5–7x faster | SIMD-parallel CDC at ~70–95 MB/s |
| Hashing | 3.8x faster | BLAKE3 with BLAKE2b fallback |
| Compression | 3.7x faster | ZSTD adaptive at 6.9 GB/s for text |
| Summarization | 50–80% savings | Semantic segment selection, structure preserved |
| Glob queries | N→1 DB calls | Batch `SELECT ... WHERE IN` replaces per-file lookups |
| Batch read | 2x fewer lookups | Pre-computed cache set eliminates double lookup |

See [docs/performance.md](docs/performance.md) for benchmarks and methodology.

---

## Package Structure

```
src/semantic_cache_mcp/
├── cache/              # Orchestration facade
│   ├── __init__.py     # Public API
│   ├── store.py        # Cache entry management
│   ├── read.py         # smart_read logic
│   ├── write.py        # smart_write / smart_edit
│   ├── search.py       # Embedding search + similar
│   └── _helpers.py     # Internal utilities
├── server/             # MCP interface (FastMCP 3.0)
│   ├── __init__.py
│   ├── _mcp.py         # FastMCP app setup
│   ├── response.py     # Response formatting + token budget
│   └── tools.py        # All 11 tool definitions
├── core/
│   ├── chunking/       # HyperCDC (Gear hash) + SIMD parallel CDC
│   ├── similarity/     # Cosine, LSH, int8/binary/ternary quantization
│   └── text/           # Diff generation, semantic summarization
├── storage/            # SQLite content-addressable storage
├── config.py           # Environment-driven configuration
└── types.py            # Shared type definitions
```

---

## Documentation

| Guide | Description |
|-------|-------------|
| [Architecture](docs/architecture.md) | Component design, algorithms, data flow |
| [Performance](docs/performance.md) | Optimization techniques, benchmarks |
| [Security](docs/security.md) | Threat model, input validation, size limits |
| [Advanced Usage](docs/advanced-usage.md) | Programmatic API, custom storage backends |
| [Troubleshooting](docs/troubleshooting.md) | Common issues, debug logging |

---

## Contributing

```bash
git clone https://github.com/CoderDayton/semantic-cache-mcp.git
cd semantic-cache-mcp
uv sync
uv run pytest
```

This project uses Python 3.12+, strict type hints throughout, Ruff for formatting and linting, and pytest for testing. See [CONTRIBUTING.md](CONTRIBUTING.md) for commit conventions, pre-commit hooks, and code standards.

---

## License

[MIT License](LICENSE) — use freely in personal and commercial projects.

---

## Credits

Built with [FastMCP 3.0](https://github.com/modelcontextprotocol/python-sdk) and:

- [FastEmbed](https://github.com/qdrant/fastembed) — local embeddings (nomic-embed-text-v1.5)
- SIMD-accelerated Parallel CDC — 5–7x faster than serial HyperCDC
- Semantic summarization based on TCRA-LLM ([arXiv:2310.15556](https://arxiv.org/abs/2310.15556))
- LSH approximate nearest-neighbor search
- int8/binary/ternary quantization for extreme compression
- BLAKE3 cryptographic hashing
- ZSTD/LZ4/Brotli adaptive compression
- LRU-K frequency-aware cache eviction
