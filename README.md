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

**Cut your MCP client's token usage by 98% on cached reads. Respond in milliseconds.**

Semantic Cache MCP is a [Model Context Protocol](https://modelcontextprotocol.io) server that replaces redundant full-file reads with marker hits, unified diffs, and semantic summaries. Fifteen tools (read, read_image, batch_read, write, edit, edit_preview, batch_edit, search, grep, glob, similar, diff, delete, clear, stats) route every file operation through one cache-aware layer, so an MCP-capable agent skips files it has already seen.

---

## Why this exists

In order of impact:

**1. Reads stop costing tokens.** The first read seeds the cache. Re-reads of unchanged files return a 5-token marker (`mtime` match, no disk I/O). Modified files return a unified diff. Files larger than the budget collapse to a semantic skeleton that preserves structure rather than slicing at a byte offset.

**2. Search and grep run on the cache, not the disk.** Semantic search (hybrid BM25 + HNSW), similar-file lookup, glob, and grep all read from the same indexed corpus that `read`/`batch_read` populate. An in-session result LRU collapses repeated queries to sub-millisecond hits.

**3. Mutations are bounded by default.** `write`, `edit`, and `batch_edit` enforce size and match limits, support `dry_run`, can run formatters, and refresh the cache atomically. Local FastEmbed is the default embedding provider; OpenAI-compatible endpoints are opt-in.

---

## Installation

Add to Claude Code settings (`~/.claude.json`):

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

### OpenAI-Compatible Embeddings

Local FastEmbed remains the default. To route embeddings through an OpenAI-compatible provider instead, enable it in the MCP env block. Defaults target Ollama:

```json
"env": {
  "OPENAI_EMBEDDINGS_ENABLED": "true",
  "OPENAI_BASE_URL": "http://localhost:11434/v1",
  "OPENAI_API_KEY": "ollama",
  "OPENAI_EMBEDDING_MODEL": "nomic-embed-text"
}
```

Run `ollama pull nomic-embed-text` first if the model is not installed. For hosted OpenAI, set `OPENAI_BASE_URL=https://api.openai.com/v1`, use a real `OPENAI_API_KEY`, and choose an embedding model such as `text-embedding-3-small`. `OPENAI_EMBEDDING_DIMENSIONS` is optional; leave it unset to infer the returned vector size.

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

- MUST use `semantic-cache-mcp` instead of native I/O tools (98% token savings on cached reads)
```
---

## Tools

### Core

| Tool | Description |
|------|-------------|
| `read` | Single-file cache-aware read. Returns full content on first read, unchanged markers on cache hits, diffs on modifications, and supports `offset`/`limit` for targeted recovery. |
| `read_image` | Pass-through for image files. Returns an MCP image content block (base64 + mime) so vision models can see the pixels; sidecar metadata holds size and mime. Bypasses the semantic cache. Capped at 5 MiB (`SCMCP_MAX_IMAGE_BYTES`). |
| `delete` | Single-path delete for one file or symlink, with cache eviction and `dry_run=true`. Intentionally does not support globs, recursive delete, or real-directory delete. |
| `write` | Full-file create or replace with cache refresh. Returns creation status or an overwrite diff, supports `append=true`, and can run formatters. |
| `edit` | Single-file exact edit using cached content. Supports scoped and line-range replacement plus `dry_run=true`. For multiple edits to the same file, prefer `batch_edit`. |
| `batch_edit` | Multiple exact edits in one file with partial success reporting. Preferred over repeated `edit` calls on the same file: single response, atomic, faster on large files. |
| `edit_preview` | Read-only probe that returns match count, line numbers, and small context snippets for a candidate `old_string`. Use before a costly `edit` to confirm anchor uniqueness. |

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

The table above is the authoritative tool map. This section only shows the common call shapes.

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
<summary><strong>batch_read</strong> — Multiple files with token budget</summary>

```
batch_read paths="/src/a.py,/src/b.py" max_total_tokens=50000
batch_read paths='["/src/a.py","/src/b.py"]' priority="/src/main.py"
batch_read paths="/src/*.py" max_total_tokens=30000
```

- Expands simple globs, honors `priority`, enforces `max_total_tokens`, and reports skipped paths with recovery hints.
- Unchanged files are collapsed into the summary instead of repeating content.

</details>

<details>
<summary><strong>discovery</strong> — Search, similar, glob, grep, diff</summary>

```
search query="authentication middleware logic" k=5
similar path="/src/auth.py" k=3
glob pattern="**/*.py" directory="./src" cached_only=true
grep pattern="class Cache" path="src/**/*.py"
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
| `OPENAI_EMBEDDINGS_ENABLED` | `false` | Use OpenAI-compatible remote embeddings instead of local FastEmbed |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible base URL; default targets Ollama |
| `OPENAI_API_KEY` | `ollama` | API key for the remote embedding provider |
| `OPENAI_EMBEDDING_MODEL` | `nomic-embed-text` | Remote embedding model name |
| `OPENAI_EMBEDDING_DIMENSIONS` | *(inferred)* | Optional requested/expected remote embedding dimension |
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

`search` works the same way. An in-session LRU keyed on `(query, k, directory)`
returns warm hits in ~10 µs; misses fall through to embed + BM25 + HNSW. Every
cache mutation (`put`, `clear`, `delete_path`, `update_mtime`) bumps the LRU, so
callers never see a result that predates a write.

---

## Performance

Measured on this project's 43 source files (**168,614 tokens**), CPU embeddings, i9-13900K, commit `5cd7100`. Reproducible via `--json` output for CI diffing.

### Token savings — **98.5%** overall (phases 2–6)

| Phase | Scenario | Savings |
|-------|----------|--------:|
| **Overall (cached, phases 2–6)** | **Aggregate token reduction** | **98.5%** |
| Unchanged re-read | mtime match — fast path skips disk I/O | 98.9% |
| Content hash | mtime drifted, BLAKE3 still matches | 98.9% |
| Batch read | All files via `batch_read`, 200K budget | 98.9% |
| Search previews | 5 queries × k=5, previews vs full reads | 98.3% |
| Small edits | Real ~5% line changes in 30% of files | 97.3% |
| Cold read | First read, no cache (baseline) | 0% |

### Latency — **unchanged reads ~1 ms; repeat searches ~10 µs**

| Operation | p50 | Notes |
|-----------|----:|-------|
| Single unchanged read (fast path) | **1.1 ms** | mtime + cache hit; no disk I/O |
| Single diff read (changed file) | 1.0 ms | hash check + unified diff |
| Search k=5 (cache **hit**) | **< 0.01 ms** | in-session LRU; **2,000×+ vs cold** |
| Search k=5 (cache **miss**) | 5.6 ms | embed query + hybrid BM25/HNSW |
| Edit (scoped find/replace) | 3.3 ms | uses cached content |
| Find similar (k=3) | 2.2 ms | cached embedding reused |
| Grep (literal `def `) | 1.4 ms | FTS5 over cached corpus |
| Grep (regex) | 2.1 ms | regex compiled once |
| Batch read (43 files, diff mode) | 40.2 ms | one ONNX inference for all new/changed files |
| Unchanged re-read (43 files) | 26.9 ms | whole-corpus pass |
| Cold read (43 files, total) | 1,990 ms | includes disk I/O, tokenisation, embedding |
| Write (200-line file) | 49.1 ms | creates + caches + embeds |
| Single embedding (largest file) | 47 ms | ONNX, single thread |
| Model warmup (one-time) | 195 ms | startup only |

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
- [SimpleVecDB ≥ 2.6.0](https://github.com/CoderDayton/SimpleVecDB) — HNSW vector storage with FTS5 keyword search, atomic `delete_collection`, and opt-in embedding persistence (`store_embeddings=True`)
- Semantic summarization based on TCRA-LLM ([arXiv:2310.15556](https://arxiv.org/abs/2310.15556))
- BLAKE3 cryptographic hashing for content freshness
- LRU-K frequency-aware cache eviction
