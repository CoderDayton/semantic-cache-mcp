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

Semantic Cache MCP is a [Model Context Protocol](https://modelcontextprotocol.io) server that dramatically cuts token consumption when Claude reads files. Instead of sending full file contents every time, it returns diffs for changed files, finds semantically similar cached files, and intelligently truncates large files—all transparently.

---

## ✨ Features

- **80%+ Token Reduction** — Returns diffs instead of full files when content changes
- **Cached Write/Edit** — File modifications use cache for reading, return diffs
- **Local Embeddings** — No API keys needed, runs entirely offline
- **Semantic Similarity** — Finds related files using fast vector search
- **Content-Addressable Storage** — Efficient deduplication and delta compression
- **Smart Truncation** — Preserves code structure when cutting large files
- **LRU-K Eviction** — Keeps frequently accessed files in cache
- **Accurate Token Counting** — GPT-4o compatible tokenizer
- **DoS Protection** — Size limits and match count validation

---

## 📦 Installation

```bash
# Install from GitHub (recommended)
uv tool install git+https://github.com/CoderDayton/semantic-cache-mcp.git

# Or clone and install locally
git clone https://github.com/CoderDayton/semantic-cache-mcp.git
cd semantic-cache-mcp && uv tool install .
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

### Recommended: Block Native Tools

Add to `~/.claude/settings.json` to force semantic-cache usage:

```json
{
  "permissions": {
    "deny": ["Read", "Write", "Edit"]
  }
}
```

This blocks native file tools, ensuring Claude uses semantic-cache MCP tools for all file operations.

### Recommended: CLAUDE.md Configuration

Add to your `~/.claude/CLAUDE.md` to enforce semantic-cache usage globally:

```markdown
## External Tools

- semantic-cache: MUST use instead of native file tools (80%+ token savings)
  - `read` → single-file inspection (diffs by default)
    - `diff_mode=true` → keep during iteration; set `false` only when full uncached content is needed
    - `offset`/`limit` → read specific line ranges without loading full content into context
  - `batch_read` → use for 2+ files (better than repeated `read`)
  - `write` → replaces Write tool (caches result, returns diff on overwrite)
    - `auto_format=true` → runs formatter after write (ruff/prettier/gofmt)
    - Prefer for full rewrites/create; use `edit` for targeted substitutions
  - `edit` → replaces Edit tool (uses cached read, returns diff)
    - `auto_format=true` → runs formatter after edit
  - `multi_edit` → use for 2+ edits in one file (better than repeated `edit`)
  - `search`/`similar` → use after cache is seeded via `read`/`batch_read`
    - Start with small `k` (3-5), increase only if needed
```

This tells Claude to prefer semantic-cache tools over the built-in Read, Write, and Edit tools, maximizing token savings across all file operations.

---

## 🚀 Tools

### Core Tools

| Tool | Description |
|------|-------------|
| `read` | Smart file reading with diffs (99% savings unchanged, 80-95% changed) |
| `write` | Write files, returns diff on overwrite. `auto_format=true` runs formatter |
| `edit` | Find/replace using cached reads (zero token read cost). `auto_format=true` runs formatter |
| `multi_edit` | Batch find/replace, partial success supported. `auto_format=true` runs formatter |

All tools return JSON. Response detail and token cap are controlled globally via environment variables.

### Discovery Tools

| Tool | Description |
|------|-------------|
| `search` | Semantic search across cached files by meaning |
| `similar` | Find files semantically similar to a given file |
| `glob` | Find files by pattern with cache status |
| `batch_read` | Read multiple files with token budget, priority ordering, glob expansion |
| `diff` | Compare two files using cache |

### Management Tools

| Tool | Description |
|------|-------------|
| `stats` | Cache metrics (files, tokens, compression ratio) |
| `clear` | Reset all cache entries |

<details>
<summary><strong>Tool Examples</strong></summary>

#### read

```bash
read path="/src/app.py"
read path="/src/app.py" offset=120 limit=80
```

- First read → full content, cached
- Unchanged → `// File unchanged (1,234 tokens cached)`
- Modified → unified diff only
- Use `offset`/`limit` after identifying a target region

#### write

```bash
write path="/src/app.py" content="..."
write path="/src/app.py" content="..." auto_format=true
```

Returns diff of changes, updates cache for instant reads. With `auto_format=true`, runs formatter (ruff/prettier/gofmt) after write.

#### edit

```bash
edit path="/src/app.py" old_string="old" new_string="new"
edit path="/src/app.py" old_string="old" new_string="new" auto_format=true
edit path="/src/app.py" old_string="old" new_string="new" dry_run=true
```

Uses cached content (no token cost), returns diff. Use `replace_all=true` for multiple matches. With `auto_format=true`, runs formatter after edit.
Use `multi_edit` when applying 2+ independent edits in one file.

#### multi_edit

```bash
multi_edit path="/src/app.py" edits='[["old1", "new1"], ["old2", "new2"]]'
multi_edit path="/src/app.py" edits='[["old1", "new1"], ["old2", "new2"]]' dry_run=true
```

Independent edits—some can fail while others succeed.

#### search

```bash
search query="authentication logic" k=5
```

Searches cached files by semantic meaning, not keywords.
Seed cache first with `read` or `batch_read`.

#### similar

```bash
similar path="/src/auth.py" k=5
```

Finds related code, tests, or documentation.
Use after source and nearby files have been cached.

#### batch_read

```bash
batch_read paths="/src/a.py,/src/b.py" max_total_tokens=50000
batch_read paths='["/src/a.py","/src/b.py","/src/c.py"]' max_total_tokens=30000
batch_read paths="/src/*.py" priority="/src/main.py,/src/config.py"
```

- **Glob expansion**: paths like `src/*.py` are expanded (max 50 files)
- **Priority ordering**: `priority` paths are read first (order preserved), remainder sorted smallest-first
- **Unchanged collapse**: previously-read unchanged files reported in `summary.unchanged` with no content (saves tokens)
- **Skipped enrichment**: skipped files include `est_tokens` and a hint to use `read` with offset/limit
- **Structured response**: `summary` (metadata), `skipped` (actionable), `files` (only entries with content)

#### glob

```bash
glob pattern="**/*.py" directory="./src"
```

Max 1000 matches, 5s thread-safe timeout (cross-platform). Shows which files are cached.

#### diff

```bash
diff path1="/src/old.py" path2="/src/new.py"
```

Returns unified diff with similarity score.

</details>

---

## ⚙️ Configuration

| Environment Variable | Default | Description                                             |
| -------------------- | ------- | ------------------------------------------------------- |
| `LOG_LEVEL`          | `INFO`  | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `TOOL_OUTPUT_MODE`   | `compact` | Tool response detail level (`compact`, `normal`, `debug`) |
| `TOOL_MAX_RESPONSE_TOKENS` | `0` | Global response cap (`0` disables cap) |
| `MAX_CONTENT_SIZE`   | `100000` | Default max bytes returned by `read`/`smart_read` |
| `MAX_CACHE_ENTRIES`  | `10000` | Max cache entries before LRU-K eviction |

Environment example:

```bash
TOOL_OUTPUT_MODE=compact
TOOL_MAX_RESPONSE_TOKENS=6000
MAX_CONTENT_SIZE=100000
MAX_CACHE_ENTRIES=10000
```

**Embeddings:** Uses local [FastEmbed](https://github.com/qdrant/fastembed) with `nomic-ai/nomic-embed-text-v1.5` model. No API keys or external services needed.

Safety limits in `cache.py`:

| Setting          | Default | Description                            |
| ---------------- | ------- | -------------------------------------- |
| `MAX_WRITE_SIZE` | 10MB    | Maximum content size for write tool    |
| `MAX_EDIT_SIZE`  | 10MB    | Maximum file size for edit tool        |
| `MAX_MATCHES`    | 10,000  | Maximum occurrences for replace_all    |

---

## 🏗️ How It Works

```text/plain
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Claude    │────▶│  smart_read  │────▶│   Cache     │
│   Code      │     │              │     │   Lookup    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ Unchanged│     │   Diff   │     │ Semantic │
   │   99%    │     │  80-95%  │     │  70-90%  │
   └──────────┘     └──────────┘     └──────────┘
```

**Caching strategies (in order of preference):**

1. **File unchanged** — mtime matches cache → return "no changes" message
2. **File changed** — compute unified diff → return diff only
3. **Similar file** — find semantically similar cached file → return diff from it
4. **Large file** — semantic summarization preserving important segments (docstrings, key functions)
5. **New file** — return full content, store in cache with SIMD-accelerated chunking

---

## ⚡ Performance

Recent optimizations deliver **5-22x improvements** across core operations:

| Component | Improvement | Details |
|-----------|-------------|---------|
| **Embeddings** | 22x smaller | int8 quantization (772 vs 17KB/vector) |
| **Array conversion** | ~100x faster | frombytes() memcpy replaces tolist() iteration |
| **Glob queries** | N→1 DB calls | Batch `SELECT ... WHERE IN` replaces per-file lookups |
| **Batch read** | 2x fewer lookups | Pre-computed cache set eliminates double lookup |
| **Chunking** | 5-7x faster | SIMD-accelerated parallel CDC |
| **Similarity** | 1.7x faster | Pre-quantized binary search |
| **Summarization** | 50-80% savings | Semantic segment selection preserves structure |
| **Hashing** | 3.8x faster | BLAKE3 with BLAKE2b fallback |
| **Compression** | 3.7x faster | ZSTD with adaptive quality |

See [Performance Docs](docs/performance.md) for benchmarks and detailed analysis.

---

## 📚 Documentation

| Guide                                      | Description                                |
| ------------------------------------------ | ------------------------------------------ |
| [Architecture](docs/architecture.md)       | Component design, algorithms, data flow    |
| [Performance](docs/performance.md)         | Optimization techniques, memory efficiency |
| [Security](docs/security.md)               | Security considerations and threat model   |
| [Advanced Usage](docs/advanced-usage.md)   | Programmatic API, custom storage backends  |
| [Troubleshooting](docs/troubleshooting.md) | Common issues, debug logging               |

---

## 🤝 Contributing

Contributions welcome! This project uses:

- **Python 3.12+** with strict type hints
- **Ruff** for formatting and linting
- **pytest** for testing

```bash
git clone https://github.com/CoderDayton/semantic-cache-mcp.git
cd semantic-cache-mcp
uv sync && uv run pytest
```

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for setup, commit conventions, and code standards.

---

## 📄 License

[MIT License](LICENSE) — use freely in personal and commercial projects.

---

## 🙏 Credits

Built with [FastMCP 3.0](https://github.com/modelcontextprotocol/python-sdk) and powered by:

- [FastEmbed](https://github.com/qdrant/fastembed) for local embeddings (nomic-embed-text-v1.5)
- SIMD-accelerated Parallel CDC (5-7x faster than serial HyperCDC)
- Semantic summarization based on TCRA-LLM (arXiv:2310.15556)
- LSH approximate similarity search for fast nearest-neighbor lookups
- Binary/ternary quantization for extreme compression (up to 100x)
- BLAKE3 for cryptographic hashing
- ZSTD/LZ4/Brotli adaptive compression
- int8 quantized similarity search (22x storage reduction)
- LRU-K frequency-aware cache eviction
