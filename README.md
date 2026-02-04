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
  - `read` → replaces Read tool (returns diffs, not full content)
  - `write` → replaces Write tool (caches result, returns diff on overwrite)
  - `edit` → replaces Edit tool (uses cached read, returns diff)
```

This tells Claude to prefer semantic-cache tools over the built-in Read, Write, and Edit tools, maximizing token savings across all file operations.

---

## 🚀 Usage

The server provides eleven tools:

### `read` — Smart File Reading

```bash
read path="/src/app.py"
```

**What happens:**

- First read: Full content returned, cached for future
- Same file again: "File unchanged" (99% token savings)
- File modified: Unified diff only (80-95% savings)
- Similar file exists: Diff from similar file (70-90% savings)

**Example output for unchanged file:**

```bash
// File unchanged: /src/app.py (1,234 tokens cached)
// [cache:true diff:false saved:1,200]
```

**Example output for modified file:**

```diff
// Diff for /src/app.py (changed since cache):
// Stats: +3 -1 ~2 lines, 8.5% size
--- cached
+++ current
@@ -42,7 +42,7 @@
 def process():
-    return old_value
+    return new_value
// [cache:true diff:true saved:950]
```

### `write` — Smart File Writing

```bash
write path="/src/app.py" content="new file content"
```

**What happens:**

- New file: Creates and caches, returns metadata
- Existing file: Returns diff of changes (not full content!)
- Updates cache for instant subsequent reads

**Example output for update:**

```diff
// Updated: /src/app.py
// Stats: +5 -2 ~1 lines
// Tokens saved: 890 (diff vs full content)
// Hash: a1b2c3d4e5f6...
--- old
+++ new
@@ -10,3 +10,6 @@
 unchanged line
-removed line
+added line
// [created:false dry_run:false cached:true]
```

### `edit` — Smart Find/Replace

```bash
edit path="/src/app.py" old_string="old_value" new_string="new_value"
```

**What happens:**

- Uses cached content for reading (no token cost!)
- Returns diff showing exactly what changed
- Validates match uniqueness (use `replace_all=true` for multiple)

**Example output:**

```diff
// Edited: /src/app.py
// Replaced 1 of 1 match at line 42
// Stats: +1 -1 ~1 lines
// Tokens saved: 1,234 (cached read)
// Hash: a1b2c3d4e5f6...
--- old
+++ new
@@ -40,5 +40,5 @@
 context before
-    old_value
+    new_value
 context after
// [replace_all:false dry_run:false cached:true]
```

### `stats` — Cache Metrics

```json
{
  "files_cached": 42,
  "total_tokens_cached": 125000,
  "compression_ratio": 0.19,
  "dedup_ratio": 5.3
}
```

### `multi_edit` — Batch Find/Replace

```bash
multi_edit path="/src/app.py" edits='[["old1", "new1"], ["old2", "new2"]]'
```

Apply multiple independent edits. Each can succeed/fail independently—partial success applies working edits.

### `search` — Semantic Search

```bash
search query="authentication logic" k=10
```

Search cached files by meaning, not keywords. Returns ranked matches with similarity scores.

### `similar` — Find Related Files

```bash
similar path="/src/auth.py" k=5
```

Find cached files semantically similar to a given file. Great for finding tests, implementations, or related modules.

### `diff` — Compare Files

```bash
diff path1="/src/old.py" path2="/src/new.py"
```

Compare two files using cache. Returns unified diff with similarity score.

### `batch_read` — Read Multiple Files

```bash
batch_read paths="/src/a.py,/src/b.py" max_total_tokens=50000
```

Read multiple files with token budget. Skips files if budget exceeded.

### `glob` — Find Files by Pattern

```bash
glob pattern="**/*.py" directory="./src"
```

Find files matching pattern. Shows cache status and token counts. Max 1000 matches, 5s timeout.

### `stats` — Cache Metrics

```json
{
  "files_cached": 42,
  "total_tokens_cached": 125000,
  "compression_ratio": 0.19,
  "dedup_ratio": 5.3
}
```

### `clear` — Reset Cache

```text/plain
Cleared 42 cache entries
```

---

## ⚙️ Configuration

| Environment Variable | Default | Description                                             |
| -------------------- | ------- | ------------------------------------------------------- |
| `LOG_LEVEL`          | `INFO`  | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

**Embeddings:** Uses local [FastEmbed](https://github.com/qdrant/fastembed) with `nomic-ai/nomic-embed-text-v1.5` model. No API keys or external services needed.

Cache settings in `config.py`:

| Setting                | Default | Description                            |
| ---------------------- | ------- | -------------------------------------- |
| `MAX_CONTENT_SIZE`     | 100KB   | Maximum content size returned          |
| `MAX_CACHE_ENTRIES`    | 10,000  | LRU-K eviction threshold               |
| `SIMILARITY_THRESHOLD` | 0.85    | Minimum cosine similarity for matching |

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
