# Semantic Cache MCP

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP 3.0](https://img.shields.io/badge/FastMCP-3.0-green.svg)](https://github.com/modelcontextprotocol/python-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Reduce Claude Code token usage by 80%+ with intelligent file caching.**

Semantic Cache MCP is a [Model Context Protocol](https://modelcontextprotocol.io) server that dramatically cuts token consumption when Claude reads files. Instead of sending full file contents every time, it returns diffs for changed files, finds semantically similar cached files, and intelligently truncates large files—all transparently. Works with any Claude Code session out of the box.

---

## 📦 Installation

```bash
# Install with uv (recommended)
uv tool install /path/to/semantic-cache-mcp

# Or with pip
pip install /path/to/semantic-cache-mcp
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

---

## 🚀 Usage

The server provides three tools that Claude can use:

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

## ✨ Features

- **80%+ Token Reduction** — Returns diffs instead of full files when content changes
- **Local Embeddings** — No API keys needed, runs entirely offline
- **Semantic Similarity** — Finds related files using fast vector search
- **Content-Addressable Storage** — Efficient deduplication and delta compression
- **Smart Truncation** — Preserves code structure when cutting large files
- **LRU-K Eviction** — Keeps frequently accessed files in cache
- **Accurate Token Counting** — GPT-4o compatible tokenizer

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
4. **Large file** — smart truncation preserving structure
5. **New file** — return full content, store in cache

---

## ⚡ Performance

Recent optimizations deliver **2-20x speedups** across core operations:

| Component | Improvement | Details |
|-----------|-------------|---------|
| **Chunking** | 2.7x faster | HyperCDC vs Rabin fingerprinting |
| **Hashing** | 3.8-4.9x faster | BLAKE3 vs BLAKE2b (8KB+ chunks) |
| **Compression** | 3.7x faster | ZSTD caching + native threading |
| **Similarity** | 8-14x faster | Batch SIMD + int8 quantization |

See [Performance Docs](docs/performance.md) for benchmarks and detailed analysis.

---

## 📚 Documentation

| Guide                                      | Description                                |
| ------------------------------------------ | ------------------------------------------ |
| [Architecture](docs/architecture.md)       | Component design, algorithms, data flow    |
| [Performance](docs/performance.md)         | Optimization techniques, memory efficiency |
| [Advanced Usage](docs/advanced-usage.md)   | Programmatic API, custom storage backends  |
| [Troubleshooting](docs/troubleshooting.md) | Common issues, debug logging               |

---

## 🤝 Contributing

Contributions welcome! This project uses:

- **Python 3.12+** with strict type hints
- **Ruff** for formatting and linting
- **mypy** for type checking

```bash
# Development setup
git clone https://github.com/CoderDayton/semantic-cache-mcp.git && cd semantic-cache-mcp
uv sync
uv run ruff check src/
uv run mypy src/
```

See [Contributing Guide](docs/contributing.md) for commit conventions and code standards.

---

## 📄 License

[MIT License](LICENSE) — use freely in personal and commercial projects.

---

## 🙏 Credits

Built with [FastMCP 3.0](https://github.com/modelcontextprotocol/python-sdk) and powered by:

- HyperCDC (Gear hash) for content-defined chunking
- BLAKE3 for cryptographic hashing (with BLAKE2b fallback)
- ZSTD/LZ4/Brotli multi-codec compression
- int8 quantization for fast similarity search
- LRU-K for frequency-aware eviction

See [Performance](docs/performance.md) for benchmarks and optimization details.
