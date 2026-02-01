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
- **Local Embeddings** — FastEmbed with nomic-embed-text-v1.5 (no API keys needed)
- **Semantic Similarity** — Finds related cached files using embeddings (cosine similarity > 0.85)
- **Content-Addressable Storage** — Rabin fingerprinting CDC + BLAKE2b hashing for deduplication
- **Adaptive Compression** — Brotli compression tuned by Shannon entropy
- **LRU-K Eviction** — Frequency-aware cache management (keeps frequently accessed files)
- **o200k_base Tokenizer** — Accurate GPT-4o token counting with auto-download
- **Structured Logging** — Debug, info, and warning levels via `LOG_LEVEL` env var

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

- Rabin fingerprinting for content-defined chunking
- BLAKE2b for cryptographic hashing
- Shannon entropy for adaptive compression
- LRU-K for frequency-aware eviction
