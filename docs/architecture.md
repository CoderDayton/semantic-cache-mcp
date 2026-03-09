# Architecture

## Package Structure

```
src/semantic_cache_mcp/
├── config.py               # Constants and environment-variable configuration
├── types.py                # All shared data models (ReadResult, WriteResult, etc.)
├── cache/                  # Orchestration facade — coordinates all components
│   ├── __init__.py         # Public API re-exports (including embed_batch)
│   ├── store.py            # SemanticCache class: embedding, storage, metrics coordination
│   ├── read.py             # smart_read, batch_smart_read (batch pre-scan + embed)
│   ├── write.py            # smart_write, smart_edit, smart_batch_edit
│   ├── search.py           # semantic_search, find_similar_files, glob_with_cache_status, compare_files
│   ├── metrics.py          # SessionMetrics: per-session and lifetime metric tracking
│   └── _helpers.py         # Internal utilities: _suppress_large_diff, formatter dispatch
├── server/                 # MCP interface — thin translation layer only
│   ├── __init__.py
│   ├── _mcp.py             # FastMCP app instance, lifespan, startup warmup
│   ├── response.py         # Response formatting, TOOL_OUTPUT_MODE handling
│   └── tools.py            # All 12 MCP tool definitions
├── core/                   # Pure algorithms — stateless, zero I/O
│   ├── __init__.py         # Flat re-exports from all sub-packages
│   ├── chunking/           # Content-defined chunking (used for large file splitting)
│   │   ├── __init__.py
│   │   ├── _gear.py        # Serial HyperCDC (Gear hash rolling window)
│   │   └── _simd.py        # SIMD-accelerated parallel CDC
│   ├── embeddings.py       # FastEmbed local embeddings (embed, embed_batch, embed_query)
│   ├── hashing.py          # BLAKE3/BLAKE2b content hashing, DeduplicateIndex
│   ├── similarity/         # Cosine similarity utilities
│   │   ├── __init__.py
│   │   └── _cosine.py      # cosine_similarity, batch operations
│   ├── text/               # Diff generation and semantic summarization
│   │   ├── __init__.py
│   │   ├── _diff.py        # generate_diff, compute_delta, diff_stats
│   │   └── _summarize.py   # summarize_semantic (TCRA-LLM based)
│   └── tokenizer.py        # BPE token counting (o200k_base)
└── storage/                # Persistence layer
    ├── __init__.py
    ├── vector.py           # VectorStorage: simplevecdb with HNSW + FTS5
    └── sqlite.py           # SQLiteStorage: session metrics persistence only
```

## Design Principles

- **Separation of concerns** — `core/` is stateless pure algorithms; `storage/` is persistence only; `cache/` orchestrates; `server/` translates MCP ↔ Python
- **Dependency injection** — storage and config are passed explicitly; no hidden globals
- **Facade pattern** — `cache/` exposes a clean API; callers never touch `storage/` directly
- **Performance first in hot paths** — embedding, hashing, similarity, and tokenization are optimized; everything else prioritizes clarity
- **Type safety** — strict mypy, no `Any` in public APIs

---

## Storage (`storage/vector.py`)

### VectorStorage (simplevecdb)

The primary storage backend uses [SimpleVecDB](https://github.com/CoderDayton/SimpleVecDB):

- **HNSW index** — O(log N) approximate nearest neighbor search for semantic similarity
- **FTS5 full-text search** — BM25 keyword search for grep and hybrid queries
- **Hybrid search** — Reciprocal Rank Fusion (RRF) combines HNSW + BM25 results
- **Raw text storage** — File contents stored as plain text in `page_content` (no compression)
- **Float16 quantization** — 2× smaller vectors in the HNSW index with negligible quality loss

### Document Model

Files are stored as simplevecdb `Document` objects:

```
Small file (< 8KB):
  └── Single document: page_content=full_text, embedding=file_vector

Large file (≥ 8KB):
  ├── Parent document: page_content="", embedding=file_vector, is_parent=True
  └── Child documents (per CDC chunk):
      ├── page_content=chunk_text, embedding=zero_vector, chunk_index=0
      ├── page_content=chunk_text, embedding=zero_vector, chunk_index=1
      └── ...
```

Large files are split via HyperCDC (content-defined chunking) into multiple child documents. The parent holds the file-level embedding for semantic search; children hold raw text for content retrieval and grep. Child documents use zero embeddings — per-chunk embedding would require N model calls.

### Metadata

Each document carries metadata for cache management:

| Key | Type | Description |
|-----|------|-------------|
| `path` | `str` | Absolute file path |
| `content_hash` | `str` | BLAKE3 hex digest of full file content |
| `mtime` | `float` | File modification time |
| `tokens` | `int` | Token count (BPE o200k_base) |
| `chunk_index` | `int` | Chunk ordering (-1 for parent) |
| `total_chunks` | `int` | Number of chunks (1 for small files) |
| `access_history` | `JSON` | Last 5 access timestamps (LRU-K) |
| `is_parent` | `bool` | Parent document marker (large files only) |

### LRU-K Eviction (K=2)

When `MAX_CACHE_ENTRIES` is exceeded, eviction uses the **K-th most recent** access time rather than the most recent:

- A file accessed only once has no second access time → evicted first
- A file accessed regularly has a recent second access time → retained
- This correctly handles large one-time reads (e.g., grepping) without polluting the cache

### Session Metrics (`storage/sqlite.py`)

A separate lightweight SQLite database stores session metrics only. This is not used for file content — just for tracking token savings, cache hits/misses, and tool call counts across sessions.

---

## Core Algorithms

### Chunking (`core/chunking/`)

Used by `VectorStorage._put_chunked()` to split large files (≥ 8KB) into content-defined chunks.

#### Serial HyperCDC — `_gear.py`

Content-defined chunking using a Gear hash rolling window:

- Pre-computed 256-entry gear table for O(1) byte lookups
- Rolling hash: `h = ((h << 1) + gear[byte]) & MASK_64`
- Boundary when `(h & mask) == 0` (normalized chunking)
- Skip-min: no boundary checks in first 2KB per chunk
- ~8KB average chunk size

Key property: similar files produce identical chunks even when bytes shift position, enabling efficient re-chunking on file changes.

#### SIMD Parallel CDC — `_simd.py`

Faster chunking via CPU-core-level parallelism:

1. Divide content into N segments (one per available core)
2. Each worker finds CDC boundaries in its segment independently
3. Merge and de-duplicate overlapping boundaries at segment edges

`get_optimal_chunker(prefer_simd=True)` auto-selects; falls back gracefully to serial HyperCDC.

---

### Hashing (`core/hashing.py`)

BLAKE3 primary, BLAKE2b fallback. Used for:

- **Content hash freshness** — detecting when mtime changes but content is identical (touch, git checkout)
- **Deduplication** — `DeduplicateIndex` for fingerprint-based dedup
- **Change detection** — comparing cached vs current content hash

LRU cache avoids re-hashing identical data.

---

### Tokenizer (`core/tokenizer.py`)

GPT-4o compatible (o200k_base) BPE tokenizer.

- **O(N log M)** priority-queue merge algorithm vs O(N²) naive
- Merge results memoized to skip repeated BPE sequences
- Single-byte tokens skip BPE entirely (fast path)
- Files > 50KB use sampling for O(1) estimate
- Auto-downloads tiktoken file (~3.5MB), SHA256-verified
- Heuristic fallback (chars / 4) when model unavailable

---

### Embeddings (`core/embeddings.py`)

Local text embeddings via FastEmbed (configurable model, default `BAAI/bge-small-en-v1.5`).

- **384-dimensional** (default), 33M parameters, 512 token context window
- Runs via ONNX Runtime — `cpu` by default, `cuda` when configured
- Singleton model instance — loaded once, reused across all calls
- Warmup pass at server startup for predictable first-request latency
- `"Represent this sentence for searching relevant passages:"` prefix for query embedding
- File-type semantic labels prepended to document content before embedding (e.g., `"python source file: ..."`)
- Model stored in `~/.cache/semantic-cache-mcp/models/`
- No API keys; fully offline after initial download

#### Batch Embedding

`embed_batch(texts)` amortizes ONNX Runtime overhead across N texts in a single model call — critical for `batch_smart_read` where multiple files need embedding on first cache miss.

**Pre-scan flow in `batch_smart_read`:**
1. Before reading any file, scan all requested paths for new or changed files (mtime check)
2. Collect their content and apply the same file-type label prefix used by single-file embedding
3. Call `embed_batch()` once with all texts — a single ONNX inference pass
4. Store prefetched embeddings; `smart_read` picks them up instead of calling the model individually

This reduces N model calls (one per new file) to exactly 1, regardless of batch size.

---

### Text (`core/text/`)

#### `_diff.py` — Diff and Delta Compression

- Unified diffs via Python `difflib` with diff statistics (insertions, deletions, modifications)
- Delta compression: store only changed lines (10–100× smaller for small edits to large files)
- `_suppress_large_diff()` (in `cache/_helpers.py`): caps diff output at a token budget to prevent context overflow

#### `_summarize.py` — Semantic Summarization

Based on TCRA-LLM (arXiv:2310.15556). Preserves structural integrity when files exceed the size budget:

**Algorithm:**
1. Split file at semantic boundaries (function/class definitions, paragraphs)
2. Score each segment:
   - **Position score**: U-shaped curve — highest at start and end, lowest in middle
   - **Density score**: unique token ratio + syntax character density + non-whitespace ratio
   - **Diversity penalty**: cosine similarity to already-selected segments (avoids redundancy)
3. Greedily select highest-scoring segments that fit the budget
4. Always preserve the first segment (docstrings, imports, module header)
5. Reassemble with `# ... [N lines omitted] ...` markers

**Result:** 50–80% token savings on large files vs simple truncation, while preserving code skeleton and intent.

---

## Data Flow

### Read

```
Client ──→ smart_read(path, diff_mode=True)
                │
                ▼
         cache.get(path)
                │
        ┌───────┴───────────────┬──────────────────┐
        │                       │                  │
        ▼                       ▼                  ▼
  mtime match             content hash          not found
  "unchanged"             match (touch)         read disk
  (99% savings)           update mtime          embed + store
                          "unchanged"           return full
                          (99% savings)
                                │
                          hash changed
                          compute diff
                          (80-95% savings)
```

After context compression, call with `diff_mode=False` to bypass the "unchanged" path and always receive full content.

### Batch Read

```
Client ──→ batch_smart_read(paths, diff_mode=True)
                │
         1. Pre-scan: filter new/changed paths (mtime check)
                │
         2. embed_batch([...all new/changed texts...])
            ── single ONNX inference call ──
                │
         3. smart_read() per file (embedding prefetched, no model call)
                │
         4. Return BatchReadResult with per-file status
```

### Write / Edit

```
Client ──→ smart_write(path, content)
                │
      read existing content (cache → disk)
                │
         apply new content
                │
        ┌───────┴──────────┐
        ▼                  ▼
   write to disk      update cache
        │                  │
        │             embed new content
        │             store in VectorStorage
        │
   return diff (not full content)
```

### Semantic Search

```
query ──→ embed_query() ──→ query_vec (384D float32)
                                    │
                     ┌──────────────┴──────────────┐
                     │                              │
                     ▼                              ▼
              BM25 keyword search           HNSW vector search
              (FTS5 full-text)              (simplevecdb)
                     │                              │
                     └──────────┬───────────────────┘
                                │
                        Reciprocal Rank Fusion
                        (combine + deduplicate)
                                │
                           top-k results
                        (path, preview, score)
```

---

[← Back to README](../README.md)
