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
│   └── tools/              # All 12 MCP tool definitions + _shielded_write helper
├── core/                   # Pure algorithms — stateless, zero I/O
│   ├── __init__.py         # Flat re-exports from all sub-packages
│   ├── chunking/           # Content-defined chunking (used for large file splitting)
│   │   ├── __init__.py
│   │   ├── _gear.py        # Serial HyperCDC (Gear hash rolling window)
│   │   └── _simd.py        # SIMD-accelerated parallel CDC
│   ├── embeddings/         # FastEmbed local embeddings (embed, embed_batch, embed_query)
│   ├── hashing/            # BLAKE3/BLAKE2b content hashing, DeduplicateIndex
│   ├── similarity/         # Cosine similarity utilities
│   │   ├── __init__.py
│   │   └���─ _cosine.py      # cosine_similarity, batch operations
│   ├── text/               # Diff generation and semantic summarization
│   │   ├── __init__.py
│   │   ├── _diff.py        # generate_diff, compute_delta, diff_stats
│   │   └── _summarize.py   # summarize_semantic (TCRA-LLM based)
│   └── tokenizer.py        # BPE token counting (o200k_base)
└── storage/                # Persistence layer
    ├── __init__.py
    ├── vector/             # VectorStorage: simplevecdb with HNSW + FTS5
    └── sqlite.py           # SQLiteStorage: session metrics persistence only
```

## Design Principles

- **Separation of concerns** — `core/` is stateless pure algorithms; `storage/` is persistence only; `cache/` orchestrates; `server/` translates MCP ↔ Python
- **Dependency injection** — storage and config passed explicitly; no hidden globals
- **Facade pattern** — `cache/` exposes a clean API; callers never touch `storage/` directly
- **Performance first in hot paths** — embedding, hashing, similarity, and tokenization are optimized; everything else prioritizes clarity

---

## Storage (`storage/vector.py`)

### VectorStorage (simplevecdb)

The primary storage backend uses [SimpleVecDB](https://github.com/CoderDayton/SimpleVecDB):

- **HNSW index** — O(log N) approximate nearest neighbor search for semantic similarity
- **FTS5 full-text search** — BM25 keyword search for grep and hybrid queries
- **Hybrid search** — Reciprocal Rank Fusion (RRF) combines HNSW + BM25 results
- **Raw text storage** — File contents stored as plain text in `page_content` (no compression)
- **INT8 quantization** — 4× smaller vectors in the HNSW index with negligible quality loss

### Document Model

Files are stored as simplevecdb `Document` objects:

```
Small file (< 8KB):
  └── Single document: page_content=full_text, embedding=file_vector

Large file (≥ 8KB):
  ├── Parent document: page_content="", embedding=file_vector, is_parent=True
  └���─ Child documents (per CDC chunk):
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

Separate SQLite database for token savings, cache hits/misses, and tool call counts. Not used for file content.

---

## Core Algorithms

### Chunking (`core/chunking/`)

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

BLAKE3 primary, BLAKE2b fallback. LRU-cached to avoid re-hashing identical data.

- **Content hash freshness** — detects mtime changes with identical content (touch, git checkout)
- **Deduplication** — `DeduplicateIndex` for fingerprint-based dedup
- **Change detection** — cached vs current content hash

---

### Tokenizer (`core/tokenizer.py`)

GPT-4o compatible (o200k_base) BPE tokenizer.

- **O(N log M)** priority-queue merge vs O(N²) naive
- Merge results memoized to skip repeated BPE sequences
- Single-byte tokens skip BPE entirely (fast path)
- Files > 50KB use sampling for O(1) estimate

---

### Embeddings (`core/embeddings.py`)

Local text embeddings via FastEmbed (configurable model, default `BAAI/bge-small-en-v1.5`).

- **384-dimensional** (default), 33M parameters, 512 token context window
- Runs via ONNX Runtime — `cpu` by default, `cuda` when configured
- Singleton model instance — loaded once, reused across all calls
- Dedicated single-thread `ThreadPoolExecutor` — ONNX serializes inference internally, so embedding calls don't starve the default thread pool
- Warmup pass at server startup (in `asyncio.to_thread`) for predictable first-request latency
- `"Represent this sentence for searching relevant passages:"` prefix for query embedding
- File-type semantic labels prepended to document content before embedding (e.g., `"python source file: ..."`)

#### Batch Embedding

`embed_batch(texts)` amortizes ONNX Runtime overhead across N texts in a single model call.

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

## Threading Model & Graceful Shutdown

### Thread Pools

The server runs a single asyncio event loop. Blocking operations are offloaded to thread pools:

| Executor | Workers | Used for |
|----------|---------|----------|
| **Embed executor** | 1 | ONNX `model.embed()` — serialized because ONNX Runtime holds a session lock |
| **Default executor** | N (OS-dependent) | `VectorStorage` catalog ops, `sync_coll.save()`, `summarize_semantic()` |
| **Async subprocess** | — | `_format_file()` (ruff, prettier, etc.) |

Embedding uses a dedicated single-thread executor so concurrent ONNX calls don't consume default pool slots. Storage I/O runs on the default pool and is never blocked by embedding work.

### Graceful Shutdown

On SIGTERM/SIGINT:

1. `cache.request_shutdown()` — sets `_shutting_down` flag, new `begin_operation()` calls return `False`
2. Signal handler cancels all asyncio tasks — `CancelledError` propagates, running `finally` blocks
3. Write/edit tool handlers use `asyncio.shield()` via `_shielded_write()` — the inner task completes even if the outer handler is cancelled
4. Lifespan `finally` calls `async_close()`:
   - Waits up to 8 seconds for in-flight operations to drain (`_drained` event)
   - Catches `CancelledError` during drain so close always proceeds
   - Persists session metrics → closes VectorStorage → closes SQLite pool → shuts down embed executor
5. All `VectorStorage` async methods guard `_closed` — return safe defaults instead of crashing
6. Second signal forces `os._exit()` for hard termination

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

After context compression, use `diff_mode=False` to force full content.

### Batch Read

```
Client ──→ batch_smart_read(paths, diff_mode=True)
                │
         1. Gather all cache.get() in parallel (asyncio.gather)
                │
         2. Pre-scan: filter new/changed paths (mtime check, reuses gathered entries)
                │
         3. embed_batch([...all new/changed texts...])
            ── single ONNX inference call ──
                │
         4. smart_read() per file (embedding prefetched, no model call)
                │
         5. Return BatchReadResult with per-file status
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
