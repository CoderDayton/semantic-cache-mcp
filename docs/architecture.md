# Architecture

## Package Structure

```
src/semantic_cache_mcp/
├── config.py               # Constants and environment-variable configuration
├── types.py                # All shared data models (ReadResult, WriteResult, etc.)
├── cache/                  # Orchestration facade, coordinates all components
│   ├── __init__.py         # Public API re-exports
│   ├── store.py            # SemanticCache class: storage and metrics coordination
│   ├── read.py             # smart_read, batch_smart_read
│   ├── write.py            # smart_write, smart_edit, smart_batch_edit
│   ├── search.py           # semantic_search, glob_with_cache_status, compare_files
│   ├── metrics.py          # SessionMetrics: per-session and lifetime metric tracking
│   └── _helpers.py         # Internal utilities: _suppress_large_diff, formatter dispatch
├── server/                 # MCP interface, thin translation layer only
│   ├── __init__.py
│   ├── _mcp.py             # FastMCP app instance, lifespan, startup
│   ├── response.py         # Response formatting, TOOL_OUTPUT_MODE handling
│   └── tools/              # All 13 MCP tool definitions + _shielded_write helper
├── core/                   # Pure algorithms, stateless, zero I/O
│   ├── __init__.py         # Flat re-exports from all sub-packages
│   ├── chunking/           # Content-defined chunking (used for large file splitting)
│   │   ├── __init__.py
│   │   ├── _gear.py        # Serial HyperCDC (Gear hash rolling window)
│   │   └── _simd.py        # SIMD-accelerated parallel CDC

│   ├── hashing/            # BLAKE3/BLAKE2b content hashing, DeduplicateIndex

│   ├── text/               # Diff generation and semantic summarization
│   │   ├── __init__.py
│   │   ├── _diff.py        # generate_diff, diff_with_stats, compute_delta, diff_stats
│   │   └���─ _summarize.py   # summarize_semantic (TCRA-LLM based)
│   └���─ tokenizer.py        # BPE token counting (o200k_base)
└── storage/                # Persistence layer
    ├── __init__.py
    ├── docstore/           # ContentStorage: vendored SQLite + FTS5 keyword store
    └── sqlite.py           # SQLiteStorage: session metrics persistence only
```

## Design Principles

- **Separation of concerns.** `core/` is stateless pure algorithms, `storage/` is persistence only, `cache/` orchestrates, and `server/` translates between MCP and Python
- **Dependency injection.** Storage and config are passed explicitly, with no hidden globals
- **Facade pattern.** `cache/` exposes a clean API, and callers never touch `storage/` directly
- **Performance first in hot paths.** Hashing, chunking, and tokenization are optimized, and everything else favors clarity

---

## Storage (`storage/docstore/`)

### ContentStorage (vendored SQLite + FTS5)

The storage backend is a small SQLite store with FTS5, vendored into the package as `DocStore`. It holds text and metadata only:

- **FTS5 full-text search.** BM25 keyword ranking powers `search` and `grep`
- **Raw text storage.** File contents are stored as plain text in `page_content`, with no compression
- **Metadata filtering.** Path and chunk lookups go through JSON metadata columns

### Document Model

Files are stored as `Document` rows:

```
Small file (< 8KB):
  └── Single document: page_content=full_text

Large file (≥ 8KB):
  ├── Parent document: page_content="", is_parent=True
  └���─ Child documents (per CDC chunk):
      ├── page_content=chunk_text, chunk_index=0
      ├── page_content=chunk_text, chunk_index=1
      └── ...
```

Large files are split via HyperCDC (content-defined chunking) into multiple child documents. The parent holds file-level metadata; children hold raw text for content retrieval, search, and grep.

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
| `access_history` | `JSON` | Recent access timestamps, used by W-TinyLFU eviction |
| `is_parent` | `bool` | Parent document marker (large files only) |
| `preview` | `str` | First ~200 chars of file content, pre-stored at index time so search results don't re-slice chunked `page_content` at query time |

### W-TinyLFU Eviction

When `MAX_CACHE_ENTRIES` is exceeded, eviction uses W-TinyLFU, the policy Caffeine uses, which scores entries by both frequency and recency:

- Frequency comes from a small 4-bit Count-Min sketch that ages over time, so a file read many times is kept even when it was not the most recent
- Recency keeps a freshly read file from being dropped before it has a chance to prove useful
- The in-memory index bootstraps from each entry's `access_history` metadata on first need, so it survives a restart without a separate table
- This keeps a large one-time read (for example a wide grep seed) from pushing out the files you actually work on

### Session Metrics (`storage/sqlite.py`)

Separate SQLite database for token savings, cache hits/misses, and tool call counts. Not used for file content.

---

## Core Algorithms

### Chunking (`core/chunking/`)

#### Serial HyperCDC (`_gear.py`)

Content-defined chunking using a Gear hash rolling window:

- Pre-computed 256-entry gear table for O(1) byte lookups
- Rolling hash: `h = ((h << 1) + gear[byte]) & MASK_64`
- Boundary when `(h & mask) == 0` (normalized chunking)
- Skip-min: no boundary checks in first 2KB per chunk
- ~8KB average chunk size

Key property: similar files produce identical chunks even when bytes shift position, enabling efficient re-chunking on file changes.

#### SIMD Parallel CDC (`_simd.py`)

Faster chunking via CPU-core-level parallelism:

1. Divide content into N segments (one per available core)
2. Each worker finds CDC boundaries in its segment independently
3. Merge and de-duplicate overlapping boundaries at segment edges

`get_optimal_chunker(prefer_simd=True)` auto-selects; falls back gracefully to serial HyperCDC.

---

### Hashing (`core/hashing.py`)

BLAKE3 primary, BLAKE2b fallback. LRU-cached to avoid re-hashing identical data.

- **Content hash freshness.** Detects mtime changes with identical content (touch, git checkout)
- **Deduplication.** `DeduplicateIndex` for fingerprint-based dedup
- **Change detection.** Cached versus current content hash

---

### Tokenizer (`core/tokenizer.py`)

GPT-4o compatible (o200k_base) BPE tokenizer.

- **O(N log M)** priority-queue merge vs O(N²) naive
- Merge results memoized to skip repeated BPE sequences
- Single-byte tokens skip BPE entirely (fast path)
- Files > 50KB use sampling for O(1) estimate

---

### Text (`core/text/`)

#### `_diff.py`: Diff and Delta Compression

- Unified diffs via Python `difflib` with diff statistics (insertions, deletions, modifications)
- `diff_with_stats()`: computes the unified diff and its statistics from a single line-matcher pass, shared by the write/edit/batch-edit and compare paths that need both (the output matches calling `generate_diff` and `diff_stats` separately)
- Adaptive context width: files under 100 lines use 2 lines of diff context instead of 3, where the extra line is a large share of a small payload; larger files keep 3
- Delta compression: store only changed lines (10 to 100x smaller for small edits to large files)
- `_suppress_large_diff()` (in `cache/_helpers.py`): caps diff output at a token budget to prevent context overflow. A middle tier keeps the per-hunk `@@` headers (which regions changed and by how much) when a diff is over budget but has few hunks, so the caller can fetch specifics with a ranged `read`; beyond a hunk cap it falls back to a bare count summary

#### `_summarize.py`: Semantic Summarization

Based on TCRA-LLM (arXiv:2310.15556). Preserves structural integrity when files exceed the size budget:

**Algorithm:**
1. Split file at semantic boundaries (function/class definitions, paragraphs)
2. Score each segment:
   - **Position score**: U-shaped curve, highest at the start and end, lowest in the middle
   - **Density score**: unique token ratio + syntax character density + non-whitespace ratio
3. Greedily select highest-scoring segments that fit the budget
4. Always preserve the first segment (docstrings, imports, module header)
5. Reassemble selected segments in original order; `# ... [N lines omitted] ...` markers
   are emitted only when `SummarizationConfig.include_markers=True` (default `False`
   since 0.4.6, because markers added no LLM-visible value and consumed token budget)

**Result:** 50 to 80% token savings on large files versus simple truncation, while preserving the code skeleton and intent.

---

## Threading Model & Graceful Shutdown

### Thread Pools

The server runs a single asyncio event loop. Blocking operations are offloaded to thread pools:

| Executor | Workers | Used for |
|----------|---------|----------|
| **IO executor** | 1 | All `ContentStorage` reads and writes. Single-threaded so the one SQLite connection is never touched concurrently |
| **Default executor** | N (OS-dependent) | `summarize_semantic()` and other CPU-bound work |
| **Async subprocess** | n/a | `_format_file()` (ruff, prettier, etc.) |

Storage operations run on a dedicated single-thread executor so the single SQLite connection is only ever used from one thread, which keeps writes serialized and safe.

### Graceful Shutdown

On SIGTERM/SIGINT:

1. `cache.request_shutdown()` sets the `_shutting_down` flag, and new `begin_operation()` calls return `False`
2. The signal handler cancels all asyncio tasks, so `CancelledError` propagates and runs `finally` blocks
3. Write and edit tool handlers use `asyncio.shield()` via `_shielded_write()`, so the inner task completes even if the outer handler is cancelled
4. Lifespan `finally` calls `async_close()`:
   - Waits up to 8 seconds for in-flight operations to drain (`_drained` event)
   - Catches `CancelledError` during drain so close always proceeds
   - Persists session metrics, then closes ContentStorage, then the SQLite pool, then the IO executor
5. All `ContentStorage` async methods guard `_closed` and return safe defaults instead of crashing
6. Second signal forces `os._exit()` for hard termination

---

## Data Flow

### Read

```
Client ──→ smart_read(path, diff_mode=True)
                │
        1. astat(path)          ◀─ cheap stat() only
        2. cache.get(path)
                │
        ┌───────┴────────────────┐
        │                        │
        ▼                        ▼
  cached + mtime match    cached but mtime drifted, OR not cached
  ── FAST PATH ──         ── SLOW PATH ──
  return "unchanged"      aread_bytes(path) ──→ hash + decode
  no aread_bytes call             │
  (99% savings,           ┌───────┴────────────┐
   ~1 ms latency)         ▼                    ▼
                  hash matches cached    hash changed
                  update_mtime           generate diff
                  return "unchanged"     refresh_path + return diff
                  (99% savings)          (80-95% savings)
                                                │
                                        not cached at all
                                          read full
                                          + return full
```

**Why the fast path matters:** the unchanged case is the most common in interactive
sessions (the LLM re-reads files it already has). Skipping `aread_bytes`, `count_tokens`,
and the hash compute keeps single-file unchanged reads to **~1 ms** (vs ~2 ms when
the disk read was unconditional in pre-0.4.6 builds).

After context compression, use `diff_mode=False` to force full content.

### Batch Read

```
Client ──→ batch_smart_read(paths, diff_mode=True)
                │
         1. Gather all cache.get() in parallel (asyncio.gather)
                │
         2. Pre-fetch stat results in parallel (asyncio.gather)
                │
         3. smart_read() per file (reuses the gathered entries)
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
        │             store in ContentStorage
        │
   return diff (not full content)
```

### Search (BM25)

```
query ──→ semantic_search(cache, query, k, directory)
              │
       ┌──────┴──────────────────────────┐
       ▼                                 ▼
 in-session result cache hit?      MISS, BM25 retrieve
 (LRU keyed on q,k,dir)                  │
   YES → return immediately       BM25 keyword search (FTS5 full-text)
   (< 0.01 ms, 2,000×+ faster            │
    than a cold search)            deduplicate by path
                                         │
                                  store in result LRU
                                         │
                                  top-k results
                                  (path, preview, score)
```

The in-session result LRU lives on `SemanticCache._search_cache` (32-entry
`OrderedDict`). It is invalidated on every cache mutation: `put`, `clear`,
`delete_path`, and `update_mtime` all call `_bump_search_cache()`, which
clears the LRU. So callers never see a result that predates a write.

---

[← Back to README](../README.md)
