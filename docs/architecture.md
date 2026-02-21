# Architecture

## Package Structure

```
src/semantic_cache_mcp/
├── config.py               # Constants and environment-variable configuration
├── types.py                # All shared data models (ReadResult, WriteResult, etc.)
├── cache/                  # Orchestration facade — coordinates all components
│   ├── __init__.py         # Public API re-exports (including embed_batch)
│   ├── store.py            # SemanticCache class: get_or_build_lsh(), get_embeddings_batch(), _invalidate_lsh()
│   ├── read.py             # smart_read, batch_smart_read (batch pre-scan + embed)
│   ├── write.py            # smart_write, smart_edit, smart_batch_edit
│   ├── search.py           # semantic_search, find_similar_files, glob_with_cache_status, compare_files
│   └── _helpers.py         # Internal utilities: _suppress_large_diff, formatter dispatch
├── server/                 # MCP interface — thin translation layer only
│   ├── __init__.py
│   ├── _mcp.py             # FastMCP app instance, lifespan, startup warmup
│   ├── response.py         # Response formatting, TOOL_OUTPUT_MODE handling
│   └── tools.py            # All 11 MCP tool definitions
├── core/                   # Pure algorithms — stateless, zero I/O
│   ├── __init__.py         # Flat re-exports from all sub-packages (includes embed_batch)
│   ├── chunking/           # Content-defined chunking
│   │   ├── __init__.py
│   │   ├── _gear.py        # Serial HyperCDC (Gear hash rolling window)
│   │   └── _simd.py        # SIMD-accelerated parallel CDC
│   ├── compression.py      # Adaptive multi-codec (ZSTD/LZ4/Brotli)
│   ├── embeddings.py       # FastEmbed local embeddings (embed, embed_batch, embed_query)
│   ├── hashing.py          # BLAKE3/BLAKE2b, DeduplicateIndex, HierarchicalHasher
│   ├── similarity/         # Cosine similarity, LSH, and quantization
│   │   ├── __init__.py
│   │   ├── _cosine.py      # cosine_similarity, int8 quantization, top_k_from_quantized
│   │   ├── _lsh.py         # LSHIndex, LSHConfig, SimHash projections, serialize/deserialize
│   │   └── _quantization.py # Binary/ternary quantization
│   ├── text/               # Diff generation and semantic summarization
│   │   ├── __init__.py
│   │   ├── _diff.py        # generate_diff, compute_delta, diff_stats
│   │   └── _summarize.py   # summarize_semantic (TCRA-LLM based)
│   └── tokenizer.py        # BPE token counting (o200k_base)
└── storage/                # Persistence layer
    ├── __init__.py
    └── sqlite.py           # SQLiteStorage: content-addressable + LRU-K eviction + LSH persistence
```

## Design Principles

- **Separation of concerns** — `core/` is stateless pure algorithms; `storage/` is persistence only; `cache/` orchestrates; `server/` translates MCP ↔ Python
- **Dependency injection** — storage and config are passed explicitly; no hidden globals
- **Facade pattern** — `cache/` exposes a clean API; callers never touch `storage/` directly
- **Performance first in hot paths** — chunking, hashing, similarity, and tokenization are optimized; everything else prioritizes clarity
- **Type safety** — strict mypy, `@overload` for conditional return types, no `Any` in public APIs

---

## Core Algorithms

### Chunking (`core/chunking/`)

#### Serial HyperCDC — `_gear.py`

Content-defined chunking using a Gear hash rolling window.

- Pre-computed 256-entry gear table for O(1) byte lookups
- Rolling hash: `h = ((h << 1) + gear[byte]) & MASK_64`
- Boundary when `(h & mask) == 0` (normalized chunking)
- Skip-min: no boundary checks in first 2KB per chunk
- ~8KB average chunk size, ~13–14 MB/s throughput
- **2.7× faster than Rabin fingerprinting**

Key property: similar files produce identical chunks even when bytes shift position, enabling cross-file deduplication.

#### SIMD Parallel CDC — `_simd.py`

**5–7× faster** than serial via boundary-level CPU parallelism:

1. Divide content into N segments (one per available core)
2. Each worker finds CDC boundaries in its segment independently
3. Merge and de-duplicate overlapping boundaries at segment edges
4. Stitch final chunk list

- ~70–95 MB/s on 4-core systems (vs ~13–14 MB/s serial)
- `get_optimal_chunker(prefer_simd=True)` auto-selects; falls back gracefully to serial HyperCDC

---

### Compression (`core/compression.py`)

Adaptive multi-codec compression. Codec and level are chosen per chunk based on a fast entropy estimate:

| Entropy  | Codec   | Level | Write Throughput |
|----------|---------|-------|-----------------|
| > 7.5    | STORE   | —     | 29 GB/s         |
| > 6.5    | ZSTD    | 1     | 6.9 GB/s        |
| > 4.0    | ZSTD    | 3     | 5.3 GB/s        |
| ≤ 4.0    | ZSTD    | 9     | 4.8 GB/s        |

Key optimizations:
- Compressor instances are cached to avoid object creation per call
- Native ZSTD multi-threading for files > 4MB
- O(1) magic-byte detection skips already-compressed data
- Fallback chain: ZSTD → Brotli → LZ4 → STORE

---

### Hashing (`core/hashing.py`)

BLAKE3 primary, BLAKE2b fallback.

| Chunk Size | BLAKE2b   | BLAKE3    | Speedup |
|------------|-----------|-----------|---------|
| 256 B      | 543 MB/s  | 468 MB/s  | 0.86×   |
| 8 KB       | 1,094 MB/s | 4,195 MB/s | **3.8×** |
| 64 KB      | 1,150 MB/s | 5,597 MB/s | **4.9×** |

LRU cache: 16K chunks, 4K blocks, 2K content hashes — avoids re-hashing identical data.

Additional components:
- `DeduplicateIndex` — fast fingerprint-based dedup (~966K lookups/sec)
- `HierarchicalHasher` — chunk → block → content multi-level hashing
- `StreamingHasher` — incremental hashing for large files

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

Local text embeddings via FastEmbed with `BAAI/bge-small-en-v1.5`.

- **384-dimensional**, 33M parameters, 512 token context window
- Runs via ONNX Runtime — CPU by default, CUDA when available
- Singleton model instance — loaded once, reused across all calls
- Warmup pass at server startup for predictable first-request latency
- `"Represent this sentence for searching relevant passages:"` prefix for query embedding
- File-type semantic labels prepended to document content before embedding (e.g., `"python source file: ..."`)
- Model stored in `~/.cache/semantic-cache-mcp/models/`
- No API keys; fully offline after initial download (~500MB)

#### Batch Embedding

`embed_batch(texts)` amortizes ONNX Runtime overhead across N texts in a single model call — critical for `batch_smart_read` where multiple files need embedding on first cache miss.

**Pre-scan flow in `batch_smart_read`:**
1. Before reading any file, scan all requested paths for new or changed files (mtime check)
2. Collect their content and apply the same file-type label prefix used by single-file embedding
3. Call `embed_batch()` once with all texts — a single ONNX inference pass
4. Store prefetched embeddings; `smart_read` picks them up instead of calling the model individually

This reduces N model calls (one per new file) to exactly 1, regardless of batch size.

`SemanticCache.get_embeddings_batch(path_content_pairs)` provides the same optimization at the cache facade level, applying file-type labels automatically.

---

### Similarity (`core/similarity/`)

#### `_cosine.py` — int8 Quantized Cosine Similarity

int8 quantization stores each 384-dimensional embedding as **388 bytes** instead of ~6KB (4× in raw bytes; 22× vs Python `list[float]`) with < 0.3% accuracy loss:

1. Scale 384-dim float32 vector to `[-128, 127]` int8 range
2. Store as `struct.pack('<f', scale) + quantized.tobytes()` — 4 bytes scale + 384 bytes int8 = 388 bytes total
3. At query time: load blob, dequantize, compute cosine via SIMD batch matrix op

Performance (1000 vectors, 384D, bge-small-en-v1.5):

| Method                    | Time    | Speedup |
|---------------------------|---------|---------|
| Per-vector loop           | ~5 ms   | 1×      |
| Batch matrix (float32)    | ~0.8 ms | 6×      |
| Batch matrix (int8)       | ~0.6 ms | 8×      |
| Quantized + dim pruning   | ~0.35 ms | 14×    |

#### `_lsh.py` — LSH Approximate Search (Persisted)

SimHash-based Locality-Sensitive Hashing for fast approximate nearest-neighbor search.

**How it works:**
1. Project each vector onto random hyperplanes → binary signature (SimHash)
2. Index signatures in multiple hash tables (band decomposition for recall)
3. At query time: compute query signature, retrieve candidates from all tables, run exact cosine only on candidates

**Persistence:** The LSH index is persisted to SQLite rather than rebuilt from scratch on every search. After building, it is serialized via `serialize_lsh_index()` and stored as a BLOB in the `lsh_index` table (SQLite singleton, `id=1`). On subsequent search calls or server restarts, `get_or_build_lsh()` loads the persisted index rather than rebuilding from embedding blobs.

**Load order in `SemanticCache.get_or_build_lsh()`:**
1. In-memory `_lsh_index` — if present and size matches, return immediately
2. SQLite `lsh_index` row — if `blob_count` matches current file count, deserialize and return
3. Fresh build — compute from all embedding BLOBs, persist to SQLite

**Invalidation:** Any `put()` or `clear()` call invokes `_invalidate_lsh()`, which clears both the in-memory reference and the persisted SQLite row. The index is rebuilt lazily on the next `search` or `similar` call.

`_LSH_THRESHOLD = 100` still applies:
- n < 100: exhaustive `top_k_from_quantized()` (already fast at this scale)
- n ≥ 100: LSH candidates (4× k), then exact cosine re-ranking on candidates

`SemanticCache` details:
- `__slots__ = ("_storage", "_lsh_index")`
- `get_or_build_lsh(blobs, dim)` — returns consistent LSH index, loading from SQLite if available
- `get_embeddings_batch(path_content_pairs)` — batch embed with file-type labels applied automatically
- `_invalidate_lsh()` — clears in-memory index and SQLite row on any cache mutation

- `@overload` decorators provide precise return types: `Literal[True]` → `list[tuple[int, float]]`, `Literal[False]` → `list[int]`

#### `_quantization.py` — Binary and Ternary Quantization

For extreme compression or pre-filtering in two-stage search:

| Method   | Compression | Accuracy | Primary Use         |
|----------|-------------|----------|---------------------|
| int8     | 4×          | 99.7%    | Production similarity |
| Ternary  | 16×         | 85–90%   | Approximate pre-filter |
| Binary   | 32×         | 80–85%   | Coarse pre-filter     |

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

## Storage (`storage/sqlite.py`)

Content-addressable chunk store — conceptually similar to Git's object store.

### Schema

```sql
CREATE TABLE chunks (
    hash     TEXT PRIMARY KEY,   -- BLAKE3 hex digest
    data     BLOB NOT NULL,      -- compressed chunk bytes
    size     INTEGER NOT NULL,   -- uncompressed size
    ref_count INTEGER DEFAULT 1  -- number of files referencing this chunk
) WITHOUT ROWID;

CREATE TABLE files (
    path          TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    chunk_hashes  TEXT NOT NULL,  -- JSON array of chunk hashes
    mtime         REAL NOT NULL,
    tokens        INTEGER NOT NULL,
    embedding     BLOB,           -- int8 quantized: 4B scale + 384×int8 = 388 bytes
    created_at    REAL NOT NULL,
    access_history TEXT NOT NULL  -- JSON array of access timestamps (LRU-K)
) WITHOUT ROWID;

CREATE INDEX idx_embedding ON files(embedding) WHERE embedding IS NOT NULL;

CREATE TABLE lsh_index (
    id         INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row
    data       BLOB NOT NULL,                        -- serialize_lsh_index() output
    blob_count INTEGER NOT NULL DEFAULT 0,           -- file count when built
    updated_at REAL NOT NULL
) WITHOUT ROWID;

CREATE TABLE session_metrics (
    session_id       TEXT PRIMARY KEY,               -- UUID per server session
    started_at       REAL NOT NULL,
    ended_at         REAL,                           -- NULL until persist()
    tokens_saved     INTEGER NOT NULL DEFAULT 0,
    tokens_original  INTEGER NOT NULL DEFAULT 0,
    tokens_returned  INTEGER NOT NULL DEFAULT 0,
    cache_hits       INTEGER NOT NULL DEFAULT 0,
    cache_misses     INTEGER NOT NULL DEFAULT 0,
    files_read       INTEGER NOT NULL DEFAULT 0,
    files_written    INTEGER NOT NULL DEFAULT 0,
    files_edited     INTEGER NOT NULL DEFAULT 0,
    diffs_served     INTEGER NOT NULL DEFAULT 0,
    tool_calls_json  TEXT NOT NULL DEFAULT '{}'      -- JSON: {"read": 5, "edit": 2, ...}
) WITHOUT ROWID;
```

> **Note:** `clear()` deletes `files`, `chunks`, and `lsh_index` — `session_metrics` is preserved so lifetime aggregates survive cache resets.

### LRU-K Eviction (K=2)

Uses the **K-th most recent** access time rather than the most recent, making eviction decisions robust against sequential scans:

- A file accessed only once has no second access time → evicted first
- A file accessed regularly has a recent second access time → retained
- This correctly handles large one-time reads (e.g., grepping) without polluting the cache

### SQLite Optimizations

```sql
PRAGMA journal_mode = WAL;       -- concurrent reads + writes
PRAGMA synchronous   = NORMAL;   -- safe in WAL, 2-3× faster commits
PRAGMA cache_size    = -64000;   -- 64MB page cache
PRAGMA temp_store    = MEMORY;   -- avoid disk I/O for temp tables
PRAGMA mmap_size     = 268435456; -- 256MB memory-mapped I/O
```

- **WITHOUT ROWID** tables: 20–30% space savings for text-keyed tables
- **Partial index** on `embedding IS NOT NULL`: faster similarity scans skip null rows
- **Connection pool** (queue-based, 5 connections): eliminates ~5–10ms connection overhead per request
- **Batch operations**: `executemany` for inserts; `IN` clause for multi-path fetches/deletes
- **WAL checkpoint** (`TRUNCATE` mode) after commits for read-after-write consistency

---

## Data Flow

### Read

```
Client ──→ smart_read(path, diff_mode=True)
                │
                ▼
         cache.get(path)
                │
        ┌───────┴────────────┬──────────────────┐
        │                    │                  │
        ▼                    ▼                  ▼
  mtime match          hash changed          not found
  "unchanged"          compute diff          read disk
  (99% savings)        (80-95% savings)      store + return
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
        │             _invalidate_lsh()
        │             (next search rebuilds index)
        │
   return diff (not full content)
```

### Semantic Search

```
query ──→ embed_query() ──→ query_vec (384D float32)
                                    │
                   ┌────────────────┴────────────────┐
                   │ n < 100 files                   │ n ≥ 100 files
                   │ exhaustive cosine scan           │ get_or_build_lsh()
                   │ top_k_from_quantized()           │   ├─ in-memory hit → reuse
                   │                                  │   ├─ SQLite hit → deserialize
                   │                                  │   └─ miss → build + persist
                   │                                  │ LSH candidates (4×k) → exact re-rank
                   └────────────────┬────────────────┘
                                    │
                               top-k results
                            (path, similarity score)
```

---

[← Back to README](../README.md)
