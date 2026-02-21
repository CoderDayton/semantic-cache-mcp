# Performance

## Token Savings Benchmark

Measured end-to-end on this project's own source files (30 `.py` files, ~136K tokens total). Benchmark script: [`benchmarks/benchmark_token_savings.py`](../benchmarks/benchmark_token_savings.py).

| Phase | Scenario | Tokens returned | Original | Savings |
|-------|----------|----------------:|---------:|--------:|
| 1. Cold read | First read, no cache | 135,643 | 135,643 | 0.0% |
| 2. Unchanged re-read | Same files, no modifications | 1,247 | 135,643 | 99.1% |
| 3. Small edits | ~5% of lines changed in 30% of files | 2,383 | 135,736 | 98.2% |
| 4. Batch read | All files via `batch_smart_read` | 1,247 | 135,736 | 99.1% |

**Overall (phases 2–4): 98.8% token reduction.**

Run it yourself:

```bash
uv run python benchmarks/benchmark_token_savings.py
uv run pytest tests/test_benchmark_token_savings.py -v  # CI-verifiable assertion: savings ≥ 80%
```

### Token Reduction by Strategy

| Strategy          | Savings | Trigger                           |
|-------------------|---------|-----------------------------------|
| Unchanged (mtime) | ~99%    | File mtime matches cached entry   |
| Diff (changed)    | 80–95%  | File modified since last cache    |
| Semantic match    | 70–90%  | Similar file found in cache       |
| Summarized        | 50–80%  | File exceeds `MAX_CONTENT_SIZE`   |

---

## Optimization Summary

| Technique                          | Benefit                          | Details                                                        |
|------------------------------------|----------------------------------|----------------------------------------------------------------|
| **SIMD parallel CDC chunking**     | 5–7× faster                      | Boundary detection across CPU cores; ~70–95 MB/s on 4-core    |
| **Serial HyperCDC (Gear hash)**    | 2.7× vs Rabin                    | Skip-min optimization; ~13–14 MB/s                            |
| **int8 quantized embeddings**      | 22× storage reduction            | 388 bytes vs ~12KB per 384D vector; <0.3% accuracy loss       |
| **Persistent LSH index**           | Eliminates per-search rebuild    | Serialized to SQLite; reloaded on next search or restart; O(N·dim) build amortized across all searches until next write |
| **Batch embedding**                | N ONNX calls → 1                 | Pre-scan in `batch_smart_read` collects all new/changed files; single `model.embed()` call; unchanged files skipped entirely |
| **LSH approximate search**         | O(1) candidate retrieval         | SimHash index for caches ≥ 100 files; persisted between sessions |
| **BLAKE3 hashing**                 | 3.8–4.9× vs BLAKE2b              | Hardware-accelerated on 8KB+ chunks                           |
| **LRU hash cache**                 | Avoid re-hashing                 | 16K chunks, 4K blocks, 2K content hashes                      |
| **Adaptive ZSTD compression**      | Entropy-matched codec            | STORE/ZSTD-1/ZSTD-3/ZSTD-9 selected per chunk                |
| **Cached ZSTD compressors**        | 2× faster compression            | Avoid object creation overhead per call                       |
| **O(1) magic-byte detection**      | 15–20× faster                    | First-byte table skips already-compressed data                |
| **Native ZSTD multi-threading**    | 19× faster large files           | Uses ZSTD threads, not Python threads; triggered at >4MB      |
| **O(N log M) BPE tokenizer**       | vs O(N²) naive                   | Priority-queue merge with memoization                         |
| **Batch matrix similarity**        | 6–14× vs per-vector loop         | Single SIMD operation over all cached embeddings              |
| **frombytes() array conversion**   | ~100× faster                     | Direct memcpy vs tolist() for embedding deserialization        |
| **Batch SQL IN clause**            | N→1 DB round-trips               | Single `SELECT ... WHERE IN` replaces per-file lookups        |
| **Pre-computed cache set in batch**| 2× fewer lookups                 | Eliminates double cache.get() in batch_smart_read sort        |
| **Partial index on embedding**     | Faster similarity scans          | Indexes only rows with `embedding IS NOT NULL`                |
| **WITHOUT ROWID tables**           | 20–30% space savings             | Text primary keys stored directly in B-tree                   |
| **SQLite WAL mode**                | Concurrent reads                 | Reads never block; multiple readers + one writer              |
| **Connection pooling**             | Eliminate ~5–10ms overhead       | Queue-based pool, 5 connections; reused across requests       |
| **array.array for embeddings**     | ~50% less memory                 | Typed arrays vs Python lists                                  |
| **`__slots__` on dataclasses**     | Eliminate `__dict__`             | Memory-efficient data models                                  |
| **Generator expressions**          | Avoid intermediate lists         | Used in hot paths (chunk iteration, similarity ranking)       |
| **Semantic summarization**         | 50–80% savings on large files    | Segment scoring + greedy selection preserves structure        |

---

## Chunking

### SIMD Parallel CDC

**5–7× speedup** via CPU-core-level parallelism:

1. Divide content into N segments (one per core)
2. Each worker independently finds CDC boundaries in its segment
3. Merge overlapping boundaries at segment edges

```python
# Hot path
boundaries = _parallel_cdc_boundaries(content, num_threads=4)
chunks = [content[start:end] for start, end in boundaries]
```

**Benchmarks (10MB file, 4 cores):**

| Method              | Throughput    | Time     |
|---------------------|---------------|----------|
| Serial HyperCDC     | ~13–14 MB/s   | ~750 ms  |
| SIMD Parallel CDC   | ~70–95 MB/s   | ~110 ms  |
| **Speedup**         |               | **5–7×** |

Falls back to serial HyperCDC automatically if SIMD is unavailable.

### Serial HyperCDC (Gear Hash)

**2.7× faster than Rabin fingerprinting:**

```python
# Pre-computed gear table; skip-min skips first 2KB per chunk
h = ((h << 1) + gear[content[i]]) & 0xFFFFFFFFFFFFFFFF
if (h & mask) == 0:
    # boundary found
```

- ~8KB average chunk size (configurable via mask bits)
- ~13–14 MB/s throughput
- Content-defined boundaries → similar files share chunks even when bytes shift

---

## Hashing

BLAKE3 vs BLAKE2b (no cache hits):

| Chunk Size | BLAKE2b    | BLAKE3     | Speedup  |
|------------|------------|------------|----------|
| 256 B      | 543 MB/s   | 468 MB/s   | 0.86×    |
| 8 KB       | 1,094 MB/s | 4,195 MB/s | **3.8×** |
| 64 KB      | 1,150 MB/s | 5,597 MB/s | **4.9×** |
| 1 MB (stream) | 1,140 MB/s | 5,121 MB/s | **4.5×** |

BLAKE3 dominates on typical CDC chunk sizes (8KB+). The slight regression at 256B is due to the larger digest size (32 vs 20 bytes) — irrelevant in practice since chunks are rarely that small.

`DeduplicateIndex`: ~966K lookups/sec using thread-safe binary fingerprints.

---

## Compression

Adaptive codec selection based on fast entropy estimation:

```python
# Ultra-fast entropy approximation (80× faster than Shannon entropy)
def _fast_entropy(data: bytes) -> float:
    unique = len(set(data[:256]))
    return (unique.bit_length() - 1) + (unique & (unique - 1) > 0) * 0.5
```

| Entropy  | Codec    | Level | Write Throughput |
|----------|----------|-------|-----------------|
| > 7.5    | STORE    | —     | 29 GB/s         |
| > 6.5    | ZSTD     | 1     | 6.9 GB/s        |
| > 4.0    | ZSTD     | 3     | 5.3 GB/s        |
| ≤ 4.0    | ZSTD     | 9     | 4.8 GB/s        |

Source code (low entropy) uses ZSTD-9. Already-compressed binary files use STORE to avoid wasted CPU.

---

## Embeddings and Similarity

### Model

`BAAI/bge-small-en-v1.5` — 384-dimensional, 33M parameters, 512 token context. Runs locally via FastEmbed (ONNX Runtime). Device selection is controlled by the `EMBEDDING_DEVICE` environment variable: `cpu` (default), `cuda` (GPU), or `auto` (detect).

### int8 Quantization

384-dimensional float32 embedding → int8 quantized:

```
float32: 384 dims × 4 bytes = 1,536 bytes
int8:    4 bytes (scale) + 384 bytes (int8) = 388 bytes
Reduction: ~4× in raw bytes; ~22× vs Python list[float] (~12KB with CPython overhead)
```

Accuracy: <0.3% mean error vs exact float32 cosine similarity.

### Batch Matrix Operations

```
1000 vectors, 384D (BAAI/bge-small-en-v1.5):

Per-vector loop:               ~5.0 ms  (1×)
Batch matrix (float32):        ~0.8 ms  (6×)
Batch matrix (int8 quantized): ~0.6 ms  (8×)
Quantized + dimension pruning: ~0.35 ms (14×)
```

### Batch Embedding (`embed_batch`)

`batch_smart_read` pre-scans all requested paths before the main read loop:

1. Identify new/changed files via mtime check (unchanged files need no embedding)
2. Read content from disk and apply file-type semantic labels
3. Call `embed_batch()` once — a single `model.embed(texts)` ONNX inference call
4. Distribute results into `smart_read` via `_embedding=` parameter

For a batch of 20 files where 12 are new: 12 model calls → 1. Unchanged files pay zero embedding cost.

### LSH Approximate Search (Persisted)

For caches ≥ 100 files, `_top_k_with_lsh()` uses a persisted SimHash index:

1. Project each vector onto random hyperplanes (64 bits, 4 tables, band_size=8)
2. Index persisted to SQLite after first build (`serialize_lsh_index()`)
3. Query retrieves candidates in O(1) per table
4. Exact cosine re-ranking on candidates (4× k)

**Persistence benefit:** Without this, every `search` or `similar` call rebuilds the index from scratch (O(N·dim) to dequantize and hash all embeddings). With persistence, the rebuild cost is paid at most once per write session. For read-heavy workflows (read 100 files, search repeatedly), this eliminates the rebuild entirely between searches.

**Invalidation:** Any `put()` or `clear()` clears both the in-memory `_lsh_index` and the SQLite row. The index rebuilds lazily on the next search call.

---

## Memory Efficiency

### Embedding Storage

| Format                | Size (384D) | Notes                           |
|-----------------------|-------------|---------------------------------|
| `list[float]`         | ~12 KB      | CPython list + float overhead   |
| `array.array('f')`    | ~1.5 KB     | 8× reduction                   |
| `int8 quantized blob` | ~388 B      | 31× reduction, <0.3% error     |

```python
# Before: Python list (one float = ~32 bytes in CPython including GC overhead)
embedding = [0.1, 0.2, 0.3]  # 3 floats ≈ 96 bytes in a list

# After: typed array (4 bytes/float, no per-element overhead)
embedding = array.array('f', [0.1, 0.2, 0.3])  # 12 bytes
```

### Dataclasses with `__slots__`

```python
@dataclass(slots=True)
class CacheEntry:
    path: str
    content_hash: str
    tokens: int
    # No __dict__ per instance — significant savings at scale
```

---

## SQLite

### PRAGMA Configuration

```sql
PRAGMA journal_mode  = WAL;         -- concurrent reads, no reader locks
PRAGMA synchronous   = NORMAL;      -- safe in WAL, 2-3× faster writes
PRAGMA cache_size    = -64000;      -- 64MB page cache (vs 2MB default)
PRAGMA temp_store    = MEMORY;      -- avoid temp disk I/O
PRAGMA mmap_size     = 268435456;   -- 256MB memory-mapped I/O
```

### Schema Optimizations

```sql
-- WITHOUT ROWID: 20-30% space savings for text-keyed tables
CREATE TABLE chunks (...) WITHOUT ROWID;
CREATE TABLE files  (...) WITHOUT ROWID;
CREATE TABLE lsh_index (...) WITHOUT ROWID;  -- singleton LSH persistence

-- Partial index: only index rows with embeddings
CREATE INDEX idx_embedding ON files(embedding) WHERE embedding IS NOT NULL;
```

### Query Patterns

```python
# Batch fetch (1 round-trip vs N)
placeholders = ",".join("?" * len(hashes))
conn.execute(f"SELECT hash, data FROM chunks WHERE hash IN ({placeholders})", hashes)

# Batch insert
conn.executemany("INSERT OR IGNORE INTO chunks VALUES (?, ?, ?, ?)", chunks_data)

# Batch delete
conn.execute(f"DELETE FROM files WHERE path IN ({placeholders})", paths)
```

---

## Semantic Summarization

Research-based content selection (TCRA-LLM, arXiv:2310.15556) for large files:

**Algorithm:**
1. Split at semantic boundaries (function/class definitions, paragraphs)
2. Score each segment:
   - **Position**: U-shaped curve — high at start and end, low in middle
   - **Density**: unique token ratio + syntax density + non-whitespace ratio
   - **Diversity**: cosine penalty for similarity to already-selected segments
3. Greedily select highest-scoring segments that fit the budget
4. Always preserve the first segment (module docstring, imports)
5. Reassemble with omission markers

**Output example (10KB budget, 25KB file):**

```python
"""Module docstring."""   # always preserved

def func_0():
    pass

# ... [2067 lines omitted] ...

def func_999():
    pass
```

**Result:** 50–80% token savings vs simple truncation, while preserving code structure and intent.

---

## Profiling

```bash
# CPU profiling
python -m cProfile -o profile.prof -m semantic_cache_mcp
python -m pstats profile.prof

# Memory profiling
pip install memory-profiler
python -m memory_profiler your_script.py

# Line-level profiling (hot paths)
pip install line-profiler
kernprof -l -v your_script.py
```

---

[← Back to README](../README.md)
