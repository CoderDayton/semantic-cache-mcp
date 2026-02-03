# Performance

## Token Reduction

| Strategy | Savings | When Used |
|----------|---------|-----------|
| Unchanged file | 99% | File mtime matches cache |
| Diff (changed) | 80-95% | File modified since cache |
| Semantic match | 70-90% | Similar file in cache |
| Truncation | 50-80% | Large files > 100KB |

---

## Optimization Techniques

| Technique | Benefit | Implementation |
|-----------|---------|----------------|
| **HyperCDC chunking** | 2.7x faster than Rabin | Gear hash with skip-min optimization |
| **Cached ZSTD compressors** | 2x faster compression | Avoid object creation per call |
| **O(1) magic detection** | 15-20x faster detection | First-byte lookup table |
| **Native ZSTD threading** | 19x faster large files | Use ZSTD threads, not Python |
| **BLAKE3 hashing** | 3.8-4.9x faster than BLAKE2b | Hardware-accelerated, parallelizable |
| **LRU cache for hashing** | Skip repeated hashing | 16K chunks, 4K blocks, 2K content |
| **Priority queue tokenizer** | O(N log M) vs O(N²) | Heap-based BPE merging |
| **SQLite WAL mode** | Better read concurrency | No reader locks, parallel access |
| **Connection pooling** | Eliminate connection overhead | Queue-based pool, 5-10ms savings per query |
| **WAL checkpointing** | Read-after-write consistency | TRUNCATE mode after commits |
| **Partial indexes** | Faster filtered scans | Index on `embedding IS NOT NULL` |
| **Batch SQL operations** | 2-5x faster updates | `executemany` + `IN` clause |
| **WITHOUT ROWID tables** | 20-30% space savings | Text primary keys optimized |
| **array.array for embeddings** | ~50% less memory | Typed arrays vs Python lists |
| **Generator expressions** | Avoid intermediate lists | Used in hot paths |
| **`__slots__` on dataclasses** | Eliminate `__dict__` | Memory-efficient models |

---

## Memory Efficiency

### Embeddings

```python
# Before: list[float] — 72 bytes for 3 floats
embedding = [0.1, 0.2, 0.3]

# After: array.array('f') — 12 bytes for 3 floats
embedding = array.array('f', [0.1, 0.2, 0.3])  # 6x reduction
```

For a 1536-dimension embedding:
- `list[float]`: ~12KB per embedding
- `array.array('f')`: ~6KB per embedding (50% reduction)

### Dataclasses with `__slots__`

```python
@dataclass(slots=True)
class CacheEntry:
    path: str
    content_hash: str
    # ... no __dict__ overhead
```

---

## Chunking Performance

### SIMD-Accelerated Parallel CDC

**5-7x speedup** over serial HyperCDC through boundary-level parallelism:

```python
# Hot path - parallel boundary detection across segments
boundaries = _parallel_find_boundaries(content, num_threads=4)
chunks = [content[start:end] for start, end in boundaries]
```

**Benchmarks** (10MB file, 4 cores):
- Serial HyperCDC: ~13-14 MB/s (~750ms)
- Parallel CDC: ~70-95 MB/s (~105-140ms)
- Speedup: 5-7x on multi-core systems

**Fallback:** Gracefully falls back to serial HyperCDC if SIMD unavailable.

### Serial HyperCDC (Gear Hash)

2.7x faster than Rabin fingerprinting:

```python
# Hot path - pre-computed gear table, skip min_size bytes
h = ((h << 1) + gear[content[i]]) & 0xFFFFFFFFFFFFFFFF
if (h & mask) == 0:
    # Boundary found
```

**Benchmarks** (representative, not guaranteed):
- Throughput: ~13-14 MB/s
- Average chunk size: ~8KB (configurable via mask bits)
- 1MB file: ~75ms chunking time

---

## SQLite Optimizations

### Batch Inserts

```python
# Instead of individual inserts
cursor.executemany(
    "INSERT OR IGNORE INTO chunks VALUES (?, ?, ?, ?)",
    chunks_data
)
```

### Efficient Lookups

```python
# Use IN clause for batch fetches
placeholders = ",".join("?" * len(hashes))
cursor.execute(f"SELECT * FROM chunks WHERE hash IN ({placeholders})", hashes)
```

---

## Compression Strategy

Multi-codec adaptive compression with ZSTD (primary), LZ4, and Brotli fallbacks:

```python
# Ultra-fast entropy approximation (80x faster than Shannon)
def _fast_entropy(data: bytes) -> float:
    unique = len(set(data[:256]))
    return (unique.bit_length() - 1) + (unique & (unique - 1) > 0) * 0.5
```

| Entropy | Codec | Level | Throughput |
|---------|-------|-------|------------|
| > 7.5 | STORE | - | 29 GB/s |
| > 6.5 | ZSTD | 1 | 6.9 GB/s |
| > 4.0 | ZSTD | 3 | 5.3 GB/s |
| <= 4.0 | ZSTD | 9 | 4.8 GB/s |

**Features:**
- Cached ZSTD compressors (avoid object creation)
- Native multi-threading for files > 4MB
- O(1) magic-byte detection for pre-compressed data
- Automatic codec fallback (ZSTD → Brotli → LZ4 → STORE)

---

## Hashing Performance

BLAKE3 vs BLAKE2b throughput (no cache):

| Chunk Size | BLAKE2b | BLAKE3 | Speedup |
|------------|---------|--------|---------|
| 256B | 543 MB/s | 468 MB/s | 0.86x |
| 8KB | 1,094 MB/s | 4,195 MB/s | **3.83x** |
| 64KB | 1,150 MB/s | 5,597 MB/s | **4.87x** |

**Streaming (1MB file):** 1,140 → 5,121 MB/s (**4.5x faster**)

BLAKE3 excels on typical CDC chunk sizes (8KB+). The slight overhead on tiny chunks (<256B) is due to larger digest size (32 bytes vs 20 bytes).

**DeduplicateIndex:** ~966K lookups/sec with thread-safe binary fingerprints.

---

## SQLite Query Optimization

**WAL Mode Benefits:**
- No reader locks: Reads never block reads or writes
- Better concurrency: Multiple readers + one writer simultaneously
- Faster writes: Asynchronous checkpointing

**PRAGMA Settings:**
```sql
journal_mode = WAL        -- Write-Ahead Logging
synchronous = NORMAL      -- Safe in WAL mode, 2-3x faster writes
cache_size = -64000       -- 64MB cache (vs 2MB default)
temp_store = MEMORY       -- Avoid disk I/O for temp tables
mmap_size = 268435456     -- 256MB memory-mapped I/O
```

**Index Optimizations:**
- **Partial index**: `CREATE INDEX idx_embedding ON files(embedding) WHERE embedding IS NOT NULL`
  - Only indexes rows with embeddings (saves space, faster scans)
  - Similarity search uses index for filtered `SELECT`
- **WITHOUT ROWID**: Tables with text primary keys use B-tree directly (20-30% space savings)

**Query Patterns:**
- Batch deletes: `DELETE FROM files WHERE path IN (?, ?, ...)` vs individual deletes
- Batch updates: `executemany()` for bulk ref_count changes
- Order by created_at: Prioritize recent files in similarity search

**Connection Pooling:**
- Queue-based pool with bounded size (default: 5 connections)
- Eliminates connection creation overhead (~5-10ms per connection)
- `check_same_thread=False` allows cross-thread connection reuse
- `PRAGMA wal_checkpoint(TRUNCATE)` after commits ensures read-after-write consistency
- Thread-safe via `queue.Queue` for checkout/checkin
- Automatic cleanup on shutdown

---

## Similarity Search Performance

Batch similarity with optional int8 quantization (1000 vectors, 384D embeddings):

| Method | Time | Speedup |
|--------|------|---------|
| Per-vector loop | ~5ms | baseline |
| Batch matrix (float32) | ~0.8ms | 6x |
| Batch matrix (int8 quantized) | ~0.6ms | 8x |
| Quantized + dimension pruning | ~0.35ms | 14x |

**Quantization accuracy:** <0.3% error vs exact computation.

**find_similar() optimization:**
- Before: O(N) individual `cosine_similarity()` calls
- After: Single `top_k_similarities()` SIMD matrix operation

---

## Text Processing Optimizations

### Semantic Summarization

**Research-based** content selection (TCRA-LLM approach, arXiv:2310.15556):

**Algorithm:**
1. Segment file by semantic boundaries (functions, classes, paragraphs)
2. Score segments by position, information density, and diversity
3. Greedily select highest-scoring segments that fit budget
4. Always preserve first segment (docstrings, imports, headers)
5. Reassemble with omission markers

**Performance:**
- **50-80% token savings** on large files vs simple truncation
- Preserves structural integrity (first/last segments prioritized)
- Language-agnostic boundary detection (Python, TS, Go, Rust, etc.)

**Example (10KB limit on 25KB file):**
```python
"""Module docstring."""  # Always preserved

def func_0():
    pass

# ... [2067 lines omitted] ...

def func_999():
    pass
```

**Benefits over simple truncation:**
- Preserves docstrings and imports
- Maintains code skeleton (function signatures)
- LLM can understand file structure
- U-shaped priority curve (high value at start/end)

### Delta Compression

Delta compression and diff generation for efficient change tracking:

| Feature | Benefit | Use Case |
|---------|---------|----------|
| **Delta compression** | 10-100x smaller | Minimal changes to large files |
| **Semantic summarization** | 50-80% savings | Large file truncation |
| **Diff stats** | Track changes | Metadata for cache decisions |

**Delta compression example:**
```python
# 15KB file with 5 line changes
delta = compute_delta(old, new)
# Result: 245 bytes (98% compression)
```

**Cache diff output:**
```
// Stats: +5 -2 ~3 lines, 12.3% size
// Diff for file.py (changed since cache):
```

---

## Profiling Tips

```bash
# Profile with py-spy
py-spy record -o profile.svg -- semantic-cache-mcp

# Memory profiling
python -m memory_profiler your_script.py
```

---

[← Back to README](../README.md)
