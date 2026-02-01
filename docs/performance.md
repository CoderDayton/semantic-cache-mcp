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
| **Batch SQLite queries** | 2-5x faster inserts | `executemany` + `IN` clause |
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

HyperCDC with Gear hash (2.7x faster than Rabin fingerprinting):

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

Delta compression and semantic truncation for efficient diffs:

| Feature | Benefit | Use Case |
|---------|---------|----------|
| **Delta compression** | 10-100x smaller | Minimal changes to large files |
| **Semantic truncation** | Preserves structure | Code files (Python, TS, Go) |
| **Diff stats** | Track changes | Metadata for cache decisions |

**Delta compression example:**
```python
# 15KB file with 5 line changes
delta = compute_delta(old, new)
# Result: 245 bytes (98% compression)
```

**Semantic truncation:**
- Cuts at function/class boundaries instead of arbitrary lines
- Detects language (Python, TypeScript, Go)
- Preserves code integrity for LLM parsing

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
