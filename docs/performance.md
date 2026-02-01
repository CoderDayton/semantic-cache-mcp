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
| **Inlined rolling hash** | 2-3x faster chunking | Eliminated method call overhead |
| **LRU cache for hashing** | Skip repeated hashing | `@lru_cache(maxsize=1024)` on pure functions |
| **Counter for entropy** | 2-3x faster calculation | C-implemented `collections.Counter` |
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

Rabin fingerprinting with inlined rolling hash:

```python
# Hot path - inlined for performance
h = (h * RH_PRIME + data[i] - data[i - RH_WINDOW] * RH_POW_OUT) & RH_MOD
```

**Benchmarks** (representative, not guaranteed):
- 1MB file: ~50ms chunking time
- Average chunk size: ~8KB (configurable via `RH_MASK`)

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

Adaptive Brotli quality based on Shannon entropy:

```python
def estimate_entropy(data: bytes) -> float:
    counts = Counter(data[:ENTROPY_SAMPLE_SIZE])
    total = sum(counts.values())
    return -sum(c/total * log2(c/total) for c in counts.values() if c > 0)
```

| Entropy | Quality | Ratio | Speed |
|---------|---------|-------|-------|
| > 7.0 | 1 | Low | Fast |
| > 5.5 | 4 | Medium | Medium |
| <= 5.5 | 6 | High | Slower |

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
