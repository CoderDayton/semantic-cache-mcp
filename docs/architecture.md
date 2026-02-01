# Architecture

## Component-Based Structure

```
semantic_cache_mcp/
├── config.py           # Configuration constants
├── types.py            # Data models (CacheEntry, ReadResult)
├── cache.py            # SemanticCache facade (orchestration)
├── server.py           # FastMCP tools (read, stats, clear)
├── core/               # Core algorithms (single responsibility)
│   ├── chunking.py     # HyperCDC gear-hash chunking
│   ├── compression.py  # Multi-codec (ZSTD/LZ4/Brotli)
│   ├── embeddings.py   # FastEmbed local embeddings
│   ├── hashing.py      # BLAKE2b with LRU cache
│   ├── similarity.py   # Cosine similarity (NumPy optimized)
│   ├── tokenizer.py    # o200k_base BPE tokenizer
│   └── text.py         # Diff generation, smart truncation
└── storage/            # Persistence layer
    └── sqlite.py       # SQLite content-addressable storage
```

## Design Principles

- **Single Responsibility** — Each module has one clear purpose
- **Facade Pattern** — `SemanticCache` coordinates all components
- **Dependency Inversion** — Storage backend is swappable
- **Performance First** — Optimized hot paths, minimal allocations
- **Type Safety** — Strict typing with domain type aliases

---

## Core Algorithms

### Chunking (`core/chunking.py`)

HyperCDC with Gear hash for high-performance content-defined chunking (CDC).

**How it works:**
1. Pre-computed 256-entry gear table for fast byte lookups
2. Rolling hash: `h = ((h << 1) + gear[byte]) & MASK_64`
3. Split when `(h & mask) == 0` (normalized chunking)
4. Skip-min optimization: no boundary checks in first 2KB
5. Results in ~8KB average chunks that align on content boundaries

**Performance:** ~13-14 MB/s (2.7x faster than Rabin fingerprinting)

**Benefits:**
- Similar files share chunks even if bytes shift
- Enables efficient deduplication across files
- Gear hash is simpler and faster than Rabin polynomial

### Compression (`core/compression.py`)

Multi-codec adaptive compression with ZSTD (primary), LZ4, and Brotli.

| Entropy Level | Codec | Level | Throughput |
|---------------|-------|-------|------------|
| Very High (>7.5) | STORE | - | 29 GB/s |
| High (>6.5) | ZSTD | 1 | 6.9 GB/s |
| Medium (>4.0) | ZSTD | 3 | 5.3 GB/s |
| Low (<=4.0) | ZSTD | 9 | 4.8 GB/s |

**Features:**
- Cached compressors (avoid object creation overhead)
- Native ZSTD multi-threading for large files (>4MB)
- O(1) magic-byte detection for pre-compressed data
- Automatic codec fallback chain

### Hashing (`core/hashing.py`)

BLAKE3 hashing (with BLAKE2b fallback) for high-performance content addressing.

| Feature | Description |
|---------|-------------|
| **BLAKE3** | 3.8-4.9x faster than BLAKE2b on 8KB+ chunks |
| **LRU caching** | 16K chunks, 4K blocks, 2K content hashes |
| **Streaming** | Memory-efficient hashing for large files |
| **Hierarchical** | Chunk → block → content multi-level hashing |
| **DeduplicateIndex** | Fast fingerprint-based dedup lookups (~1M ops/sec) |

**API:**
- `hash_chunk(data)` — Hash CDC chunk (cached)
- `hash_content(content)` — Hash full content (str or bytes)
- `DeduplicateIndex` — Fast dedup with binary fingerprints
- `HierarchicalHasher` — Multi-level chunk→block→content hashing
- `StreamingHasher` — Incremental hashing for large files

### Tokenizer (`core/tokenizer.py`)

Optimized BPE tokenizer with O(N log M) merge algorithm.

| Feature | Description |
|---------|-------------|
| **Priority queue merging** | O(N log M) vs O(N²) naive search |
| **Merge caching** | Memoize BPE operations |
| **Fast path** | Single-byte tokens skip BPE |
| **Sample estimation** | O(1) token count for >50KB text |
| **Lazy compilation** | Regex pattern compiled once |

**Compatibility:**
- OpenAI o200k_base encoding (GPT-4o)
- Auto-downloads tiktoken file (~3.5MB)
- SHA256 verification
- Heuristic fallback if unavailable

### Embeddings (`core/embeddings.py`)

Local text embeddings using FastEmbed with nomic-embed-text-v1.5.

- Singleton model instance (loaded once, reused)
- Warmup at startup for predictable latency
- 768-dimensional embeddings
- 8192 token context window
- Uses `search_document:` and `search_query:` prefixes

### Similarity (`core/similarity.py`)

SIMD-optimized similarity with optional int8 quantization and dimension pruning.

| Feature | Speedup | Description |
|---------|---------|-------------|
| **NumPy baseline** | 19-27x vs Python | Vectorized dot product |
| **int8 quantization** | 4-8x additional | Scale to [-128,127], SIMD int8 ops |
| **Dimension pruning** | 20-40% additional | Skip low-magnitude dims (PDX) |
| **Batch matrix ops** | Eliminates loop | Single SIMD operation for N vectors |

**API:**
- `cosine_similarity(a, b)` — Single pair comparison
- `cosine_similarity_batch_matrix(query, vectors)` — SIMD batch
- `top_k_similarities(query, vectors, k)` — Efficient top-K retrieval

**Benchmarks (1000 vectors, 384D):**
- Baseline: ~5ms
- Quantized: ~0.6ms (8x faster)
- Quantized + pruned: ~0.35ms (14x faster)

### Text Processing (`core/text.py`)

Advanced diff and truncation with delta compression and syntax awareness.

| Feature | Description |
|---------|-------------|
| **Delta compression** | Store changes only (10-100x smaller) |
| **Semantic truncation** | Cut at function/class boundaries |
| **Diff statistics** | Track insertions/deletions/modifications |
| **Streaming diff** | Memory-efficient for multi-GB files |
| **Language detection** | Python, TypeScript, Go support |

**API:**
- `generate_diff(old, new)` — Unified diff with stats
- `compute_delta(old, new)` — Compressed change representation
- `truncate_semantic(content, max_size)` — Syntax-aware truncation
- `diff_stats(old, new)` — Change metrics

**Example:**
```python
delta = compute_delta(old, new)
# Delta: 245 bytes vs 15KB original (98% compression)
```

---

## Storage Layer

### SQLiteStorage (`storage/sqlite.py`)

Content-addressable chunk store (like Git).

**Features:**
- Chunks stored by BLAKE2b hash (deduplication)
- File metadata with chunk references
- LRU-K eviction (frequency-aware cache management)
- Batch operations with `executemany`

### Schema

```sql
CREATE TABLE chunks (
    hash TEXT PRIMARY KEY,
    data BLOB NOT NULL,
    size INTEGER NOT NULL,
    ref_count INTEGER DEFAULT 1
);

CREATE TABLE files (
    path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    chunk_hashes TEXT NOT NULL,  -- JSON array
    mtime REAL NOT NULL,
    tokens INTEGER NOT NULL,
    embedding BLOB,              -- array.array('f') serialized
    created_at REAL NOT NULL,
    access_history TEXT NOT NULL -- JSON array for LRU-K
);
```

### LRU-K Eviction

Uses the K-th most recent access time (K=2) for eviction decisions:
- Files accessed only once are evicted first
- Frequently accessed files are retained
- More accurate than simple LRU for workloads with scans

---

## Data Flow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Client    │────▶│  smart_read  │────▶│    Cache    │
│  (Claude)   │     │   (facade)   │     │   Lookup    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   Strategy   │     │   SQLite    │
                    │   Selection  │     │   Storage   │
                    └──────────────┘     └─────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │ Unchanged│    │   Diff   │    │ Semantic │
    │  (99%)   │    │ (80-95%) │    │  Match   │
    └──────────┘    └──────────┘    └──────────┘
```

---

[← Back to README](../README.md)
