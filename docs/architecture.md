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

BLAKE2b with `@lru_cache` for repeated chunks.

- 32-byte digests (256-bit security)
- LRU cache prevents rehashing identical content
- Used for both chunk addressing and content verification

### Tokenizer (`core/tokenizer.py`)

Self-contained BPE tokenizer compatible with OpenAI's o200k_base encoding.

- Auto-downloads tiktoken file from OpenAI (~3.5MB)
- SHA256 verification for security
- Falls back to heuristic if unavailable
- 200,000 token vocabulary (GPT-4o compatible)

### Embeddings (`core/embeddings.py`)

Local text embeddings using FastEmbed with nomic-embed-text-v1.5.

- Singleton model instance (loaded once, reused)
- Warmup at startup for predictable latency
- 768-dimensional embeddings
- 8192 token context window
- Uses `search_document:` and `search_query:` prefixes

### Similarity (`core/similarity.py`)

NumPy-optimized cosine similarity (19-27x faster than pure Python).

```python
def cosine_similarity(a, b) -> float:
    # NumPy vectorized dot product
    return float(np.dot(a, b))  # Pre-normalized vectors
```

Includes batch similarity for comparing against multiple vectors efficiently.

### Text Processing (`core/text.py`)

- **Diff generation**: Unified diff format with context
- **Smart truncation**: Preserves structure (functions, classes) when cutting

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
