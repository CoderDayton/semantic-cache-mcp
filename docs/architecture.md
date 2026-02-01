# Architecture

## Component-Based Structure

```
semantic_cache_mcp/
├── config.py           # Configuration constants
├── types.py            # Data models (CacheEntry, ReadResult)
├── cache.py            # SemanticCache facade (orchestration)
├── server.py           # FastMCP tools (read, stats, clear)
├── core/               # Core algorithms (single responsibility)
│   ├── chunking.py     # Rabin fingerprinting CDC
│   ├── compression.py  # Adaptive Brotli compression
│   ├── embeddings.py   # FastEmbed local embeddings
│   ├── hashing.py      # BLAKE2b with LRU cache
│   ├── similarity.py   # Cosine similarity
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

Rabin fingerprinting with rolling hash for content-defined chunking (CDC).

**How it works:**
1. Slide a window over the content
2. Compute rolling hash using Rabin polynomial
3. Split when hash matches boundary condition (mask)
4. Results in ~8KB average chunks that align on content boundaries

**Benefits:**
- Similar files share chunks even if bytes shift
- Enables efficient deduplication across files

### Compression (`core/compression.py`)

Adaptive Brotli compression based on Shannon entropy estimation.

| Entropy Level | Quality | Use Case |
|---------------|---------|----------|
| High (>7.0) | 1 | Already compressed data |
| Medium (>5.5) | 4 | Mixed content |
| Low (<=5.5) | 6 | Highly compressible text |

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

Cosine similarity on normalized embeddings.

```python
def cosine_similarity(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))  # Normalized vectors
```

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
