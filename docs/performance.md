# Performance

## Token Savings Benchmark

Measured end-to-end on this project's own source files (30 `.py` files, ~136K tokens total). Benchmark script: [`benchmarks/benchmark_token_savings.py`](../benchmarks/benchmark_token_savings.py).

| Phase | Scenario | Tokens returned | Original | Savings |
|-------|----------|----------------:|---------:|--------:|
| 1. Cold read | First read, no cache | 122,402 | 122,402 | 0.0% |
| 2. Unchanged re-read | Same files, no modifications | 1,134 | 122,402 | 99.1% |
| 3. Content hash | Touch files (mtime changed, content identical) | 1,134 | 122,402 | 99.1% |
| 4. Small edits | ~5% of lines changed in 30% of files | 2,297 | 122,475 | 98.1% |
| 5. Batch read | All files via `batch_smart_read` | 1,134 | 122,475 | 99.1% |
| 6. Search | 5 queries × k=5, previews vs full reads | 1,690 | 105,500 | 98.4% |

**Overall (phases 2–6): 98.8% token reduction.**

Run it yourself:

```bash
uv run python benchmarks/benchmark_token_savings.py
uv run pytest tests/test_benchmark_token_savings.py -v  # CI-verifiable assertion: savings ≥ 80%
```

### Token Reduction by Strategy

| Strategy          | Savings | Trigger                           |
|-------------------|---------|-----------------------------------|
| Unchanged (mtime) | ~99%    | File mtime matches cached entry   |
| Content hash      | ~99%    | mtime changed but BLAKE3 hash matches (touch, git checkout) |
| Diff (changed)    | 80–95%  | File modified since last cache    |
| Search previews   | ~98%    | Semantic search returns previews, not full files |
| Semantic match    | 70–90%  | Similar file found in cache       |
| Summarized        | 50–80%  | File exceeds `MAX_CONTENT_SIZE`   |

---

## Operation Latency

Measured on this project's 30 source files with CPU embeddings (BAAI/bge-small-en-v1.5 via ONNX Runtime). Benchmark script: [`benchmarks/benchmark_performance.py`](../benchmarks/benchmark_performance.py).

### Cache Read

| Operation | Time | Notes |
|-----------|-----:|-------|
| Single unchanged read | 2 ms | mtime check only, no I/O |
| Unchanged re-read (29 files) | 25 ms | All files cached and unmodified |
| Batch read (29 files, diff mode) | 35 ms | Pre-scans + batch diff in one call |
| Cold read (29 files) | 2,554 ms | Includes disk I/O, tokenization, and embedding |

### Cache Write + Edit

| Operation | Time | Notes |
|-----------|-----:|-------|
| Write (200-line file) | 47 ms | Creates file + caches + embeds |
| Edit (scoped find/replace) | 48 ms | Uses cached content, writes diff |

### Search + Similarity

| Operation | Time | Notes |
|-----------|-----:|-------|
| Semantic search (k=5) | 4 ms | Hybrid BM25 + HNSW via simplevecdb |
| Semantic search (k=10) | 5 ms | Scales well with k |
| Find similar (k=3) | 49 ms | Embeds query file + HNSW lookup |
| Find similar (k=10) | 50 ms | Embedding dominates, k is cheap |

### Grep

| Operation | Time | Notes |
|-----------|-----:|-------|
| Grep (literal, `def `) | 1 ms | Fixed-string match across all cached files |
| Grep (regex, `class\s+\w+`) | 2 ms | Regex compiled once, scanned in-memory |

### Embedding + Tokenizer

| Operation | Time | Notes |
|-----------|-----:|-------|
| Model warmup | 206 ms | One-time ONNX model load |
| Tokenizer (40K chars) | 0.3 ms | Warm; cold first call ~36 ms |
| Single embed (largest file) | 47 ms | ~3K token file |
| Batch embed (10 files) | 469 ms | Single `model.embed()` call |
| Single embed (short string) | 2 ms | `"def hello(): pass"` |

Run it yourself:

```bash
uv run python benchmarks/benchmark_performance.py
```

---

## Optimization Summary

| Technique | Benefit | Details |
|-----------|---------|---------|
| **Three-state read model** | 80–99% token savings | Unchanged → message only; changed → diff only; new → full content |
| **Content hash freshness** | Avoids false re-reads | BLAKE3 hash detects when mtime changes but content is identical (touch, git checkout) |
| **Batch embedding** | N ONNX calls → 1 | `batch_read` pre-scans all new/changed files; single `model.embed()` call |
| **Hybrid search (BM25 + HNSW)** | Sub-5ms search | simplevecdb combines keyword and vector search via Reciprocal Rank Fusion |
| **O(N log M) BPE tokenizer** | vs O(N²) naive | Priority-queue merge with memoization |
| **LRU-K cache eviction** | Frequency-aware | Keeps frequently accessed files; evicts cold entries first |
| **`__slots__` on dataclasses** | Eliminate `__dict__` | Memory-efficient data models |
| **Generator expressions** | Avoid intermediate lists | Used in hot paths (similarity ranking, batch operations) |
| **In-memory grep** | Sub-2ms pattern search | Cached file contents searched without disk I/O |
| **BLAKE3 hashing** | Fast content fingerprinting | Hardware-accelerated; used for change detection |

---

## Embeddings

### Model

`BAAI/bge-small-en-v1.5` (default) — 384-dimensional, 33M parameters, 512 token context. Runs locally via [FastEmbed](https://github.com/qdrant/fastembed) (ONNX Runtime). Configurable via `EMBEDDING_MODEL` env var.

Device selection via `EMBEDDING_DEVICE`: `cpu` (default), `cuda` (GPU), or `auto` (detect).

### Batch Embedding (`embed_batch`)

`batch_read` pre-scans all requested paths before the main read loop:

1. Identify new/changed files via mtime check (unchanged files need no embedding)
2. Read content from disk and prepare for embedding
3. Call `embed_batch()` once — a single `model.embed(texts)` ONNX inference call
4. Distribute results into `smart_read` via `_embedding=` parameter

For a batch of 20 files where 12 are new: 12 model calls → 1. Unchanged files pay zero embedding cost.

---

## Storage

### VectorStorage (simplevecdb)

The storage backend uses [SimpleVecDB](https://github.com/CoderDayton/SimpleVecDB), which provides:

- **HNSW index** — Approximate nearest neighbor search for semantic similarity
- **FTS5 full-text search** — BM25 keyword search for grep and hybrid queries
- **Hybrid search** — Reciprocal Rank Fusion combines HNSW + BM25 results
- **Raw text storage** — File contents stored as plain text (no compression layer)

### Why Raw Text?

v0.3.0 replaced the previous compressed-chunk architecture (SQLiteStorage with CDC chunking, ZSTD compression, int8 quantized embeddings, LSH index) with raw text + vector embeddings. The tradeoffs:

| | v0.2.x (SQLiteStorage) | v0.3.0 (VectorStorage) |
|---|---|---|
| Storage format | Compressed CDC chunks | Raw text + embeddings |
| Search | LSH approximate + exact re-rank | HNSW + BM25 hybrid |
| Complexity | ~2,500 LOC across 6 modules | ~400 LOC, single module |
| Disk usage | Lower (compression) | Higher (raw text) |
| Search quality | Good (int8 cosine) | Better (hybrid RRF) |
| Maintenance | Complex (chunk/compress/quantize pipeline) | Simple (store/embed/search) |

The simplicity and search quality improvements outweigh the modest increase in disk usage. Source code files compress well but are small enough that the compression overhead isn't justified.

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
