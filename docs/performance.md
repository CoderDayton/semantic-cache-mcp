# Performance

## Token Savings Benchmark

30 `.py` files, ~136K tokens total. Script: [`benchmarks/benchmark_token_savings.py`](../benchmarks/benchmark_token_savings.py).

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

CPU embeddings (BAAI/bge-small-en-v1.5 via ONNX Runtime), 30 source files. Script: [`benchmarks/benchmark_performance.py`](../benchmarks/benchmark_performance.py).

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

## Concurrency Optimizations (0.3.4)

| Optimization | Effect |
|--------------|--------|
| Dedicated embed executor (1 thread) | ONNX calls don't starve the default thread pool |
| `asyncio.gather()` in `batch_smart_read` | N cache lookups run in parallel instead of serial |
| No double-fetch on diff path | Eliminates 1 SQLite query per changed-file read |
| Embedding reuse in `find_similar_files` | Skips ONNX when `cached.embedding` is available |
| All blocking calls in `asyncio.to_thread()` | Event loop never blocked by I/O or ONNX |
| Async subprocess for formatters | `_format_file` no longer freezes the event loop |

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
