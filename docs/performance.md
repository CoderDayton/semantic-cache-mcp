# Performance

Two benchmark suites characterise the cache:

- [`benchmarks/benchmark_token_savings.py`](../benchmarks/benchmark_token_savings.py) — measures the **token reduction** delivered by each cache hit path.
- [`benchmarks/benchmark_performance.py`](../benchmarks/benchmark_performance.py) — measures the **wall-clock latency** of every core operation, reporting p50 / p95 / p99.

Both write a JSON report (`--json <path>`) for diffing across runs and produce reproducible results from a deterministic seed.

The numbers below were captured on:

| | |
|---|---|
| **CPU** | Intel Core i9-13900K (32 cores) |
| **Python** | 3.12 |
| **Embeddings** | BAAI/bge-small-en-v1.5 (ONNX Runtime, CPU) |
| **Corpus** | 43 source files, **168,614 tokens** |
| **Commit** | `5cd7100` (release/0.4.6) |

Reproduce with:

```bash
uv run python benchmarks/benchmark_token_savings.py --json out.json
uv run python benchmarks/benchmark_performance.py --json out.json --iterations 15
```

---

## Token Savings

Each phase reads the same 43-file corpus through `smart_read` / `batch_smart_read` / `semantic_search` and reports tokens emitted vs. tokens that would have been read in the absence of the cache.

| # | Phase | Trigger | Tokens returned | Original | Savings |
|---|-------|---------|----------------:|---------:|--------:|
| 1 | Cold read | First read, no cache (baseline) | 168,614 | 168,614 | 0.0% |
| 2 | Unchanged re-read | mtime match — **fast path skips disk I/O** | 1,792 | 168,614 | **98.9%** |
| 3 | Content hash | mtime drifted (e.g. `git checkout`), BLAKE3 still matches | 1,792 | 168,614 | **98.9%** |
| 4 | Small edits (12/43 changed) | Real ~5% line changes on 30% of files | 4,589 | 168,812 | **97.3%** |
| 4a |  → changed files only | Returned as unified diff | 3,298 | 69,982 | 95.3% |
| 4b |  → unchanged files | Fast path | 1,291 | 98,830 | 98.7% |
| 5 | Batch read (200K budget) | `batch_smart_read` over the whole corpus | 1,792 | 168,812 | **98.9%** |
| 6 | Search previews | 5 semantic queries × k=5, previews vs. full reads | 1,673 | 98,025 | **98.3%** |

**Aggregate (phases 2–6): 98.5% token reduction.**

The CI test [`tests/test_benchmark_token_savings.py`](../tests/test_benchmark_token_savings.py) asserts ≥ 80% overall as a regression gate.

### Token reduction by strategy

| Strategy | Savings | Trigger |
|----------|--------:|---------|
| Unchanged (mtime) | ~99% | `cached.mtime >= file.mtime` — disk read skipped entirely |
| Content hash | ~99% | mtime drifted but BLAKE3 hash still matches |
| Diff (changed) | 80–95% | File modified since last cache; emitted as unified diff |
| Search previews | ~98% | `search` returns 200-char previews, never full files |
| Semantic match | 70–90% | Similar file already cached; emitted as diff vs. neighbour |
| Summarised | 50–80% | File exceeds `MAX_CONTENT_SIZE`; semantic skeleton retained |

---

## Latency

All numbers are p50 unless otherwise noted; p95/p99 are reported in the raw output. Cold-read totals include disk I/O, tokenisation, and embedding for the entire corpus.

### Cache read

| Operation | p50 | p95 | Notes |
|-----------|----:|----:|-------|
| Single unchanged read (fast path) | **1.1 ms** | 14.9 ms | mtime check + cache hit; **no disk I/O** |
| Single diff read (changed file) | 1.0 ms | 1.1 ms | Hash check + unified diff |
| Unchanged re-read (43 files) | 26.9 ms | 41.5 ms | Whole-corpus pass |
| Cold read (43 files, total) | — | — | 1,990 ms one-shot (47 ms/file avg) |

### Batch read

| Operation | p50 | p95 |
|-----------|----:|----:|
| `batch_read` (43 files, diff mode) | 40.2 ms | 66.8 ms |

### Write + edit

| Operation | p50 | p95 |
|-----------|----:|----:|
| Write (200-line file) | 49.1 ms | 50.0 ms |
| Edit (scoped find/replace) | 3.3 ms | 3.9 ms |

### Search

| Operation | p50 | p95 | Notes |
|-----------|----:|----:|-------|
| Search k=5 (cache **miss**) | 5.6 ms | — | Embed query + hybrid BM25/HNSW |
| Search k=5 (cache **hit**) | **< 0.01 ms** | < 0.01 ms | In-session result LRU |
| Search k=10 (cache hit) | < 0.01 ms | < 0.01 ms | |

The in-session search cache delivers a **2,000×+ speedup** on repeated queries (warm 0.013 ms vs. cold 26.5 ms over 5 queries).

### Grep

| Operation | p50 | p95 |
|-----------|----:|----:|
| Literal (`def `) | 1.4 ms | 1.5 ms |
| Regex (`class\s+\w+`) | 2.1 ms | 2.5 ms |

### Response shaping

`_finalize_payload` runs on every tool response. The `chars/4` fast-exit (added in 0.4.6) skips the BPE encode entirely when a payload is safely under the response token cap.

| Payload | p50 | p95 |
|---------|----:|----:|
| Small (single match, 25K cap) | < 0.01 ms | < 0.01 ms |
| Large (40 files × 5 matches) | 0.03 ms | 0.04 ms |

### Embedding + tokenizer

| Operation | p50 | Notes |
|-----------|----:|-------|
| Model warmup | 195 ms | One-time ONNX model load |
| Tokeniser (68 KB) | 0.2 ms | Warm BPE encode |
| Tokeniser (392 KB, all files) | 0.23 ms | Merge cache amortises full sweeps |
| Single embed (largest file, ~3K tokens) | 47 ms | ONNX inference, single thread |
| Batch embed (10 files) | 487 ms | One `model.embed()` call |
| Single embed (short string) | 1.9 ms | `"def hello(): pass"` |

---

## Why these numbers

The 0.4.6 audit pass landed several specific optimisations that show up directly in the table above:

| Optimisation | Where it lands | Visible effect |
|--------------|----------------|----------------|
| `stat` + cache lookup before `aread_bytes` | `cache/read.py` | Single unchanged read drops to ~1 ms (no disk I/O) |
| In-session search-result LRU | `cache/search.py`, `cache/store.py` | Repeat-query hits at < 0.01 ms |
| Drop `// Stats:` line from diff content | `cache/read.py` | ~15 tokens trimmed per changed file in phase 4 |
| Char/4 fast-exit in `_finalize_payload` | `server/response.py` | Response shaping is sub-microsecond on small payloads |
| Char-budget grep truncation | `server/tools/__init__.py` | Large grep results stay under the response cap |
| Pre-stored search previews | `storage/vector/__init__.py` | No re-slicing of chunked content at query time |
| `include_markers=False` default | `core/text/_summarize.py` | Summarisation no longer wastes tokens on `[N lines omitted]` markers |
| `EMBED_REUSE_THRESHOLD = 0.20` | `cache/write.py` | Skips ONNX re-embed for small edits (saves ~23 ms/edit) |

---

## Concurrency model

| Decision | Effect |
|----------|--------|
| Single-thread `DetachedExecutor` for ONNX + usearch | Required: both segfault under multi-threaded access. All blocking I/O routes through this one thread. |
| `asyncio.gather()` in `batch_smart_read` | Cache lookups and stats run in parallel; smart-read calls themselves serialise on the single executor. |
| Pre-batched embedding in `batch_smart_read` | One `embed_batch` call for all new/changed files in a batch; amortises ONNX overhead. |
| Cache-aware short-circuit in `smart_read` | Skips `aread_bytes` and `count_tokens` on the unchanged fast path. |
| Async subprocess for formatters | `_format_file` doesn't freeze the event loop. |

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

For benchmark results in machine-readable form (CI / regression diffing):

```bash
uv run python benchmarks/benchmark_performance.py    --json perf.json --samples
uv run python benchmarks/benchmark_token_savings.py  --json tok.json
```

`--samples` includes raw per-iteration timings for distribution analysis.

---

[← Back to README](../README.md)
