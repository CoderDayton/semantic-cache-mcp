# Performance

Two benchmark suites characterise the cache:

- [`benchmarks/benchmark_token_savings.py`](../benchmarks/benchmark_token_savings.py): measures the **token reduction** delivered by each cache hit path.
- [`benchmarks/benchmark_performance.py`](../benchmarks/benchmark_performance.py): measures the **wall-clock latency** of every core operation, reporting p50 / p95 / p99.

Both write a JSON report (`--json <path>`) for diffing across runs and produce reproducible results from a deterministic seed.

The numbers below were captured on:

| | |
|---|---|
| **CPU** | Intel Core i9-13900K (32 cores) |
| **Python** | 3.12 |
| **Search** | BM25 keyword (FTS5), no embedding model |
| **Corpus** | 40 source files, **177,509 tokens**, 213 documents |
| **Commit** | `71252b3` |

Reproduce with:

```bash
uv run python benchmarks/benchmark_token_savings.py --json out.json
uv run python benchmarks/benchmark_performance.py --json out.json --iterations 15
```

---

## Token Savings

Each phase reads the same 40-file corpus through `smart_read` / `batch_smart_read` / `semantic_search` and reports tokens emitted vs. tokens that would have been read in the absence of the cache.

| # | Phase | Trigger | Tokens returned | Original | Savings |
|---|-------|---------|----------------:|---------:|--------:|
| 1 | Cold read | First read, no cache (baseline) | 177,509 | 177,509 | 0.0% |
| 2 | Unchanged re-read | mtime match, **fast path skips disk I/O** | 1,580 | 177,509 | **99.1%** |
| 3 | Content hash | mtime drifted (e.g. `git checkout`), BLAKE3 still matches | 1,580 | 177,509 | **99.1%** |
| 4 | Small edits (12/40 changed) | Real ~5% line changes on 30% of files | 3,921 | 177,747 | **97.8%** |
| 4a |  → changed files only | Returned as unified diff (bare hunks, no file headers) | 2,850 | 108,783 | 97.4% |
| 4b |  → unchanged files | Fast path | 1,071 | 68,964 | 98.4% |
| 5 | Batch read (200K budget) | `batch_smart_read` over the whole corpus | 1,537 | 177,747 | **99.1%** |
| 6 | Search previews | 5 keyword queries × k=5, previews vs. full reads | 301 | 110,925 | **99.7%** |

**Aggregate (phases 2 to 6): 98.9% token reduction.**

The CI test [`tests/test_benchmark_token_savings.py`](../tests/test_benchmark_token_savings.py) asserts ≥ 80% overall as a regression gate.

### Token reduction by strategy

| Strategy | Savings | Trigger |
|----------|--------:|---------|
| Unchanged (mtime) | ~99% | `cached.mtime >= file.mtime`, disk read skipped entirely |
| Content hash | ~99% | mtime drifted but BLAKE3 hash still matches |
| Diff (changed) | 80 to 95% | File modified since last cache; emitted as unified diff |
| Search previews | ~100% | `search` returns 200-char previews, never full files |
| Summarised | 50 to 80% | File exceeds `MAX_CONTENT_SIZE`; semantic skeleton retained |

---

## Latency

All numbers are p50 unless otherwise noted; p95/p99 are reported in the raw output. Cold-read totals include disk I/O and tokenisation for the entire corpus. Every phase, including search and grep, runs against the same fixed 40-file corpus, so scan latency does not grow with the benchmark's iteration count.

### Cache read

| Operation | p50 | p95 | Notes |
|-----------|----:|----:|-------|
| Single unchanged read (fast path) | **0.9 ms** | 1.0 ms | mtime check + cache hit; **no disk I/O** |
| Single diff read (changed file) | 0.7 ms | 0.8 ms | Hash check + unified diff |
| Unchanged re-read (40 files) | 18 ms | 30 ms | Whole-corpus pass |
| Cold read (40 files, total) | n/a | n/a | 125 ms one-shot (~3.1 ms/file avg) |

### Batch read

| Operation | p50 | p95 |
|-----------|----:|----:|
| `batch_read` (40 files, diff mode) | 26.0 ms | 37.5 ms |

### Write + edit

| Operation | p50 | p95 |
|-----------|----:|----:|
| Write (200-line file) | 1.8 ms | 11.3 ms |
| Edit (scoped find/replace) | 2.4 ms | 2.5 ms |

### Chunked write (large files, CDC-split)

| Operation | p50 | p95 |
|-----------|----:|----:|
| Chunked write (72 KB, ~25 chunks) | 4.2 ms | 19.0 ms |
| Chunked write (360 KB, ~125 chunks) | 21 ms | 28.5 ms |
| Chunked re-read (72 KB, record_access fan-out) | 1.4 ms | 1.6 ms |

### Search

| Operation | p50 | p95 | Notes |
|-----------|----:|----:|-------|
| Search k=5 (cache **miss**) | 1.5 ms | n/a | BM25 keyword search (FTS5) |
| Search k=5 (cache **hit**) | **< 0.01 ms** | < 0.01 ms | In-session result LRU |
| Search k=10 (cache hit) | < 0.01 ms | < 0.01 ms | |

The in-session search cache delivers a **hundreds-fold speedup** on repeated queries (warm < 0.01 ms vs. cold ~4.6 ms over 5 queries).

### Grep

| Operation | p50 | p95 |
|-----------|----:|----:|
| Literal (`def `) | 1.3 ms | 1.4 ms |
| Regex (`class\s+\w+`) | 3.7 ms | 3.9 ms |

### Response shaping

`_finalize_payload` runs on every tool response. The `chars/4` fast-exit (added in 0.4.6) skips the BPE encode entirely when a payload is safely under the response token cap.

| Payload | p50 | p95 |
|---------|----:|----:|
| Small (single match, 25K cap) | < 0.01 ms | < 0.01 ms |
| Large (40 files × 5 matches) | 0.03 ms | 0.03 ms |

### Tokenizer

| Operation | p50 | Notes |
|-----------|----:|-------|
| Tokeniser (81 KB) | 0.20 ms | Warm BPE encode |
| Tokeniser (398 KB, all files) | 0.22 ms | Merge cache amortises full sweeps |

---

## Why these numbers

Removing the embedding and vector layer made the write and cold-read paths much
cheaper, with no ONNX inference on the hot path, while the cache's token savings
stayed the same. The optimisations below still land directly in the table above:

| Optimisation | Where it lands | Visible effect |
|--------------|----------------|----------------|
| `stat` + cache lookup before `aread_bytes` | `cache/read.py` | Single unchanged read drops to ~0.9 ms (no disk I/O) |
| No embedding on write/refresh | `cache/store.py`, `cache/write.py` | Write (200-line file) drops to ~1.8 ms; cold read to ~125 ms |
| In-session search-result LRU | `cache/search.py`, `cache/store.py` | Repeat-query hits at < 0.01 ms |
| Drop `// Stats:` line from diff content | `cache/read.py` | ~15 tokens trimmed per changed file in phase 4 |
| Char/4 fast-exit in `_finalize_payload` | `server/response.py` | Response shaping is sub-microsecond on small payloads |
| Char-budget grep truncation | `server/tools/__init__.py` | Large grep results stay under the response cap |
| Pre-stored search previews | `storage/docstore/__init__.py` | No re-slicing of chunked content at query time |
| `include_markers=False` default | `core/text/_summarize.py` | Summarisation no longer wastes tokens on `[N lines omitted]` markers |

---

## Concurrency model

| Decision | Effect |
|----------|--------|
| Single-thread `DetachedExecutor` for storage I/O | All blocking storage I/O (SQLite reads and writes) routes through one thread, so the single connection is never touched concurrently. |
| `asyncio.gather()` in `batch_smart_read` | Cache lookups and stat pre-fetch run in parallel; smart-read calls themselves serialise on the single executor. |
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
