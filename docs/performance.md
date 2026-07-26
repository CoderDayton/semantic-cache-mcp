# Performance

Two benchmark suites characterise the cache:

- [`benchmarks/benchmark_token_savings.py`](../benchmarks/benchmark_token_savings.py): measures the **token reduction** delivered by each cache hit path.
- [`benchmarks/benchmark_performance.py`](../benchmarks/benchmark_performance.py): measures the **wall-clock latency** of every core operation, reporting p50 / p95 / p99.

Both write a JSON report (`--json <path>`) for diffing across runs and produce reproducible results from a deterministic seed.

The numbers below were captured on:

| | |
|---|---|
| **CPU** | Intel Core i9-13900K (32 cores) |
| **Python** | 3.13 |
| **Search** | BM25 keyword (FTS5), no embedding model |
| **Corpus** | 41 source files, **200,756 tokens**, 238 documents |
| **Version** | `0.5.3` |

> **Comparing these against 0.5.2.** The corpus is this repository's own `src/`,
> so it moves with the code. 0.5.3 added a file and pushed
> `server/tools/__init__.py` from 98,737 to 105,577 bytes — across the
> 100,000-byte `MAX_CONTENT_SIZE` threshold. That one file is now semantically
> summarised on every cold or full read, which is the whole reason phase 1 shows
> a 5.6% "saving" where 0.5.2 showed 0.0%, and why cold read (87 → 114 ms) and
> `batch_read` (33 → 51 ms) both got slower. Neither is a regression in the read
> path; both are the cost of summarising one newly-oversized file. The cache-hit
> paths — what this page is actually about — are unchanged at 99.2%.

Reproduce with:

```bash
uv run python benchmarks/benchmark_token_savings.py --json out.json
uv run python benchmarks/benchmark_performance.py --json out.json --iterations 15
```

---

## Token Savings

Each phase reads the same 41-file corpus through `smart_read` / `batch_smart_read` / `semantic_search` and reports tokens emitted vs. tokens that would have been read in the absence of the cache.

Every saving below is earned by a caller that echoes back the `content_hash` it was given (`known_hash` on `read`, `known_hashes` on `batch_read`). Since 0.5.2 that is a hard requirement, not an optimization: a caller that cannot prove it still holds a file is sent the file. The phases model an agent that keeps its hashes, which is the flow the numbers describe.

Since 0.5.3 a ranged read earns a `coverage_token` on the same terms, redeemable for the lines it delivered. These phases read whole files, so that path is not exercised by the table below and contributes nothing to these numbers — the row it would change is a repeated read of one region of a large file, which this corpus does not model.

| # | Phase | Trigger | Tokens returned | Original | Savings |
|---|-------|---------|----------------:|---------:|--------:|
| 1 | Cold read | First read, no cache (baseline) | 189,478 | 200,756 | 5.6% |
| 2 | Unchanged re-read | mtime match, **fast path skips disk I/O** | 1,658 | 200,756 | **99.2%** |
| 3 | Content hash | mtime drifted (e.g. `git checkout`), BLAKE3 still matches | 1,658 | 200,756 | **99.2%** |
| 4 | Small edits (12/41 changed) | Real ~5% line changes on 30% of files | 3,898 | 201,079 | **98.1%** |
| 4a |  → changed files only | Returned as unified diff (bare hunks, no file headers) | 2,735 | 109,265 | 97.5% |
| 4b |  → unchanged files | Fast path | 1,163 | 91,814 | 98.7% |
| 5 | Batch read (200K budget) | `batch_smart_read` over the whole corpus, echoing the hashes phase 4 returned | 1,660 | 201,079 | **99.2%** |
| 6 | Search previews | 5 keyword queries × k=5, previews vs. full reads | 1,783 | 122,400 | **98.5%** |

**Aggregate (phases 2 to 6): 98.8% token reduction.**

Phase 1 is the no-cache baseline and used to sit at exactly 0.0%. It now returns
5.6% fewer tokens than the corpus holds because one file crossed
`MAX_CONTENT_SIZE` and comes back summarised — a first read of an oversized file
is not a cache saving, and should not be read as one.

Phase 6 costs more than it did in 0.5.1 (98.5% vs. 99.7%), and deliberately so:
`search` now joins query terms with `OR`, so a query returns the files that match
*some* of it rather than only the files that match all of it. The extra ~1,400
tokens buy back the queries that used to return nothing at all.

The CI test [`tests/test_benchmark_token_savings.py`](../tests/test_benchmark_token_savings.py) asserts ≥ 80% overall as a regression gate.

### Token reduction by strategy

| Strategy | Savings | Trigger |
|----------|--------:|---------|
| Unchanged (mtime) | ~99% | Caller's hash matches and `cached.mtime >= file.mtime`; disk read skipped entirely |
| Content hash | ~99% | Caller's hash matches; mtime drifted but BLAKE3 hash still matches |
| Diff (changed) | 80 to 95% | File modified since last cache, and the caller holds the base content; emitted as unified diff |
| No possession proof | 0% | Caller sent no matching hash — the file is returned in full, served from cache when fresh |
| Search previews | ~100% | `search` returns 200-char previews, never full files |
| Summarised | 50 to 80% | File exceeds `MAX_CONTENT_SIZE`; semantic skeleton retained |

---

## Latency

All numbers are p50 unless otherwise noted; p95/p99 are reported in the raw output. Cold-read totals include disk I/O and tokenisation for the entire corpus. Every phase, including search and grep, runs against the same fixed 41-file corpus, so scan latency does not grow with the benchmark's iteration count.

### Cache read

| Operation | p50 | p95 | Notes |
|-----------|----:|----:|-------|
| Single unchanged read (fast path) | **1.2 ms** | 2.0 ms | mtime check + cache hit; **no disk I/O** |
| Single diff read (changed file) | 0.8 ms | 1.1 ms | Hash check + unified diff |
| Unchanged re-read (41 files) | 21 ms | 22 ms | Whole-corpus pass |
| Cold read (41 files, total) | n/a | n/a | 114 ms one-shot (~2.8 ms/file avg), including the summarisation of the one file over `MAX_CONTENT_SIZE` |

### Batch read

| Operation | p50 | p95 |
|-----------|----:|----:|
| `batch_read` (41 files, diff mode) | 51.2 ms | 52.2 ms |

The jump from 33.1 ms in 0.5.2 is the same oversized file: an unproven caller is
read with `force_full`, so that file is summarised on every pass.

### Write + edit

| Operation | p50 | p95 |
|-----------|----:|----:|
| Write (200-line file) | 1.9 ms | 2.4 ms |
| Edit (scoped find/replace) | 2.2 ms | 2.5 ms |

### Chunked write (large files, CDC-split)

| Operation | p50 | p95 |
|-----------|----:|----:|
| Chunked write (72 KB, ~25 chunks) | 3.2 ms | 3.8 ms |
| Chunked write (360 KB, ~125 chunks) | 9.7 ms | 12.1 ms |
| Chunked re-read (72 KB, record_access fan-out) | 1.0 ms | 1.3 ms |

### Search

| Operation | p50 | p95 | Notes |
|-----------|----:|----:|-------|
| Search k=5 (cache **miss**) | 2.2 ms | n/a | BM25 keyword search (FTS5) |
| Search k=5 (cache **hit**) | **< 0.01 ms** | < 0.01 ms | In-session result LRU |
| Search k=10 (cache hit) | < 0.01 ms | < 0.01 ms | |

The in-session search cache delivers a **hundreds-fold speedup** on repeated queries (warm < 0.01 ms vs. cold ~10.0 ms over 5 queries — about 900× faster).

### Grep

| Operation | p50 | p95 |
|-----------|----:|----:|
| Literal (`def `) | 1.5 ms | 1.8 ms |
| Regex (`class\s+\w+`) | 3.6 ms | 4.2 ms |

### Response shaping

`_finalize_payload` runs on every tool response. The `chars/4` fast-exit (added in 0.4.6) skips the BPE encode entirely when a payload is safely under the response token cap.

| Payload | p50 | p95 |
|---------|----:|----:|
| Small (single match, 25K cap) | < 0.01 ms | < 0.01 ms |
| Large (40 files × 5 matches) | 0.02 ms | 0.03 ms |

### Tokenizer

| Operation | p50 | Notes |
|-----------|----:|-------|
| Tokeniser (~105 KB) | 0.19 ms | Warm BPE encode |
| Tokeniser (~467 KB, all files) | 0.21 ms | Merge cache amortises full sweeps |

---

## Why these numbers

Removing the embedding and vector layer made the write and cold-read paths much
cheaper, with no ONNX inference on the hot path, while the cache's token savings
stayed the same. The optimisations below still land directly in the table above:

| Optimisation | Where it lands | Visible effect |
|--------------|----------------|----------------|
| `stat` + cache lookup before `aread_bytes` | `cache/read.py` | Single unchanged read drops to ~1.0 ms (no disk I/O); the stat is taken before the read so a concurrent write is never cached as fresh |
| No embedding on write/refresh | `cache/store.py`, `cache/write.py` | Write (200-line file) drops to ~1.7 ms; cold read to ~87 ms |
| Single-pass diff + stats (`diff_with_stats`) | `core/text/_diff.py`, `cache/write.py` | Write/edit no longer run the line-matcher twice; 360 KB chunked write drops from ~21 ms to ~9 ms |
| Adaptive diff context (2 lines under 100-line files) | `core/text/_diff.py`, `cache/_helpers.py` | Small-file diffs carry less context overhead; suppressed diffs keep per-hunk headers |
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
