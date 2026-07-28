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
| **Filesystem** | ext4 on NVMe SSD — **not tmpfs**, see below |
| **Search** | BM25 keyword (FTS5), no embedding model |
| **Corpus** | 41 source files, **212,499 tokens**, 250 documents |
| **Version** | `0.5.3` |

> **The filesystem is part of the measurement.** The suite works in a
> `tempfile.TemporaryDirectory`, which follows `TMPDIR` — and on most Linux
> systems `/tmp` is tmpfs, i.e. RAM. tmpfs discards `fsync`, so a benchmark run
> there reports a write latency the durable path never actually achieves. Since
> 0.5.3 `awrite_atomic` flushes the file *and* its parent directory before the
> rename, so the write path now costs a real disk round-trip and the difference
> is no longer noise: **+1.1 ms per write, +1.1 ms per edit** on this machine.
> Every latency number on this page is from an ext4/NVMe run; the tmpfs column
> in [Write + edit](#write--edit) is kept only to show what the durability
> guarantee costs.

Reproduce with:

```bash
# Pin the work directory to a real disk. Without this you are very likely
# measuring tmpfs, and the write/edit rows will read ~40% low.
BENCH_TMP="$HOME/.cache/scmcp-bench" && mkdir -p "$BENCH_TMP"
TMPDIR="$BENCH_TMP" uv run python benchmarks/benchmark_performance.py --json perf.json --iterations 15

uv run python benchmarks/benchmark_token_savings.py --json tok.json
```

`benchmark_performance.py` prints and records the work directory's filesystem
(`workdir_fs` in the JSON), so a report always says which one it measured.

---

## Token Savings

Each phase reads the same 41-file corpus through `smart_read` / `batch_smart_read` / `semantic_search` and reports tokens emitted vs. tokens that would have been read in the absence of the cache.

Every saving below is earned by a caller that echoes back the `content_hash` it was given (`known_hash` on `read`, `known_hashes` on `batch_read`). Since 0.5.2 that is a hard requirement, not an optimization: a caller that cannot prove it still holds a file is sent the file. The phases model an agent that keeps its hashes, which is the flow the numbers describe.

Since 0.5.3 a ranged read earns a `coverage_token` on the same terms, redeemable for the lines it delivered. These phases read whole files, so that path is not exercised by the table below and contributes nothing to these numbers — the row it would change is a repeated read of one region of a large file, which this corpus does not model.

| # | Phase | Trigger | Tokens returned | Original | Savings |
|---|-------|---------|----------------:|---------:|--------:|
| 1 | Cold read | First read, no cache (baseline) | 200,039 | 212,499 | 5.9% |
| 2 | Unchanged re-read | mtime match, **fast path skips disk I/O** | 1,535 | 212,499 | **99.3%** |
| 3 | Content hash | mtime drifted (e.g. `git checkout`), BLAKE3 still matches | 1,535 | 212,499 | **99.3%** |
| 4 | Small edits (12/41 changed) | Real ~5% line changes on 30% of files | 3,999 | 212,829 | **98.1%** |
| 4a |  → changed files only | Returned as unified diff (bare hunks, no file headers) | 2,929 | 114,682 | 97.4% |
| 4b |  → unchanged files | Fast path | 1,070 | 98,147 | 98.9% |
| 5 | Batch read (200K budget) | `batch_smart_read` over the whole corpus, echoing the hashes phase 4 returned | 1,537 | 212,829 | **99.3%** |
| 6 | Search previews | 5 keyword queries × k=5, previews vs. full reads | 1,753 | 129,550 | **98.6%** |

**Aggregate (phases 2 to 6): 98.9% token reduction.**

Phase 1 is the no-cache baseline and used to sit at exactly 0.0%. It returns
5.9% fewer tokens than the corpus holds because one file
(`server/tools/__init__.py`) exceeds `MAX_CONTENT_SIZE` and comes back
summarised — a first read of an oversized file is not a cache saving, and
should not be read as one. It is also why the cold-read and `batch_read`
latency rows are higher than they were in 0.5.2: that one file is summarised on
every unproven pass.

Phase 6 costs more than it did in 0.5.1 (98.6% vs. 99.7%), and deliberately so:
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
| Single unchanged read (fast path) | **1.1 ms** | 1.1 ms | mtime check + cache hit; **no disk I/O** |
| Single diff read (changed file) | 0.7 ms | 0.9 ms | Hash check + unified diff |
| Unchanged re-read (41 files) | 19.5 ms | 22.7 ms | Whole-corpus pass |
| Cold read (41 files, total) | n/a | n/a | 100 ms for a single unrepeated pass (~2.4 ms/file), including summarising the one file over `MAX_CONTENT_SIZE` |

The cold-read total is one pass, not a distribution — it is the only row here
with n=1, and it moves ±20% between runs. Treat it as an order of magnitude.

### Batch read

| Operation | p50 | p95 |
|-----------|----:|----:|
| `batch_read` (41 files, diff mode) | 45.6 ms | 49.7 ms |

### Write + edit

The only rows on this page where the filesystem changes the answer. Both paths
go through `awrite_atomic`, which since 0.5.3 fsyncs the temp file and then the
parent directory before returning — so a rename can never land ahead of the
bytes it points at.

| Operation | p50 (ext4/NVMe) | p95 (ext4/NVMe) | p50 (tmpfs) | Cost of durability |
|-----------|----:|----:|----:|----:|
| Write (200-line file) | **2.7 ms** | 2.8 ms | 1.6 ms | +1.07 ms |
| Edit (scoped find/replace) | **3.1 ms** | 3.2 ms | 2.0 ms | +1.12 ms |

Two `fsync` calls per write is the price of the atomic-write guarantee, and it
is charged once per mutation regardless of file size. It is not tunable: a
write that returns before its data is durable is a write that can be lost by a
power cut while the cache still reports it as committed.

### Chunked write (large files, CDC-split)

| Operation | p50 (ext4/NVMe) | p95 (ext4/NVMe) | p50 (tmpfs) |
|-----------|----:|----:|----:|
| Chunked write (72 KB, ~25 chunks) | 3.9 ms | 7.4 ms | 3.0 ms |
| Chunked write (360 KB, ~125 chunks) | 11.3 ms | 19.5 ms | 8.8 ms |
| Chunked re-read (72 KB, record_access fan-out) | 0.9 ms | 0.9 ms | 0.9 ms |

### Search

| Operation | p50 | p95 | Notes |
|-----------|----:|----:|-------|
| Search k=5 (cache **miss**) | 1.4 ms | n/a | BM25 keyword search (FTS5) |
| Search k=5 (cache **hit**) | **< 0.01 ms** | < 0.01 ms | In-session result LRU |
| Search k=10 (cache hit) | < 0.01 ms | < 0.01 ms | |

The in-session search cache delivers a **hundreds-fold speedup** on repeated queries (warm 0.007 ms vs. cold 6.8 ms over 5 queries — about 960× faster).

### Grep

| Operation | p50 | p95 |
|-----------|----:|----:|
| Literal (`def `) | 1.5 ms | 1.7 ms |
| Regex (`class\s+\w+`) | 3.4 ms | 3.5 ms |

Since 0.5.3 a pattern whose shape can backtrack catastrophically (a repeatable
group wrapping an unbounded quantifier, e.g. `(a+)+$`) is rejected before
`re.compile`, in ~0.01 ms, rather than being compiled and run. See
[security.md](security.md#regular-expression-safety).

### Response shaping

`_finalize_payload` runs on every tool response. The `chars/4` fast-exit (added in 0.4.6) skips the BPE encode entirely when a payload is safely under the response token cap.

| Payload | p50 | p95 |
|---------|----:|----:|
| Small (single match, 25K cap) | < 0.01 ms | < 0.01 ms |
| Large (40 files × 5 matches) | 0.02 ms | 0.03 ms |

### Tokenizer

| Operation | p50 | Notes |
|-----------|----:|-------|
| Tokeniser (~108 KB) | 0.19 ms | Warm BPE encode |
| Tokeniser (~494 KB, all files) | 0.21 ms | Merge cache amortises full sweeps |

---

## Cache footprint on disk

`docstore.db` is a cache, so its size is a cost rather than a payload — and it
is dominated by accounting, not by content. A long-lived store measured here
held **242 files and 4.2 MB of text in a 139.2 MB file**, 33× its own payload.

### What the file is made of

Vacuuming at each stage isolates how much of the file is live. On a snapshot of
that store:

| Contents | Live size | |
|---|---:|---|
| Everything | 39.0 MB | |
| minus FTS5 delete markers | 15.4 MB | `optimize` — **runs at shutdown since 0.5.3** |
| minus the duplicated text copy | 8.6 MB | external-content FTS5 — not implemented |

FTS5 records a deletion as an index entry rather than removing one, so a store
that has evicted a lot of files carries the vocabulary of every file it ever
held. Only `optimize` (a full segment merge) discards them — the incremental
`merge` command does not: 50 successive `merge` calls moved the index
24.69 MB → 24.68 MB, while one `optimize` took it to 1.16 MB in 144 ms.

External-content FTS5 is the smallest of the wins and the only one needing a
schema migration. It also needs a code change that is easy to get wrong: with
external content FTS5 reads the content table to work out which tokens a delete
should remove, so `delete_by_ids` would have to drop the FTS row *before* the
document row rather than after. Getting that backwards leaves orphaned index
entries that `integrity-check` reports as clean.

### What the file actually does

The table above is not what you see on disk, because SQLite keeps freed pages
in the file and reuses them rather than returning them to the OS. Since 0.5.3
the store is opened in incremental auto-vacuum mode and shutdown merges the
index then hands the freed pages back. Driving a copy of that store through a
real `ContentStorage` open and close:

| Step | File | Cost |
|---|---:|---:|
| Before | 153.4 MB | — |
| Open — one-time `auto_vacuum=INCREMENTAL` + `VACUUM` | 37.4 MB | 50.7 ms, once per store |
| Close — `optimize` then `incremental_vacuum` | **17.5 MB** | 19.8 ms |
| Reopen | 17.5 MB | 0.4 ms |

All 244 files and 2,174 documents survive the rewrite; `grep` and `stats` are
unaffected.

**The two halves are independent and neither is sufficient alone.** `optimize`
stops the index carrying the vocabulary of every file ever evicted, but SQLite
retains the pages it frees and reuses them, so on its own it moved that store
from 139.2 MB to 139.2 MB. Only the vacuum hands pages back — and a bare
`PRAGMA incremental_vacuum` reclaims exactly one page per step, so it has to be
driven to completion or nothing measurably changes. A store created before
0.5.3 pays the rewrite once, on its first open; the mode is recorded in the
database header, so the check is self-describing, needs no marker file, and
never repeats.

Age-based pruning is deliberately absent. The store is bounded by
`MAX_CACHE_ENTRIES` (10,000 files) through W-TinyLFU, and that bound has room
to spare in practice — the 153 MB store above held **244 files and 4.2 MB of
text**, 2.4% of capacity. Its size was accounting, not content, so expiring
entries by age would have reclaimed almost nothing while adding delete markers
to do it.

---

## Why these numbers

Removing the embedding and vector layer made the write and cold-read paths much
cheaper, with no ONNX inference on the hot path, while the cache's token savings
stayed the same. The optimisations below still land directly in the table above:

| Optimisation | Where it lands | Visible effect |
|--------------|----------------|----------------|
| `stat` + cache lookup before `aread_bytes` | `cache/read.py` | Single unchanged read drops to ~1.1 ms (no disk I/O); the stat is taken before the read so a concurrent write is never cached as fresh |
| No embedding on write/refresh | `cache/store.py`, `cache/write.py` | Write and cold read no longer pay for inference |
| Single-pass diff + stats (`diff_with_stats`) | `core/text/_diff.py`, `cache/write.py` | Write/edit no longer run the line-matcher twice; 360 KB chunked write dropped from ~21 ms to ~9 ms |
| Adaptive diff context (2 lines under 100-line files) | `core/text/_diff.py`, `cache/_helpers.py` | Small-file diffs carry less context overhead; suppressed diffs keep per-hunk headers |
| In-session search-result LRU | `cache/search.py`, `cache/store.py` | Repeat-query hits at < 0.01 ms |
| Drop `// Stats:` line from diff content | `cache/read.py` | ~15 tokens trimmed per changed file in phase 4 |
| Char/4 fast-exit in `_finalize_payload` | `server/response.py` | Response shaping is sub-microsecond on small payloads |
| Char-budget grep truncation | `server/tools/__init__.py` | Large grep results stay under the response cap |
| Pre-stored search previews | `storage/docstore/__init__.py` | No re-slicing of chunked content at query time |
| `include_markers=False` default | `core/text/_summarize.py` | Summarisation no longer wastes tokens on `[N lines omitted]` markers |
| Shared line index across a batch (0.5.3) | `cache/_helpers.py`, `cache/write.py` | `batch_edit` rebuilt the line table once per edit; a 30-edit batch on a 20K-line file dropped 1210 ms → 656 ms |
| Metadata-only projections (0.5.3) | `storage/docstore/_docstore.py` | `get_stats` and `has_cached_paths_under` were loading every cached file's text to read one field: 11.0 → 5.6 ms and 12.6 → 3.1 ms |
| Bounded hash-cache retention (0.5.3) | `core/hashing/_blake.py` | The LRUs are keyed on the buffer they hashed, so entry counts were a memory bound of ~1.2 GB; sized to ~96 MB worst case, keeping the 1.6–3.1× hit speedup |
| Bounded binary sniff (0.5.3) | `cache/_helpers.py`, `cache/write.py` | The write/edit paths read whole files to inspect the first 8 KB |

---

## Concurrency model

| Decision | Effect |
|----------|--------|
| Single-thread `DetachedExecutor` for storage I/O | All blocking storage I/O (SQLite reads and writes) routes through one thread, so the single connection is never touched concurrently. |
| `asyncio.gather()` in `batch_smart_read` | Cache lookups and stat pre-fetch run in parallel; smart-read calls themselves serialise on the single executor. |
| Cache-aware short-circuit in `smart_read` | Skips `aread_bytes` and `count_tokens` on the unchanged fast path. |
| Async subprocess for formatters | `_format_file` doesn't freeze the event loop. |
| Regex matching stays on the calling thread | `re` holds the GIL for the duration of a match, so offloading a scan to the executor does not keep the event loop responsive — it only occupies the thread the store needs. Pattern shapes that can backtrack catastrophically are rejected up front instead. |

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
TMPDIR="$HOME/.cache/scmcp-bench" \
  uv run python benchmarks/benchmark_performance.py --json perf.json --samples
uv run python benchmarks/benchmark_token_savings.py --json tok.json
```

`--samples` includes raw per-iteration timings for distribution analysis.

---

[← Back to README](../README.md)
