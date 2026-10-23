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
| **Corpus** | 48 source files, **131,899 tokens**, 287 documents |
| **Version** | `0.6.0` + unreleased (`e65c5bf`) |

Latency rows are the per-row median of three ext4 runs; the tmpfs column is one
run. Token counts are exact o200k counts, matching tiktoken. Before `571937a`
they were estimates that ran about 1.7× high, which is why this corpus measures
fewer tokens than the 44-file one did.

> **The filesystem is part of the measurement.** The suite works in a
> `tempfile.TemporaryDirectory`, which follows `TMPDIR` — and on most Linux
> systems `/tmp` is tmpfs, i.e. RAM. tmpfs discards `fsync`, so a benchmark run
> there reports a write latency the durable path never actually achieves. Since
> 0.5.3 `awrite_atomic` flushes the file *and* its parent directory before the
> rename, so the write path now costs a real disk round-trip and the difference
> is no longer noise: **+1.1 ms per write, +1.7 ms per edit** on this machine.
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

Each phase reads the same 48-file corpus through `smart_read` / `batch_smart_read` / `semantic_search` and reports tokens emitted vs. tokens that would have been read in the absence of the cache.

Every saving below is earned by a caller that echoes back the `content_hash` it was given (`known_hash` on `read`, `known_hashes` on `batch_read`). Since 0.5.2 that is a hard requirement, not an optimization: a caller that cannot prove it still holds a file is sent the file. The phases model an agent that keeps its hashes, which is the flow the numbers describe.

Since 0.5.3 a ranged read earns a `coverage_token` on the same terms, redeemable for the lines it delivered. These phases read whole files, so that path is not exercised by the table below and contributes nothing to these numbers — the row it would change is a repeated read of one region of a large file, which this corpus does not model.

| # | Phase | Trigger | Tokens returned | Original | Savings |
|---|-------|---------|----------------:|---------:|--------:|
| 1 | Cold read | First read, no cache (baseline) | 103,627 | 131,899 | 21.4% |
| 2 | Unchanged re-read | mtime match, **fast path skips disk I/O** | 1,313 | 131,899 | **99.0%** |
| 3 | Content hash | mtime drifted (e.g. `git checkout`), BLAKE3 still matches | 1,313 | 131,899 | **99.0%** |
| 4 | Small edits (14/48 changed) | Real ~5% line changes on 30% of files | 2,928 | 132,073 | **97.8%** |
| 4a |  → changed files only | Returned as unified diff (bare hunks, no file headers) | 2,009 | 50,073 | 96.0% |
| 4b |  → unchanged files | Fast path | 919 | 82,000 | 98.9% |
| 5 | Batch read (200K budget) | `batch_smart_read` over the whole corpus, echoing the hashes phase 4 returned | 1,313 | 132,073 | **99.0%** |
| 6 | Search previews | 5 keyword queries × k=5, previews vs. full reads | 1,086 | 68,675 | **98.4%** |

**Aggregate (phases 2 to 6): 98.7% token reduction.**

Phase 1 is the no-cache baseline. It returns 21.4% fewer tokens than the corpus
holds because one file (`server/tools/__init__.py`) exceeds `MAX_CONTENT_SIZE`
and comes back as its outline — a first read of an oversized file is not a
cache saving, and should not be read as one. That one file is outlined on every
unproven pass, including cold reads and `batch_read`.

Phase 6 costs more than it did in 0.5.1 (98.4% vs. 99.7%), and deliberately so:
`search` joins query terms with `OR`, so a query returns the files that match
*some* of it rather than only the files that match all of it. The extra tokens
buy back the queries that would otherwise return nothing at all.

The CI test [`tests/test_benchmark_token_savings.py`](../tests/test_benchmark_token_savings.py) asserts ≥ 80% overall as a regression gate.

### Token reduction by strategy

| Strategy | Savings | Trigger |
|----------|--------:|---------|
| Unchanged (mtime) | ~99% | Caller's hash matches and `cached.mtime >= file.mtime`; disk read skipped entirely |
| Content hash | ~99% | Caller's hash matches; mtime drifted but BLAKE3 hash still matches |
| Diff (changed) | 80 to 95% | File modified since last cache, and the caller holds the base content; emitted as unified diff |
| No possession proof | 0% | Caller sent no matching hash — the file is returned in full, served from cache when fresh |
| Search previews | ~100% | `search` returns 200-char previews, never full files |
| Mapped | ~97% | File exceeds `MAX_CONTENT_SIZE`; outline returned, or first/last 20 lines when it has no definitions |

---

## Latency

All numbers are p50 unless otherwise noted; p95/p99 are reported in the raw output. Cold-read totals include disk I/O and tokenisation for the entire corpus. Every phase, including search and grep, runs against the same fixed 48-file corpus, so scan latency does not grow with the benchmark's iteration count.

### Cache read

| Operation | p50 | p95 | Notes |
|-----------|----:|----:|-------|
| Single unchanged read (fast path) | **0.17 ms** | 0.21 ms | mtime check + cache hit; **no disk I/O** |
| Single diff read (changed file) | 0.17 ms | 0.25 ms | Hash check + unified diff |
| Unchanged re-read (48 files) | 8.2 ms | 12.5 ms | Whole-corpus pass |
| Cold read (48 files, total) | n/a | n/a | 130 ms for a single unrepeated pass (~2.7 ms/file), including outlining the one file over `MAX_CONTENT_SIZE` |

The cold-read total is one pass, not a distribution — it is the only row here
with n=1, and it moves ±20% between runs (111–137 ms across these three). Treat
it as an order of magnitude.

The cached-read rows dropped about 7× with the indexed `path` column
(`c38e219`), which turned a by-path lookup from a scan of every document's
metadata into an index probe. Measured at the commit before it and at it, same
machine: single unchanged read 1.27 → 0.17 ms, whole-corpus re-read 21.4 →
7.4 ms, `batch_read` 58.4 → 41.4 ms.

### Batch read

| Operation | p50 | p95 |
|-----------|----:|----:|
| `batch_read` (48 files, diff mode) | 45.0 ms | 74.5 ms |

### Write + edit

The only rows on this page where the filesystem changes the answer. Both paths
go through `awrite_atomic`, which since 0.5.3 fsyncs the temp file and then the
parent directory before returning — so a rename can never land ahead of the
bytes it points at.

| Operation | p50 (ext4/NVMe) | p95 (ext4/NVMe) | p50 (tmpfs) | Cost of durability |
|-----------|----:|----:|----:|----:|
| Write (200-line file) | **1.9 ms** | 2.1 ms | 0.75 ms | +1.13 ms |
| Edit (scoped find/replace) | **2.5 ms** | 5.0 ms | 0.84 ms | +1.66 ms |

Two `fsync` calls per write is the price of the atomic-write guarantee, and it
is charged once per mutation regardless of file size. It is not tunable: a
write that returns before its data is durable is a write that can be lost by a
power cut while the cache still reports it as committed.

### Chunked write (large files, CDC-split)

| Operation | p50 (ext4/NVMe) | p95 (ext4/NVMe) | p50 (tmpfs) |
|-----------|----:|----:|----:|
| Chunked write (72 KB, ~25 chunks) | 3.8 ms | 11.9 ms | 1.7 ms |
| Chunked write (360 KB, ~125 chunks) | 10.8 ms | 21.2 ms | 5.5 ms |
| Chunked re-read (72 KB, record_access fan-out) | 0.17 ms | 0.18 ms | 0.15 ms |

Chunked writes are the noisiest rows on ext4: the 72 KB p50 was 3.7, 11.8 and
3.7 ms across the three runs, the 360 KB p50 7.3, 17.8 and 10.8 ms.

### Search

| Operation | p50 | p95 | Notes |
|-----------|----:|----:|-------|
| Search k=5 (cache **miss**) | 0.72 ms | n/a | BM25 keyword search (FTS5) |
| Search k=5 (cache **hit**) | **< 0.01 ms** | < 0.01 ms | In-session result LRU |
| Search k=10 (cache hit) | < 0.01 ms | < 0.01 ms | |

The in-session search cache delivers a **hundreds-fold speedup** on repeated queries (warm vs. cold 3.8 ms over 5 queries — about 510× faster).

### Grep

| Operation | p50 | p95 |
|-----------|----:|----:|
| Literal (`def `) | 2.2 ms | 2.9 ms |
| Regex (`class\s+\w+`) | 2.2 ms | 2.5 ms |

On a 197-file, 4.8 MB store, median of 7 runs, the trigram index against the
BM25 prefilter it falls back to:

| Pattern | BM25 prefilter | Trigram index |
|---------|---------------:|--------------:|
| `zz_not_present_anywhere` (miss) | 13.4 ms | 4.7 ms |
| `content_hash` (11 files) | 17.3 ms | 8.1 ms |
| `TODO` (9 files) | 12.8 ms | 10.2 ms |
| `def \w+\(` | 22.7 ms | 19.1 ms |
| `function` (93 files) | 23.0 ms | 20.7 ms |
| `\bparse` (35 files) | 32.2 ms | 31.7 ms |

The index adds 8.4 MB to that store (19.4 → 27.8 MB). The first grep after an
upgrade indexes every file it scans: 246 ms there, once, written in batches of
about 256K characters (20 batches, none holding the store lock over 18 ms). Positions are what make
a phrase an exact substring test; without them (`detail=none`, 0.8 MB) `parse`
admits 59 candidate files instead of 35.

Since 0.5.3 a pattern whose shape can backtrack catastrophically (a repeatable
group wrapping an unbounded quantifier, e.g. `(a+)+$`) is rejected before
`re.compile`, in ~0.01 ms, rather than being compiled and run. See
[security.md](security.md#regular-expression-safety).

### Response shaping

`_finalize_payload` runs on every tool response. It skips the BPE encode when the
rendered payload has at most `cap // 4` characters, since one character can cost
up to 4 byte-level tokens. Above that it counts, stopping as soon as the count
passes the cap (`tokens_exceed`).

| Payload | p50 | p95 |
|---------|----:|----:|
| Small (single match, 25K cap) | < 0.01 ms | < 0.01 ms |
| Large (40 files × 5 matches) | 0.24 ms | 0.26 ms |

The large payload is about 10K characters, over the 6,250-character bound for a
25K cap, so it is counted.

### Tokenizer

| Operation | p50 | Notes |
|-----------|----:|-------|
| Tokeniser (~131 KB) | 6.88 ms | Exact count, count memo cleared; 0.004 ms on a memo hit |
| Tokeniser (~576 KB, all files) | 3.32 ms | Sampled estimate (above 256,000 chars), 0.1% under exact; 0.002 ms on a memo hit |

The suite's own `count_tokens` rows time repeat calls on the same text, so they
measure the memo hit. The memo-cleared figures above are timed separately,
clearing the tokenizer's count memo before each call.

---

## Through the MCP server

The rows above call the cache in-process. These drive the installed
`semantic-cache-mcp` command over MCP stdio, the way a client does, so each
includes JSON-RPC framing, the server's tool worker and, when shared, the relay
hop. Work directory on ext4, processes pinned to the P-cores, 10 warm-up calls,
then p50 of 100 calls; each mode run three times, alternating, and the median
of the three p50s reported.

| Call | In-process | Shared daemon |
|------|-----------:|--------------:|
| `read`, unchanged (hash held) | 1.58 ms | 1.25 ms |
| `read`, small file in full | 1.49 ms | 1.31 ms |
| `edit` with `known_hash` | 1.92 ms | 1.71 ms |
| `write`, 50 lines | 7.26 ms | 5.54 ms |
| `grep`, fixed string | 3.13 ms | 3.57 ms |
| `grep`, regex `def \w+\(` | 5.21 ms | 2.91 ms |
| `search` | 1.03 ms | 0.87 ms |
| `glob` `**/*.py` | 4.26 ms | 5.42 ms |
| `stats` | 1.23 ms | 1.16 ms |
| Cold `read` of all 48 files | 280 ms | 274 ms |

The relay adds no measurable cost per call; the differences are within the
spread of the three runs.

| | In-process | Shared daemon |
|---|---:|---:|
| Client connect, no server running | 1.38 s | 1.39 s (starts the daemon) |
| Client connect, daemon already running | — | **31 ms** |
| 4 clients connected: processes, RSS | 12, 1,201 MB | **7, 385 MB** |

The same run checks behavior end to end: every tool through the real command,
two clients in different directories resolving relative paths against their
own, one daemon between them, and the daemon exiting after
`SCMCP_IDLE_TIMEOUT`.

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
| `stat` + cache lookup before `aread_bytes` | `cache/read.py` | Single unchanged read skips disk I/O; the stat is taken before the read so a concurrent write is never cached as fresh |
| Indexed `path` column | `storage/docstore/_docstore.py` | By-path lookups probe an index instead of scanning document metadata: single unchanged read 1.27 → 0.17 ms |
| No embedding on write/refresh | `cache/store.py`, `cache/write.py` | Write and cold read no longer pay for inference |
| Single-pass diff + stats (`diff_with_stats`) | `core/text/_diff.py`, `cache/write.py` | Write/edit no longer run the line-matcher twice; 360 KB chunked write dropped from ~21 ms to ~9 ms |
| Adaptive diff context (2 lines under 100-line files) | `core/text/_diff.py`, `cache/_helpers.py` | Small-file diffs carry less context overhead; suppressed diffs keep per-hunk headers |
| In-session search-result LRU | `cache/search.py`, `cache/store.py` | Repeat-query hits at < 0.01 ms |
| Drop `// Stats:` line from diff content | `cache/read.py` | ~15 tokens trimmed per changed file in phase 4 |
| Char/4 fast-exit in `_finalize_payload` | `server/response.py` | Response shaping is sub-microsecond on small payloads |
| Char-budget grep truncation | `server/tools/__init__.py` | Large grep results stay under the response cap |
| Pre-stored search previews | `storage/docstore/__init__.py` | No re-slicing of chunked content at query time |
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
