#!/usr/bin/env python3
"""Latency benchmarks for semantic-cache-mcp.

Reports p50 / p95 / p99 across all core operations:

  - Tokenizer (BPE)
  - Cache read (cold, unchanged-fast-path, diff)
  - Batch read
  - Search (cold + warm via in-session cache)
  - Grep (literal + regex)
  - Write + edit
  - Chunked write + re-read
  - Response shaping (`_finalize_payload`)

Methodology: every operation is measured against a fixed corpus of the
project's own source files. The read-only scan phases (search, grep) run before
the mutation phases, and each mutation phase evicts the temp documents it
creates, so corpus size, and therefore scan latency, never grows with
`--iterations`. The measured corpus size is printed and recorded.

Usage:
    uv run python benchmarks/benchmark_performance.py
    uv run python benchmarks/benchmark_performance.py --json results.json
    uv run python benchmarks/benchmark_performance.py --iterations 20
"""

from __future__ import annotations

import asyncio
import itertools
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from _bench_lib import (  # noqa: E402
    BenchmarkReport,
    collect_metadata,
    common_argparser,
    print_header,
    time_async,
    time_sync,
)

from semantic_cache_mcp.cache import (  # noqa: E402, I001
    SemanticCache,
    batch_smart_read,
    semantic_search,
    smart_edit,
    smart_read,
    smart_write,
)
from semantic_cache_mcp.core.tokenizer import count_tokens  # noqa: E402
from semantic_cache_mcp.server.response import _finalize_payload  # noqa: E402


def _collect_source_files(limit: int | None = None) -> list[Path]:
    src = PROJECT_ROOT / "src" / "semantic_cache_mcp"
    files = sorted(src.rglob("*.py"), key=lambda p: p.stat().st_size, reverse=True)
    return files[:limit] if limit else files


def _copy_to_tmp(files: list[Path], src_root: Path, tmp: Path) -> list[Path]:
    copies: list[Path] = []
    for f in files:
        rel = f.relative_to(src_root)
        dest = tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copies.append(dest)
    return copies


async def _corpus_size(cache: SemanticCache) -> tuple[int, int]:
    """Current (files_cached, total_documents) in the cache.

    Printed at the scan phase so the corpus the search/grep ops run against is
    visible and verifiable, and any future accumulation regression is obvious.
    """
    stats = await cache._storage.get_stats()
    return int(stats.get("files_cached", 0)), int(stats.get("total_documents", 0))


# ---------------------------------------------------------------------------
# Benchmark groups
# ---------------------------------------------------------------------------


def bench_tokenizer(report: BenchmarkReport, files: list[Path], iters: int) -> None:
    text = files[0].read_text()
    _, t = time_sync(f"count_tokens ({len(text)} chars)", count_tokens, text, iterations=iters)
    report.add_timing(t)
    print(t.render())

    big = "\n".join(f.read_text() for f in files)
    _, t = time_sync(
        f"count_tokens ({len(big)} chars, all)", count_tokens, big, iterations=max(3, iters // 4)
    )
    report.add_timing(t)
    print(t.render())


async def bench_cache_read(
    report: BenchmarkReport, cache: SemanticCache, files: list[Path], iters: int
) -> None:
    # Cold-read pass (one-shot, populates cache).
    t0 = time.perf_counter()
    for f in files:
        await smart_read(cache, str(f))
    cold_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  {'Cold read (' + str(len(files)) + ' files, total)':<42s}  {cold_ms:>7.2f} ms")
    report.measurements["cold_read_total_ms"] = cold_ms
    report.measurements["cold_read_files"] = len(files)
    report.measurements["cold_read_per_file_ms"] = cold_ms / max(1, len(files))

    # Unchanged-fast-path single read (now skips disk I/O entirely).
    last, stats = await time_async(
        "Single unchanged read (fast path)", smart_read, cache, str(files[0]), iterations=iters
    )
    report.add_timing(stats)
    print(stats.render())

    # Whole-corpus unchanged re-read.
    async def _reread_all() -> None:
        for f in files:
            await smart_read(cache, str(f))

    _, stats = await time_async(
        f"Unchanged re-read ({len(files)} files)",
        _reread_all,
        iterations=max(3, iters // 2),
    )
    report.add_timing(stats)
    print(stats.render())

    # Diff path: change one file mid-corpus, measure the re-read.
    target = files[1]
    original = target.read_text()
    target.write_text(original + "\n# bench-edit-marker\n")
    _, stats = await time_async(
        "Single diff read (changed file)",
        smart_read,
        cache,
        str(target),
        iterations=max(3, iters // 2),
    )
    report.add_timing(stats)
    print(stats.render())
    target.write_text(original)
    # Resync the cache to the restored content so the read-only phases that
    # follow see the pristine baseline corpus, not the transient edit.
    await smart_read(cache, str(target))


async def bench_batch_read(
    report: BenchmarkReport, cache: SemanticCache, files: list[Path], iters: int
) -> None:
    paths = [str(f) for f in files]

    async def _batch() -> None:
        await batch_smart_read(cache, paths, max_total_tokens=200_000, diff_mode=True)

    _, stats = await time_async(
        f"batch_read ({len(files)} files, diff_mode)",
        _batch,
        iterations=max(3, iters // 2),
    )
    report.add_timing(stats)
    print(stats.render())


async def bench_write_edit(
    report: BenchmarkReport, cache: SemanticCache, tmp: Path, iters: int
) -> None:
    test_file = tmp / "bench_write_test.py"
    content = "def hello():\n    return 'world'\n" * 100

    async def _do_write() -> None:
        await smart_write(cache, str(test_file), content)

    _, stats = await time_async("Write (200-line file)", _do_write, iterations=max(3, iters // 2))
    report.add_timing(stats)
    print(stats.render())

    await smart_read(cache, str(test_file))

    async def _do_edit() -> None:
        await smart_edit(
            cache,
            str(test_file),
            old_string="def hello():\n    return 'world'",
            new_string="def greet():\n    return 'earth'",
            start_line=1,
            end_line=2,
        )
        # Reset for next iteration
        test_file.write_text(content)

    _, stats = await time_async(
        "Edit (scoped find/replace)", _do_edit, iterations=max(3, iters // 2)
    )
    report.add_timing(stats)
    print(stats.render())

    # Keep the corpus at the seeded baseline: drop the temp file we created.
    await cache.delete_path(str(test_file))


async def bench_chunked_write(
    report: BenchmarkReport, cache: SemanticCache, tmp: Path, iters: int
) -> None:
    """Files >= CHUNK_THRESHOLD (8 KB) are CDC-chunked on write.

    The chunked path does N per-chunk operations (BPE token counts, JSON
    history fan-out, per-chunk FTS5 inserts). This case exercises the
    code paths most affected by the chunked-write hot path and is
    therefore where any future regression would surface first.
    """
    # ~50 KB -> ~25 chunks at CHUNK_MIN_SIZE=2048.
    medium_text = "".join(f"def func_{i}():\n    return {i} * 2\n\n" for i in range(2000))

    # ~250 KB -> ~125 chunks; stresses the per-chunk fan-out paths.
    big_text = medium_text * 5

    medium_kb = len(medium_text.encode("utf-8")) // 1024
    big_kb = len(big_text.encode("utf-8")) // 1024

    # Every temp path created in this phase; evicted at the end so they never
    # inflate a later phase's corpus.
    created: list[str] = []

    # Chunked re-read measured first, on the seeded baseline corpus, so its
    # record_access path-lookup cost reflects a fixed corpus and not the
    # throughput writes accumulated below.
    reread_path = tmp / "bench_chunked_reread.py"
    created.append(str(reread_path))
    await smart_write(cache, str(reread_path), medium_text)
    await smart_read(cache, str(reread_path))

    async def _reread_chunked() -> None:
        await smart_read(cache, str(reread_path))

    _, stats = await time_async(
        f"Chunked re-read ({medium_kb} KB, record_access fan-out)",
        _reread_chunked,
        iterations=max(3, iters // 2),
    )
    report.add_timing(stats)
    print(stats.render())

    # Chunked write throughput: each iteration targets a fresh path so it
    # measures a cold CDC ingest, not a warm overwrite.
    medium_counter = itertools.count()
    big_counter = itertools.count()

    async def _write_medium() -> None:
        path = tmp / f"bench_chunked_medium_{next(medium_counter)}.py"
        created.append(str(path))
        await smart_write(cache, str(path), medium_text)

    _, stats = await time_async(
        f"Chunked write ({medium_kb} KB, ~25 chunks)",
        _write_medium,
        iterations=max(3, iters // 2),
    )
    report.add_timing(stats)
    print(stats.render())

    async def _write_big() -> None:
        path = tmp / f"bench_chunked_big_{next(big_counter)}.py"
        created.append(str(path))
        await smart_write(cache, str(path), big_text)

    _, stats = await time_async(
        f"Chunked write ({big_kb} KB, ~125 chunks)",
        _write_big,
        iterations=max(3, iters // 2),
    )
    report.add_timing(stats)
    print(stats.render())

    # Restore the seeded baseline: evict every temp doc created in this phase.
    for path_str in created:
        await cache.delete_path(path_str)


async def bench_search(report: BenchmarkReport, cache: SemanticCache, iters: int) -> None:
    files_cached, total_docs = await _corpus_size(cache)
    print(f"  {'corpus (fixed)':<42s}  {files_cached} files / {total_docs} docs")
    report.measurements["search_corpus_files"] = files_cached
    report.measurements["search_corpus_docs"] = total_docs

    query = "content storage docstore"

    # Cold search: each iteration evicts the cache by mutating the store.
    miss_samples: list[float] = []
    for _ in range(max(3, iters // 2)):
        cache._search_cache.clear()
        t0 = time.perf_counter()
        await semantic_search(cache, query, k=5)
        miss_samples.append((time.perf_counter() - t0) * 1000.0)
    miss_p50 = sorted(miss_samples)[len(miss_samples) // 2]
    print(f"  {'Search k=5 (cache miss)':<42s}  p50={miss_p50:>7.2f}  (n={len(miss_samples)})")
    report.measurements["search_miss_p50_ms"] = miss_p50

    # Warm search: same query repeated — hits the in-session result cache.
    _, stats = await time_async(
        "Search k=5 (cache hit)",
        semantic_search,
        cache,
        query,
        k=5,
        iterations=iters,
    )
    report.add_timing(stats)
    print(stats.render())

    _, stats = await time_async(
        "Search k=10 (cache hit)",
        semantic_search,
        cache,
        "file caching and diff",
        k=10,
        iterations=iters,
    )
    report.add_timing(stats)
    print(stats.render())


async def bench_grep(report: BenchmarkReport, cache: SemanticCache, iters: int) -> None:
    storage = cache._storage

    async def _literal() -> None:
        await storage.grep("def ", fixed_string=True, max_matches=100)

    _, stats = await time_async("Grep literal 'def '", _literal, iterations=iters)
    report.add_timing(stats)
    print(stats.render())

    async def _regex() -> None:
        await storage.grep(r"class\s+\w+", max_matches=100)

    _, stats = await time_async("Grep regex 'class\\s+\\w+'", _regex, iterations=iters)
    report.add_timing(stats)
    print(stats.render())


def bench_response_shaping(report: BenchmarkReport, iters: int) -> None:
    """`_finalize_payload` is hit on every tool response; chars/4 fast-exit
    should make small payloads near-free."""
    small = {"ok": True, "tool": "read", "matches": [{"path": "x.py", "similarity": 0.9}]}
    big_files = [{"path": f"file_{i}.py", "matches": ["x" * 40] * 5} for i in range(40)]
    big = {"ok": True, "tool": "grep", "files": big_files}

    _, stats = time_sync(
        "_finalize_payload (small, 25K cap)",
        _finalize_payload,
        small,
        25_000,
        iterations=iters,
    )
    report.add_timing(stats)
    print(stats.render())

    _, stats = time_sync(
        "_finalize_payload (large, 25K cap)",
        _finalize_payload,
        big,
        25_000,
        iterations=iters,
    )
    report.add_timing(stats)
    print(stats.render())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    ap = common_argparser("Semantic Cache — Latency Benchmark")
    args = ap.parse_args()
    iters = args.iterations or 10

    metadata = collect_metadata(PROJECT_ROOT)
    report = BenchmarkReport(name="latency", metadata=metadata)

    if not args.quiet:
        print_header(report)

    source_files = _collect_source_files()
    src_root = PROJECT_ROOT / "src" / "semantic_cache_mcp"
    if not args.quiet:
        print(f"  files: {len(source_files)}    iterations: {iters}\n")

    with tempfile.TemporaryDirectory(prefix="scmcp_perf_") as tmp_str:
        tmp = Path(tmp_str)
        work_dir = tmp / "src"
        work_dir.mkdir()
        files = _copy_to_tmp(source_files, src_root, work_dir)
        cache = SemanticCache(db_path=tmp / "cache.db")

        if not args.quiet:
            print("\n--- Tokenizer ---")
        bench_tokenizer(report, files, iters)

        if not args.quiet:
            print("\n--- Cache Read ---")
        await bench_cache_read(report, cache, files, iters)

        if not args.quiet:
            print("\n--- Batch Read ---")
        await bench_batch_read(report, cache, files, iters)

        # Read-only scan phases run on the pristine seeded corpus, before any
        # mutation phase below can add documents to the store.
        if not args.quiet:
            print("\n--- Search ---")
        await bench_search(report, cache, iters)

        if not args.quiet:
            print("\n--- Grep ---")
        await bench_grep(report, cache, iters)

        if not args.quiet:
            print("\n--- Write + Edit ---")
        await bench_write_edit(report, cache, tmp, iters)

        if not args.quiet:
            print("\n--- Chunked Write ---")
        await bench_chunked_write(report, cache, tmp, iters)

        if not args.quiet:
            print("\n--- Response Shaping ---")
        bench_response_shaping(report, max(50, iters * 5))

    if args.json is not None:
        report.write_json(args.json, include_samples=args.samples)
        if not args.quiet:
            print(f"\n  json: {args.json}")

    if not args.quiet:
        print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
