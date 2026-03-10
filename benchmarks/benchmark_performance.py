#!/usr/bin/env python3
"""Performance benchmark for semantic-cache-mcp.

Measures wall-clock timings for all core operations:

  - Embedding (single + batch)
  - Cache read (cold + warm + unchanged)
  - Cache write + edit
  - Search (keyword + semantic + hybrid)
  - Similarity lookup
  - Grep (regex across cached files)

Usage:
    uv run python benchmarks/benchmark_performance.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from semantic_cache_mcp.cache import (  # noqa: E402, I001
    SemanticCache,
    batch_smart_read,
    find_similar_files,
    semantic_search,
    smart_edit,
    smart_read,
    smart_write,
)
from semantic_cache_mcp.core.embeddings import embed, embed_batch, warmup  # noqa: E402
from semantic_cache_mcp.core.tokenizer import count_tokens  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _timed(label: str, fn, *args, iterations: int = 1, **kwargs):
    """Run fn, print timing, return result of last call."""
    times: list[float] = []
    result = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    if iterations > 1:
        print(f"  {label:<40s}  {avg*1000:>8.1f} ms  (avg of {iterations})")
    else:
        print(f"  {label:<40s}  {avg*1000:>8.1f} ms")
    return result


async def _timed_async(label: str, fn, *args, iterations: int = 1, **kwargs):
    """Run async fn, print timing, return result of last call."""
    times: list[float] = []
    result = None
    for _ in range(iterations):
        t0 = time.perf_counter()
        result = await fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    if iterations > 1:
        print(f"  {label:<40s}  {avg*1000:>8.1f} ms  (avg of {iterations})")
    else:
        print(f"  {label:<40s}  {avg*1000:>8.1f} ms")
    return result


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def bench_embedding(files: list[Path]) -> None:
    print("\n--- Embedding ---")

    # Single embed
    text = files[0].read_text()
    _timed("Single embed (largest file)", embed, text, iterations=5)

    # Batch embed
    texts = [f.read_text() for f in files[:10]]
    _timed(f"Batch embed ({len(texts)} files)", embed_batch, texts, iterations=3)

    # Small text
    _timed("Single embed (short string)", embed, "def hello(): pass", iterations=10)


def bench_tokenizer(files: list[Path]) -> None:
    print("\n--- Tokenizer ---")
    text = files[0].read_text()
    _timed(f"count_tokens ({len(text)} chars)", count_tokens, text, iterations=10)

    all_text = "\n".join(f.read_text() for f in files)
    _timed(f"count_tokens ({len(all_text)} chars, all files)", count_tokens, all_text, iterations=3)


async def bench_cache_read(cache: SemanticCache, files: list[Path]) -> None:
    print("\n--- Cache Read ---")

    # Cold read (first time, populates cache)
    t0 = time.perf_counter()
    for f in files:
        await smart_read(cache, str(f))
    elapsed = time.perf_counter() - t0
    print(f"  {'Cold read (' + str(len(files)) + ' files)':<40s}  {elapsed*1000:>8.1f} ms")

    # Warm read (unchanged, should be near-instant)
    t0 = time.perf_counter()
    for f in files:
        await smart_read(cache, str(f))
    elapsed = time.perf_counter() - t0
    print(f"  {'Unchanged re-read (' + str(len(files)) + ' files)':<40s}  {elapsed*1000:>8.1f} ms")

    # Single file unchanged
    await _timed_async("Single unchanged read", smart_read, cache, str(files[0]), iterations=20)


async def bench_batch_read(cache: SemanticCache, files: list[Path]) -> None:
    print("\n--- Batch Read ---")
    paths = [str(f) for f in files]
    await _timed_async(
        f"batch_read ({len(files)} files, diff_mode)",
        batch_smart_read, cache, paths,
        max_total_tokens=200_000, diff_mode=True,
    )


async def bench_write_edit(cache: SemanticCache, tmp: Path) -> None:
    print("\n--- Write + Edit ---")

    test_file = tmp / "bench_write_test.py"
    content = "def hello():\n    return 'world'\n" * 100

    await _timed_async("Write (new file, 200 lines)", smart_write, cache, str(test_file), content)

    # Read it first so edit has cache
    await smart_read(cache, str(test_file))

    await _timed_async(
        "Edit (scoped find/replace)",
        smart_edit, cache, str(test_file),
        old_string="def hello():\n    return 'world'",
        new_string="def greet():\n    return 'earth'",
        start_line=1, end_line=2,
    )


async def bench_search(cache: SemanticCache) -> None:
    print("\n--- Search ---")
    await _timed_async(
        "Semantic search k=5", semantic_search, cache,
        "embedding model configuration", k=5, iterations=3,
    )
    await _timed_async(
        "Semantic search k=10", semantic_search, cache,
        "file caching and diff", k=10, iterations=3,
    )


async def bench_similar(cache: SemanticCache, files: list[Path]) -> None:
    print("\n--- Similar ---")
    await _timed_async("Find similar k=3", find_similar_files, cache, str(files[0]), k=3, iterations=3)
    await _timed_async("Find similar k=10", find_similar_files, cache, str(files[0]), k=10, iterations=3)


async def bench_grep(cache: SemanticCache) -> None:
    print("\n--- Grep ---")
    storage = cache._storage
    await _timed_async(
        "Grep (literal, 'def ')",
        storage.grep, "def ", fixed_string=True, max_matches=100,
        iterations=3,
    )
    await _timed_async(
        "Grep (regex, 'class\\s+\\w+')",
        storage.grep, r"class\s+\w+", max_matches=100,
        iterations=3,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    print("Semantic Cache Performance Benchmark")
    print("=" * 55)

    source_files = _collect_source_files()
    src_root = PROJECT_ROOT / "src" / "semantic_cache_mcp"
    print(f"Source files: {len(source_files)}")

    # Warmup embedding model first
    print("\nWarming up embedding model...")
    t0 = time.perf_counter()
    warmup()
    print(f"  Model warmup: {(time.perf_counter() - t0)*1000:.1f} ms")

    with tempfile.TemporaryDirectory(prefix="scmcp_perf_") as tmp_str:
        tmp = Path(tmp_str)
        work_dir = tmp / "src"
        work_dir.mkdir()
        files = _copy_to_tmp(source_files, src_root, work_dir)

        cache = SemanticCache(db_path=tmp / "cache.db")

        bench_tokenizer(files)
        bench_embedding(files)
        await bench_cache_read(cache, files)
        await bench_batch_read(cache, files)
        await bench_write_edit(cache, tmp)
        await bench_search(cache)
        await bench_similar(cache, files)
        await bench_grep(cache)

    print(f"\n{'=' * 55}")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
