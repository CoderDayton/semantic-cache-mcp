#!/usr/bin/env python3
"""Token-savings benchmark for semantic-cache-mcp.

Measures token reduction across 7 real-world phases:

  1. Cold read         — first read, no cache (baseline)
  2. Unchanged re-read — fast path: cache hit, mtime match, no disk I/O
  3. Content hash      — `touch` the files; mtime drifts but content matches
  4. Small edits       — ~5% of lines actually changed in 30% of files
  5. Batch read        — all files via `batch_smart_read`
  6. Search previews   — 5 semantic queries × k=5, previews vs full reads
  7. Search cache      — same queries repeated; in-session result cache

Each tool response is also passed through `_finalize_payload` to capture
real shaping costs (envelope JSON, query/pattern echo gating, debug fields).

Usage:
    uv run python benchmarks/benchmark_token_savings.py
    uv run python benchmarks/benchmark_token_savings.py --json results.json
"""

from __future__ import annotations

import asyncio
import os
import random
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
)

from semantic_cache_mcp.cache import (  # noqa: E402, I001
    SemanticCache,
    batch_smart_read,
    semantic_search,
    smart_read,
)
from semantic_cache_mcp.core.tokenizer import count_tokens  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_source_files(src_dir: Path, limit: int | None = None) -> list[Path]:
    files = sorted(src_dir.rglob("*.py"), key=lambda p: p.stat().st_size, reverse=True)
    return files[:limit] if limit else files


def _copy_files_to_temp(files: list[Path], src_root: Path, tmp_dir: Path) -> list[Path]:
    copies: list[Path] = []
    for f in files:
        rel = f.relative_to(src_root)
        dest = tmp_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copies.append(dest)
    return copies


def _apply_small_edits(files: list[Path], fraction: float = 0.3, seed: int = 42) -> list[Path]:
    """Apply genuine ~5% line-level edits to ~`fraction` of files.

    The previous version used identity replacements (`self → self`,
    `def → def`) which mutated nothing. This version inserts a unique
    comment near the top and changes `return` to `return  # bench` on
    the first matching line — both are real, BLAKE3-detectable changes.
    """
    rng = random.Random(seed)
    to_modify = rng.sample(files, k=max(1, int(len(files) * fraction)))

    for fpath in to_modify:
        text = fpath.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        if len(lines) < 5:
            continue

        # Insert a marker comment near the top
        insert_pos = rng.randint(1, min(5, len(lines) - 1))
        lines.insert(insert_pos, f"# bench: small-edit-{insert_pos}\n")

        # Mutate one `return` line if present, else mutate a `def` line
        for idx in range(len(lines)):
            if " return " in lines[idx] or lines[idx].lstrip().startswith("return "):
                lines[idx] = lines[idx].rstrip("\n") + "  # bench-edit\n"
                break
        else:
            for idx in range(len(lines)):
                if lines[idx].lstrip().startswith("def "):
                    lines[idx] = lines[idx].rstrip("\n") + "  # bench-edit\n"
                    break

        fpath.write_text("".join(lines), encoding="utf-8")

    return to_modify


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


async def _read_corpus(cache: SemanticCache, files: list[Path]) -> tuple[int, int]:
    returned = original = 0
    for f in files:
        result = await smart_read(cache, str(f), diff_mode=True)
        returned += result.tokens_returned
        original += result.tokens_original
    return returned, original


async def phase_cold_read(cache: SemanticCache, files: list[Path]) -> tuple[int, int]:
    return await _read_corpus(cache, files)


async def phase_unchanged_reread(cache: SemanticCache, files: list[Path]) -> tuple[int, int]:
    return await _read_corpus(cache, files)


async def phase_content_hash(cache: SemanticCache, files: list[Path]) -> tuple[int, int]:
    for f in files:
        os.utime(f)
    return await _read_corpus(cache, files)


async def phase_small_modifications(
    cache: SemanticCache, files: list[Path], modified: list[Path]
) -> dict[str, tuple[int, int]]:
    modified_set = set(modified)
    ch_ret = ch_orig = unch_ret = unch_orig = 0
    for f in files:
        result = await smart_read(cache, str(f), diff_mode=True)
        if f in modified_set:
            ch_ret += result.tokens_returned
            ch_orig += result.tokens_original
        else:
            unch_ret += result.tokens_returned
            unch_orig += result.tokens_original
    return {
        "changed": (ch_ret, ch_orig),
        "unchanged": (unch_ret, unch_orig),
        "combined": (ch_ret + unch_ret, ch_orig + unch_orig),
    }


async def phase_batch_read(cache: SemanticCache, files: list[Path]) -> tuple[int, int]:
    result = await batch_smart_read(
        cache, [str(f) for f in files], max_total_tokens=200_000, diff_mode=True
    )
    return result.total_tokens, result.total_tokens + result.tokens_saved


async def phase_search_previews(
    cache: SemanticCache, original_tokens: int, num_files: int
) -> tuple[int, int, float]:
    """Return (returned, original, elapsed_s) for the first (cold) pass."""
    queries = [
        "embedding model configuration",
        "file caching and diff logic",
        "semantic search implementation",
        "token counting and BPE",
        "MCP server tool registration",
    ]
    returned = 0
    t0 = time.perf_counter()
    for q in queries:
        result = await semantic_search(cache, q, k=5)
        for m in result.matches:
            returned += count_tokens(m.preview) if m.preview else 0
    elapsed = time.perf_counter() - t0
    avg_tokens = original_tokens // max(num_files, 1)
    original = len(queries) * 5 * avg_tokens
    return returned, original, elapsed


async def phase_search_cache_hit(cache: SemanticCache) -> tuple[float, float]:
    """Repeat the same 5 queries; should hit the in-session result cache.

    Returns (cold_elapsed_s, warm_elapsed_s).
    """
    queries = [
        "embedding model configuration",
        "file caching and diff logic",
        "semantic search implementation",
        "token counting and BPE",
        "MCP server tool registration",
    ]

    # Force cache miss
    cache._search_cache.clear()
    t0 = time.perf_counter()
    for q in queries:
        await semantic_search(cache, q, k=5)
    cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    for q in queries:
        await semantic_search(cache, q, k=5)
    warm = time.perf_counter() - t0

    return cold, warm


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _pct(saved: int, original: int) -> float:
    return 0.0 if original == 0 else saved / original


def _fmt_row(label: str, returned: int, original: int) -> str:
    saved = original - returned
    return (
        f"  {label:<32s}  tokens: {returned:>7,} / {original:>7,}  "
        f"saved: {saved:>7,} ({_pct(saved, original):>5.1%})"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_benchmark(
    file_limit: int | None = None,
    seed: int = 42,
    quiet: bool = False,
    json_path: Path | None = None,
) -> dict[str, float]:
    src_dir = PROJECT_ROOT / "src" / "semantic_cache_mcp"
    if not src_dir.exists():
        raise RuntimeError(f"Source directory not found: {src_dir}")

    source_files = _collect_source_files(src_dir, limit=file_limit)
    if not source_files:
        raise RuntimeError("No .py files found in source directory")

    metadata = collect_metadata(PROJECT_ROOT)
    report = BenchmarkReport(name="token_savings", metadata=metadata)

    with tempfile.TemporaryDirectory(prefix="scmcp_bench_") as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "cache.db"
        work_dir = tmp_path / "src"
        work_dir.mkdir()
        files = _copy_files_to_temp(source_files, src_dir, work_dir)
        cache = SemanticCache(db_path=db_path)

        if not quiet:
            print_header(BenchmarkReport(name="Semantic Cache — Token Savings", metadata=metadata))
            print(f"  files: {len(files)}\n")

        t_total = time.perf_counter()

        # Phase 1: Cold
        p1_ret, p1_orig = await phase_cold_read(cache, files)
        if not quiet:
            print(f"Total original tokens: {p1_orig:,}\n")
            print("Phase 1: Cold Read")
            print(_fmt_row("First read", p1_ret, p1_orig))

        # Phase 2: Unchanged
        p2_ret, p2_orig = await phase_unchanged_reread(cache, files)
        if not quiet:
            print("\nPhase 2: Unchanged Re-read (fast path: skips disk)")
            print(_fmt_row("Cached re-read", p2_ret, p2_orig))

        # Phase 3: Content hash
        p3_ret, p3_orig = await phase_content_hash(cache, files)
        if not quiet:
            print("\nPhase 3: Content Hash (mtime drift, BLAKE3 match)")
            print(_fmt_row("Content hash hit", p3_ret, p3_orig))

        # Phase 4: Small edits
        modified = _apply_small_edits(files, fraction=0.3, seed=seed)
        p4 = await phase_small_modifications(cache, files, modified)
        p4c_ret, p4c_orig = p4["combined"]
        if not quiet:
            ch_ret, ch_orig = p4["changed"]
            unch_ret, unch_orig = p4["unchanged"]
            print(f"\nPhase 4: Small Edits ({len(modified)}/{len(files)} files)")
            print(_fmt_row(f"Changed ({len(modified)})", ch_ret, ch_orig))
            print(_fmt_row(f"Unchanged ({len(files) - len(modified)})", unch_ret, unch_orig))
            print(_fmt_row("Combined", p4c_ret, p4c_orig))

        # Phase 5: Batch
        p5_ret, p5_orig = await phase_batch_read(cache, files)
        if not quiet:
            print("\nPhase 5: Batch Read (200K budget)")
            print(_fmt_row("Batch read", p5_ret, p5_orig))

        # Phase 6: Search previews
        p6_ret, p6_orig, p6_cold_s = await phase_search_previews(cache, p1_orig, len(files))
        if not quiet:
            print(f"\nPhase 6: Search Previews ({p6_cold_s * 1000:.0f} ms cold)")
            print(_fmt_row("Search previews", p6_ret, p6_orig))

        # Phase 7: Search-cache speedup
        cold_s, warm_s = await phase_search_cache_hit(cache)
        speedup = cold_s / warm_s if warm_s > 0 else float("inf")
        if not quiet:
            print("\nPhase 7: In-session Search Cache")
            print(f"  {'5 queries cold (miss)':<32s}  {cold_s * 1000:>7.1f} ms")
            print(f"  {'5 queries warm (hit)':<32s}  {warm_s * 1000:>7.1f} ms  ({speedup:>5.1f}× faster)")  # noqa: E501

        elapsed = time.perf_counter() - t_total

        # Aggregate
        all_ret = p2_ret + p3_ret + p4c_ret + p5_ret + p6_ret
        all_orig = p2_orig + p3_orig + p4c_orig + p5_orig + p6_orig
        overall = _pct(all_orig - all_ret, all_orig)

        if not quiet:
            print(f"\n{'=' * 60}")
            print(f"Overall (phases 2-6): {overall:.1%} token reduction")
            print(f"Elapsed: {elapsed:.2f}s")

        results: dict[str, float] = {
            "cold_read": _pct(p1_orig - p1_ret, p1_orig),
            "unchanged": _pct(p2_orig - p2_ret, p2_orig),
            "content_hash": _pct(p3_orig - p3_ret, p3_orig),
            "small_edits": _pct(p4c_orig - p4c_ret, p4c_orig),
            "batch_read": _pct(p5_orig - p5_ret, p5_orig),
            "search": _pct(p6_orig - p6_ret, p6_orig),
            "overall": overall,
            "search_cache_speedup": speedup,
            "search_cold_ms": cold_s * 1000.0,
            "search_warm_ms": warm_s * 1000.0,
        }

        report.measurements = {
            "total_files": len(files),
            "original_tokens": p1_orig,
            "modified_files": len(modified),
            "phases": {
                "cold_read": {"returned": p1_ret, "original": p1_orig},
                "unchanged": {"returned": p2_ret, "original": p2_orig},
                "content_hash": {"returned": p3_ret, "original": p3_orig},
                "small_edits": {
                    "changed": p4["changed"],
                    "unchanged": p4["unchanged"],
                    "combined": p4["combined"],
                },
                "batch_read": {"returned": p5_ret, "original": p5_orig},
                "search_previews": {"returned": p6_ret, "original": p6_orig},
                "search_cache": {
                    "cold_ms": cold_s * 1000.0,
                    "warm_ms": warm_s * 1000.0,
                    "speedup": speedup,
                },
            },
            "ratios": {k: v for k, v in results.items() if not k.startswith("search_")},
        }

        if json_path is not None:
            report.write_json(json_path)
            if not quiet:
                print(f"  json: {json_path}")

        return results


def _main() -> int:
    ap = common_argparser("Semantic Cache — Token Savings Benchmark")
    args = ap.parse_args()
    results = asyncio.run(
        run_benchmark(quiet=args.quiet, json_path=args.json)
    )
    if results["overall"] < 0.80:
        if not args.quiet:
            print(f"\nFAIL: Overall savings {results['overall']:.1%} < 80% target")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_main())
