"""Pytest wrapper for token savings benchmark.

Runs the benchmark on a small file set (5 files) and gates the savings the
README advertises. The floors sit just under what the benchmark actually
measures (99.8% on every phase at this file limit, 98.9% over the full
corpus), because a floor 19 points below the published number defends
nothing: a collapse from 99% to 81% would pass it silently. Each floor keeps
only enough headroom for the corpus itself changing, since the benchmark
reads this repo's own `src/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure benchmark module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from benchmark_token_savings import run_benchmark


@pytest.fixture(scope="module")
async def benchmark_results() -> dict[str, float]:
    """Run benchmark once for all tests in this module."""
    return await run_benchmark(file_limit=5, seed=42, quiet=True)


def test_unchanged_reread_savings(benchmark_results: dict[str, float]) -> None:
    """Phase 2: an unchanged re-read sends a marker, not the file."""
    assert benchmark_results["unchanged"] >= 0.99, (
        f"Unchanged re-read savings {benchmark_results['unchanged']:.1%} < 99%"
    )


def test_content_hash_savings(benchmark_results: dict[str, float]) -> None:
    """Phase 3: a drifted mtime over identical bytes costs the same marker."""
    assert benchmark_results["content_hash"] >= 0.99, (
        f"Content hash savings {benchmark_results['content_hash']:.1%} < 99%"
    )


def test_small_edits_savings(benchmark_results: dict[str, float]) -> None:
    """Phase 4: real edits in 30% of files come back as diffs."""
    assert benchmark_results["small_edits"] >= 0.95, (
        f"Small edits savings {benchmark_results['small_edits']:.1%} < 95%"
    )


def test_batch_read_savings(benchmark_results: dict[str, float]) -> None:
    """Phase 5: a whole-corpus batch_read with hashes echoed back."""
    assert benchmark_results["batch_read"] >= 0.95, (
        f"Batch read savings {benchmark_results['batch_read']:.1%} < 95%"
    )


def test_search_savings(benchmark_results: dict[str, float]) -> None:
    """Phase 6: previews instead of full reads. The loosest floor of the six —
    preview size tracks where the matching term lands, so it moves with the
    corpus more than the hash-driven phases do."""
    assert benchmark_results["search"] >= 0.90, (
        f"Search savings {benchmark_results['search']:.1%} < 90%"
    )


def test_overall_savings(benchmark_results: dict[str, float]) -> None:
    """Phases 2-6 aggregate: the README's headline number."""
    assert benchmark_results["overall"] >= 0.97, (
        f"Overall savings {benchmark_results['overall']:.1%} < 97%, "
        "which is the number README.md publishes"
    )
