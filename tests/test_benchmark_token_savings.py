"""Pytest wrapper for token savings benchmark.

Runs the benchmark on a small file set (5 files) and asserts ≥80% savings
for cached read phases. This ensures the README's "80%+ token reduction"
claim is continuously verified in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure benchmark module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from benchmark_token_savings import run_benchmark


@pytest.fixture(scope="module")
def benchmark_results() -> dict[str, float]:
    """Run benchmark once for all tests in this module."""
    return run_benchmark(file_limit=5, seed=42, quiet=True)


def test_unchanged_reread_savings(benchmark_results: dict[str, float]) -> None:
    """Phase 2: Unchanged re-reads should save ≥95% tokens."""
    assert benchmark_results["unchanged"] >= 0.95, (
        f"Unchanged re-read savings {benchmark_results['unchanged']:.1%} < 95%"
    )


def test_small_edits_savings(benchmark_results: dict[str, float]) -> None:
    """Phase 3: Mixed changed/unchanged should save ≥80% tokens."""
    assert benchmark_results["small_edits"] >= 0.80, (
        f"Small edits savings {benchmark_results['small_edits']:.1%} < 80%"
    )


def test_batch_read_savings(benchmark_results: dict[str, float]) -> None:
    """Phase 4: Batch read should save ≥80% tokens."""
    assert benchmark_results["batch_read"] >= 0.80, (
        f"Batch read savings {benchmark_results['batch_read']:.1%} < 80%"
    )


def test_overall_savings(benchmark_results: dict[str, float]) -> None:
    """Overall savings across phases 2-4 should meet the 80% claim."""
    assert benchmark_results["overall"] >= 0.80, (
        f"Overall savings {benchmark_results['overall']:.1%} < 80%"
    )
