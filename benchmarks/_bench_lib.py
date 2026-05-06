"""Shared helpers for the semantic-cache-mcp benchmark suite.

Provides:
  - statistics-aware timing (mean, median, p95, p99, stdev, min, max)
  - environment metadata capture (git SHA, Python, OS, CPU)
  - JSON result writer for CI / diffing across runs
  - small CLI wrapper for `--json <path>` and `--quiet`

Designed to be imported by both `benchmark_performance.py` and
`benchmark_token_savings.py` so they share output format and methodology.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess  # nosec B404 — used for `git rev-parse` only
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


@dataclass
class TimingStats:
    """Distributional summary of a timed operation."""

    label: str
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    samples_ms: list[float] = field(default_factory=list)

    def render(self) -> str:
        """Single-line human-readable rendering."""
        if self.iterations >= 5:
            return (
                f"  {self.label:<42s}  "
                f"p50={self.median_ms:>7.2f}  "
                f"p95={self.p95_ms:>7.2f}  "
                f"p99={self.p99_ms:>7.2f}  "
                f"(n={self.iterations})"
            )
        return f"  {self.label:<42s}  {self.mean_ms:>7.2f} ms  (n={self.iterations})"

    def to_dict(self, include_samples: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_samples:
            d.pop("samples_ms", None)
        return d


def _stats(label: str, samples_s: list[float]) -> TimingStats:
    samples_ms = [s * 1000.0 for s in samples_s]
    if not samples_ms:
        raise ValueError("no samples")
    n = len(samples_ms)
    sorted_ms = sorted(samples_ms)

    def _percentile(p: float) -> float:
        # Nearest-rank percentile (NIST definition); robust for small n.
        idx = max(0, min(n - 1, int(round(p / 100.0 * n + 0.5)) - 1))
        return sorted_ms[idx]

    return TimingStats(
        label=label,
        iterations=n,
        mean_ms=statistics.fmean(samples_ms),
        median_ms=statistics.median(samples_ms),
        p95_ms=_percentile(95.0),
        p99_ms=_percentile(99.0),
        min_ms=sorted_ms[0],
        max_ms=sorted_ms[-1],
        stdev_ms=statistics.pstdev(samples_ms) if n > 1 else 0.0,
        samples_ms=samples_ms,
    )


def time_sync(
    label: str,
    fn: Callable[..., Any],
    *args: Any,
    iterations: int = 5,
    warmup: int = 1,
    **kwargs: Any,
) -> tuple[Any, TimingStats]:
    """Run a sync callable under timing; returns (last_result, stats)."""
    last: Any = None
    for _ in range(max(0, warmup)):
        last = fn(*args, **kwargs)
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        last = fn(*args, **kwargs)
        samples.append(time.perf_counter() - t0)
    return last, _stats(label, samples)


async def time_async(
    label: str,
    fn: Callable[..., Awaitable[Any]],
    *args: Any,
    iterations: int = 5,
    warmup: int = 1,
    **kwargs: Any,
) -> tuple[Any, TimingStats]:
    """Run an async callable under timing; returns (last_result, stats)."""
    last: Any = None
    for _ in range(max(0, warmup)):
        last = await fn(*args, **kwargs)
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        last = await fn(*args, **kwargs)
        samples.append(time.perf_counter() - t0)
    return last, _stats(label, samples)


# ---------------------------------------------------------------------------
# Environment metadata
# ---------------------------------------------------------------------------


def _git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.run(  # nosec B603 — fixed argv, no shell
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git_dirty(repo: Path) -> bool:
    try:
        out = subprocess.run(  # nosec B603
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def _cpu_brand() -> str:
    """Best-effort CPU identifier across platforms."""
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        _, _, brand = line.partition(":")
                        return brand.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def collect_metadata(repo_root: Path) -> dict[str, Any]:
    """Capture machine + repo metadata for benchmark provenance."""
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "git_sha": _git_sha(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": _cpu_brand(),
        "cpu_count": os.cpu_count(),
    }


# ---------------------------------------------------------------------------
# Result writer
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    """Top-level benchmark report serialised to JSON."""

    name: str
    metadata: dict[str, Any]
    timings: list[TimingStats] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)

    def add_timing(self, stats: TimingStats) -> None:
        self.timings.append(stats)

    def to_dict(self, include_samples: bool = False) -> dict[str, Any]:
        return {
            "name": self.name,
            "metadata": self.metadata,
            "timings": [t.to_dict(include_samples=include_samples) for t in self.timings],
            "measurements": self.measurements,
        }

    def write_json(self, path: Path, include_samples: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(include_samples=include_samples), fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def common_argparser(name: str) -> argparse.ArgumentParser:
    """Argparser with `--json`, `--quiet`, `--iterations` flags."""
    ap = argparse.ArgumentParser(description=name)
    ap.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write machine-readable results to this JSON file.",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress human-readable output.")
    ap.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override default iteration count for fast/slow benchmarks.",
    )
    ap.add_argument(
        "--samples",
        action="store_true",
        help="Include raw sample timings in the JSON output.",
    )
    return ap


def print_header(report: BenchmarkReport) -> None:
    print(f"\n{report.name}")
    print("=" * 60)
    md = report.metadata
    print(f"  date: {md['timestamp']}")
    sha = md.get("git_sha") or "?"
    if md.get("git_dirty"):
        sha = f"{sha}*"
    print(f"  git:  {sha}")
    print(f"  py:   {md['python']}   |   cpu: {md['cpu']} ({md['cpu_count']} cores)")
    print()


# ---------------------------------------------------------------------------
# Tiny utility re-exports
# ---------------------------------------------------------------------------


def run_async(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)
