#!/usr/bin/env python3
"""
Comprehensive benchmark of CDC algorithms.

Compares:
- HyperCDC (Gear hash) - current implementation
- HyperCDC SIMD - parallel fingerprint implementation
- FastCDC-style (normalized chunking)
- Rabin fingerprint (classic polynomial hash)
- Fixed-size chunking (baseline)

Metrics:
1. Throughput (MB/s)
2. Deduplication ratio (unique chunks / total chunks for modified content)
3. Average chunk size
4. Chunk size variance (coefficient of variation)

Based on CDC Investigation paper (arXiv:2409.06066v3) methodology.
"""

from __future__ import annotations

import os
import random
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

# Import our implementations
from semantic_cache_mcp.core.chunking import hypercdc_chunks
from semantic_cache_mcp.core.chunking import hypercdc_simd_chunks
from semantic_cache_mcp.core.hashing import hash_chunk

# ---------------------------------------------------------------------------
# Additional CDC implementations for comparison
# ---------------------------------------------------------------------------


def _rabin_table(poly: int = 0x3DA3358B4DC173) -> tuple[int, ...]:
    """Generate Rabin fingerprint lookup table."""
    table = []
    for i in range(256):
        fp = i
        for _ in range(8):
            if fp & 1:
                fp = (fp >> 1) ^ poly
            else:
                fp >>= 1
        table.append(fp)
    return tuple(table)


_RABIN_TABLE = _rabin_table()


def rabin_chunks(
    content: bytes,
    min_size: int = 2048,
    max_size: int = 65536,
    mask_bits: int = 13,
) -> Iterator[bytes]:
    """
    Classic Rabin fingerprint CDC.

    Uses polynomial rolling hash for content-defined chunking.
    Slower than Gear hash but historically significant.

    Args:
        content: Raw bytes to chunk
        min_size: Minimum chunk size
        max_size: Maximum chunk size
        mask_bits: Selectivity (chunk size ~ 2^mask_bits)

    Yields:
        Content chunks
    """
    n = len(content)
    if n == 0:
        return

    table = _RABIN_TABLE
    mask = (1 << mask_bits) - 1

    chunk_start = 0

    while chunk_start < n:
        i = chunk_start
        end_min = min(chunk_start + min_size, n)
        end_max = min(chunk_start + max_size, n)

        # Initialize Rabin fingerprint
        fp = 0

        # Phase 1: Hash min_size bytes without boundary check
        while i < end_min:
            # Slide out oldest byte (simplified - not true sliding window)
            fp = ((fp << 8) ^ table[content[i]]) & 0xFFFFFFFFFFFFFFFF
            i += 1

        # Phase 2: Check for boundaries
        while i < end_max:
            fp = ((fp << 8) ^ table[content[i]]) & 0xFFFFFFFFFFFFFFFF
            i += 1
            if (fp & mask) == 0:
                yield content[chunk_start:i]
                chunk_start = i
                break
        else:
            yield content[chunk_start:i]
            chunk_start = i


def fastcdc_chunks(
    content: bytes,
    min_size: int = 2048,
    avg_size: int = 8192,
    max_size: int = 65536,
) -> Iterator[bytes]:
    """
    FastCDC-style normalized chunking.

    Uses gear hash with two-phase masking (weak before avg, strong after)
    for better chunk size distribution around the target average.

    Args:
        content: Raw bytes to chunk
        min_size: Minimum chunk size
        avg_size: Target average chunk size
        max_size: Maximum chunk size

    Yields:
        Content chunks
    """
    n = len(content)
    if n == 0:
        return

    # Gear table
    rnd = random.Random(0x9E3779B97F4A7C15)
    gear = tuple(rnd.getrandbits(64) for _ in range(256))

    # Calculate masks for normalized chunking
    # Before avg: weaker mask (easier to hit boundary)
    # After avg: stronger mask (harder to hit)
    bits = max(1, (avg_size - 1).bit_length())
    mask_weak = (1 << (bits - 1)) - 1  # ~avg_size/2 expected
    mask_strong = (1 << (bits + 1)) - 1  # ~avg_size*2 expected

    chunk_start = 0

    while chunk_start < n:
        i = chunk_start
        end_min = min(chunk_start + min_size, n)
        end_max = min(chunk_start + max_size, n)
        norm_size = chunk_start + avg_size

        h = 0

        # Phase 1: Hash min_size bytes
        while i < end_min:
            h = ((h << 1) + gear[content[i]]) & 0xFFFFFFFFFFFFFFFF
            i += 1

        # Phase 2: Normalized chunking with two masks
        while i < end_max:
            h = ((h << 1) + gear[content[i]]) & 0xFFFFFFFFFFFFFFFF
            i += 1

            # Use weak mask before avg_size, strong mask after
            mask = mask_weak if i < norm_size else mask_strong

            if (h & mask) == 0:
                yield content[chunk_start:i]
                chunk_start = i
                break
        else:
            yield content[chunk_start:i]
            chunk_start = i


def fixed_chunks(
    content: bytes,
    chunk_size: int = 8192,
) -> Iterator[bytes]:
    """
    Fixed-size chunking (baseline for comparison).

    No content-awareness - just splits at fixed intervals.
    Maximum throughput, zero deduplication benefit.

    Args:
        content: Raw bytes to chunk
        chunk_size: Fixed chunk size

    Yields:
        Content chunks
    """
    n = len(content)
    for i in range(0, n, chunk_size):
        yield content[i : i + chunk_size]


# ---------------------------------------------------------------------------
# Benchmark framework
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    algorithm: str
    throughput_mbs: float
    avg_chunk_size: float
    chunk_count: int
    size_variance_cv: float  # Coefficient of variation
    dedup_ratio: float  # unique_chunks / total_chunks after modification


def measure_throughput(
    chunker: Callable[[bytes], Iterator[bytes]],
    data: bytes,
    iterations: int = 5,
) -> tuple[float, list[int]]:
    """
    Measure chunking throughput.

    Returns (throughput_mbs, chunk_sizes)
    """
    size_mb = len(data) / (1024 * 1024)

    # Warmup
    list(chunker(data))

    # Timed runs
    times: list[float] = []
    chunk_sizes: list[int] = []

    for _ in range(iterations):
        start = time.perf_counter()
        chunks = list(chunker(data))
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        if not chunk_sizes:
            chunk_sizes = [len(c) for c in chunks]

    avg_time = statistics.mean(times)
    throughput = size_mb / avg_time if avg_time > 0 else 0

    return throughput, chunk_sizes


def measure_dedup_ratio(
    chunker: Callable[[bytes], Iterator[bytes]],
    original: bytes,
    modified: bytes,
) -> float:
    """
    Measure deduplication effectiveness.

    Chunks both original and modified, returns ratio of unique chunks
    when processing modified content that could be deduplicated against original.
    """
    # Hash all chunks from original
    original_hashes = {hash_chunk(c) for c in chunker(original)}

    # Count chunks in modified
    modified_chunks = list(chunker(modified))
    total = len(modified_chunks)

    if total == 0:
        return 1.0

    # Count unique (not in original)
    unique = sum(1 for c in modified_chunks if hash_chunk(c) not in original_hashes)

    # Dedup ratio = reused / total (higher = better deduplication)
    return 1.0 - (unique / total)


def create_test_data(size: int, pattern: str = "random") -> bytes:
    """Create test data of specified size and pattern."""
    if pattern == "random":
        return os.urandom(size)
    elif pattern == "text":
        # Simulated text with some repetition
        words = [
            b"the ",
            b"quick ",
            b"brown ",
            b"fox ",
            b"jumps ",
            b"over ",
            b"lazy ",
            b"dog.\n",
        ]
        result = bytearray()
        while len(result) < size:
            result.extend(random.choice(words))
        return bytes(result[:size])
    elif pattern == "code":
        # Simulated code
        lines = [
            b"def function():\n",
            b"    if condition:\n",
            b"        return value\n",
            b"    else:\n",
            b"        pass\n",
            b"\n",
            b"class MyClass:\n",
            b'    """Docstring."""\n',
        ]
        result = bytearray()
        while len(result) < size:
            result.extend(random.choice(lines))
        return bytes(result[:size])
    elif pattern == "binary":
        # Mix of zeros and random (simulates binary files)
        result = bytearray()
        while len(result) < size:
            if random.random() < 0.3:
                result.extend(b"\x00" * random.randint(100, 1000))
            else:
                result.extend(os.urandom(random.randint(100, 500)))
        return bytes(result[:size])
    else:
        return os.urandom(size)


def modify_data(data: bytes, change_ratio: float = 5.0) -> bytes:
    """
    Create modified version of data for dedup testing.

    Simulates realistic modifications by changing contiguous regions.
    This tests CDC's ability to preserve unchanged chunk boundaries.

    Args:
        data: Original data
        change_ratio: Percentage of data to modify (e.g., 5.0 = 5%)
    """
    result = bytearray(data)
    n = len(result)

    # Change a few contiguous regions (simulates editing a few functions/sections)
    total_change_bytes = int(n * change_ratio / 100)
    n_regions = max(1, total_change_bytes // 500)  # ~500 bytes per modified region
    bytes_per_region = total_change_bytes // n_regions

    for _ in range(n_regions):
        # Pick a random position and modify a contiguous block
        start = random.randint(0, max(0, n - bytes_per_region - 1))
        end = min(start + bytes_per_region, n)

        for i in range(start, end):
            result[i] = random.randint(0, 255)

    return bytes(result)


def run_benchmark(
    data_size: int = 10 * 1024 * 1024,  # 10 MB
    data_pattern: str = "random",
    change_ratio: float = 5.0,  # 5% changes
) -> list[BenchmarkResult]:
    """
    Run comprehensive benchmark on all algorithms.

    Args:
        data_size: Size of test data in bytes
        data_pattern: Type of test data (random, text, code, binary)
        change_ratio: Percentage of data to modify for dedup test

    Returns:
        List of BenchmarkResult for each algorithm
    """
    # Generate test data
    print(f"Generating {data_size / 1024 / 1024:.1f} MB {data_pattern} test data...")
    original = create_test_data(data_size, data_pattern)
    modified = modify_data(original, change_ratio)

    # Define algorithms to test
    algorithms = [
        ("Fixed (baseline)", lambda d: fixed_chunks(d, chunk_size=8192)),
        ("Rabin fingerprint", lambda d: rabin_chunks(d)),
        ("FastCDC-style", lambda d: fastcdc_chunks(d)),
        ("HyperCDC (Gear)", lambda d: hypercdc_chunks(d)),
        ("HyperCDC SIMD", lambda d: hypercdc_simd_chunks(d)),
    ]

    results = []

    for name, chunker in algorithms:
        print(f"  Benchmarking {name}...")

        # Measure throughput
        throughput, chunk_sizes = measure_throughput(chunker, original)

        # Calculate statistics
        avg_size = statistics.mean(chunk_sizes) if chunk_sizes else 0
        chunk_count = len(chunk_sizes)

        # Coefficient of variation for chunk sizes
        if len(chunk_sizes) > 1 and avg_size > 0:
            std_dev = statistics.stdev(chunk_sizes)
            cv = std_dev / avg_size
        else:
            cv = 0.0

        # Measure dedup ratio
        dedup = measure_dedup_ratio(chunker, original, modified)

        results.append(
            BenchmarkResult(
                algorithm=name,
                throughput_mbs=throughput,
                avg_chunk_size=avg_size,
                chunk_count=chunk_count,
                size_variance_cv=cv,
                dedup_ratio=dedup,
            )
        )

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Pretty-print benchmark results."""
    print("\n" + "=" * 80)
    print("CDC ALGORITHM BENCHMARK RESULTS")
    print("=" * 80)

    # Header
    print(
        f"{'Algorithm':<22} {'Throughput':>12} {'Avg Size':>10} "
        f"{'Chunks':>8} {'CV':>8} {'Dedup':>8}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r.algorithm:<22} {r.throughput_mbs:>10.2f} MB/s "
            f"{r.avg_chunk_size:>10,.0f} {r.chunk_count:>8,} "
            f"{r.size_variance_cv:>8.3f} {r.dedup_ratio:>7.1%}"
        )

    print("-" * 80)
    print("\nLegend:")
    print("  Throughput: Processing speed in MB/s (higher = faster)")
    print("  Avg Size: Average chunk size in bytes")
    print("  Chunks: Number of chunks produced")
    print("  CV: Coefficient of variation (lower = more consistent sizes)")
    print("  Dedup: Deduplication ratio after 5% modification (higher = better)")


def main() -> None:
    """Run benchmarks with different data patterns."""
    print("CDC Algorithm Benchmark")
    print("Based on methodology from arXiv:2409.06066v3\n")

    # Test with different patterns
    patterns = ["random", "text", "code", "binary"]

    all_results: dict[str, list[BenchmarkResult]] = {}

    for pattern in patterns:
        print(f"\n{'=' * 40}")
        print(f"Pattern: {pattern.upper()}")
        print("=" * 40)

        results = run_benchmark(
            data_size=5 * 1024 * 1024,  # 5 MB for faster testing
            data_pattern=pattern,
            change_ratio=5.0,
        )
        all_results[pattern] = results
        print_results(results)

    # Summary across all patterns
    print("\n\n" + "=" * 80)
    print("SUMMARY: Average across all patterns")
    print("=" * 80)

    algorithm_names = [r.algorithm for r in all_results["random"]]
    summary = []

    for name in algorithm_names:
        throughputs = []
        dedups = []

        for pattern in patterns:
            for r in all_results[pattern]:
                if r.algorithm == name:
                    throughputs.append(r.throughput_mbs)
                    dedups.append(r.dedup_ratio)

        summary.append(
            {
                "algorithm": name,
                "avg_throughput": statistics.mean(throughputs),
                "avg_dedup": statistics.mean(dedups),
            }
        )

    print(f"\n{'Algorithm':<22} {'Avg Throughput':>15} {'Avg Dedup':>12}")
    print("-" * 50)
    for s in summary:
        print(f"{s['algorithm']:<22} {s['avg_throughput']:>13.2f} MB/s {s['avg_dedup']:>11.1%}")


if __name__ == "__main__":
    main()
