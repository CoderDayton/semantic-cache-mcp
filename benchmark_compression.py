#!/usr/bin/env python3
"""Compression module benchmarks for optimization."""

import cProfile
import io
import pstats
import random
import time
from typing import Callable

# Configure before import
import sys
sys.path.insert(0, "src")

from semantic_cache_mcp.core.compression import (
    compress_adaptive,
    decompress,
    _fast_entropy,
    _detect_incompressible,
    _compress_zstd,
    _compress_lz4,
    _compress_brotli,
    _compress_store,
    HAS_ZSTD,
    HAS_LZ4,
    HAS_BROTLI,
    Codec,
    DEFAULT_CONFIG,
)


def generate_test_data():
    """Generate various test data types."""
    random.seed(42)
    return {
        # Low entropy - highly compressible
        "zeros": b"\x00" * 100_000,
        "repeated": b"Hello World! " * 10_000,
        "low_entropy": bytes([i % 16 for i in range(100_000)]),

        # Medium entropy - typical text/code
        "text": ("The quick brown fox jumps over the lazy dog. " * 500).encode(),
        "code": ("""
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
""" * 200).encode(),

        # High entropy - less compressible
        "random": bytes(random.getrandbits(8) for _ in range(100_000)),
        "binary": bytes(range(256)) * 400,

        # Edge cases
        "tiny": b"small",
        "empty": b"",

        # Large for parallel
        "large_text": (("Log entry: " + "x" * 100 + "\n") * 2000).encode(),
    }


def benchmark_function(func: Callable, data: bytes, iterations: int = 100) -> dict:
    """Benchmark a single function."""
    # Warmup
    for _ in range(5):
        func(data)

    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        func(data)
    elapsed = time.perf_counter() - start

    throughput = (len(data) * iterations) / elapsed / 1e6  # MB/s

    return {
        "iterations": iterations,
        "total_time": elapsed,
        "avg_time_us": elapsed / iterations * 1e6,
        "throughput_mbs": throughput,
    }


def benchmark_entropy():
    """Profile entropy estimation performance."""
    print("\n=== Entropy Estimation Benchmarks ===")

    test_data = generate_test_data()

    for name, data in test_data.items():
        if len(data) < 10:
            continue
        result = benchmark_function(_fast_entropy, data, iterations=10000)
        entropy = _fast_entropy(data)
        print(f"  {name:15} entropy={entropy:.2f}  {result['throughput_mbs']:8.1f} MB/s  {result['avg_time_us']:6.1f} µs")


def benchmark_incompressible_detection():
    """Profile magic byte detection."""
    print("\n=== Incompressible Detection Benchmarks ===")

    test_data = generate_test_data()

    for name, data in test_data.items():
        if len(data) < 10:
            continue
        result = benchmark_function(
            lambda d: _detect_incompressible(d), data, iterations=10000
        )
        detected, fmt = _detect_incompressible(data)
        print(f"  {name:15} detected={str(detected):5} fmt={str(fmt):12}  {result['throughput_mbs']:8.1f} MB/s")


def benchmark_codecs():
    """Profile individual codec performance."""
    print("\n=== Codec Benchmarks ===")

    test_data = generate_test_data()

    codecs = []
    if HAS_ZSTD:
        codecs.append(("ZSTD-1", lambda d: _compress_zstd(d, 1, None)))
        codecs.append(("ZSTD-3", lambda d: _compress_zstd(d, 3, None)))
        codecs.append(("ZSTD-9", lambda d: _compress_zstd(d, 9, None)))
    if HAS_LZ4:
        codecs.append(("LZ4-0", lambda d: _compress_lz4(d, 0)))
        codecs.append(("LZ4-4", lambda d: _compress_lz4(d, 4)))
    if HAS_BROTLI:
        codecs.append(("BROTLI-0", lambda d: _compress_brotli(d, 0)))
        codecs.append(("BROTLI-4", lambda d: _compress_brotli(d, 4)))
    codecs.append(("STORE", lambda d: _compress_store(d)))

    # Medium entropy test data
    data = test_data["text"]

    print(f"\n  Test data: text ({len(data):,} bytes)")
    print(f"  {'Codec':12} {'Throughput':>12} {'Ratio':>8} {'Comp Size':>12}")
    print("  " + "-" * 50)

    for name, compress_fn in codecs:
        try:
            result = benchmark_function(compress_fn, data, iterations=100)
            compressed = compress_fn(data)
            ratio = len(compressed) / len(data)
            print(f"  {name:12} {result['throughput_mbs']:>10.1f} MB/s {ratio:>7.2%} {len(compressed):>10,} B")
        except Exception as e:
            print(f"  {name:12} ERROR: {e}")

    # Low entropy test
    data = test_data["repeated"]
    print(f"\n  Test data: repeated ({len(data):,} bytes)")
    print(f"  {'Codec':12} {'Throughput':>12} {'Ratio':>8} {'Comp Size':>12}")
    print("  " + "-" * 50)

    for name, compress_fn in codecs:
        try:
            result = benchmark_function(compress_fn, data, iterations=100)
            compressed = compress_fn(data)
            ratio = len(compressed) / len(data)
            print(f"  {name:12} {result['throughput_mbs']:>10.1f} MB/s {ratio:>7.2%} {len(compressed):>10,} B")
        except Exception as e:
            print(f"  {name:12} ERROR: {e}")


def benchmark_adaptive():
    """Profile full adaptive compression pipeline."""
    print("\n=== Adaptive Compression Pipeline ===")

    test_data = generate_test_data()

    print(f"  {'Data':15} {'Size':>10} {'Comp':>10} {'Ratio':>8} {'Throughput':>12}")
    print("  " + "-" * 60)

    for name, data in test_data.items():
        if len(data) < 10:
            continue

        result = benchmark_function(compress_adaptive, data, iterations=100)
        compressed = compress_adaptive(data)
        ratio = len(compressed) / len(data) if len(data) > 0 else 1.0

        # Verify roundtrip
        decompressed = decompress(compressed)
        if decompressed != data:
            print(f"  {name:15} ROUNDTRIP FAILURE!")
            continue

        print(f"  {name:15} {len(data):>10,} {len(compressed):>10,} {ratio:>7.2%} {result['throughput_mbs']:>10.1f} MB/s")


def benchmark_parallel():
    """Profile parallel compression threshold."""
    print("\n=== Parallel Compression Scaling ===")

    # Create large compressible data
    base = ("Log entry: timestamp=2024-01-01T00:00:00 level=INFO message=" + "x" * 100 + "\n").encode()

    sizes = [64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024]

    print(f"  {'Size':>12} {'Throughput':>12} {'Ratio':>8}")
    print("  " + "-" * 40)

    for size in sizes:
        data = base * (size // len(base))
        result = benchmark_function(compress_adaptive, data, iterations=20)
        compressed = compress_adaptive(data)
        ratio = len(compressed) / len(data)
        print(f"  {len(data):>10,} B {result['throughput_mbs']:>10.1f} MB/s {ratio:>7.2%}")


def profile_hotspots():
    """Profile to find optimization targets."""
    print("\n=== Profiling Hotspots ===")

    data = generate_test_data()["text"]

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(1000):
        compress_adaptive(data)

    profiler.disable()

    output = io.StringIO()
    stats = pstats.Stats(profiler, stream=output)
    stats.sort_stats("cumtime")
    stats.print_stats(20)

    print(output.getvalue())


def main():
    print("=" * 60)
    print("Compression Module Benchmark Suite")
    print("=" * 60)
    print(f"\nCodec availability: ZSTD={HAS_ZSTD} LZ4={HAS_LZ4} BROTLI={HAS_BROTLI}")

    benchmark_entropy()
    benchmark_incompressible_detection()
    benchmark_codecs()
    benchmark_adaptive()
    benchmark_parallel()
    profile_hotspots()


if __name__ == "__main__":
    main()
