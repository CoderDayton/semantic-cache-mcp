"""Tests for CDC benchmark utilities."""

from __future__ import annotations

import pytest

from benchmarks.benchmark_cdc import (
    create_test_data,
    fastcdc_chunks,
    fixed_chunks,
    measure_dedup_ratio,
    measure_throughput,
    modify_data,
    rabin_chunks,
)


class TestCDCAlgorithms:
    """Tests for additional CDC algorithm implementations."""

    def test_rabin_chunks_reassembly(self) -> None:
        """Rabin chunks reassemble to original."""
        content = create_test_data(50000, "random")
        chunks = list(rabin_chunks(content))
        assert b"".join(chunks) == content

    def test_fastcdc_chunks_reassembly(self) -> None:
        """FastCDC chunks reassemble to original."""
        content = create_test_data(50000, "random")
        chunks = list(fastcdc_chunks(content))
        assert b"".join(chunks) == content

    def test_fixed_chunks_reassembly(self) -> None:
        """Fixed chunks reassemble to original."""
        content = create_test_data(50000, "random")
        chunks = list(fixed_chunks(content))
        assert b"".join(chunks) == content

    def test_rabin_respects_min_size(self) -> None:
        """Rabin chunks respect minimum size."""
        content = create_test_data(100000, "random")
        min_size = 2048
        chunks = list(rabin_chunks(content, min_size=min_size))

        for chunk in chunks[:-1]:  # Last chunk may be smaller
            assert len(chunk) >= min_size

    def test_rabin_respects_max_size(self) -> None:
        """Rabin chunks respect maximum size."""
        content = create_test_data(100000, "random")
        max_size = 8192
        chunks = list(rabin_chunks(content, max_size=max_size))

        for chunk in chunks:
            assert len(chunk) <= max_size


class TestTestDataGeneration:
    """Tests for test data generation."""

    @pytest.mark.parametrize("pattern", ["random", "text", "code", "binary"])
    def test_create_test_data_size(self, pattern: str) -> None:
        """Test data is correct size."""
        size = 10000
        data = create_test_data(size, pattern)
        assert len(data) == size

    def test_modify_data_changes_some_bytes(self) -> None:
        """Modified data differs from original."""
        original = create_test_data(10000, "random")
        modified = modify_data(original, change_ratio=10.0)

        # Should be different
        assert original != modified

        # But most bytes should be the same
        same_count = sum(1 for a, b in zip(original, modified, strict=False) if a == b)
        assert same_count > len(original) * 0.8  # At least 80% same


class TestBenchmarkMeasurements:
    """Tests for benchmark measurement functions."""

    def test_measure_throughput_returns_positive(self) -> None:
        """Throughput measurement returns positive value."""
        data = create_test_data(50000, "random")
        throughput, sizes = measure_throughput(fixed_chunks, data, iterations=2)

        assert throughput > 0
        assert len(sizes) > 0

    def test_measure_dedup_ratio_identical(self) -> None:
        """Identical data has 100% dedup ratio."""
        data = create_test_data(50000, "random")
        ratio = measure_dedup_ratio(fixed_chunks, data, data)

        # Same data should have perfect dedup
        assert ratio == 1.0

    def test_measure_dedup_ratio_completely_different(self) -> None:
        """Completely different data has low dedup ratio."""
        data1 = create_test_data(50000, "random")
        data2 = create_test_data(50000, "random")  # Different random data

        ratio = measure_dedup_ratio(fixed_chunks, data1, data2)

        # Random data should have low dedup (likely 0)
        assert ratio < 0.5
