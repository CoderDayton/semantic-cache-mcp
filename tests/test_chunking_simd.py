"""Tests for SIMD-accelerated parallel CDC chunking."""

from __future__ import annotations

import os

import pytest

from semantic_cache_mcp.core.chunking import hypercdc_chunks
from semantic_cache_mcp.core.chunking_simd import (
    _parallel_cdc_boundaries,
    get_optimal_chunker,
    hypercdc_simd_boundaries,
    hypercdc_simd_chunks,
)


class TestParallelCDCBoundaries:
    """Tests for parallel CDC boundary detection algorithm."""

    def test_empty_data(self) -> None:
        """Empty data returns single boundary at 0."""
        import numpy as np

        data = np.array([], dtype=np.uint8)
        result = _parallel_cdc_boundaries(data, min_size=100)
        assert result == [0]

    def test_small_data(self) -> None:
        """Data smaller than min_size returns single chunk."""
        import numpy as np

        data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        result = _parallel_cdc_boundaries(data, min_size=100)
        assert result == [5]

    def test_respects_min_size(self) -> None:
        """Boundaries respect minimum chunk size."""
        import numpy as np

        data = np.frombuffer(os.urandom(10000), dtype=np.uint8)
        result = _parallel_cdc_boundaries(data, min_size=500, max_size=5000)

        prev = 0
        for b in result:
            chunk_size = b - prev
            # All chunks except last should be >= min_size
            if b != result[-1]:
                assert chunk_size >= 500, f"Chunk {prev}-{b} too small: {chunk_size}"
            prev = b

    def test_respects_max_size(self) -> None:
        """All chunks are within max_size limit."""
        import numpy as np

        data = np.frombuffer(os.urandom(20000), dtype=np.uint8)
        result = _parallel_cdc_boundaries(data, min_size=100, max_size=1000)

        prev = 0
        for b in result:
            chunk_size = b - prev
            assert chunk_size <= 1000, f"Chunk {prev}-{b} exceeds max_size: {chunk_size}"
            prev = b

    def test_produces_reasonable_chunks(self) -> None:
        """Algorithm produces reasonable number of chunks."""
        import numpy as np

        data = np.frombuffer(os.urandom(50000), dtype=np.uint8)
        result = _parallel_cdc_boundaries(data, min_size=2048, max_size=8192)

        # Should cover the full data
        assert result[-1] == len(data)
        # Should produce reasonable chunk count for 50KB with 2-8KB chunks
        assert 5 < len(result) < 100


class TestHyperCDCSIMD:
    """Integration tests for SIMD chunking."""

    def test_empty_content(self) -> None:
        """Empty content yields no chunks."""
        chunks = list(hypercdc_simd_chunks(b""))
        assert chunks == []

    def test_small_content(self) -> None:
        """Content smaller than min_size yields single chunk."""
        content = b"Hello, World!"
        chunks = list(hypercdc_simd_chunks(content, min_size=100))
        assert len(chunks) == 1
        assert chunks[0] == content

    def test_reassembly(self) -> None:
        """Chunks reassemble to original content."""
        content = os.urandom(50000)
        chunks = list(hypercdc_simd_chunks(content))
        reassembled = b"".join(chunks)
        assert reassembled == content

    def test_deterministic(self) -> None:
        """Same content produces same chunks."""
        content = os.urandom(30000)
        chunks1 = list(hypercdc_simd_chunks(content))
        chunks2 = list(hypercdc_simd_chunks(content))
        assert chunks1 == chunks2

    def test_boundaries_match_chunks(self) -> None:
        """Boundary indices match actual chunk positions."""
        content = os.urandom(20000)
        boundaries = list(hypercdc_simd_boundaries(content))
        chunks = list(hypercdc_simd_chunks(content))

        assert len(boundaries) == len(chunks)
        for (start, end), chunk in zip(boundaries, chunks, strict=True):
            assert content[start:end] == chunk

    def test_chunk_sizes_within_bounds(self) -> None:
        """All chunks respect min/max size constraints."""
        content = os.urandom(100000)
        min_size = 2048
        max_size = 8192

        chunks = list(hypercdc_simd_chunks(content, min_size=min_size, max_size=max_size))

        for i, chunk in enumerate(chunks):
            # Last chunk can be smaller
            if i < len(chunks) - 1:
                assert len(chunk) >= min_size, f"Chunk {i} too small: {len(chunk)}"
            assert len(chunk) <= max_size, f"Chunk {i} too large: {len(chunk)}"

    def test_mask_bits_affects_chunk_size(self) -> None:
        """Higher mask_bits produces larger chunks."""
        content = os.urandom(100000)

        chunks_10 = list(hypercdc_simd_chunks(content, mask_bits=10))
        chunks_14 = list(hypercdc_simd_chunks(content, mask_bits=14))

        # Higher mask_bits = fewer, larger chunks
        assert len(chunks_10) > len(chunks_14)


class TestOptimalChunker:
    """Tests for chunker selection."""

    def test_returns_simd_when_available(self) -> None:
        """Returns SIMD chunker when numpy available."""
        chunker = get_optimal_chunker(prefer_simd=True)
        assert chunker == hypercdc_simd_chunks

    def test_respects_prefer_simd_false(self) -> None:
        """Returns hash-based chunker when prefer_simd=False."""
        chunker = get_optimal_chunker(prefer_simd=False)
        assert chunker == hypercdc_chunks


class TestSIMDvsHashComparison:
    """Compare SIMD and hash-based chunking behavior."""

    def test_reasonable_chunk_count(self) -> None:
        """SIMD produces reasonable number of chunks for deduplication."""
        content = os.urandom(50000)

        # Use mask_bits=13 for similar chunk size to Gear hash
        simd_chunks = list(
            hypercdc_simd_chunks(content, min_size=2048, max_size=16384, mask_bits=13)
        )
        hash_chunks = list(hypercdc_chunks(content, min_size=2048, max_size=16384))

        # Both should produce reasonable results
        assert 3 < len(simd_chunks) < 50, f"SIMD chunks: {len(simd_chunks)}"
        assert 3 < len(hash_chunks) < 50, f"Hash chunks: {len(hash_chunks)}"

    def test_both_reassemble_correctly(self) -> None:
        """Both implementations reassemble to original content."""
        content = os.urandom(30000)

        simd_reassembled = b"".join(hypercdc_simd_chunks(content))
        hash_reassembled = b"".join(hypercdc_chunks(content))

        assert simd_reassembled == content
        assert hash_reassembled == content

    def test_content_defined_not_fixed_size(self) -> None:
        """Parallel CDC produces variable-size chunks (not fixed-size)."""
        content = os.urandom(50000)
        chunks = list(hypercdc_simd_chunks(content, min_size=1024, max_size=16384))

        # Chunks should vary in size (not all same size like fixed chunking)
        sizes = [len(c) for c in chunks]
        unique_sizes = len(set(sizes))

        # Should have multiple different chunk sizes
        assert unique_sizes > 1, f"Expected variable sizes, got {unique_sizes} unique sizes"

        # Sizes should span a reasonable range
        size_range = max(sizes) - min(sizes)
        assert size_range > 1000, f"Expected size variation, got range {size_range}"


@pytest.mark.parametrize("size", [1000, 10000, 100000])
def test_simd_various_sizes(size: int) -> None:
    """SIMD chunking works for various content sizes."""
    content = os.urandom(size)
    chunks = list(hypercdc_simd_chunks(content))
    assert b"".join(chunks) == content


@pytest.mark.parametrize("mask_bits", [10, 11, 12, 13, 14, 15])
def test_simd_various_mask_bits(mask_bits: int) -> None:
    """SIMD chunking works for various mask_bits values."""
    content = os.urandom(50000)
    chunks = list(hypercdc_simd_chunks(content, mask_bits=mask_bits))
    assert b"".join(chunks) == content
    assert len(chunks) > 0
