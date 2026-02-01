"""Tests for core algorithms: chunking, hashing, compression, similarity."""

from __future__ import annotations

import array
import math

import pytest

from semantic_cache_mcp.core.chunking import hypercdc_chunks
from semantic_cache_mcp.core.compression import (
    compress_adaptive,
    decompress,
    estimate_entropy,
)
from semantic_cache_mcp.core.hashing import hash_chunk, hash_content
from semantic_cache_mcp.core.similarity import cosine_similarity
from semantic_cache_mcp.core.text import generate_diff, truncate_smart


class TestContentDefinedChunking:
    """Tests for content-defined chunking."""

    def test_empty_content_yields_nothing(self) -> None:
        """Empty content should yield no chunks."""
        chunks = list(hypercdc_chunks(b""))
        assert chunks == []

    def test_single_byte_yields_single_chunk(self) -> None:
        """Single byte should yield one chunk."""
        chunks = list(hypercdc_chunks(b"x"))
        assert len(chunks) == 1
        assert chunks[0] == b"x"

    def test_small_content_single_chunk(self) -> None:
        """Small content below min_size should be single chunk."""
        data = b"Hello, World!"
        chunks = list(hypercdc_chunks(data, min_size=100))
        assert len(chunks) == 1
        assert chunks[0] == data

    def test_deterministic_output(self) -> None:
        """Same input should always produce same chunks."""
        data = b"The quick brown fox jumps over the lazy dog. " * 100
        chunks1 = list(hypercdc_chunks(data))
        chunks2 = list(hypercdc_chunks(data))
        assert chunks1 == chunks2

    def test_reassembly_matches_original(self) -> None:
        """Reassembled chunks should match original content."""
        data = b"Test data for chunking. " * 500
        chunks = list(hypercdc_chunks(data))
        reassembled = b"".join(chunks)
        assert reassembled == data

    def test_respects_max_size(self) -> None:
        """All chunks should be at most max_size."""
        data = b"x" * 100000
        max_size = 8192
        chunks = list(hypercdc_chunks(data, max_size=max_size))
        for chunk in chunks:
            assert len(chunk) <= max_size

    def test_chunks_at_least_min_size(self) -> None:
        """Non-final chunks should be at least min_size."""
        data = b"y" * 50000
        min_size = 1024
        chunks = list(hypercdc_chunks(data, min_size=min_size))
        # All but last chunk should meet min_size
        for chunk in chunks[:-1]:
            assert len(chunk) >= min_size

    def test_binary_content(self) -> None:
        """Binary content should chunk correctly."""
        data = bytes(range(256)) * 100
        chunks = list(hypercdc_chunks(data))
        assert b"".join(chunks) == data


class TestHashing:
    """Tests for BLAKE2b hashing."""

    def test_hash_chunk_consistent(self) -> None:
        """Same data should produce same hash."""
        data = b"Test data"
        hash1 = hash_chunk(data)
        hash2 = hash_chunk(data)
        assert hash1 == hash2

    def test_hash_chunk_format(self) -> None:
        """Hash should be 40-character hex string."""
        data = b"Test"
        result = hash_chunk(data)
        assert len(result) == 40
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_chunk_different_data(self) -> None:
        """Different data should produce different hashes."""
        hash1 = hash_chunk(b"data1")
        hash2 = hash_chunk(b"data2")
        assert hash1 != hash2

    def test_hash_content_consistent(self) -> None:
        """Same content should produce same hash."""
        content = "Hello, World!"
        hash1 = hash_content(content)
        hash2 = hash_content(content)
        assert hash1 == hash2

    def test_hash_content_format(self) -> None:
        """Content hash should be 32-character hex string."""
        result = hash_content("Test")
        assert len(result) == 32
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_content_empty_string(self) -> None:
        """Empty string should have a valid hash."""
        result = hash_content("")
        assert len(result) == 32


class TestCompression:
    """Tests for adaptive Brotli compression."""

    def test_compress_decompress_roundtrip(self) -> None:
        """Compressing then decompressing should return original."""
        original = b"Test data for compression roundtrip. " * 100
        compressed = compress_adaptive(original)
        decompressed = decompress(compressed)
        assert decompressed == original

    def test_compression_reduces_size(self) -> None:
        """Highly compressible data should compress well."""
        original = b"aaaaaaaaaa" * 1000
        compressed = compress_adaptive(original)
        assert len(compressed) < len(original)

    def test_empty_data_roundtrip(self) -> None:
        """Empty data should roundtrip correctly."""
        original = b""
        compressed = compress_adaptive(original)
        decompressed = decompress(compressed)
        assert decompressed == original

    def test_binary_data_roundtrip(self) -> None:
        """Binary data should roundtrip correctly."""
        original = bytes(range(256)) * 10
        compressed = compress_adaptive(original)
        decompressed = decompress(compressed)
        assert decompressed == original

    def test_entropy_estimation_low(self) -> None:
        """Repetitive data should have low entropy."""
        data = b"aaaa" * 1000
        entropy = estimate_entropy(data)
        assert entropy < 1.0  # Very low entropy

    def test_entropy_estimation_high(self) -> None:
        """Random-like data should have high entropy."""
        data = bytes(range(256)) * 10
        entropy = estimate_entropy(data)
        assert entropy > 7.0  # High entropy (close to 8 bits)

    def test_entropy_empty_data(self) -> None:
        """Empty data should have zero entropy."""
        assert estimate_entropy(b"") == 0.0

    def test_adaptive_quality_selection(self) -> None:
        """Different entropy levels should work with adaptive compression."""
        # Low entropy (highly compressible)
        low_entropy = b"x" * 10000
        comp_low = compress_adaptive(low_entropy)

        # High entropy (less compressible)
        high_entropy = bytes(range(256)) * 40
        comp_high = compress_adaptive(high_entropy)

        # Both should decompress correctly
        assert decompress(comp_low) == low_entropy
        assert decompress(comp_high) == high_entropy


class TestSimilarity:
    """Tests for cosine similarity."""

    def test_identical_vectors_similarity_one(self) -> None:
        """Identical normalized vectors should have similarity 1.0."""
        vec = [0.5, 0.5, 0.5, 0.5]
        # Normalize
        mag = math.sqrt(sum(x * x for x in vec))
        normalized = [x / mag for x in vec]
        sim = cosine_similarity(normalized, normalized)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors_similarity_zero(self) -> None:
        """Orthogonal vectors should have similarity 0.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim) < 1e-6

    def test_opposite_vectors_similarity_negative(self) -> None:
        """Opposite vectors should have similarity -1.0."""
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim - (-1.0)) < 1e-6

    def test_similar_vectors_high_similarity(self) -> None:
        """Similar vectors should have high similarity."""
        vec1 = [0.9, 0.1, 0.0]
        vec2 = [0.85, 0.15, 0.0]
        # Normalize both
        mag1 = math.sqrt(sum(x * x for x in vec1))
        mag2 = math.sqrt(sum(x * x for x in vec2))
        norm1 = [x / mag1 for x in vec1]
        norm2 = [x / mag2 for x in vec2]
        sim = cosine_similarity(norm1, norm2)
        assert sim > 0.9

    def test_array_type_vectors(self) -> None:
        """Should work with array.array type."""
        vec1 = array.array("f", [0.6, 0.8])
        vec2 = array.array("f", [0.6, 0.8])
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim - 1.0) < 1e-6

    def test_zero_vector_handling(self) -> None:
        """Zero vectors should return 0 similarity."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        sim = cosine_similarity(vec1, vec2)
        assert sim == 0.0


class TestDiffGeneration:
    """Tests for unified diff generation."""

    def test_unchanged_file_no_diff(self) -> None:
        """Unchanged content should return no changes message."""
        text = "Same content\nLine 2\n"
        result = generate_diff(text, text)
        assert result == "// No changes"

    def test_changed_file_produces_diff(self) -> None:
        """Changed content should produce unified diff."""
        old = "Line 1\nLine 2\n"
        new = "Line 1\nLine 2 modified\n"
        result = generate_diff(old, new)
        assert "---" in result or "-Line 2" in result

    def test_new_file_diff(self) -> None:
        """Adding to empty file should show additions."""
        old = ""
        new = "New line 1\nNew line 2\n"
        result = generate_diff(old, new)
        assert "+New line" in result or "+" in result

    def test_empty_files_no_diff(self) -> None:
        """Two empty files should have no diff."""
        result = generate_diff("", "")
        assert result == "// No changes"

    def test_diff_context_lines(self) -> None:
        """Diff should include context lines."""
        old = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
        new = "Line 1\nLine 2\nLine 3 CHANGED\nLine 4\nLine 5\n"
        result = generate_diff(old, new, context_lines=1)
        assert "Line 2" in result or "Line 4" in result


class TestSmartTruncation:
    """Tests for smart truncation."""

    def test_small_content_unchanged(self) -> None:
        """Small content should not be truncated."""
        content = "Short content\n"
        result = truncate_smart(content, max_size=1000)
        assert result == content

    def test_large_content_truncated(self) -> None:
        """Large content should be truncated."""
        lines = [f"Line {i}\n" for i in range(200)]
        content = "".join(lines)
        result = truncate_smart(content, max_size=500)
        assert len(result) <= 500
        assert "truncated" in result.lower() or "TRUNCATED" in result

    def test_preserves_top_lines(self) -> None:
        """Truncation should preserve top lines."""
        lines = [f"Line {i}\n" for i in range(200)]
        content = "".join(lines)
        result = truncate_smart(content, max_size=2000, keep_top=10)
        assert "Line 0" in result
        assert "Line 9" in result

    def test_preserves_bottom_lines(self) -> None:
        """Truncation should preserve bottom lines."""
        lines = [f"Line {i}\n" for i in range(200)]
        content = "".join(lines)
        result = truncate_smart(content, max_size=2000, keep_top=10, keep_bottom=10)
        assert "Line 199" in result
        assert "Line 190" in result

    def test_empty_content(self) -> None:
        """Empty content should return empty."""
        result = truncate_smart("", max_size=1000)
        assert result == ""
