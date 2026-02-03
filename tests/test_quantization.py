"""Tests for extreme quantization: binary and ternary embeddings."""

from __future__ import annotations

import array

import numpy as np
import pytest

from semantic_cache_mcp.core.quantization import (
    batch_dot_product_ternary,
    batch_hamming_similarity_binary,
    dequantize_binary,
    dequantize_ternary,
    dot_product_ternary,
    evaluate_quantization_accuracy,
    hamming_similarity_binary,
    quantize_binary,
    quantize_hybrid,
    quantize_ternary,
)


class TestBinaryQuantization:
    """Tests for 1-bit binary quantization."""

    def test_quantize_basic(self) -> None:
        """Basic quantization produces correct size."""
        vec = np.array([0.5, -0.3, 0.1, -0.8, 0.2, -0.1, 0.7, -0.9], dtype=np.float32)
        blob = quantize_binary(vec)
        # 8 values = 1 byte
        assert len(blob) == 1

    def test_quantize_384d(self) -> None:
        """384D embedding produces 48 bytes."""
        vec = np.random.randn(384).astype(np.float32)
        blob = quantize_binary(vec)
        assert len(blob) == 48

    def test_roundtrip(self) -> None:
        """Quantize then dequantize produces correct signs."""
        vec = np.array([0.5, -0.3, 0.1, -0.8], dtype=np.float32)
        blob = quantize_binary(vec)
        recovered = dequantize_binary(blob, len(vec))

        # Signs should match
        assert np.all(np.sign(recovered) == np.sign(vec))

    def test_dequantize_values(self) -> None:
        """Dequantized values are {-1, +1}."""
        vec = np.random.randn(100).astype(np.float32)
        blob = quantize_binary(vec)
        recovered = dequantize_binary(blob, len(vec))

        unique_vals = set(recovered.tolist())
        assert unique_vals <= {-1.0, 1.0}

    def test_identical_similarity_is_one(self) -> None:
        """Same vector has similarity 1.0."""
        vec = np.random.randn(384).astype(np.float32)
        blob = quantize_binary(vec)
        sim = hamming_similarity_binary(blob, blob)
        assert sim == 1.0

    def test_opposite_similarity_is_zero(self) -> None:
        """Opposite signs have similarity 0.0."""
        vec = np.random.randn(384).astype(np.float32)
        blob1 = quantize_binary(vec)
        blob2 = quantize_binary(-vec)
        sim = hamming_similarity_binary(blob1, blob2)
        assert sim == 0.0

    def test_similar_vectors_high_similarity(self) -> None:
        """Similar vectors have high similarity."""
        vec1 = np.random.randn(384).astype(np.float32)
        vec2 = vec1 + np.random.randn(384).astype(np.float32) * 0.1

        blob1 = quantize_binary(vec1)
        blob2 = quantize_binary(vec2)
        sim = hamming_similarity_binary(blob1, blob2)

        # Similar vectors should have > 0.8 similarity
        assert sim > 0.8

    def test_batch_matches_single(self) -> None:
        """Batch similarity matches single computation."""
        query = np.random.randn(384).astype(np.float32)
        targets = [np.random.randn(384).astype(np.float32) for _ in range(10)]

        query_blob = quantize_binary(query)
        target_blobs = [quantize_binary(t) for t in targets]

        # Single computation
        single_sims = [hamming_similarity_binary(query_blob, b) for b in target_blobs]

        # Batch computation
        batch_sims = batch_hamming_similarity_binary(query_blob, target_blobs)

        np.testing.assert_allclose(single_sims, batch_sims, rtol=1e-5)

    def test_array_array_input(self) -> None:
        """Works with array.array input."""
        vec_np = np.random.randn(128).astype(np.float32)
        vec_arr = array.array("f", vec_np.tolist())

        blob_np = quantize_binary(vec_np)
        blob_arr = quantize_binary(vec_arr)

        assert blob_np == blob_arr


class TestTernaryQuantization:
    """Tests for 1.58-bit ternary quantization."""

    def test_quantize_basic(self) -> None:
        """Basic quantization produces correct size."""
        vec = np.array([0.5, -0.3, 0.1, -0.8], dtype=np.float32)
        blob = quantize_ternary(vec)
        # 4 values = 1 byte (2 bits per value)
        assert len(blob) == 1

    def test_quantize_384d(self) -> None:
        """384D embedding produces 96 bytes."""
        vec = np.random.randn(384).astype(np.float32)
        blob = quantize_ternary(vec)
        assert len(blob) == 96

    def test_dequantize_values(self) -> None:
        """Dequantized values are {-1, 0, +1}."""
        vec = np.random.randn(100).astype(np.float32)
        blob = quantize_ternary(vec)
        recovered = dequantize_ternary(blob, len(vec))

        unique_vals = set(recovered.tolist())
        assert unique_vals <= {-1.0, 0.0, 1.0}

    def test_threshold_produces_zeros(self) -> None:
        """Threshold creates some zero values."""
        vec = np.random.randn(384).astype(np.float32)
        blob = quantize_ternary(vec, threshold_percentile=33.0)
        recovered = dequantize_ternary(blob, len(vec))

        # Should have some zeros
        n_zeros = np.sum(recovered == 0)
        # Roughly 1/3 should be zero with 33% threshold
        assert 0.1 < n_zeros / len(vec) < 0.5

    def test_roundtrip_sign_preservation(self) -> None:
        """Large values preserve sign after roundtrip."""
        vec = np.array([2.0, -2.0, 0.01, -0.01], dtype=np.float32)
        blob = quantize_ternary(vec, threshold_percentile=50.0)
        recovered = dequantize_ternary(blob, len(vec))

        # Large values should preserve sign
        assert np.sign(recovered[0]) == np.sign(vec[0])
        assert np.sign(recovered[1]) == np.sign(vec[1])

    def test_dot_product_identical(self) -> None:
        """Same vector dot product is maximum."""
        vec = np.random.randn(384).astype(np.float32)
        blob = quantize_ternary(vec)
        dot = dot_product_ternary(blob, blob, 384)

        # Should be positive (sum of squares of ternary values)
        assert dot > 0

    def test_dot_product_orthogonal(self) -> None:
        """Opposite vectors have negative dot product."""
        vec = np.random.randn(384).astype(np.float32)
        blob1 = quantize_ternary(vec)
        blob2 = quantize_ternary(-vec)
        dot = dot_product_ternary(blob1, blob2, 384)

        # Should be negative
        assert dot < 0

    def test_batch_matches_single(self) -> None:
        """Batch dot product matches single computation."""
        dim = 384
        query = np.random.randn(dim).astype(np.float32)
        targets = [np.random.randn(dim).astype(np.float32) for _ in range(10)]

        query_blob = quantize_ternary(query)
        target_blobs = [quantize_ternary(t) for t in targets]

        # Single computation
        single_dots = [dot_product_ternary(query_blob, b, dim) for b in target_blobs]

        # Batch computation
        batch_dots = batch_dot_product_ternary(query_blob, target_blobs, dim)

        np.testing.assert_allclose(single_dots, batch_dots, rtol=1e-5)

    def test_array_array_input(self) -> None:
        """Works with array.array input."""
        vec_np = np.random.randn(128).astype(np.float32)
        vec_arr = array.array("f", vec_np.tolist())

        blob_np = quantize_ternary(vec_np)
        blob_arr = quantize_ternary(vec_arr)

        assert blob_np == blob_arr


class TestHybridQuantization:
    """Tests for hybrid (binary + int8) storage."""

    def test_produces_both_formats(self) -> None:
        """Hybrid returns both binary and int8 blobs."""
        vec = np.random.randn(384).astype(np.float32)
        binary_blob, int8_blob = quantize_hybrid(vec)

        assert len(binary_blob) == 48  # 384 / 8
        assert len(int8_blob) == 388  # 4 + 384


class TestStorageSizes:
    """Tests for storage size claims."""

    @pytest.mark.parametrize(
        "dim,expected_binary,expected_ternary,expected_int8",
        [
            (384, 48, 96, 388),
            (768, 96, 192, 772),
            (128, 16, 32, 132),
        ],
    )
    def test_storage_sizes(
        self,
        dim: int,
        expected_binary: int,
        expected_ternary: int,
        expected_int8: int,
    ) -> None:
        """Verify storage sizes for different dimensions."""
        from semantic_cache_mcp.core.similarity import quantize_embedding

        vec = np.random.randn(dim).astype(np.float32)

        binary_blob = quantize_binary(vec)
        ternary_blob = quantize_ternary(vec)
        int8_blob = quantize_embedding(vec)

        assert len(binary_blob) == expected_binary
        assert len(ternary_blob) == expected_ternary
        assert len(int8_blob) == expected_int8


class TestEvaluationAccuracy:
    """Tests for quantization accuracy evaluation."""

    def test_int8_high_correlation(self) -> None:
        """int8 should have very high correlation with float32."""
        np.random.seed(42)
        embeddings = [np.random.randn(384).astype(np.float32) for _ in range(100)]

        result = evaluate_quantization_accuracy(embeddings, method="int8", k=10)

        assert result["spearman_correlation"] > 0.99

    def test_ternary_reasonable_correlation(self) -> None:
        """Ternary should have reasonable correlation."""
        np.random.seed(42)
        embeddings = [np.random.randn(384).astype(np.float32) for _ in range(100)]

        result = evaluate_quantization_accuracy(embeddings, method="ternary", k=10)

        # Ternary should still preserve ranking order well
        assert result["spearman_correlation"] > 0.7

    def test_binary_maintains_some_correlation(self) -> None:
        """Binary should maintain some ranking correlation."""
        np.random.seed(42)
        embeddings = [np.random.randn(384).astype(np.float32) for _ in range(100)]

        result = evaluate_quantization_accuracy(embeddings, method="binary", k=10)

        # Binary is more aggressive, but should still be useful
        assert result["spearman_correlation"] > 0.5

    def test_compression_ratios(self) -> None:
        """Compression ratios are as expected."""
        np.random.seed(42)
        embeddings = [np.random.randn(384).astype(np.float32) for _ in range(50)]

        binary_result = evaluate_quantization_accuracy(embeddings, method="binary")
        ternary_result = evaluate_quantization_accuracy(embeddings, method="ternary")
        int8_result = evaluate_quantization_accuracy(embeddings, method="int8")

        # Binary: 384 * 4 / 48 = 32x vs float32
        assert binary_result["compression_vs_float32"] > 30

        # Ternary: 384 * 4 / 96 = 16x vs float32
        assert ternary_result["compression_vs_float32"] > 15

        # int8: 384 * 4 / 388 ≈ 4x vs float32
        assert int8_result["compression_vs_float32"] > 3

        # Binary vs int8: 388 / 48 ≈ 8x
        assert binary_result["compression_vs_int8"] > 7

        # Ternary vs int8: 388 / 96 ≈ 4x
        assert ternary_result["compression_vs_int8"] > 3


class TestRankingPreservation:
    """Tests for similarity ranking preservation."""

    def test_top_k_recall_int8(self) -> None:
        """int8 should have perfect or near-perfect recall@10."""
        np.random.seed(42)
        embeddings = [np.random.randn(384).astype(np.float32) for _ in range(100)]

        result = evaluate_quantization_accuracy(embeddings, method="int8", k=10)

        # int8 should have very high recall
        assert result["recall_at_k"] >= 0.9

    def test_ternary_reasonable_recall(self) -> None:
        """Ternary should have reasonable recall."""
        np.random.seed(42)
        embeddings = [np.random.randn(384).astype(np.float32) for _ in range(100)]

        result = evaluate_quantization_accuracy(embeddings, method="ternary", k=10)

        # Ternary should have reasonable recall
        assert result["recall_at_k"] >= 0.5

    def test_binary_useful_recall(self) -> None:
        """Binary should have useful recall for pre-filtering."""
        np.random.seed(42)
        embeddings = [np.random.randn(384).astype(np.float32) for _ in range(100)]

        result = evaluate_quantization_accuracy(embeddings, method="binary", k=10)

        # Binary should be useful for pre-filtering (recall >= 0.3)
        assert result["recall_at_k"] >= 0.3


@pytest.mark.parametrize("dim", [128, 384, 768])
def test_different_dimensions(dim: int) -> None:
    """Quantization works for different embedding dimensions."""
    vec = np.random.randn(dim).astype(np.float32)

    binary_blob = quantize_binary(vec)
    ternary_blob = quantize_ternary(vec)

    assert len(binary_blob) == (dim + 7) // 8
    assert len(ternary_blob) == (dim + 3) // 4

    # Roundtrip works
    recovered_binary = dequantize_binary(binary_blob, dim)
    recovered_ternary = dequantize_ternary(ternary_blob, dim)

    assert len(recovered_binary) == dim
    assert len(recovered_ternary) == dim
