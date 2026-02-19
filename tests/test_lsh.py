"""Tests for LSH-based similarity matching."""

from __future__ import annotations

import array

import numpy as np
import pytest

from semantic_cache_mcp.core.similarity import (
    LSHConfig,
    LSHIndex,
    compute_simhash,
    compute_simhash_batch,
    create_lsh_index,
    deserialize_lsh_index,
    hamming_distance,
    hamming_distance_batch,
    serialize_lsh_index,
)


class TestSimHash:
    """Tests for SimHash computation."""

    def test_deterministic(self) -> None:
        """Same embedding produces same hash."""
        from semantic_cache_mcp.core.similarity._lsh import _generate_hyperplanes

        vec = np.random.randn(384).astype(np.float32)
        hyperplanes = _generate_hyperplanes(384, 64, seed=42)

        hash1 = compute_simhash(vec, hyperplanes)
        hash2 = compute_simhash(vec, hyperplanes)

        assert hash1 == hash2

    def test_similar_vectors_similar_hashes(self) -> None:
        """Similar vectors should have small Hamming distance."""
        from semantic_cache_mcp.core.similarity._lsh import _generate_hyperplanes

        # Create two similar vectors
        base = np.random.randn(384).astype(np.float32)
        similar = base + np.random.randn(384).astype(np.float32) * 0.1  # Small noise

        hyperplanes = _generate_hyperplanes(384, 64, seed=42)

        hash1 = compute_simhash(base, hyperplanes)
        hash2 = compute_simhash(similar, hyperplanes)

        distance = hamming_distance(hash1, hash2)
        # Similar vectors should have Hamming distance < 20 (out of 64 bits)
        assert distance < 20, f"Similar vectors have distance {distance}"

    def test_different_vectors_different_hashes(self) -> None:
        """Very different vectors should have larger Hamming distance."""
        from semantic_cache_mcp.core.similarity._lsh import _generate_hyperplanes

        vec1 = np.random.randn(384).astype(np.float32)
        vec2 = np.random.randn(384).astype(np.float32)

        hyperplanes = _generate_hyperplanes(384, 64, seed=42)

        hash1 = compute_simhash(vec1, hyperplanes)
        hash2 = compute_simhash(vec2, hyperplanes)

        # Random vectors should have distance ~32 (half the bits)
        distance = hamming_distance(hash1, hash2)
        assert 20 < distance < 44, f"Random vectors have distance {distance}"

    def test_batch_matches_single(self) -> None:
        """Batch computation matches single computation."""
        from semantic_cache_mcp.core.similarity._lsh import _generate_hyperplanes

        vectors = [np.random.randn(384).astype(np.float32) for _ in range(10)]
        hyperplanes = _generate_hyperplanes(384, 64, seed=42)

        # Single computation
        single_hashes = [compute_simhash(v, hyperplanes) for v in vectors]

        # Batch computation
        batch_hashes = compute_simhash_batch(vectors, hyperplanes)

        for i, (single, batch) in enumerate(zip(single_hashes, batch_hashes)):
            assert single == batch, f"Mismatch at index {i}"

    def test_array_array_input(self) -> None:
        """Works with array.array input."""
        from semantic_cache_mcp.core.similarity._lsh import _generate_hyperplanes

        vec_np = np.random.randn(384).astype(np.float32)
        vec_arr = array.array("f", vec_np.tolist())

        hyperplanes = _generate_hyperplanes(384, 64, seed=42)

        hash_np = compute_simhash(vec_np, hyperplanes)
        hash_arr = compute_simhash(vec_arr, hyperplanes)

        assert hash_np == hash_arr


class TestHammingDistance:
    """Tests for Hamming distance computation."""

    def test_zero_distance_same_hash(self) -> None:
        """Same hash has distance 0."""
        assert hamming_distance(0xABCD, 0xABCD) == 0

    def test_all_bits_different(self) -> None:
        """All bits different gives max distance."""
        # 8 bits all different
        assert hamming_distance(0xFF, 0x00) == 8

    def test_specific_distance(self) -> None:
        """Test specific known distance."""
        # 0b1010 vs 0b1111 = 2 bits different
        assert hamming_distance(0b1010, 0b1111) == 2

    def test_batch_matches_single(self) -> None:
        """Batch distance matches single computation."""
        query = 0x123456789ABCDEF0
        targets = np.array([0x123456789ABCDEF0, 0x0, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)

        batch_distances = hamming_distance_batch(query, targets)

        for i, target in enumerate(targets):
            single = hamming_distance(query, int(target))
            assert batch_distances[i] == single


class TestLSHIndex:
    """Tests for LSH index operations."""

    def test_add_and_query(self) -> None:
        """Basic add and query operations."""
        index = LSHIndex(config=LSHConfig(num_bits=64, num_tables=2))

        # Add some vectors
        vec1 = np.random.randn(384).astype(np.float32)
        vec2 = vec1 + np.random.randn(384).astype(np.float32) * 0.1  # Similar

        index.add(1, vec1)
        index.add(2, vec2)

        # Query with similar vector
        results = index.query(vec1, k=5, return_distances=True)

        assert len(results) > 0
        # Should find the original vector
        assert any(r[0] == 1 for r in results)

    def test_remove(self) -> None:
        """Remove items from index."""
        index = LSHIndex()

        vec = np.random.randn(384).astype(np.float32)
        index.add(1, vec)
        index.add(2, vec + 0.01)

        assert index.remove(1)
        assert not index.remove(999)  # Non-existent

        # Item 1 should no longer be found
        results = index.query(vec, k=10)
        assert 1 not in results

    def test_query_empty_index(self) -> None:
        """Query on empty index returns empty."""
        index = LSHIndex()
        results = index.query(np.random.randn(384).astype(np.float32), k=10)
        assert results == []

    def test_query_with_hamming(self) -> None:
        """Query with Hamming distance threshold."""
        index = LSHIndex()

        base = np.random.randn(384).astype(np.float32)
        index.add(1, base)
        index.add(2, base + np.random.randn(384).astype(np.float32) * 0.05)
        index.add(3, np.random.randn(384).astype(np.float32))  # Different

        results = index.query_with_hamming(base, max_distance=10)

        # Should find item 1 (exact match) and item 2 (similar)
        ids = [r[0] for r in results]
        assert 1 in ids
        # Item 1 should have distance 0
        for item_id, dist in results:
            if item_id == 1:
                assert dist == 0

    def test_clear(self) -> None:
        """Clear removes all items."""
        index = LSHIndex()

        for i in range(10):
            index.add(i, np.random.randn(384).astype(np.float32))

        count = index.clear()
        assert count == 10
        assert index.get_stats()["num_items"] == 0

    def test_get_stats(self) -> None:
        """Stats reflect index state."""
        config = LSHConfig(num_bits=64, num_tables=4, band_size=8)
        index = LSHIndex(config=config)

        for i in range(5):
            index.add(i, np.random.randn(384).astype(np.float32))

        stats = index.get_stats()
        assert stats["num_items"] == 5
        assert stats["num_tables"] == 4
        assert stats["num_bits"] == 64
        assert stats["embeddings_stored"] == 5


class TestLSHIndexSerialization:
    """Tests for LSH index serialization."""

    def test_serialize_deserialize(self) -> None:
        """Index survives serialization round-trip."""
        index = LSHIndex(config=LSHConfig(num_bits=32, num_tables=2))

        vectors = [np.random.randn(128).astype(np.float32) for _ in range(5)]
        for i, vec in enumerate(vectors):
            index.add(i, vec)

        # Serialize
        data = serialize_lsh_index(index)

        # Deserialize
        restored = deserialize_lsh_index(data)

        # Check structure
        assert restored.dim == index.dim
        assert restored.config.num_bits == index.config.num_bits
        assert restored.config.num_tables == index.config.num_tables
        assert len(restored._signatures) == len(index._signatures)
        assert len(restored._embeddings) == len(index._embeddings)

        # Query should work
        results = restored.query(vectors[0], k=5, return_distances=True)
        assert len(results) > 0
        assert any(r[0] == 0 for r in results)

    def test_empty_index_serialization(self) -> None:
        """Empty index serializes and deserializes."""
        index = LSHIndex()
        data = serialize_lsh_index(index)
        restored = deserialize_lsh_index(data)
        assert restored.get_stats()["num_items"] == 0


class TestCreateLSHIndex:
    """Tests for the create_lsh_index helper."""

    def test_creates_populated_index(self) -> None:
        """Creates index with all items."""
        items = [(i, np.random.randn(384).astype(np.float32)) for i in range(10)]

        index = create_lsh_index(items)

        assert index.get_stats()["num_items"] == 10

    def test_custom_config(self) -> None:
        """Respects custom config."""
        items = [(i, np.random.randn(128).astype(np.float32)) for i in range(5)]
        config = LSHConfig(num_bits=32, num_tables=2)

        index = create_lsh_index(items, config=config)

        assert index.config.num_bits == 32
        assert index.config.num_tables == 2


class TestLSHPrecisionRecall:
    """Tests for LSH precision and recall characteristics."""

    def test_finds_most_similar(self) -> None:
        """LSH finds the most similar item with high probability."""
        np.random.seed(42)

        # Create a base vector and variations
        base = np.random.randn(384).astype(np.float32)
        base = base / np.linalg.norm(base)  # Normalize

        # Create items with varying similarity
        items = []
        items.append((0, base))  # Exact match
        items.append((1, base + np.random.randn(384).astype(np.float32) * 0.1))  # Very similar
        items.append(
            (2, base + np.random.randn(384).astype(np.float32) * 0.3)
        )  # Moderately similar

        # Add random items
        for i in range(3, 20):
            items.append((i, np.random.randn(384).astype(np.float32)))

        config = LSHConfig(
            num_bits=64,
            num_tables=4,
            band_size=8,
            similarity_threshold=0.5,
        )
        index = create_lsh_index(items, config=config)

        # Query with base vector
        results = index.query(base, k=5, return_distances=True)

        # Should find the exact match
        result_ids = [r[0] for r in results]
        assert 0 in result_ids, f"Exact match not found in {result_ids}"

        # The most similar should be first
        if results:
            assert results[0][0] == 0, f"Exact match not first: {results}"


@pytest.mark.parametrize("num_bits", [32, 64, 128])
def test_different_bit_sizes(num_bits: int) -> None:
    """LSH works with different hash bit sizes."""
    config = LSHConfig(num_bits=num_bits, num_tables=2, band_size=min(8, num_bits // 4))
    index = LSHIndex(config=config)

    vec = np.random.randn(256).astype(np.float32)
    index.add(1, vec)

    results = index.query(vec, k=1)
    assert 1 in results


@pytest.mark.parametrize("dim", [128, 384, 768])
def test_different_dimensions(dim: int) -> None:
    """LSH works with different embedding dimensions."""
    index = LSHIndex()

    vec = np.random.randn(dim).astype(np.float32)
    index.add(1, vec)

    results = index.query(vec, k=1)
    assert 1 in results
