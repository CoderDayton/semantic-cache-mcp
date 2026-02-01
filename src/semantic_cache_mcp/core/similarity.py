"""Similarity utilities for embedding comparisons."""

from __future__ import annotations

from ..types import EmbeddingVector


def cosine_similarity(a: EmbeddingVector, b: EmbeddingVector) -> float:
    """Compute cosine similarity for normalized embeddings.

    For normalized vectors, cosine similarity equals dot product.
    Uses sum() with generator for C-level performance.

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Similarity score in [-1, 1] range
    """
    return sum(x * y for x, y in zip(a, b))
