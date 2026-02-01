"""Similarity and token counting utilities."""

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


def count_tokens(content: str) -> int:
    """Approximate token count using BPE heuristic.

    Counts whitespace to estimate words without allocating a list.
    More accurate than char/4 for code content.

    Args:
        content: Text to count tokens for

    Returns:
        Estimated token count
    """
    if not content:
        return 0

    spaces = content.count(" ") + content.count("\n") + content.count("\t")
    words = spaces + 1
    return int(words * 1.3 + len(content) * 0.1)
