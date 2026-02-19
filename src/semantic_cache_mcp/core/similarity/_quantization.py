"""
Extreme quantization for embeddings: binary (1-bit) and ternary (1.58-bit).

Based on research showing embeddings can be aggressively quantized while
maintaining strong correlation for similarity ranking:

- Binary ({-1, +1}): 1 bit per dim, 8x compression vs int8
- Ternary ({-1, 0, +1}): ~1.58 bits per dim, 4x compression vs int8

Storage comparison for 384D embedding:
- float32 JSON: ~16,970 bytes (baseline)
- int8 blob: 388 bytes (scale + quantized)
- ternary: 96-128 bytes (packed 2 bits per value)
- binary: 48 bytes (packed 1 bit per value)

Trade-offs:
- Higher compression = lower precision for similarity scores
- Ranking correlation (Spearman) typically preserved well
- Best for candidate pre-filtering, not final ranking
"""

from __future__ import annotations

import array

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Binary Quantization (1-bit)
# ---------------------------------------------------------------------------


def quantize_binary(v: array.array | list | NDArray[np.float32]) -> bytes:
    """
    Quantize embedding to binary (1 bit per dimension).

    Uses sign-based quantization: +1 if value >= 0, -1 otherwise.
    Storage: ceil(dim/8) bytes = 48 bytes for 384D.

    For similarity: use Hamming distance or popcount-based dot product.

    Args:
        v: Embedding vector

    Returns:
        Packed binary blob (bits packed into bytes, MSB first)
    """
    # Convert to numpy
    if isinstance(v, array.array):
        arr = np.frombuffer(v, dtype=np.float32).copy()
    else:
        arr = np.asarray(v, dtype=np.float32)

    # Sign-based quantization: True if >= 0
    bits = arr >= 0

    # Pack bits into bytes
    dim = len(arr)
    n_bytes = (dim + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)

    for i, bit in enumerate(bits):
        if bit:
            packed[i // 8] |= 1 << (7 - (i % 8))

    return packed.tobytes()


def dequantize_binary(blob: bytes, dim: int) -> NDArray[np.float32]:
    """
    Dequantize binary blob to float32.

    Note: Dequantized values are {-1.0, +1.0}, not original magnitude.

    Args:
        blob: Binary blob from quantize_binary()
        dim: Original embedding dimension

    Returns:
        Float32 array with values {-1.0, +1.0}
    """
    packed = np.frombuffer(blob, dtype=np.uint8)
    result = np.zeros(dim, dtype=np.float32)

    for i in range(dim):
        bit = (packed[i // 8] >> (7 - (i % 8))) & 1
        result[i] = 1.0 if bit else -1.0

    return result


def hamming_similarity_binary(blob1: bytes, blob2: bytes) -> float:
    """
    Compute similarity from two binary blobs using Hamming distance.

    Similarity = 1 - (hamming_distance / total_bits)

    This correlates with cosine similarity for normalized embeddings.

    Args:
        blob1: First binary blob
        blob2: Second binary blob

    Returns:
        Similarity in [0, 1] where 1 = identical
    """
    arr1 = np.frombuffer(blob1, dtype=np.uint8)
    arr2 = np.frombuffer(blob2, dtype=np.uint8)

    # XOR and count differing bits
    xor = arr1 ^ arr2
    hamming = np.unpackbits(xor).sum()

    total_bits = len(blob1) * 8
    return 1.0 - (hamming / total_bits)


def batch_hamming_similarity_binary(
    query_blob: bytes, target_blobs: list[bytes]
) -> NDArray[np.float32]:
    """
    Batch Hamming similarity from query to multiple targets.

    Args:
        query_blob: Query binary blob
        target_blobs: List of target binary blobs

    Returns:
        Array of similarities in [0, 1]
    """
    if not target_blobs:
        return np.array([], dtype=np.float32)

    query = np.frombuffer(query_blob, dtype=np.uint8)
    total_bits = len(query_blob) * 8

    # Stack targets
    targets = np.vstack([np.frombuffer(b, dtype=np.uint8) for b in target_blobs])

    # Vectorized XOR and popcount
    xor = targets ^ query
    hamming_distances = np.unpackbits(xor, axis=1).sum(axis=1)

    return 1.0 - (hamming_distances / total_bits).astype(np.float32)


# ---------------------------------------------------------------------------
# Ternary Quantization (1.58-bit)
# ---------------------------------------------------------------------------


def quantize_ternary(
    v: array.array | list | NDArray[np.float32],
    threshold_percentile: float = 33.0,
) -> bytes:
    """
    Quantize embedding to ternary ({-1, 0, +1}).

    Uses threshold-based quantization:
    - +1 if value > threshold
    - -1 if value < -threshold
    - 0 otherwise

    Threshold is computed as the percentile of absolute values.

    Storage format: 2 bits per value = dim/4 bytes = 96 bytes for 384D
    Encoding: 00 = -1, 01 = 0, 10 = +1

    Args:
        v: Embedding vector
        threshold_percentile: Percentile for threshold (default: 33% = ~1/3 zeros)

    Returns:
        Packed ternary blob
    """
    # Convert to numpy
    if isinstance(v, array.array):
        arr = np.frombuffer(v, dtype=np.float32).copy()
    else:
        arr = np.asarray(v, dtype=np.float32)

    # Compute threshold
    abs_vals = np.abs(arr)
    threshold = np.percentile(abs_vals, threshold_percentile) if len(arr) > 0 else 0.0

    # Quantize to {-1, 0, 1}
    ternary = np.zeros(len(arr), dtype=np.int8)
    ternary[arr > threshold] = 1
    ternary[arr < -threshold] = -1

    # Pack into 2 bits per value: -1=00, 0=01, 1=10
    # Each byte holds 4 values
    dim = len(arr)
    n_bytes = (dim + 3) // 4
    packed = np.zeros(n_bytes, dtype=np.uint8)

    for i, val in enumerate(ternary):
        encoded = val + 1  # -1 -> 0, 0 -> 1, 1 -> 2
        byte_idx = i // 4
        bit_offset = (3 - (i % 4)) * 2  # MSB first
        packed[byte_idx] |= encoded << bit_offset

    return packed.tobytes()


def dequantize_ternary(blob: bytes, dim: int) -> NDArray[np.float32]:
    """
    Dequantize ternary blob to float32.

    Note: Dequantized values are {-1.0, 0.0, +1.0}, not original magnitude.

    Args:
        blob: Ternary blob from quantize_ternary()
        dim: Original embedding dimension

    Returns:
        Float32 array with values {-1.0, 0.0, +1.0}
    """
    packed = np.frombuffer(blob, dtype=np.uint8)
    result = np.zeros(dim, dtype=np.float32)

    for i in range(dim):
        byte_idx = i // 4
        bit_offset = (3 - (i % 4)) * 2
        encoded = (packed[byte_idx] >> bit_offset) & 0b11
        result[i] = float(int(encoded) - 1)  # 0 -> -1, 1 -> 0, 2 -> 1

    return result


def dot_product_ternary(blob1: bytes, blob2: bytes, dim: int) -> float:
    """
    Compute dot product from two ternary blobs.

    For ternary values, dot product is fast to compute:
    - Only count where both are non-zero
    - Result is sum of (v1[i] * v2[i]) for each position

    Args:
        blob1: First ternary blob
        blob2: Second ternary blob
        dim: Embedding dimension

    Returns:
        Dot product (proportional to cosine similarity for normalized vectors)
    """
    v1 = dequantize_ternary(blob1, dim)
    v2 = dequantize_ternary(blob2, dim)
    return float(np.dot(v1, v2))


def batch_dot_product_ternary(
    query_blob: bytes, target_blobs: list[bytes], dim: int
) -> NDArray[np.float32]:
    """
    Batch dot product from query to multiple ternary targets.

    Args:
        query_blob: Query ternary blob
        target_blobs: List of target ternary blobs
        dim: Embedding dimension

    Returns:
        Array of dot products
    """
    if not target_blobs:
        return np.array([], dtype=np.float32)

    query = dequantize_ternary(query_blob, dim)
    targets = np.vstack([dequantize_ternary(b, dim) for b in target_blobs])

    return (targets @ query).astype(np.float32)


# ---------------------------------------------------------------------------
# Hybrid Storage (int8 with binary/ternary index)
# ---------------------------------------------------------------------------


def quantize_hybrid(
    v: array.array | list | NDArray[np.float32],
) -> tuple[bytes, bytes]:
    """
    Create hybrid quantization: int8 for precision + binary for fast filtering.

    Returns both representations for two-phase search:
    1. Binary for fast candidate filtering (Hamming distance)
    2. int8 for precise ranking on candidates

    Args:
        v: Embedding vector

    Returns:
        (binary_blob, int8_blob) tuple
    """
    # Import here to avoid circular dependency
    from ._cosine import quantize_embedding  # noqa: PLC0415

    binary_blob = quantize_binary(v)
    int8_blob = quantize_embedding(v)

    return binary_blob, int8_blob


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


def _spearman_correlation(x: NDArray[np.float32], y: NDArray[np.float32]) -> float:
    """
    Compute Spearman rank correlation between two arrays.

    Spearman = Pearson correlation of rank values.
    """
    n = len(x)
    if n < 2:
        return 0.0

    # Compute ranks (1-based, average ties)
    def rankdata(arr: NDArray[np.float32]) -> NDArray[np.float32]:
        """Assign ranks to data, handling ties by averaging."""
        order = arr.argsort()
        ranks = np.empty_like(arr)
        ranks[order] = np.arange(1, n + 1, dtype=np.float32)
        return ranks

    rank_x = rankdata(x)
    rank_y = rankdata(y)

    # Pearson correlation on ranks
    mean_x = rank_x.mean()
    mean_y = rank_y.mean()
    dx = rank_x - mean_x
    dy = rank_y - mean_y

    num = np.sum(dx * dy)
    denom = np.sqrt(np.sum(dx * dx) * np.sum(dy * dy))

    if denom == 0:
        return 0.0

    return float(num / denom)


def evaluate_quantization_accuracy(
    embeddings: list[NDArray[np.float32]],
    method: str = "int8",
    k: int = 10,
) -> dict:
    """
    Evaluate quantization accuracy for similarity ranking.

    Computes:
    - Rank correlation (Spearman) vs float32 baseline
    - Recall@K for top-K retrieval
    - Storage size

    Args:
        embeddings: List of embeddings to evaluate
        method: "int8", "ternary", or "binary"
        k: K for recall@K

    Returns:
        Dict with accuracy metrics
    """
    from ._cosine import quantize_embedding, similarity_from_quantized_blob  # noqa: PLC0415

    if len(embeddings) < 2:
        return {"error": "Need at least 2 embeddings"}

    n = len(embeddings)
    dim = len(embeddings[0])

    # Compute ground truth similarities (float32)
    query = embeddings[0]
    targets = embeddings[1:]

    # Float32 baseline
    query_arr = np.asarray(query, dtype=np.float32)
    target_matrix = np.vstack([np.asarray(e, dtype=np.float32) for e in targets])
    float_sims = (target_matrix @ query_arr).astype(np.float32)

    # Quantized similarities
    if method == "int8":
        query_blob = quantize_embedding(query)
        target_blobs = [quantize_embedding(e) for e in targets]
        quant_sims = similarity_from_quantized_blob(query, target_blobs)
        storage_per_vec = 4 + dim  # scale + int8 values
    elif method == "ternary":
        query_blob = quantize_ternary(query)
        target_blobs = [quantize_ternary(e) for e in targets]
        quant_sims = batch_dot_product_ternary(query_blob, target_blobs, dim)
        storage_per_vec = (dim + 3) // 4  # 2 bits per value
    elif method == "binary":
        query_blob = quantize_binary(query)
        target_blobs = [quantize_binary(e) for e in targets]
        quant_sims = batch_hamming_similarity_binary(query_blob, target_blobs)
        # Scale to match float range for correlation
        quant_sims = quant_sims * 2 - 1  # [0,1] -> [-1,1]
        storage_per_vec = (dim + 7) // 8  # 1 bit per value
    else:
        raise ValueError(f"Unknown method: {method}")

    # Spearman rank correlation
    correlation = _spearman_correlation(float_sims, quant_sims)

    # Recall@K
    float_top_k = set(np.argsort(-float_sims)[:k])
    quant_top_k = set(np.argsort(-quant_sims)[:k])
    recall_at_k = len(float_top_k & quant_top_k) / k

    # Storage comparison
    float_storage = dim * 4  # float32
    int8_storage = 4 + dim  # scale + int8

    return {
        "method": method,
        "num_embeddings": n,
        "dimension": dim,
        "spearman_correlation": float(correlation),
        "recall_at_k": recall_at_k,
        "k": k,
        "storage_per_vector_bytes": storage_per_vec,
        "compression_vs_float32": float_storage / storage_per_vec,
        "compression_vs_int8": int8_storage / storage_per_vec,
    }


__all__ = [
    "quantize_binary",
    "dequantize_binary",
    "hamming_similarity_binary",
    "batch_hamming_similarity_binary",
    "quantize_ternary",
    "dequantize_ternary",
    "dot_product_ternary",
    "batch_dot_product_ternary",
    "quantize_hybrid",
    "evaluate_quantization_accuracy",
]
