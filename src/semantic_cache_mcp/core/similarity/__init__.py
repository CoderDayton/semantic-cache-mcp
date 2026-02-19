"""Similarity sub-package: cosine similarity, quantization, and LSH."""

from __future__ import annotations

from ._cosine import (
    SimilarityConfig,
    cosine_similarity,
    cosine_similarity_batch,
    cosine_similarity_batch_matrix,
    cosine_similarity_with_pruning,
    dequantize_embedding,
    quantize_embedding,
    similarity_from_quantized_blob,
    top_k_from_quantized,
    top_k_similarities,
)
from ._lsh import (
    DEFAULT_LSH_CONFIG,
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
from ._quantization import (
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

__all__ = [
    # cosine
    "SimilarityConfig",
    "cosine_similarity",
    "cosine_similarity_batch",
    "cosine_similarity_batch_matrix",
    "cosine_similarity_with_pruning",
    "quantize_embedding",
    "dequantize_embedding",
    "similarity_from_quantized_blob",
    "top_k_from_quantized",
    "top_k_similarities",
    # LSH
    "LSHConfig",
    "LSHIndex",
    "DEFAULT_LSH_CONFIG",
    "compute_simhash",
    "compute_simhash_batch",
    "hamming_distance",
    "hamming_distance_batch",
    "create_lsh_index",
    "serialize_lsh_index",
    "deserialize_lsh_index",
    # binary/ternary quantization
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
