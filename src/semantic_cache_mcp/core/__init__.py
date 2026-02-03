"""Core algorithms for semantic caching."""

from .chunking import hypercdc_chunks
from .chunking_simd import (
    get_optimal_chunker,
    hypercdc_simd_boundaries,
    hypercdc_simd_chunks,
)
from .compression import compress_adaptive, decompress, estimate_entropy
from .embeddings import embed, embed_query, get_model_info
from .embeddings import warmup as warmup_embeddings
from .hashing import (
    DeduplicateIndex,
    HierarchicalHasher,
    StreamingHasher,
    get_hash_stats,
    hash_chunk,
    hash_chunk_binary,
    hash_content,
)
from .lsh import (
    DEFAULT_LSH_CONFIG,
    LSHConfig,
    LSHIndex,
    compute_simhash,
    create_lsh_index,
    hamming_distance,
)
from .quantization import (
    batch_dot_product_ternary,
    batch_hamming_similarity_binary,
    dequantize_binary,
    dequantize_ternary,
    evaluate_quantization_accuracy,
    hamming_similarity_binary,
    quantize_binary,
    quantize_hybrid,
    quantize_ternary,
)
from .similarity import (
    cosine_similarity,
    cosine_similarity_batch,
    cosine_similarity_batch_matrix,
    dequantize_embedding,
    quantize_embedding,
    similarity_from_quantized_blob,
    top_k_from_quantized,
    top_k_similarities,
)
from .text import (
    DiffDelta,
    apply_delta,
    compute_delta,
    diff_stats,
    generate_diff,
    truncate_semantic,
    truncate_smart,
)
from .tokenizer import BPETokenizer, count_tokens, get_tokenizer

__all__ = [
    "hypercdc_chunks",
    "hypercdc_simd_chunks",
    "hypercdc_simd_boundaries",
    "get_optimal_chunker",
    "LSHConfig",
    "LSHIndex",
    "compute_simhash",
    "create_lsh_index",
    "hamming_distance",
    "DEFAULT_LSH_CONFIG",
    "compress_adaptive",
    "decompress",
    "estimate_entropy",
    "hash_chunk",
    "hash_content",
    "hash_chunk_binary",
    "DeduplicateIndex",
    "HierarchicalHasher",
    "StreamingHasher",
    "get_hash_stats",
    "cosine_similarity",
    "cosine_similarity_batch",
    "cosine_similarity_batch_matrix",
    "top_k_similarities",
    "quantize_embedding",
    "dequantize_embedding",
    "similarity_from_quantized_blob",
    "top_k_from_quantized",
    "count_tokens",
    "generate_diff",
    "truncate_smart",
    "truncate_semantic",
    "compute_delta",
    "apply_delta",
    "diff_stats",
    "DiffDelta",
    "BPETokenizer",
    "get_tokenizer",
    "embed",
    "embed_query",
    "warmup_embeddings",
    "get_model_info",
    # Extreme quantization (binary/ternary)
    "quantize_binary",
    "dequantize_binary",
    "hamming_similarity_binary",
    "batch_hamming_similarity_binary",
    "quantize_ternary",
    "dequantize_ternary",
    "batch_dot_product_ternary",
    "quantize_hybrid",
    "evaluate_quantization_accuracy",
]
