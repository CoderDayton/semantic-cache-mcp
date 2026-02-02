"""Core algorithms for semantic caching."""

from .chunking import hypercdc_chunks
from .compression import compress_adaptive, decompress, estimate_entropy
from .hashing import (
    hash_chunk,
    hash_content,
    hash_chunk_binary,
    DeduplicateIndex,
    HierarchicalHasher,
    StreamingHasher,
    get_hash_stats,
)
from .similarity import (
    cosine_similarity,
    cosine_similarity_batch,
    cosine_similarity_batch_matrix,
    top_k_similarities,
    quantize_embedding,
    dequantize_embedding,
    similarity_from_quantized_blob,
    top_k_from_quantized,
)
from .text import (
    generate_diff,
    truncate_smart,
    truncate_semantic,
    compute_delta,
    apply_delta,
    diff_stats,
    DiffDelta,
)
from .tokenizer import BPETokenizer, count_tokens, get_tokenizer
from .embeddings import embed, embed_query, warmup as warmup_embeddings, get_model_info

__all__ = [
    "hypercdc_chunks",
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
]
