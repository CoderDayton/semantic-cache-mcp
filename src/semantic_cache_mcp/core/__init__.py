"""Core algorithms for semantic caching."""

from .chunking import (
    get_optimal_chunker,
    hypercdc_chunks,
    hypercdc_simd_boundaries,
    hypercdc_simd_chunks,
)
from .embeddings import embed, embed_batch, embed_query, get_model_info
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
from .similarity import (
    cosine_similarity,
    cosine_similarity_batch,
    cosine_similarity_batch_matrix,
    top_k_similarities,
)
from .text import (
    DEFAULT_SUMMARIZATION_CONFIG,
    DiffDelta,
    Segment,
    SummarizationConfig,
    apply_delta,
    compute_delta,
    diff_stats,
    extract_segments,
    generate_diff,
    score_segments,
    summarize_semantic,
    truncate_semantic,
    truncate_smart,
    truncate_with_summarization,
)
from .tokenizer import BPETokenizer, count_tokens, get_tokenizer

__all__ = [
    "hypercdc_chunks",
    "hypercdc_simd_chunks",
    "hypercdc_simd_boundaries",
    "get_optimal_chunker",
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
    "embed_batch",
    "embed_query",
    "warmup_embeddings",
    "get_model_info",
    # Semantic summarization
    "SummarizationConfig",
    "DEFAULT_SUMMARIZATION_CONFIG",
    "Segment",
    "extract_segments",
    "score_segments",
    "summarize_semantic",
    "truncate_with_summarization",
]
