"""Core algorithms for semantic caching."""

from .chunking import hypercdc_chunks
from .compression import compress_adaptive, decompress, estimate_entropy
from .hashing import hash_chunk, hash_content
from .similarity import cosine_similarity
from .text import generate_diff, truncate_smart
from .tokenizer import BPETokenizer, count_tokens, get_tokenizer
from .embeddings import embed, embed_query, warmup as warmup_embeddings, get_model_info

__all__ = [
    "hypercdc_chunks",
    "compress_adaptive",
    "decompress",
    "estimate_entropy",
    "hash_chunk",
    "hash_content",
    "cosine_similarity",
    "count_tokens",
    "generate_diff",
    "truncate_smart",
    "BPETokenizer",
    "get_tokenizer",
    "embed",
    "embed_query",
    "warmup_embeddings",
    "get_model_info",
]
