"""Core algorithms for semantic caching."""

from .chunking import content_defined_chunking
from .compression import compress_adaptive, decompress
from .hashing import hash_chunk, hash_content
from .similarity import cosine_similarity, count_tokens
from .text import generate_diff, truncate_smart

__all__ = [
    "content_defined_chunking",
    "compress_adaptive",
    "decompress",
    "hash_chunk",
    "hash_content",
    "cosine_similarity",
    "count_tokens",
    "generate_diff",
    "truncate_smart",
]
