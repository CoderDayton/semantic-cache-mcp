"""Content hashing using BLAKE2b with LRU caching."""

from __future__ import annotations

import hashlib
from functools import lru_cache

from ..types import ChunkHash, ContentHash


@lru_cache(maxsize=1024)
def hash_chunk(data: bytes) -> ChunkHash:
    """Hash chunk using BLAKE2b with LRU cache.

    BLAKE2b is faster than SHA256 while remaining cryptographically secure.
    The LRU cache avoids rehashing repeated chunks.

    Args:
        data: Raw chunk bytes

    Returns:
        40-character hex digest
    """
    return hashlib.blake2b(data, digest_size=20).hexdigest()


@lru_cache(maxsize=512)
def hash_content(content: str) -> ContentHash:
    """Hash full content for change detection with LRU cache.

    Args:
        content: Text content to hash

    Returns:
        32-character hex digest
    """
    return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()
