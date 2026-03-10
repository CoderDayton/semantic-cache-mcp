"""Public API for content hashing. Implementation lives in _blake.py."""

from __future__ import annotations

from ._blake import (
    _CONTENT_CACHE_BYPASS_SIZE,
    DEFAULT_CONFIG,
    CollisionTracker,
    DeduplicateIndex,
    HashConfig,
    HierarchicalHasher,
    StreamingHasher,
    # Internal helpers re-exported for test access
    _hash_bytes,
    _hash_hex,
    get_hash_stats,
    hash_block,
    hash_chunk,
    hash_chunk_binary,
    hash_chunk_with_collision_check,
    hash_chunks_streaming,
    hash_content,
    hash_file_streaming,
    reset_collision_tracker,
)

__all__ = [
    "hash_content",
    "hash_chunk",
    "hash_chunk_binary",
    "hash_chunk_with_collision_check",
    "hash_block",
    "hash_chunks_streaming",
    "hash_file_streaming",
    "StreamingHasher",
    "HierarchicalHasher",
    "DeduplicateIndex",
    "CollisionTracker",
    "get_hash_stats",
    "reset_collision_tracker",
    # Internal — tests import these directly
    "_hash_bytes",
    "_hash_hex",
    "_CONTENT_CACHE_BYPASS_SIZE",
    "HashConfig",
    "DEFAULT_CONFIG",
]
