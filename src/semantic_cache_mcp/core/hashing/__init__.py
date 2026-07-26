from __future__ import annotations

from ._blake import (
    _CONTENT_CACHE_BYPASS_SIZE,
    DEFAULT_CONFIG,
    KEYED_HASH_KEY_SIZE,
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
    keyed_hash,
    reset_collision_tracker,
)

__all__ = [
    "keyed_hash",
    "KEYED_HASH_KEY_SIZE",
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
