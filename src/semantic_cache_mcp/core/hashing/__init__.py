from __future__ import annotations

from ._blake import (
    _CONTENT_CACHE_BYPASS_SIZE,
    DEFAULT_CONFIG,
    KEYED_HASH_KEY_SIZE,
    HashConfig,
    # Internal helpers re-exported for test access
    _hash_bytes,
    _hash_hex,
    hash_chunk,
    hash_content,
    keyed_hash,
)

__all__ = [
    "keyed_hash",
    "KEYED_HASH_KEY_SIZE",
    "hash_content",
    "hash_chunk",
    # Internal — tests import these directly
    "_hash_bytes",
    "_hash_hex",
    "_CONTENT_CACHE_BYPASS_SIZE",
    "HashConfig",
    "DEFAULT_CONFIG",
]
