"""Content hashing: BLAKE3 when available, BLAKE2b otherwise.

Three entry points carry the cache: ``hash_content`` (file identity and the
possession proof callers echo back), ``hash_chunk`` (per-CDC-chunk identity, so
a rewrite reinserts only the chunks that changed), and ``keyed_hash`` (the MAC
behind a ranged read's ``coverage_token``).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Final

# Optional: use blake3 if available (faster than BLAKE2b)
try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Type aliases for clarity
ChunkHash = str  # Hex digest (64 chars for 32-byte BLAKE3)
ContentHash = str  # Full content hash


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class HashConfig:
    # Hash function selection
    USE_BLAKE3: bool = HAS_BLAKE3

    # Digest sizes (bytes)
    CHUNK_DIGEST_SIZE: int = 32  # 256 bits for chunks (collision-free for dedup)
    CONTENT_DIGEST_SIZE: int = 32  # 256 bits for full content

    # Caching configuration.
    #
    # Both caches are keyed on the buffer they hashed, so an entry retains that
    # buffer for as long as it stays cached: the ceiling is entries x buffer
    # size, not entries x pointer. Chunks reach CHUNK_MAX_SIZE (64 KiB) and
    # content entries are capped at _CONTENT_CACHE_BYPASS_SIZE (64 KiB), so the
    # sizes below are chosen to make the worst case a number worth stating
    # rather than one that runs to a gigabyte in a long-lived server. A cache
    # hit saves 1.6-3x versus re-hashing, which is worth having for a working
    # set — but not worth an unbounded resident cost.
    CHUNK_CACHE_SIZE: int = 1024  # <= 64 MiB retained (1024 x 64 KiB)
    CONTENT_CACHE_SIZE: int = 512  # <= 32 MiB retained (512 x 64 KiB)


DEFAULT_CONFIG = HashConfig()


# ---------------------------------------------------------------------------
# Hash function selection
# ---------------------------------------------------------------------------


# Key length for keyed hashing. BLAKE3's keyed mode requires exactly 32 bytes
# and rejects anything else; BLAKE2b accepts up to 64. One size therefore works
# whichever backend is present, so callers never have to know which signed.
KEYED_HASH_KEY_SIZE: Final = 32


if HAS_BLAKE3:
    _blake3_blake3 = blake3.blake3

    def _hash_bytes(data: bytes, digest_size: int = 32) -> bytes:
        return _blake3_blake3(data).digest(length=digest_size)

    def _keyed_bytes(data: bytes, key: bytes, digest_size: int) -> bytes:
        return _blake3_blake3(data, key=key).digest(length=digest_size)
else:
    _hashlib_blake2b = hashlib.blake2b

    def _hash_bytes(data: bytes, digest_size: int = 32) -> bytes:
        return _hashlib_blake2b(data, digest_size=digest_size).digest()

    def _keyed_bytes(data: bytes, key: bytes, digest_size: int) -> bytes:
        return _hashlib_blake2b(data, key=key, digest_size=digest_size).digest()


def _hash_hex(data: bytes, digest_size: int = 32) -> str:
    digest = _hash_bytes(data, digest_size)
    return digest.hex()


def keyed_hash(data: bytes, key: bytes, digest_size: int) -> str:
    """Authenticate *data* under *key*, returning a hex tag.

    Both backends are MACs by construction in keyed mode, so this needs no
    HMAC wrapper. The tag is only comparable against one produced by the same
    process with the same key — which is the point: it proves the value came
    from here and was not assembled by hand.
    """
    if len(key) != KEYED_HASH_KEY_SIZE:
        raise ValueError(f"keyed_hash requires a {KEYED_HASH_KEY_SIZE}-byte key")
    return _keyed_bytes(data, key, digest_size).hex()


# ---------------------------------------------------------------------------
# LRU caches
# ---------------------------------------------------------------------------


@lru_cache(maxsize=DEFAULT_CONFIG.CHUNK_CACHE_SIZE)
def _cached_chunk_hash(data: bytes) -> str:
    return _hash_hex(data, DEFAULT_CONFIG.CHUNK_DIGEST_SIZE)


# Above this, an entry would dominate the cache's whole byte budget on its own,
# and re-hashing amortizes better anyway — so large content is hashed directly.
_CONTENT_CACHE_BYPASS_SIZE = 65536  # Don't cache content hashes for files > 64KB


@lru_cache(maxsize=DEFAULT_CONFIG.CONTENT_CACHE_SIZE)
def _cached_content_hash_small(data: bytes) -> str:
    return _hash_hex(data, DEFAULT_CONFIG.CONTENT_DIGEST_SIZE)


def _cached_content_hash(data: bytes) -> str:
    """Skip LRU cache for large content to avoid memory bloat."""
    if len(data) > _CONTENT_CACHE_BYPASS_SIZE:
        return _hash_hex(data, DEFAULT_CONFIG.CONTENT_DIGEST_SIZE)
    return _cached_content_hash_small(data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hash_chunk(data: bytes) -> ChunkHash:
    """BLAKE3 (BLAKE2b fallback) with LRU caching; returns 64-char hex."""
    return _cached_chunk_hash(data)


def hash_content(content: str | bytes) -> ContentHash:
    data = content.encode() if isinstance(content, str) else content
    return _cached_content_hash(data)
