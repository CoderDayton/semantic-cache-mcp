from __future__ import annotations

import hashlib
import threading
from functools import lru_cache
from typing import Protocol


class _Hasher(Protocol):
    """Common interface for blake3 and blake2b hash objects."""

    def update(self, data: bytes, /) -> object: ...  # blake3 returns self, blake2b returns None
    def hexdigest(self) -> str: ...
    def digest(self) -> bytes: ...


# Optional: use blake3 if available (faster than BLAKE2b)
try:
    import blake3

    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Type aliases for clarity
ChunkHash = str  # Hex digest (64 chars for 32-byte BLAKE3)
BlockHash = str  # Intermediate level hash
ContentHash = str  # Full content hash
Fingerprint = bytes  # Binary hash for compact storage


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class HashConfig:
    # Hash function selection
    USE_BLAKE3: bool = HAS_BLAKE3
    USE_SHA256_FALLBACK: bool = True  # Fallback if BLAKE3 unavailable

    # Digest sizes (bytes)
    CHUNK_DIGEST_SIZE: int = 32  # 256 bits for chunks (collision-free for dedup)
    BLOCK_DIGEST_SIZE: int = 32  # 256 bits for intermediate blocks
    CONTENT_DIGEST_SIZE: int = 32  # 256 bits for full content

    # Caching configuration
    CHUNK_CACHE_SIZE: int = 16384  # LRU cache for chunks (16K entries ~ 2MB pointers)
    BLOCK_CACHE_SIZE: int = 4096  # LRU cache for blocks
    CONTENT_CACHE_SIZE: int = 2048  # LRU cache for full content hashes

    # Hierarchical hashing
    ENABLE_HIERARCHICAL: bool = True
    BLOCK_SIZE: int = 256 * 1024  # 256KB blocks for mid-level hashing
    STREAM_CHUNK_SIZE: int = 64 * 1024  # 64KB streaming buffer

    # Collision tracking
    TRACK_COLLISIONS: bool = True
    COLLISION_CACHE_SIZE: int = 1024


DEFAULT_CONFIG = HashConfig()


# ---------------------------------------------------------------------------
# Hash function selection
# ---------------------------------------------------------------------------


def _hash_bytes(data: bytes, digest_size: int = 32) -> bytes:
    if DEFAULT_CONFIG.USE_BLAKE3:
        try:
            return blake3.blake3(data).digest()
        except (ImportError, AttributeError, OSError):
            pass  # BLAKE3 not available, use fallback

    # Fallback
    return hashlib.blake2b(data, digest_size=digest_size).digest()


def _hash_hex(data: bytes, digest_size: int = 32) -> str:
    digest = _hash_bytes(data, digest_size)
    return digest.hex()


# ---------------------------------------------------------------------------
# LRU Caches for deduplication
# ---------------------------------------------------------------------------


@lru_cache(maxsize=DEFAULT_CONFIG.CHUNK_CACHE_SIZE)
def _cached_chunk_hash(data: bytes) -> str:
    return _hash_hex(data, DEFAULT_CONFIG.CHUNK_DIGEST_SIZE)


@lru_cache(maxsize=DEFAULT_CONFIG.BLOCK_CACHE_SIZE)
def _cached_block_hash(data: bytes) -> str:
    return _hash_hex(data, DEFAULT_CONFIG.BLOCK_DIGEST_SIZE)


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
# Collision detection
# ---------------------------------------------------------------------------


class CollisionTracker:
    def __init__(self, max_size: int = DEFAULT_CONFIG.COLLISION_CACHE_SIZE):
        self._hash_to_data: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._collision_count = 0

    def register(self, hash_hex: str, data: bytes) -> bool:
        """
        Register a hash → data mapping.

        Returns True if collision detected (different data, same hash).
        """
        with self._lock:
            if hash_hex in self._hash_to_data:
                if self._hash_to_data[hash_hex] != data:
                    self._collision_count += 1
                    return True  # Collision!
                return False  # Same data, same hash (expected)

            # Store if cache not full
            if len(self._hash_to_data) < self._max_size:
                self._hash_to_data[hash_hex] = data
            return False

    def get_collision_count(self) -> int:
        return self._collision_count

    def clear(self) -> None:
        with self._lock:
            self._hash_to_data.clear()
            self._collision_count = 0


# Global collision tracker
_collision_tracker = CollisionTracker()


# ---------------------------------------------------------------------------
# Core API: Chunk hashing
# ---------------------------------------------------------------------------


def hash_chunk(data: bytes) -> ChunkHash:
    """BLAKE3 (BLAKE2b fallback) with LRU caching; returns 64-char hex."""
    return _cached_chunk_hash(data)


def hash_chunk_binary(data: bytes) -> Fingerprint:
    """32-byte binary fingerprint — compact for dedup index storage."""
    return _hash_bytes(data, DEFAULT_CONFIG.CHUNK_DIGEST_SIZE)


def hash_chunk_with_collision_check(data: bytes) -> tuple[ChunkHash, bool]:
    """Return (hash_hex, is_collision)."""
    hash_hex = hash_chunk(data)
    is_collision = _collision_tracker.register(hash_hex, data)
    return hash_hex, is_collision


# ---------------------------------------------------------------------------
# Core API: Block/hierarchical hashing
# ---------------------------------------------------------------------------


def hash_block(data: bytes) -> BlockHash:
    """Intermediate-level hash for 2-level dedup: chunk → block → content."""
    return _cached_block_hash(data)


def hash_content(content: str | bytes) -> ContentHash:
    data = content.encode() if isinstance(content, str) else content
    return _cached_content_hash(data)


# ---------------------------------------------------------------------------
# Streaming/incremental API
# ---------------------------------------------------------------------------


class StreamingHasher:
    """Incremental hasher for large files — avoids loading entire file into memory."""

    __slots__ = ("_hasher",)

    _hasher: _Hasher

    def __init__(self, digest_size: int = DEFAULT_CONFIG.CHUNK_DIGEST_SIZE):
        if HAS_BLAKE3:
            try:
                self._hasher = blake3.blake3()
            except (ImportError, AttributeError, OSError):
                self._hasher = hashlib.blake2b(digest_size=digest_size)
        else:
            self._hasher = hashlib.blake2b(digest_size=digest_size)

    def update(self, data: bytes) -> None:
        self._hasher.update(data)

    def finalize(self) -> str:
        return self._hasher.hexdigest()

    def finalize_binary(self) -> bytes:
        return self._hasher.digest()


def hash_file_streaming(
    file_path: str, chunk_size: int = DEFAULT_CONFIG.STREAM_CHUNK_SIZE
) -> ContentHash:
    hasher = StreamingHasher(DEFAULT_CONFIG.CONTENT_DIGEST_SIZE)
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.finalize()


def hash_chunks_streaming(
    chunks_iter, combine: bool = True
) -> tuple[list[ChunkHash], ContentHash | None]:
    chunk_hashes = []
    content_hasher = StreamingHasher(DEFAULT_CONFIG.CONTENT_DIGEST_SIZE) if combine else None

    for chunk in chunks_iter:
        ch = hash_chunk(chunk)
        chunk_hashes.append(ch)

        if combine and content_hasher:
            # Hash over chunk hashes (not raw data) to avoid re-hashing large payloads
            content_hasher.update(ch.encode())

    content_hash = content_hasher.finalize() if content_hasher else None
    return chunk_hashes, content_hash


# ---------------------------------------------------------------------------
# Multi-level hierarchical hashing (for dedup)
# ---------------------------------------------------------------------------


class HierarchicalHasher:
    """3-level hash tree: chunks → blocks → content, for deduplication."""

    def __init__(self, block_size: int = DEFAULT_CONFIG.BLOCK_SIZE):
        self._block_size = block_size
        self._chunks: list[bytes] = []
        self._chunk_hashes: list[ChunkHash] = []
        self._blocks: list[BlockHash] = []

    def add_chunk(self, chunk: bytes) -> ChunkHash:
        ch = hash_chunk(chunk)
        self._chunk_hashes.append(ch)
        self._chunks.append(chunk)
        return ch

    def finalize_block(self) -> BlockHash:
        """Hash accumulated chunks into a block hash; resets accumulator."""
        if not self._chunk_hashes:
            return ""

        combined = b"".join(ch.encode() for ch in self._chunk_hashes)
        block_hash = hash_block(combined)
        self._blocks.append(block_hash)
        self._chunk_hashes.clear()
        self._chunks.clear()
        return block_hash

    def finalize_content(self) -> tuple[ContentHash, list[BlockHash], list[ChunkHash]]:
        """Return (content_hash, block_hashes, chunk_hashes).

        Save chunk hashes before finalize_block() clears them.
        """
        remaining_chunks = list(self._chunk_hashes)
        if self._chunk_hashes:
            self.finalize_block()

        if not self._blocks:
            return "", [], []

        combined_blocks = b"".join(b.encode() for b in self._blocks)
        content_hash = hash_content(combined_blocks)

        return content_hash, self._blocks.copy(), remaining_chunks


# ---------------------------------------------------------------------------
# Deduplication index utilities
# ---------------------------------------------------------------------------


class DeduplicateIndex:
    """Fingerprint → (chunk_id, size) index for deduplication lookups."""

    def __init__(self, capacity: int = 1_000_000):
        self._fingerprints: dict[bytes, tuple[int, int]] = {}  # fp → (chunk_id, size)
        self._capacity = capacity
        self._lock = threading.Lock()

    def add(self, chunk: bytes, chunk_id: int, size: int) -> bool:
        """Return True if added, False if duplicate or index full."""
        fp = hash_chunk_binary(chunk)

        with self._lock:
            if len(self._fingerprints) >= self._capacity:
                return False  # Index full

            if fp in self._fingerprints:
                return False  # Duplicate

            self._fingerprints[fp] = (chunk_id, size)
            return True  # Successfully added

    def lookup(self, chunk: bytes) -> tuple[int, int] | None:
        fp = hash_chunk_binary(chunk)
        with self._lock:
            return self._fingerprints.get(fp)

    def size(self) -> int:
        with self._lock:
            return len(self._fingerprints)

    def clear(self) -> None:
        with self._lock:
            self._fingerprints.clear()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def get_hash_stats() -> dict[str, bool | int]:
    return {
        "use_blake3": DEFAULT_CONFIG.USE_BLAKE3,
        "chunk_cache_size": DEFAULT_CONFIG.CHUNK_CACHE_SIZE,
        "block_cache_size": DEFAULT_CONFIG.BLOCK_CACHE_SIZE,
        "content_cache_size": DEFAULT_CONFIG.CONTENT_CACHE_SIZE,
        "collisions_detected": _collision_tracker.get_collision_count(),
    }


def reset_collision_tracker() -> None:
    _collision_tracker.clear()
