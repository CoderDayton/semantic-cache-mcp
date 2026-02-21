"""
High-performance content hashing with BLAKE3, incremental streaming, and hierarchical dedup.

Features:
- BLAKE3 hash function (faster than BLAKE2b, parallelizable)
- LRU caching for chunk deduplication
- Incremental/streaming API for large files
- Multi-level hashing (chunk, block, content levels)
- Collision detection and handling
- Hardware-accelerated fallback paths
- Memory-efficient fingerprinting for dedup indices
"""

from __future__ import annotations

import hashlib
import threading
from functools import lru_cache
from typing import Any, Protocol


class _Hasher(Protocol):
    """Protocol for hash objects (blake3 or blake2b)."""

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
    """Hash function configuration and parameters."""

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
    """Hash raw bytes and return binary digest."""
    if DEFAULT_CONFIG.USE_BLAKE3:
        try:
            return blake3.blake3(data).digest()
        except (ImportError, AttributeError, OSError):
            pass  # BLAKE3 not available, use fallback

    # Fallback
    return hashlib.blake2b(data, digest_size=digest_size).digest()


def _hash_hex(data: bytes, digest_size: int = 32) -> str:
    """Hash raw bytes and return hex string."""
    digest = _hash_bytes(data, digest_size)
    return digest.hex()


# ---------------------------------------------------------------------------
# LRU Caches for deduplication
# ---------------------------------------------------------------------------


@lru_cache(maxsize=DEFAULT_CONFIG.CHUNK_CACHE_SIZE)
def _cached_chunk_hash(data: bytes) -> str:
    """LRU-cached chunk hashing."""
    return _hash_hex(data, DEFAULT_CONFIG.CHUNK_DIGEST_SIZE)


@lru_cache(maxsize=DEFAULT_CONFIG.BLOCK_CACHE_SIZE)
def _cached_block_hash(data: bytes) -> str:
    """LRU-cached block hashing for intermediate levels."""
    return _hash_hex(data, DEFAULT_CONFIG.BLOCK_DIGEST_SIZE)


@lru_cache(maxsize=DEFAULT_CONFIG.CONTENT_CACHE_SIZE)
def _cached_content_hash(data: bytes) -> str:
    """LRU-cached full-content hashing."""
    return _hash_hex(data, DEFAULT_CONFIG.CONTENT_DIGEST_SIZE)


# ---------------------------------------------------------------------------
# Collision detection
# ---------------------------------------------------------------------------


class CollisionTracker:
    """Thread-safe collision detection for hash deduplication."""

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
        """Get total collisions detected."""
        return self._collision_count

    def clear(self) -> None:
        """Clear collision cache."""
        with self._lock:
            self._hash_to_data.clear()
            self._collision_count = 0


# Global collision tracker
_collision_tracker = CollisionTracker()


# ---------------------------------------------------------------------------
# Core API: Chunk hashing
# ---------------------------------------------------------------------------


def hash_chunk(data: bytes) -> ChunkHash:
    """
    Hash a chunk using BLAKE3 (or BLAKE2b fallback) with LRU caching.

    Args:
        data: Raw chunk bytes

    Returns:
        64-character hex digest
    """
    return _cached_chunk_hash(data)


def hash_chunk_binary(data: bytes) -> Fingerprint:
    """
    Hash a chunk and return binary fingerprint (compact for dedup indices).

    Args:
        data: Raw chunk bytes

    Returns:
        32-byte binary hash
    """
    return _hash_bytes(data, DEFAULT_CONFIG.CHUNK_DIGEST_SIZE)


def hash_chunk_with_collision_check(data: bytes) -> tuple[ChunkHash, bool]:
    """
    Hash chunk and check for collisions.

    Args:
        data: Raw chunk bytes

    Returns:
        (hash_hex, is_collision)
    """
    hash_hex = hash_chunk(data)
    is_collision = _collision_tracker.register(hash_hex, data)
    return hash_hex, is_collision


# ---------------------------------------------------------------------------
# Core API: Block/hierarchical hashing
# ---------------------------------------------------------------------------


def hash_block(data: bytes) -> BlockHash:
    """
    Hash a larger block (intermediate level for hierarchical dedup).

    Used in 2-level dedup schemes: chunk → block → content.

    Args:
        data: Block bytes (typically 256KB)

    Returns:
        64-character hex digest
    """
    return _cached_block_hash(data)


def hash_content(content: str | bytes) -> ContentHash:
    """
    Hash full content for change detection.

    Args:
        content: Text content (str) or raw bytes

    Returns:
        64-character hex digest
    """
    data = content.encode() if isinstance(content, str) else content
    return _cached_content_hash(data)


# ---------------------------------------------------------------------------
# Streaming/incremental API
# ---------------------------------------------------------------------------


class StreamingHasher:
    """
    Incremental hasher for large files (streaming mode).

    Allows hashing data in chunks without loading entire file into memory.
    """

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
        """Add data to the hash."""
        self._hasher.update(data)

    def finalize(self) -> str:
        """Get final hex digest."""
        return self._hasher.hexdigest()

    def finalize_binary(self) -> bytes:
        """Get final binary digest."""
        return self._hasher.digest()


def hash_file_streaming(
    file_path: str, chunk_size: int = DEFAULT_CONFIG.STREAM_CHUNK_SIZE
) -> ContentHash:
    """
    Hash large file in streaming mode (memory-efficient).

    Args:
        file_path: Path to file to hash
        chunk_size: Read buffer size

    Returns:
        Hex digest of full file
    """
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
    """
    Hash a stream of chunks and optionally combine into content hash.

    Args:
        chunks_iter: Iterator yielding chunk bytes
        combine: Whether to compute aggregate content hash

    Returns:
        (list of chunk hashes, combined content hash or None)
    """
    chunk_hashes = []
    content_hasher = StreamingHasher(DEFAULT_CONFIG.CONTENT_DIGEST_SIZE) if combine else None

    for chunk in chunks_iter:
        ch = hash_chunk(chunk)
        chunk_hashes.append(ch)

        if combine and content_hasher:
            # Add chunk hash (not raw data) to content hash for efficiency
            content_hasher.update(ch.encode())

    content_hash = content_hasher.finalize() if content_hasher else None
    return chunk_hashes, content_hash


# ---------------------------------------------------------------------------
# Multi-level hierarchical hashing (for dedup)
# ---------------------------------------------------------------------------


class HierarchicalHasher:
    """
    Multi-level hashing for deduplication.

    Structure:
      - Level 0: Chunks (via hash_chunk)
      - Level 1: Blocks (aggregated chunk hashes)
      - Level 2: Content (aggregated block hashes)
    """

    def __init__(self, block_size: int = DEFAULT_CONFIG.BLOCK_SIZE):
        self._block_size = block_size
        self._chunks: list[bytes] = []
        self._chunk_hashes: list[ChunkHash] = []
        self._blocks: list[BlockHash] = []

    def add_chunk(self, chunk: bytes) -> ChunkHash:
        """Add chunk and return its hash."""
        ch = hash_chunk(chunk)
        self._chunk_hashes.append(ch)
        self._chunks.append(chunk)
        return ch

    def finalize_block(self) -> BlockHash:
        """
        Finalize current block from accumulated chunks.

        Returns block hash and resets accumulator.
        """
        if not self._chunk_hashes:
            return ""

        # Combine chunk hashes into block hash
        combined = b"".join(ch.encode() for ch in self._chunk_hashes)
        block_hash = hash_block(combined)
        self._blocks.append(block_hash)
        self._chunk_hashes.clear()
        self._chunks.clear()
        return block_hash

    def finalize_content(self) -> tuple[ContentHash, list[BlockHash], list[ChunkHash]]:
        """
        Finalize content from all blocks.

        Returns (content_hash, all_block_hashes, all_chunk_hashes).
        """
        # Finalize pending block
        if self._chunk_hashes:
            self.finalize_block()

        # Combine block hashes into content hash
        if not self._blocks:
            return "", [], []

        combined_blocks = b"".join(b.encode() for b in self._blocks)
        content_hash = hash_content(combined_blocks)

        return content_hash, self._blocks.copy(), self._chunk_hashes.copy()


# ---------------------------------------------------------------------------
# Deduplication index utilities
# ---------------------------------------------------------------------------


class DeduplicateIndex:
    """
    Fast deduplication index using binary fingerprints.

    Maps chunk fingerprints → metadata for quick lookups.
    """

    def __init__(self, capacity: int = 1_000_000):
        self._fingerprints: dict[bytes, tuple[int, int]] = {}  # fp → (chunk_id, size)
        self._capacity = capacity
        self._lock = threading.Lock()

    def add(self, chunk: bytes, chunk_id: int, size: int) -> bool:
        """
        Add chunk to index.

        Returns True if successfully added (not a duplicate).
        Returns False if duplicate or index full.

        Usage:
            is_dup = not index.add(chunk, chunk_id, size)
        """
        fp = hash_chunk_binary(chunk)

        with self._lock:
            if len(self._fingerprints) >= self._capacity:
                return False  # Index full

            if fp in self._fingerprints:
                return False  # Duplicate

            self._fingerprints[fp] = (chunk_id, size)
            return True  # Successfully added

    def lookup(self, chunk: bytes) -> tuple[int, int] | None:
        """
        Look up chunk fingerprint in index.

        Returns (chunk_id, size) or None if not found.
        """
        fp = hash_chunk_binary(chunk)
        with self._lock:
            return self._fingerprints.get(fp)

    def size(self) -> int:
        """Get number of fingerprints in index."""
        with self._lock:
            return len(self._fingerprints)

    def clear(self) -> None:
        """Clear index."""
        with self._lock:
            self._fingerprints.clear()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def get_hash_stats() -> dict[str, Any]:
    """Get hash function statistics."""
    return {
        "use_blake3": DEFAULT_CONFIG.USE_BLAKE3,
        "chunk_cache_size": DEFAULT_CONFIG.CHUNK_CACHE_SIZE,
        "block_cache_size": DEFAULT_CONFIG.BLOCK_CACHE_SIZE,
        "content_cache_size": DEFAULT_CONFIG.CONTENT_CACHE_SIZE,
        "collisions_detected": _collision_tracker.get_collision_count(),
    }


def reset_collision_tracker() -> None:
    """Reset collision statistics."""
    _collision_tracker.clear()
