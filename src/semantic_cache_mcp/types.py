"""Data models for semantic-cache-mcp."""

from __future__ import annotations

import array
from dataclasses import dataclass, field

# Type aliases (Python 3.12+)
type EmbeddingVector = array.array[float] | list[float]
type ChunkHash = str
type ContentHash = str


@dataclass(slots=True)
class CacheEntry:
    """Cached file entry with metadata."""

    path: str
    content_hash: ContentHash
    chunks: list[ChunkHash]
    mtime: float
    tokens: int
    embedding: EmbeddingVector | None
    created_at: float
    access_history: list[float] = field(default_factory=list)


@dataclass(slots=True)
class ReadResult:
    """Result from smart_read operation."""

    content: str
    from_cache: bool
    is_diff: bool
    tokens_original: int
    tokens_returned: int
    tokens_saved: int
    truncated: bool
    compression_ratio: float
    semantic_match: str | None = None


@dataclass(slots=True, frozen=True)
class ChunkData:
    """Immutable chunk data for storage."""

    hash: ChunkHash
    data: bytes
    size: int
