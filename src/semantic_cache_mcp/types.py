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


@dataclass(slots=True)
class WriteResult:
    """Result of a write operation.

    Attributes:
        path: Resolved absolute path to the written file.
        bytes_written: Number of bytes written to disk.
        tokens_written: Estimated token count of written content.
        created: True if new file, False if overwrite.
        diff_content: Unified diff from old content (if existed).
        diff_stats: Dictionary with insertions, deletions, modifications.
        tokens_saved: Tokens saved by returning diff instead of full content.
        content_hash: BLAKE3 hash of new content for verification.
        from_cache: Whether old content came from cache (vs disk read).
    """

    path: str
    bytes_written: int
    tokens_written: int
    created: bool
    diff_content: str | None
    diff_stats: dict[str, int] | None
    tokens_saved: int
    content_hash: ContentHash
    from_cache: bool


@dataclass(slots=True)
class EditResult:
    """Result of an edit (find/replace) operation.

    Attributes:
        path: Resolved absolute path to the edited file.
        matches_found: Total occurrences of old_string found.
        replacements_made: Number of replacements performed.
        line_numbers: Lines where replacements occurred.
        diff_content: Unified diff of changes.
        diff_stats: Dictionary with insertions, deletions, modifications.
        tokens_saved: Tokens saved by cached read + diff response.
        content_hash: BLAKE3 hash of new content for verification.
        from_cache: Whether content came from cache (vs disk read).
    """

    path: str
    matches_found: int
    replacements_made: int
    line_numbers: list[int]
    diff_content: str
    diff_stats: dict[str, int]
    tokens_saved: int
    content_hash: ContentHash
    from_cache: bool
