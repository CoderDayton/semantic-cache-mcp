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
class AppendResult:
    """Result of an append operation.

    Attributes:
        path: Resolved absolute path to the file.
        bytes_appended: Number of bytes appended.
        total_bytes: Total file size after append.
        tokens_appended: Estimated token count of appended content.
        content_hash: BLAKE3 hash of appended content chunk.
        created: True if file was created (didn't exist), False if appended.
        cache_invalidated: True if a stale cache entry was removed.
    """

    path: str
    bytes_appended: int
    total_bytes: int
    tokens_appended: int
    content_hash: ContentHash
    created: bool
    cache_invalidated: bool


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


# -----------------------------------------------------------------------------
# Search tool types
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class SearchMatch:
    """A single search result with similarity score."""

    path: str
    similarity: float  # 0.0-1.0
    tokens: int
    preview: str  # First 200 chars


@dataclass(slots=True)
class SearchResult:
    """Result from semantic_search operation."""

    query: str
    matches: list[SearchMatch]
    files_searched: int
    cached_files: int


# -----------------------------------------------------------------------------
# Diff tool types
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class DiffResult:
    """Result from compare_files operation."""

    path1: str
    path2: str
    diff_content: str
    diff_stats: dict[str, int]  # insertions, deletions, modifications
    tokens_saved: int
    similarity: float  # Semantic similarity between files
    from_cache: tuple[bool, bool]  # Which files came from cache


# -----------------------------------------------------------------------------
# Batch read tool types
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class FileReadSummary:
    """Summary of a single file read in batch."""

    path: str
    tokens: int
    status: str  # "full", "diff", "truncated", "skipped", "unchanged"
    from_cache: bool
    est_tokens: int | None = None


@dataclass(slots=True)
class BatchReadResult:
    """Result from batch_smart_read operation."""

    files: list[FileReadSummary]
    contents: dict[str, str]  # path -> content
    total_tokens: int
    tokens_saved: int
    files_read: int
    files_skipped: int
    unchanged_paths: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Similar files tool types
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class SimilarFile:
    """A file similar to the source."""

    path: str
    similarity: float
    tokens: int


@dataclass(slots=True)
class SimilarFilesResult:
    """Result from find_similar_files operation."""

    source_path: str
    source_tokens: int
    similar_files: list[SimilarFile]
    files_searched: int


# -----------------------------------------------------------------------------
# Glob tool types
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class GlobMatch:
    """A file matching glob pattern with cache status."""

    path: str
    cached: bool
    tokens: int | None  # None if not cached
    mtime: float


@dataclass(slots=True)
class GlobResult:
    """Result from glob_with_cache_status operation."""

    pattern: str
    directory: str
    matches: list[GlobMatch]
    total_matches: int
    cached_count: int
    total_cached_tokens: int


# -----------------------------------------------------------------------------
# Multi-edit tool types
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class SingleEditOutcome:
    """Outcome of a single edit within a multi-edit batch."""

    old_string: str
    new_string: str
    success: bool
    line_number: int | None  # None if not found
    error: str | None


@dataclass(slots=True)
class MultiEditResult:
    """Result from smart_multi_edit operation."""

    path: str
    outcomes: list[SingleEditOutcome]
    succeeded: int
    failed: int
    diff_content: str  # Combined diff of successful edits
    diff_stats: dict[str, int]
    tokens_saved: int
    content_hash: ContentHash
    from_cache: bool
