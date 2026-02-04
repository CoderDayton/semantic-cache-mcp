"""SemanticCache facade - orchestrates core algorithms and storage."""

from __future__ import annotations

import bisect
import logging
from pathlib import Path

import numpy as np

from .config import DB_PATH, MAX_CONTENT_SIZE
from .core import (
    count_tokens,
    diff_stats,
    generate_diff,
    get_optimal_chunker,
    summarize_semantic,
    truncate_semantic,
)
from .core.embeddings import embed
from .core.hashing import hash_content
from .storage import SQLiteStorage
from .types import CacheEntry, EditResult, EmbeddingVector, ReadResult, WriteResult

logger = logging.getLogger(__name__)


class SemanticCache:
    """High-level cache interface with semantic similarity support.

    This facade coordinates:
    - Storage backend (SQLite with content-addressable chunks)
    - Local embedding generation (FastEmbed)
    - Caching strategies (diff, truncate, semantic match)
    """

    __slots__ = ("_storage",)

    def __init__(self, db_path: Path = DB_PATH) -> None:
        """Initialize cache.

        Args:
            db_path: Path to SQLite database
        """
        self._storage = SQLiteStorage(db_path)

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    def get_embedding(self, text: str) -> EmbeddingVector | None:
        """Get embedding vector for text using local FastEmbed model.

        Args:
            text: Text to embed

        Returns:
            Embedding as array.array or None if unavailable
        """
        try:
            result = embed(text)
            if result:
                logger.debug(f"Embedding generated for {text[:50]}...")
            return result
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return None

    # -------------------------------------------------------------------------
    # Delegated operations
    # -------------------------------------------------------------------------

    def get(self, path: str) -> CacheEntry | None:
        """Get cached entry for path."""
        entry = self._storage.get(path)
        if entry:
            logger.debug(f"Cache hit: {path}")
        return entry

    def put(
        self,
        path: str,
        content: str,
        mtime: float,
        embedding: EmbeddingVector | None = None,
    ) -> None:
        """Store file in cache."""
        tokens = count_tokens(content)
        content_bytes = content.encode()

        # Use optimal chunker (SIMD if available, otherwise Gear hash)
        chunker = get_optimal_chunker(prefer_simd=True)
        chunks = sum(1 for _ in chunker(content_bytes))

        self._storage.put(path, content, mtime, embedding)
        logger.info(f"Cached file: {path} ({tokens} tokens, {chunks} chunks)")

    def get_content(self, entry: CacheEntry) -> str:
        """Get full content from cache entry."""
        return self._storage.get_content(entry)

    def record_access(self, path: str) -> None:
        """Record access for LRU-K tracking."""
        self._storage.record_access(path)

    def find_similar(
        self, embedding: EmbeddingVector, exclude_path: str | None = None
    ) -> str | None:
        """Find semantically similar cached file."""
        return self._storage.find_similar(embedding, exclude_path)

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        return self._storage.get_stats()

    def clear(self) -> int:
        """Clear all cache entries."""
        return self._storage.clear()


def smart_read(
    cache: SemanticCache,
    path: str,
    max_size: int = MAX_CONTENT_SIZE,
    diff_mode: bool = True,
    force_full: bool = False,
) -> ReadResult:
    """Read file with intelligent caching and optimization.

    Strategies (in order of token savings):
    1. File unchanged (mtime match) -> "// No changes" (99% reduction)
    2. File changed -> unified diff (80-95% reduction)
    3. Similar file in cache -> reference + diff (70-90% reduction)
    4. Large file -> smart truncation (50-80% reduction)
    5. New file -> full content with caching for future reads

    Args:
        cache: SemanticCache instance
        path: Path to file
        max_size: Maximum content size to return
        diff_mode: Enable diff-based responses
        force_full: Force full content even if cached

    Returns:
        ReadResult with content and metadata
    """
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Not a regular file: {path}")

    # Log symlink resolution for debugging
    original = Path(path).expanduser()
    if original.is_symlink():
        logger.debug(f"Following symlink: {path} -> {file_path}")

    # Check for binary file by reading first 8KB and looking for null bytes
    try:
        sample = file_path.read_bytes()[:8192]
        if b"\x00" in sample:
            raise ValueError(
                f"Binary file not supported: {path}. Semantic cache only handles text files."
            )
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with error replacement for files with mixed encoding
        content = file_path.read_text(encoding="utf-8", errors="replace")
        logger.warning(f"File {path} contains non-UTF-8 characters, using replacement")

    mtime = file_path.stat().st_mtime
    tokens_original = count_tokens(content)

    cached = cache.get(str(file_path))

    # Strategy 1 & 2: Cached file (unchanged or diff)
    if cached and diff_mode and not force_full:
        if cached.mtime >= mtime:
            # File unchanged
            cache.record_access(str(file_path))
            unchanged_msg = f"// File unchanged: {path} ({cached.tokens} tokens cached)"
            msg_tokens = count_tokens(unchanged_msg)

            if msg_tokens < tokens_original:
                return ReadResult(
                    content=unchanged_msg,
                    from_cache=True,
                    is_diff=False,
                    tokens_original=tokens_original,
                    tokens_returned=msg_tokens,
                    tokens_saved=tokens_original - msg_tokens,
                    truncated=False,
                    compression_ratio=len(unchanged_msg) / len(content) if content else 1.0,
                )

            # Small file - return full content
            cached_content = cache.get_content(cached)
            return ReadResult(
                content=cached_content,
                from_cache=True,
                is_diff=False,
                tokens_original=tokens_original,
                tokens_returned=tokens_original,
                tokens_saved=0,
                truncated=False,
                compression_ratio=1.0,
            )

        # File changed - generate diff with stats
        old_content = cache.get_content(cached)
        diff_content = generate_diff(old_content, content)
        stats = diff_stats(old_content, content)
        diff_tokens = count_tokens(diff_content)

        if diff_tokens < tokens_original * 0.6:
            stats_msg = (
                f"// Stats: +{stats['insertions']} -{stats['deletions']} "
                f"~{stats['modifications']} lines, "
                f"{stats['compression_ratio']:.1%} size\n"
            )
            result_content = f"// Diff for {path} (changed since cache):\n{stats_msg}{diff_content}"
            embedding = cache.get_embedding(content)
            cache.put(str(file_path), content, mtime, embedding)

            tokens_returned = count_tokens(result_content)
            return ReadResult(
                content=result_content,
                from_cache=True,
                is_diff=True,
                tokens_original=tokens_original,
                tokens_returned=tokens_returned,
                tokens_saved=tokens_original - tokens_returned,
                truncated=False,
                compression_ratio=len(result_content) / len(content),
            )

    # Strategy 3: Semantic similarity
    if not cached and diff_mode and not force_full:
        embedding = cache.get_embedding(content)
        if embedding:
            similar_path = cache.find_similar(embedding, str(file_path))
            if similar_path:
                similar_entry = cache.get(similar_path)
                if similar_entry:
                    similar_content = cache.get_content(similar_entry)
                    diff_content = generate_diff(similar_content, content)
                    stats = diff_stats(similar_content, content)
                    diff_tokens = count_tokens(diff_content)

                    if diff_tokens < tokens_original * 0.7:
                        stats_msg = (
                            f"// Stats: +{stats['insertions']} -{stats['deletions']} "
                            f"~{stats['modifications']} lines, "
                            f"{stats['compression_ratio']:.1%} size\n"
                        )
                        result_content = (
                            f"// Similar to cached: {similar_path}\n"
                            f"{stats_msg}"
                            f"// Diff from similar file:\n{diff_content}"
                        )
                        cache.put(str(file_path), content, mtime, embedding)

                        tokens_returned = count_tokens(result_content)
                        return ReadResult(
                            content=result_content,
                            from_cache=True,
                            is_diff=True,
                            tokens_original=tokens_original,
                            tokens_returned=tokens_returned,
                            tokens_saved=tokens_original - tokens_returned,
                            truncated=False,
                            compression_ratio=len(result_content) / len(content),
                            semantic_match=similar_path,
                        )

    # Strategy 4 & 5: Full read (with optional semantic summarization)
    truncated = False
    final_content = content

    if len(content) > max_size:
        # Use semantic summarization to preserve important content
        # Falls back to simple truncation for very small limits
        try:
            # Convert EmbeddingVector to NDArray for summarization
            def embed_fn(text: str):
                emb = cache.get_embedding(text)
                if emb is None:
                    return None
                # Convert array.array or list to numpy array
                return np.asarray(emb, dtype=np.float32)

            final_content = summarize_semantic(content, max_size, embed_fn=embed_fn)
            truncated = True
        except Exception as e:
            logger.warning(f"Semantic summarization failed: {e}, using fallback truncation")
            final_content = truncate_semantic(content, max_size)
            truncated = True

    embedding = cache.get_embedding(content)
    cache.put(str(file_path), content, mtime, embedding)

    tokens_returned = count_tokens(final_content)
    return ReadResult(
        content=final_content,
        from_cache=False,
        is_diff=False,
        tokens_original=tokens_original,
        tokens_returned=tokens_returned,
        tokens_saved=tokens_original - tokens_returned if truncated else 0,
        truncated=truncated,
        compression_ratio=len(final_content) / len(content) if content else 1.0,
    )


# Size limits for DoS protection
MAX_WRITE_SIZE = 10 * 1024 * 1024  # 10MB max content size for write
MAX_EDIT_SIZE = 10 * 1024 * 1024  # 10MB max file size for edit
MAX_MATCHES = 10000  # Max occurrences for replace_all


def _is_binary_content(data: bytes) -> bool:
    """Check if content is binary using multiple heuristics.

    Checks:
    1. Null bytes (most reliable indicator)
    2. Common binary file magic numbers
    3. High ratio of non-printable characters
    """
    if not data:
        return False

    sample = data[:8192]

    # 1. Null bytes - definitive binary indicator
    if b"\x00" in sample:
        return True

    # 2. Check for binary file magic numbers
    binary_signatures = (
        b"\x89PNG",  # PNG
        b"\xff\xd8\xff",  # JPEG
        b"GIF8",  # GIF
        b"PK\x03\x04",  # ZIP/DOCX/XLSX
        b"\x1f\x8b",  # GZIP
        b"BZ",  # BZIP2
        b"\x7fELF",  # ELF executable
        b"MZ",  # Windows executable
        b"%PDF",  # PDF (text header but binary content)
        b"\xd0\xcf\x11\xe0",  # MS Office OLE
    )
    for sig in binary_signatures:
        if sample.startswith(sig):
            return True

    # 3. High ratio of non-printable characters (>30% is suspicious)
    # Exclude common whitespace: tab(9), newline(10), carriage return(13)
    non_printable = sum(1 for b in sample if b < 32 and b not in (9, 10, 13))
    return len(sample) > 0 and non_printable / len(sample) > 0.3


def _find_match_line_numbers(content: str, search_string: str) -> list[int]:
    """Find line numbers where search_string appears.

    Returns 1-based line numbers for each occurrence.
    Uses binary search for O(M log N) complexity where M=matches, N=lines.
    """
    lines = content.splitlines(keepends=True)
    line_numbers: list[int] = []

    if not lines:
        return line_numbers

    # Build cumulative character positions for binary search
    char_pos = 0
    line_starts: list[int] = []
    for line in lines:
        line_starts.append(char_pos)
        char_pos += len(line)

    # Find all occurrences with O(log N) line lookup
    start = 0
    while True:
        idx = content.find(search_string, start)
        if idx == -1:
            break

        # Binary search for line number - O(log N) per match
        # bisect_right returns insertion point; -1 gives us the line containing idx
        line_num = bisect.bisect_right(line_starts, idx)
        line_numbers.append(line_num)  # Already 1-based due to bisect_right behavior
        start = idx + 1

    return line_numbers


def smart_write(
    cache: SemanticCache,
    path: str,
    content: str,
    create_parents: bool = True,
    dry_run: bool = False,
) -> WriteResult:
    """Write file with cache integration.

    Benefits over built-in Write:
    - Returns diff instead of echoing content (token savings)
    - Updates cache for future reads
    - Tracks operation metadata

    Args:
        cache: SemanticCache instance
        path: Absolute path to file
        content: Content to write
        create_parents: Create parent directories if missing
        dry_run: Preview changes without writing

    Returns:
        WriteResult with diff and metadata

    Raises:
        FileNotFoundError: Parent directory doesn't exist and create_parents=False
        PermissionError: Insufficient permissions
        ValueError: Path is not a file, binary content detected, or content too large
    """
    # Validate content size early (fail fast)
    if len(content) > MAX_WRITE_SIZE:
        raise ValueError(
            f"Content too large: {len(content):,} bytes exceeds {MAX_WRITE_SIZE:,} byte limit"
        )

    file_path = Path(path).expanduser().resolve()

    # Log symlink resolution
    original = Path(path).expanduser()
    if original.exists() and original.is_symlink():
        logger.debug(f"Following symlink: {path} -> {file_path}")

    # Validate path is not a directory
    if file_path.exists() and not file_path.is_file():
        raise ValueError(f"Not a regular file: {path}")

    # Check parent directory
    parent = file_path.parent
    if not parent.exists():
        if create_parents:
            if not dry_run:
                parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created parent directories: {parent}")
        else:
            raise FileNotFoundError(f"Parent directory does not exist: {parent}")

    # Calculate content metadata
    content_bytes = content.encode("utf-8")
    bytes_written = len(content_bytes)
    tokens_written = count_tokens(content)
    content_hash = hash_content(content_bytes)

    # Check if file exists and get old content
    old_content: str | None = None
    from_cache = False
    created = not file_path.exists()

    if not created:
        # Check for binary file
        try:
            sample = file_path.read_bytes()[:8192]
            if _is_binary_content(sample):
                raise ValueError(
                    f"Binary file not supported: {path}. Cannot overwrite binary with text."
                )
        except OSError as e:
            raise PermissionError(f"Cannot read existing file: {e}") from e

        # Try to get content from cache first (saves tokens!)
        cached = cache.get(str(file_path))
        if cached:
            mtime = file_path.stat().st_mtime
            if cached.mtime >= mtime:
                old_content = cache.get_content(cached)
                from_cache = True
                logger.debug(f"Using cached content for diff: {path}")

        # Fall back to disk read
        if old_content is None:
            try:
                old_content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                old_content = file_path.read_text(encoding="utf-8", errors="replace")
                logger.warning(f"File {path} contains non-UTF-8 characters")

    # Generate diff if overwriting
    diff_content: str | None = None
    diff_stats_result: dict[str, int] | None = None
    tokens_saved = 0

    if old_content is not None and old_content != content:
        diff_content = generate_diff(old_content, content)
        diff_stats_result = diff_stats(old_content, content)
        # Token savings: diff vs full content in response
        diff_tokens = count_tokens(diff_content) if diff_content else 0
        tokens_saved = max(0, tokens_written - diff_tokens)

    # Write file (unless dry_run)
    if not dry_run:
        try:
            file_path.write_text(content, encoding="utf-8")
        except OSError as e:
            raise PermissionError(f"Cannot write file: {e}") from e

        # Update cache with new content
        mtime = file_path.stat().st_mtime
        embedding = cache.get_embedding(content)
        cache.put(str(file_path), content, mtime, embedding)
        logger.info(f"{'Created' if created else 'Updated'} and cached: {path}")

    return WriteResult(
        path=str(file_path),
        bytes_written=bytes_written,
        tokens_written=tokens_written,
        created=created,
        diff_content=diff_content,
        diff_stats=diff_stats_result,
        tokens_saved=tokens_saved,
        content_hash=content_hash,
        from_cache=from_cache,
    )


def smart_edit(
    cache: SemanticCache,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    dry_run: bool = False,
) -> EditResult:
    """Edit file using find/replace with cached read.

    Benefits over built-in Edit:
    - Uses cache for reading (no token cost!)
    - Returns diff instead of confirmation
    - Tracks line numbers of changes

    Args:
        cache: SemanticCache instance
        path: Absolute path to file
        old_string: Exact string to find
        new_string: Replacement string
        replace_all: Replace all occurrences
        dry_run: Preview changes without writing

    Returns:
        EditResult with diff and match locations

    Raises:
        FileNotFoundError: File doesn't exist
        ValueError: old_string not found, multiple matches without replace_all,
                   or old_string equals new_string
        PermissionError: Insufficient permissions
    """
    # Validate inputs FIRST (fail fast before any I/O)
    if not old_string:
        raise ValueError("old_string cannot be empty")

    if old_string == new_string:
        raise ValueError("old_string and new_string are identical")

    file_path = Path(path).expanduser().resolve()

    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Not a regular file: {path}")

    # Log symlink resolution
    original = Path(path).expanduser()
    if original.is_symlink():
        logger.debug(f"Following symlink: {path} -> {file_path}")

    # Check for binary file
    try:
        sample = file_path.read_bytes()[:8192]
        if _is_binary_content(sample):
            raise ValueError(f"Binary file not supported: {path}. Edit only works with text files.")
    except OSError as e:
        raise PermissionError(f"Cannot read file: {e}") from e

    # Try to get content from cache first (huge token savings!)
    content: str
    from_cache = False
    cached = cache.get(str(file_path))

    if cached:
        mtime = file_path.stat().st_mtime
        if cached.mtime >= mtime:
            content = cache.get_content(cached)
            from_cache = True
            logger.debug(f"Using cached content for edit: {path}")
        else:
            # Cache stale, read from disk
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = file_path.read_text(encoding="utf-8", errors="replace")
    else:
        # No cache entry, read from disk
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="utf-8", errors="replace")

    # Validate content size
    if len(content) > MAX_EDIT_SIZE:
        raise ValueError(
            f"File too large for edit: {len(content):,} bytes exceeds {MAX_EDIT_SIZE:,} byte limit"
        )

    # Quick match count check before expensive operations
    quick_count = content.count(old_string)
    if quick_count > MAX_MATCHES:
        raise ValueError(
            f"Too many matches ({quick_count:,}). "
            f"Maximum {MAX_MATCHES:,} occurrences allowed for edit operations."
        )

    # Find all occurrences with line numbers
    line_numbers = _find_match_line_numbers(content, old_string)
    matches_found = len(line_numbers)

    # Validate matches
    if matches_found == 0:
        raise ValueError(
            f"old_string not found in {path}. Hint: Ensure exact whitespace and indentation match."
        )

    if matches_found > 1 and not replace_all:
        raise ValueError(
            f"old_string found {matches_found} times at lines {line_numbers} in {path}. "
            "Hint: Provide more context to make the match unique, or use replace_all=True"
        )

    # Perform replacement
    if replace_all:
        new_content = content.replace(old_string, new_string)
        replacements_made = matches_found
    else:
        new_content = content.replace(old_string, new_string, 1)
        replacements_made = 1
        line_numbers = line_numbers[:1]  # Only first match

    # Generate diff
    diff_content = generate_diff(content, new_content)
    diff_stats_result = diff_stats(content, new_content)

    # Calculate token savings from cached read
    # Only count as saved if content actually came from cache
    content_tokens = count_tokens(content)
    tokens_saved = content_tokens if from_cache else 0

    # Calculate new content hash
    content_hash = hash_content(new_content.encode("utf-8"))

    # Write file (unless dry_run)
    if not dry_run:
        try:
            file_path.write_text(new_content, encoding="utf-8")
        except OSError as e:
            raise PermissionError(f"Cannot write file: {e}") from e

        # Update cache with new content
        mtime = file_path.stat().st_mtime
        embedding = cache.get_embedding(new_content)
        cache.put(str(file_path), new_content, mtime, embedding)
        logger.info(f"Edited and cached: {path} ({replacements_made} replacement(s))")

    return EditResult(
        path=str(file_path),
        matches_found=matches_found,
        replacements_made=replacements_made,
        line_numbers=line_numbers,
        diff_content=diff_content,
        diff_stats=diff_stats_result,
        tokens_saved=tokens_saved,
        content_hash=content_hash,
        from_cache=from_cache,
    )
