"""SemanticCache facade - orchestrates core algorithms and storage."""

from __future__ import annotations

import bisect
import logging
import shutil
import subprocess  # nosec B404 - used for formatter execution with hardcoded commands
import time
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
from .core.embeddings import embed, embed_query
from .core.hashing import hash_content
from .core.similarity import cosine_similarity, top_k_from_quantized
from .storage import SQLiteStorage
from .types import (
    BatchReadResult,
    CacheEntry,
    DiffResult,
    EditResult,
    EmbeddingVector,
    FileReadSummary,
    GlobMatch,
    GlobResult,
    MultiEditResult,
    ReadResult,
    SearchMatch,
    SearchResult,
    SimilarFile,
    SimilarFilesResult,
    SingleEditOutcome,
    WriteResult,
)

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

    def get_stats(self) -> dict[str, int | float | str | bool]:
        """Get cache statistics including memory usage."""
        stats: dict[str, int | float | str | bool] = {**self._storage.get_stats()}

        # Add process memory stats
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        stats["process_rss_mb"] = round(int(line.split()[1]) / 1024, 1)
                        break
        except OSError:
            pass

        # Add merge cache stats
        from .core.tokenizer import _tokenizer
        if _tokenizer is not None:
            stats["merge_cache_entries"] = len(_tokenizer._merge_cache)
            stats["merge_cache_maxsize"] = _tokenizer._merge_cache_maxsize

        # Add embedding model readiness
        from .core.embeddings import _execution_provider, _model_ready

        stats["embedding_ready"] = _model_ready
        stats["embedding_provider"] = _execution_provider

        return stats

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
MAX_RETURN_DIFF_TOKENS = 8000  # Hard cap for emitted diff payloads
MAX_DIFF_TO_FULL_RATIO = 0.9  # Suppress diff payloads near full-content size

# Auto-format configuration: extension -> (command, args...)
# Command must accept file path as final argument
FORMATTERS: dict[str, tuple[str, ...]] = {
    ".py": ("ruff", "format"),
    ".pyi": ("ruff", "format"),
    ".js": ("prettier", "--write"),
    ".ts": ("prettier", "--write"),
    ".tsx": ("prettier", "--write"),
    ".jsx": ("prettier", "--write"),
    ".json": ("prettier", "--write"),
    ".css": ("prettier", "--write"),
    ".scss": ("prettier", "--write"),
    ".md": ("prettier", "--write"),
    ".yaml": ("prettier", "--write"),
    ".yml": ("prettier", "--write"),
    ".go": ("gofmt", "-w"),
    ".rs": ("rustfmt",),
}


def _format_file(path: Path) -> bool:
    """Format file in-place using appropriate formatter.

    Args:
        path: Path to file to format

    Returns:
        True if formatted successfully, False if formatter not found or failed
    """
    formatter = FORMATTERS.get(path.suffix.lower())
    if not formatter:
        return False

    cmd_name = formatter[0]
    # Check if formatter is installed
    if not shutil.which(cmd_name):
        logger.debug(f"Formatter not found: {cmd_name}")
        return False

    try:
        cmd = [*formatter, str(path)]
        result = subprocess.run(  # nosec B603 - commands from hardcoded FORMATTERS dict
            cmd,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            logger.debug(f"Formatted {path} with {cmd_name}")
            return True
        else:
            logger.warning(f"Formatter {cmd_name} failed: {result.stderr.decode()[:200]}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Formatter {cmd_name} timed out on {path}")
        return False
    except OSError as e:
        logger.warning(f"Failed to run {cmd_name}: {e}")
        return False


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


def _choose_min_token_content(options: dict[str, str]) -> tuple[str, str, int]:
    """Pick the response payload with the smallest token count."""
    best_kind = ""
    best_content = ""
    best_tokens: int | None = None

    for kind, content in options.items():
        tokens = count_tokens(content)
        if best_tokens is None or tokens < best_tokens:
            best_kind = kind
            best_content = content
            best_tokens = tokens

    return best_kind, best_content, best_tokens or 0


def _suppress_large_diff(diff_content: str | None, full_tokens: int) -> str | None:
    """Suppress large diff payloads to optimize returned token count.

    Returns the diff unchanged for small files, a summary string for large
    diffs that exceed the token cap, or None for empty input.
    """
    if not diff_content:
        return None

    diff_tokens = count_tokens(diff_content)
    full_tokens = max(full_tokens, 1)

    # Preserve diffs for small files where readability is more useful than suppression.
    if full_tokens <= 200:
        return diff_content

    should_suppress = (
        diff_tokens > MAX_RETURN_DIFF_TOKENS
        or diff_tokens >= int(full_tokens * MAX_DIFF_TO_FULL_RATIO)
    )
    if should_suppress:
        # Count added/removed lines and hunks from unified diff
        added = 0
        removed = 0
        hunks = 0
        for line in diff_content.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
            elif line.startswith("@@"):
                hunks += 1
        return (
            f"[diff suppressed: {diff_tokens} tokens > cap] "
            f"+{added} -{removed} lines across {hunks} hunks"
        )

    return diff_content


def _fit_content_to_max_size(
    content: str, max_size: int, cache: SemanticCache
) -> tuple[str, bool]:
    """Bound returned content to max_size using semantic summarization when needed."""
    if len(content) <= max_size:
        return content, False

    try:
        # Keep summarization embed_fn local to avoid extra allocations when unneeded.
        def embed_fn(text: str):
            emb = cache.get_embedding(text)
            if emb is None:
                return None
            return np.asarray(emb, dtype=np.float32)

        return summarize_semantic(content, max_size, embed_fn=embed_fn), True
    except Exception as e:
        logger.warning(f"Semantic summarization failed: {e}, using fallback truncation")
        return truncate_semantic(content, max_size), True


def smart_write(
    cache: SemanticCache,
    path: str,
    content: str,
    create_parents: bool = True,
    dry_run: bool = False,
    auto_format: bool = False,
) -> WriteResult:
    """Write file with cache integration.

    Benefits over built-in Write:
    - Returns diff instead of echoing content (token savings)
    - Updates cache for future reads
    - Tracks operation metadata
    - Optional auto-formatting after write

    Args:
        cache: SemanticCache instance
        path: Absolute path to file
        content: Content to write
        create_parents: Create parent directories if missing
        dry_run: Preview changes without writing
        auto_format: Run formatter after write (default: false)

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
        diff_content = _suppress_large_diff(diff_content, tokens_written)
        # Token savings: diff vs full content in response
        diff_tokens = count_tokens(diff_content) if diff_content else 0
        tokens_saved = max(0, tokens_written - diff_tokens)

    # Write file (unless dry_run)
    if not dry_run:
        try:
            file_path.write_text(content, encoding="utf-8")
        except OSError as e:
            raise PermissionError(f"Cannot write file: {e}") from e

        # Auto-format if requested
        formatted = False
        if auto_format:
            formatted = _format_file(file_path)
            if formatted:
                # Re-read formatted content
                content = file_path.read_text(encoding="utf-8")
                content_bytes = content.encode("utf-8")
                bytes_written = len(content_bytes)
                tokens_written = count_tokens(content)
                content_hash = hash_content(content_bytes)

                # Re-compute diff against original (before format)
                if old_content is not None and old_content != content:
                    diff_content = generate_diff(old_content, content)
                    diff_stats_result = diff_stats(old_content, content)
                    diff_content = _suppress_large_diff(diff_content, tokens_written)
                    diff_tokens = count_tokens(diff_content) if diff_content else 0
                    tokens_saved = max(0, tokens_written - diff_tokens)

        # Update cache with final content
        mtime = file_path.stat().st_mtime
        embedding = cache.get_embedding(content)
        cache.put(str(file_path), content, mtime, embedding)
        action = "Created" if created else "Updated"
        if formatted:
            action += " and formatted"
        logger.info(f"{action} and cached: {path}")

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
    auto_format: bool = False,
) -> EditResult:
    """Edit file using find/replace with cached read.

    Benefits over built-in Edit:
    - Uses cache for reading (no token cost!)
    - Returns diff instead of confirmation
    - Tracks line numbers of changes
    - Optional auto-formatting after edit

    Args:
        cache: SemanticCache instance
        path: Absolute path to file
        old_string: Exact string to find
        new_string: Replacement string
        replace_all: Replace all occurrences
        dry_run: Preview changes without writing
        auto_format: Run formatter after edit (default: false)

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
    content_tokens = count_tokens(content)
    diff_content = _suppress_large_diff(diff_content, content_tokens) or ""

    # Calculate token savings from cached read
    # Only count as saved if content actually came from cache
    tokens_saved = content_tokens if from_cache else 0

    # Calculate new content hash
    content_hash = hash_content(new_content.encode("utf-8"))

    # Write file (unless dry_run)
    if not dry_run:
        try:
            file_path.write_text(new_content, encoding="utf-8")
        except OSError as e:
            raise PermissionError(f"Cannot write file: {e}") from e

        # Auto-format if requested
        formatted = False
        if auto_format:
            formatted = _format_file(file_path)
            if formatted:
                # Re-read formatted content
                new_content = file_path.read_text(encoding="utf-8")
                content_hash = hash_content(new_content.encode("utf-8"))

                # Re-compute diff against original (before format)
                diff_content = generate_diff(content, new_content)
                diff_stats_result = diff_stats(content, new_content)
                diff_content = _suppress_large_diff(diff_content, content_tokens) or ""

        # Update cache with final content
        mtime = file_path.stat().st_mtime
        embedding = cache.get_embedding(new_content)
        cache.put(str(file_path), new_content, mtime, embedding)
        action = f"Edited ({replacements_made} replacement(s))"
        if formatted:
            action += " and formatted"
        logger.info(f"{action} and cached: {path}")

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


# -----------------------------------------------------------------------------
# New tool helper functions
# -----------------------------------------------------------------------------

# DoS limits
MAX_SEARCH_K = 100
MAX_SEARCH_QUERY_LEN = 8000
MAX_BATCH_FILES = 50
MAX_BATCH_TOKENS = 200_000
MAX_SIMILAR_K = 50
MAX_GLOB_MATCHES = 1000
GLOB_TIMEOUT_SECONDS = 5


def semantic_search(
    cache: SemanticCache,
    query: str,
    k: int = 10,
    directory: str | None = None,
) -> SearchResult:
    """Search cached files by semantic meaning.

    Args:
        cache: SemanticCache instance
        query: Search query text
        k: Max results (capped at 100)
        directory: Optional directory filter

    Returns:
        SearchResult with matches sorted by similarity
    """
    # DoS protection
    k = min(k, MAX_SEARCH_K)
    query = query[:MAX_SEARCH_QUERY_LEN]

    # Embed query using search_query prefix
    query_embedding = embed_query(query)
    if query_embedding is None:
        return SearchResult(query=query, matches=[], files_searched=0, cached_files=0)

    # Get all cached files with embeddings from storage
    storage = cache._storage
    with storage._pool.get_connection() as conn:
        sql = "SELECT path, tokens, embedding FROM files WHERE embedding IS NOT NULL"
        rows = conn.execute(sql).fetchall()

    if not rows:
        return SearchResult(query=query, matches=[], files_searched=0, cached_files=len(rows))

    # Filter by directory if specified
    if directory:
        dir_path = str(Path(directory).expanduser().resolve())
        rows = [r for r in rows if r[0].startswith(dir_path)]

    if not rows:
        return SearchResult(query=query, matches=[], files_searched=0, cached_files=0)

    paths = [r[0] for r in rows]
    tokens_list = [r[1] for r in rows]
    blobs = [r[2] for r in rows]

    # Batch similarity using pre-quantized vectors (optimized)
    top_results = top_k_from_quantized(query_embedding, blobs, k=k)

    # Build matches with previews
    matches: list[SearchMatch] = []
    for idx, sim in top_results:
        path = paths[idx]
        entry = cache.get(path)
        preview = ""
        if entry:
            content = cache.get_content(entry)
            preview = content[:200].replace("\n", " ")

        matches.append(
            SearchMatch(
                path=path,
                similarity=round(sim, 4),
                tokens=tokens_list[idx],
                preview=preview,
            )
        )

    return SearchResult(
        query=query,
        matches=matches,
        files_searched=len(paths),
        cached_files=len(rows),
    )


def compare_files(
    cache: SemanticCache,
    path1: str,
    path2: str,
    context_lines: int = 3,
) -> DiffResult:
    """Compare two files using cache.

    Args:
        cache: SemanticCache instance
        path1: First file path
        path2: Second file path
        context_lines: Lines of context in diff

    Returns:
        DiffResult with diff and similarity
    """
    file1 = Path(path1).expanduser().resolve()
    file2 = Path(path2).expanduser().resolve()

    # Get content for both files (from cache or disk)
    content1: str
    content2: str
    from_cache1 = False
    from_cache2 = False

    # File 1
    cached1 = cache.get(str(file1))
    if cached1 and cached1.mtime >= file1.stat().st_mtime:
        content1 = cache.get_content(cached1)
        from_cache1 = True
    else:
        content1 = file1.read_text(encoding="utf-8")
        mtime1 = file1.stat().st_mtime
        emb1 = cache.get_embedding(content1)
        cache.put(str(file1), content1, mtime1, emb1)

    # File 2
    cached2 = cache.get(str(file2))
    if cached2 and cached2.mtime >= file2.stat().st_mtime:
        content2 = cache.get_content(cached2)
        from_cache2 = True
    else:
        content2 = file2.read_text(encoding="utf-8")
        mtime2 = file2.stat().st_mtime
        emb2 = cache.get_embedding(content2)
        cache.put(str(file2), content2, mtime2, emb2)

    # Generate diff
    diff_content = generate_diff(content1, content2, context_lines=context_lines)
    stats = diff_stats(content1, content2)

    # Compute semantic similarity between embeddings (normalized)
    similarity = 0.0
    entry1 = cache.get(str(file1))
    entry2 = cache.get(str(file2))
    if entry1 and entry1.embedding and entry2 and entry2.embedding:
        # Normalize to proper cosine similarity in [0, 1] range
        raw_sim = cosine_similarity(entry1.embedding, entry2.embedding)
        # Embeddings from nomic are normalized, but clamp just in case
        similarity = max(0.0, min(1.0, float(raw_sim)))

    # Token savings: sum of cached file tokens
    tokens_saved = 0
    if from_cache1 and cached1:
        tokens_saved += cached1.tokens
    if from_cache2 and cached2:
        tokens_saved += cached2.tokens

    return DiffResult(
        path1=str(file1),
        path2=str(file2),
        diff_content=diff_content,
        diff_stats=stats,
        tokens_saved=tokens_saved,
        similarity=round(similarity, 4),
        from_cache=(from_cache1, from_cache2),
    )


def batch_smart_read(
    cache: SemanticCache,
    paths: list[str],
    max_total_tokens: int = 50000,
    priority: list[str] | None = None,
) -> BatchReadResult:
    """Read multiple files with token budget, priority ordering, and unchanged detection.

    Args:
        cache: SemanticCache instance
        paths: List of file paths
        max_total_tokens: Token budget (capped at 200K)
        priority: Paths to read first (order preserved). Does NOT override budget.

    Returns:
        BatchReadResult with contents, summaries, and unchanged_paths
    """
    # DoS protection
    paths = paths[:MAX_BATCH_FILES]
    max_total_tokens = min(max_total_tokens, MAX_BATCH_TOKENS)

    # Estimate tokens for sorting and skipped-file enrichment.
    def estimate_min_tokens(p: str) -> int:
        resolved = Path(p).expanduser().resolve()
        cached = cache.get(str(resolved))
        if cached and resolved.exists() and cached.mtime >= resolved.stat().st_mtime:
            unchanged_msg = f"// File unchanged: {p} ({cached.tokens} tokens cached)"
            return min(cached.tokens, count_tokens(unchanged_msg))
        if not resolved.exists() or not resolved.is_file():
            return 1
        # Rough estimate for uncached content: ~4 characters per token.
        return max(1, int(resolved.stat().st_size / 4))

    # Priority-aware ordering: priority paths first (in given order), then remainder smallest-first.
    if priority:
        priority_set = set(priority)
        priority_ordered = [p for p in priority if p in set(paths)]
        remainder = [p for p in paths if p not in priority_set]
        remainder_sorted = sorted(remainder, key=lambda p: (estimate_min_tokens(p), p))
        paths_sorted = priority_ordered + remainder_sorted
    else:
        paths_sorted = sorted(paths, key=lambda p: (estimate_min_tokens(p), p))

    files: list[FileReadSummary] = []
    contents: dict[str, str] = {}
    unchanged_paths: list[str] = []
    total_tokens = 0
    tokens_saved = 0
    files_skipped = 0
    processed = 0

    for path in paths_sorted:
        processed += 1

        if total_tokens >= max_total_tokens:
            # Enrich remaining paths with est_tokens
            for remaining in paths_sorted[processed - 1 :]:
                est = estimate_min_tokens(remaining)
                files.append(
                    FileReadSummary(
                        path=remaining,
                        tokens=0,
                        status="skipped",
                        from_cache=False,
                        est_tokens=est,
                    )
                )
                files_skipped += 1
            break

        try:
            result = smart_read(cache, path, diff_mode=True, force_full=False)

            # Unchanged detection: from_cache=True and is_diff=False means LLM already has content
            if result.from_cache and not result.is_diff:
                unchanged_paths.append(path)
                files.append(
                    FileReadSummary(
                        path=path,
                        tokens=result.tokens_returned,
                        status="unchanged",
                        from_cache=True,
                    )
                )
                # Count toward budget but don't emit content
                total_tokens += result.tokens_returned
                tokens_saved += result.tokens_saved
                continue

            # Determine status
            if result.truncated:
                status = "truncated"
            elif result.is_diff:
                status = "diff"
            else:
                status = "full"

            # Check token budget
            if total_tokens + result.tokens_returned > max_total_tokens:
                est = estimate_min_tokens(path)
                files.append(
                    FileReadSummary(
                        path=path,
                        tokens=0,
                        status="skipped",
                        from_cache=False,
                        est_tokens=est,
                    )
                )
                files_skipped += 1
                continue

            files.append(
                FileReadSummary(
                    path=path,
                    tokens=result.tokens_returned,
                    status=status,
                    from_cache=result.from_cache,
                )
            )
            contents[path] = result.content
            total_tokens += result.tokens_returned
            tokens_saved += result.tokens_saved

        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {path}: {e}")
            files.append(FileReadSummary(path=path, tokens=0, status="skipped", from_cache=False))
            files_skipped += 1

    return BatchReadResult(
        files=files,
        contents=contents,
        total_tokens=total_tokens,
        tokens_saved=tokens_saved,
        files_read=len(files) - files_skipped,
        files_skipped=files_skipped,
        unchanged_paths=unchanged_paths,
    )


def find_similar_files(
    cache: SemanticCache,
    path: str,
    k: int = 5,
) -> SimilarFilesResult:
    """Find files semantically similar to given file.

    Args:
        cache: SemanticCache instance
        path: Source file path
        k: Max results (capped at 50)

    Returns:
        SimilarFilesResult with similar files
    """
    k = min(k, MAX_SIMILAR_K)
    file_path = Path(path).expanduser().resolve()

    # Get/compute embedding for source file
    cached = cache.get(str(file_path))
    source_tokens = 0

    if cached and cached.mtime >= file_path.stat().st_mtime:
        source_embedding = cached.embedding
        source_tokens = cached.tokens
    else:
        content = file_path.read_text(encoding="utf-8")
        source_embedding = cache.get_embedding(content)
        source_tokens = count_tokens(content)
        mtime = file_path.stat().st_mtime
        cache.put(str(file_path), content, mtime, source_embedding)

    if source_embedding is None:
        return SimilarFilesResult(
            source_path=str(file_path),
            source_tokens=source_tokens,
            similar_files=[],
            files_searched=0,
        )

    # Get all cached files with embeddings
    storage = cache._storage
    with storage._pool.get_connection() as conn:
        rows = conn.execute(
            "SELECT path, tokens, embedding FROM files WHERE embedding IS NOT NULL AND path != ?",
            (str(file_path),),
        ).fetchall()

    if not rows:
        return SimilarFilesResult(
            source_path=str(file_path),
            source_tokens=source_tokens,
            similar_files=[],
            files_searched=0,
        )

    paths = [r[0] for r in rows]
    tokens_list = [r[1] for r in rows]
    blobs = [r[2] for r in rows]

    # Batch similarity
    top_results = top_k_from_quantized(source_embedding, blobs, k=k)

    similar_files: list[SimilarFile] = []
    for idx, sim in top_results:
        similar_files.append(
            SimilarFile(
                path=paths[idx],
                similarity=round(sim, 4),
                tokens=tokens_list[idx],
            )
        )

    return SimilarFilesResult(
        source_path=str(file_path),
        source_tokens=source_tokens,
        similar_files=similar_files,
        files_searched=len(paths),
    )


def glob_with_cache_status(
    cache: SemanticCache,
    pattern: str,
    directory: str = ".",
) -> GlobResult:
    """Find files by pattern with cache status.

    Args:
        cache: SemanticCache instance
        pattern: Glob pattern (e.g., "**/*.py")
        directory: Base directory

    Returns:
        GlobResult with matches and cache info
    """
    dir_path = Path(directory).expanduser().resolve()

    matches: list[GlobMatch] = []
    cached_count = 0
    total_cached_tokens = 0

    deadline = time.monotonic() + GLOB_TIMEOUT_SECONDS

    count = 0
    for file_path in dir_path.glob(pattern):
        if count >= MAX_GLOB_MATCHES:
            break
        if time.monotonic() > deadline:
            logger.warning(f"Glob timed out after {GLOB_TIMEOUT_SECONDS}s")
            break
        if not file_path.is_file():
            continue

        count += 1
        path_str = str(file_path)
        mtime = file_path.stat().st_mtime

        # Check cache status
        cached = cache.get(path_str)
        is_cached = cached is not None
        tokens = cached.tokens if cached else None

        if is_cached:
            cached_count += 1
            if tokens:
                total_cached_tokens += tokens

        matches.append(
            GlobMatch(
                path=path_str,
                cached=is_cached,
                tokens=tokens,
                mtime=mtime,
            )
        )

    # Sort: cached first, then by path
    matches.sort(key=lambda m: (not m.cached, m.path))

    return GlobResult(
        pattern=pattern,
        directory=str(dir_path),
        matches=matches,
        total_matches=len(matches),
        cached_count=cached_count,
        total_cached_tokens=total_cached_tokens,
    )


# DoS limits for multi_edit
MAX_MULTI_EDITS = 50


def smart_multi_edit(
    cache: SemanticCache,
    path: str,
    edits: list[tuple[str, str]],
    dry_run: bool = False,
    auto_format: bool = False,
) -> MultiEditResult:
    """Apply multiple independent edits to a file.

    Each edit is processed independently - some can succeed while others fail.
    Successful edits are applied even if some fail (partial apply).

    Args:
        cache: SemanticCache instance
        path: Absolute path to file
        edits: List of (old_string, new_string) tuples
        dry_run: Preview changes without writing
        auto_format: Run formatter after edits (default: false)

    Returns:
        MultiEditResult with per-edit outcomes and combined diff

    Raises:
        FileNotFoundError: File doesn't exist
        ValueError: No edits provided or file is binary
        PermissionError: Insufficient permissions
    """
    # Validate inputs
    if not edits:
        raise ValueError("No edits provided")

    # DoS protection
    edits = edits[:MAX_MULTI_EDITS]

    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Not a regular file: {path}")

    # Check for binary file
    try:
        sample = file_path.read_bytes()[:8192]
        if _is_binary_content(sample):
            raise ValueError(f"Binary file not supported: {path}")
    except OSError as e:
        raise PermissionError(f"Cannot read file: {e}") from e

    # Get content (from cache if possible)
    content: str
    from_cache = False
    cached = cache.get(str(file_path))

    if cached:
        mtime = file_path.stat().st_mtime
        if cached.mtime >= mtime:
            content = cache.get_content(cached)
            from_cache = True
        else:
            content = file_path.read_text(encoding="utf-8")
    else:
        content = file_path.read_text(encoding="utf-8")

    original_content = content

    # Process each edit independently
    # First pass: find all matches and validate
    edit_info: list[tuple[int, str, str, list[int]]] = []  # (idx, old, new, line_nums)

    for idx, (old_string, new_string) in enumerate(edits):
        if old_string:
            line_numbers = _find_match_line_numbers(content, old_string)
            edit_info.append((idx, old_string, new_string, line_numbers))
        else:
            edit_info.append((idx, old_string, new_string, []))

    # Build outcomes and collect successful edits
    outcomes: list[SingleEditOutcome] = []
    successful_edits: list[tuple[str, str, int]] = []  # (old, new, first_line)

    for _idx, old_string, new_string, line_numbers in edit_info:
        if not old_string:
            outcomes.append(
                SingleEditOutcome(
                    old_string=old_string,
                    new_string=new_string,
                    success=False,
                    line_number=None,
                    error="old_string cannot be empty",
                )
            )
        elif old_string == new_string:
            outcomes.append(
                SingleEditOutcome(
                    old_string=old_string,
                    new_string=new_string,
                    success=False,
                    line_number=None,
                    error="old_string and new_string are identical",
                )
            )
        elif not line_numbers:
            outcomes.append(
                SingleEditOutcome(
                    old_string=old_string,
                    new_string=new_string,
                    success=False,
                    line_number=None,
                    error="not found",
                )
            )
        else:
            # Success - record for application
            outcomes.append(
                SingleEditOutcome(
                    old_string=old_string,
                    new_string=new_string,
                    success=True,
                    line_number=line_numbers[0],
                    error=None,
                )
            )
            successful_edits.append((old_string, new_string, line_numbers[0]))

    # Apply successful edits (sort by line number descending to preserve positions)
    new_content = content
    if successful_edits:
        # Sort by line number descending (bottom-to-top)
        successful_edits.sort(key=lambda x: x[2], reverse=True)

        for old_string, new_string, _ in successful_edits:
            # Replace first occurrence only (each edit is independent)
            new_content = new_content.replace(old_string, new_string, 1)

    # Generate combined diff
    diff_content = generate_diff(original_content, new_content)
    stats = diff_stats(original_content, new_content)
    original_tokens = count_tokens(original_content)
    diff_content = _suppress_large_diff(diff_content, original_tokens) or ""

    # Calculate token savings from cached read
    tokens_saved = original_tokens if from_cache else 0

    # Calculate content hash
    content_hash = hash_content(new_content.encode("utf-8"))

    succeeded = sum(1 for o in outcomes if o.success)
    failed = len(outcomes) - succeeded

    # Write file if any edits succeeded (unless dry_run)
    if succeeded > 0 and not dry_run:
        try:
            file_path.write_text(new_content, encoding="utf-8")
        except OSError as e:
            raise PermissionError(f"Cannot write file: {e}") from e

        # Auto-format if requested
        formatted = False
        if auto_format:
            formatted = _format_file(file_path)
            if formatted:
                # Re-read formatted content
                new_content = file_path.read_text(encoding="utf-8")
                content_hash = hash_content(new_content.encode("utf-8"))

                # Re-compute diff against original (before format)
                diff_content = generate_diff(original_content, new_content)
                stats = diff_stats(original_content, new_content)
                diff_content = _suppress_large_diff(diff_content, original_tokens) or ""

        # Update cache with final content
        mtime = file_path.stat().st_mtime
        embedding = cache.get_embedding(new_content)
        cache.put(str(file_path), new_content, mtime, embedding)
        action = f"Multi-edit ({succeeded} succeeded, {failed} failed)"
        if formatted:
            action += " and formatted"
        logger.info(f"{action}: {path}")

    return MultiEditResult(
        path=str(file_path),
        outcomes=outcomes,
        succeeded=succeeded,
        failed=failed,
        diff_content=diff_content,
        diff_stats=stats,
        tokens_saved=tokens_saved,
        content_hash=content_hash,
        from_cache=from_cache,
    )
