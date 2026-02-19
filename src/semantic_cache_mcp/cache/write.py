"""Smart write and edit operations for the cache package."""

from __future__ import annotations

import logging
from pathlib import Path

from ..core import count_tokens, diff_stats, generate_diff
from ..core.hashing import hash_content
from ..types import BatchEditResult, EditResult, SingleEditOutcome, WriteResult
from ._helpers import (
    MAX_EDIT_SIZE,
    MAX_MATCHES,
    MAX_WRITE_SIZE,
    _find_match_line_numbers,
    _format_file,
    _is_binary_content,
    _suppress_large_diff,
)
from .store import SemanticCache

logger = logging.getLogger(__name__)

# DoS limit for batch_edit
MAX_BATCH_EDITS = 50


def smart_write(
    cache: SemanticCache,
    path: str,
    content: str,
    create_parents: bool = True,
    dry_run: bool = False,
    auto_format: bool = False,
    append: bool = False,
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
        content: Content to write (or append)
        create_parents: Create parent directories if missing
        dry_run: Preview changes without writing
        auto_format: Run formatter after write (default: false)
        append: Append content to existing file instead of overwriting (default: false)

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

    # Append mode: concatenate new content onto existing
    if append and old_content is not None:
        content = old_content + content
        content_bytes = content.encode("utf-8")
        bytes_written = len(content_bytes)
        tokens_written = count_tokens(content)
        content_hash = hash_content(content_bytes)

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


def smart_batch_edit(
    cache: SemanticCache,
    path: str,
    edits: list[tuple[str, str]],
    dry_run: bool = False,
    auto_format: bool = False,
) -> BatchEditResult:
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
        BatchEditResult with per-edit outcomes and combined diff

    Raises:
        FileNotFoundError: File doesn't exist
        ValueError: No edits provided or file is binary
        PermissionError: Insufficient permissions
    """
    # Validate inputs
    if not edits:
        raise ValueError("No edits provided")

    # DoS protection
    edits = edits[:MAX_BATCH_EDITS]

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

    return BatchEditResult(
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
