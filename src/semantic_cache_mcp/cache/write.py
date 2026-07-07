"""Smart write and edit operations."""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from pathlib import Path

from ..core import count_tokens, diff_stats, generate_diff
from ..core.hashing import hash_content
from ..types import BatchEditResult, EditResult, SingleEditOutcome, WriteResult
from ..utils import aread_bytes, aread_text, astat, awrite_atomic
from ..utils._async_io import (
    _atomic_write_sync as _atomic_write,  # noqa: F401 — re-exported for tests
)
from ._helpers import (
    MAX_EDIT_SIZE,
    MAX_MATCHES,
    MAX_WRITE_SIZE,
    _extract_line_range,
    _find_match_line_numbers,
    _format_file,
    _is_binary_content,
    _PhaseTimer,
    _suppress_large_diff,
)
from .store import SemanticCache

logger = logging.getLogger(__name__)

# DoS limit for batch_edit
MAX_BATCH_EDITS = 50


async def smart_write(
    cache: SemanticCache,
    path: str,
    content: str,
    create_parents: bool = True,
    dry_run: bool = False,
    auto_format: bool = False,
    append: bool = False,
) -> WriteResult:
    """Write file with cache integration.

    Returns diff instead of full content for overwrites (token savings).

    Raises:
        FileNotFoundError: Parent directory doesn't exist and create_parents=False
        PermissionError: Insufficient permissions
        ValueError: Not a regular file, binary content, or content exceeds 10MB
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
                logger.debug(f"Created parent directories: {parent}")
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
    mtime: float | None = None

    if not created:
        # Check for binary file
        try:
            sample = (await aread_bytes(file_path, cache._io_executor))[:8192]
            if _is_binary_content(sample):
                raise ValueError(
                    f"Binary file not supported: {path}. Cannot overwrite binary with text."
                )
        except OSError as e:
            raise PermissionError(f"Cannot read existing file: {e}") from e

        mtime = (await astat(file_path, cache._io_executor)).st_mtime

        # Try to get content from cache first (saves tokens!)
        cached = await cache.get(str(file_path))
        if cached:
            if cached.mtime >= mtime:
                old_content = await cache.get_content(cached)
                from_cache = True
                logger.debug(f"Using cached content for diff: {path}")
            else:
                # mtime changed — check content hash before falling back to disk
                try:
                    disk_bytes = await aread_bytes(file_path, cache._io_executor)
                    if hash_content(disk_bytes) == cached.content_hash:
                        await cache.update_mtime(str(file_path), mtime)
                        old_content = await cache.get_content(cached)
                        from_cache = True
                        logger.debug(f"Content hash match for diff: {path}")
                # Fall through to the disk read below on any failure.
                except Exception:  # nosec B110
                    pass

        # Fall back to disk read
        if old_content is None:
            try:
                old_content = await aread_text(file_path, executor=cache._io_executor)
            except UnicodeDecodeError:
                old_content = await aread_text(
                    file_path, errors="replace", executor=cache._io_executor
                )
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
            await awrite_atomic(file_path, content, cache._io_executor)
        except OSError as e:
            raise PermissionError(f"Cannot write file: {e}") from e

        # Auto-format if requested
        formatted = False
        if auto_format:
            formatted = await _format_file(file_path)
            if formatted:
                # Re-read formatted content
                content = await aread_text(file_path, executor=cache._io_executor)
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

        # Update cache with final content.
        # Re-stat after the write (and any formatter pass): awrite_atomic
        # bumps the file mtime, so the value captured for the freshness
        # check is now stale. Persisting it would make the next read treat
        # the cache as stale and needlessly re-hash from disk.
        mtime = (await astat(file_path, cache._io_executor)).st_mtime
        await cache.refresh_path(
            str(file_path),
            content,
            mtime,
        )
        action = "Created" if created else "Updated"
        if formatted:
            action += " and formatted"
        logger.debug(f"{action} and cached: {path}")

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
        dry_run=dry_run,
    )


def find_edit_anchors(
    content: str,
    old_string: str,
    *,
    max_results: int = 50,
) -> tuple[int, list[int]]:
    """Return (match_count, line_numbers) for `old_string` in `content`.

    Used by both the edit tool and the read-only `edit_preview` tool.
    `line_numbers` is 1-based and capped at `max_results`; `match_count`
    is the true total even if line numbers are truncated.
    """
    if not old_string:
        return 0, []
    count = content.count(old_string)
    if count == 0:
        return 0, []
    line_numbers = _find_match_line_numbers(content, old_string)
    if len(line_numbers) > max_results:
        line_numbers = line_numbers[:max_results]
    return count, line_numbers


def _format_anchor_miss_hint(
    content: str,
    old_string: str,
    *,
    max_suggestions: int = 3,
    max_lines: int = 5000,
    min_ratio: float = 0.6,
) -> str:
    """Produce a hint string with nearest-line matches for an edit-anchor miss.

    Returns an empty string when the file is too large to scan cheaply or
    when no line scores above `min_ratio`. The result is appended verbatim
    to the ValueError message.
    """
    if not old_string:
        return ""
    lines = content.splitlines()
    if len(lines) > max_lines:
        return ""
    # Compare against the first non-empty line of old_string — fuzzy-matching
    # the entire anchor is N^2 on long anchors; the first non-empty line is
    # almost always the distinctive part.
    needle = next((line for line in old_string.splitlines() if line.strip()), old_string)
    if not needle:
        return ""
    matcher = SequenceMatcher(a=needle, b="", autojunk=False)
    scored: list[tuple[float, int, str]] = []
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        matcher.set_seq2(line)
        ratio = matcher.ratio()
        if ratio >= min_ratio:
            scored.append((ratio, idx, line))
    if not scored:
        return ""
    scored.sort(key=lambda t: (-t[0], t[1]))
    top = scored[:max_suggestions]
    suggestions = "\n".join(f"  L{idx}: {line.rstrip()[:120]}" for _ratio, idx, line in top)
    return f"\nClosest lines in file:\n{suggestions}"


async def smart_edit(
    cache: SemanticCache,
    path: str,
    old_string: str | None,
    new_string: str,
    replace_all: bool = False,
    dry_run: bool = False,
    auto_format: bool = False,
    start_line: int | None = None,
    end_line: int | None = None,
    timer: _PhaseTimer | None = None,
) -> EditResult:
    """Edit file using find/replace with cached read.

    Three modes:
    - Mode A (find/replace): old_string + new_string — full-file search.
    - Mode B (scoped): old_string + new_string + start_line/end_line — search within range only.
    - Mode C (line replace): new_string + start_line/end_line (no old_string) — replace range.

    Raises:
        FileNotFoundError: File doesn't exist
        ValueError: old_string not found, multiple matches without replace_all,
                   invalid line range, or old_string equals new_string
        PermissionError: Insufficient permissions
    """
    if timer is not None:
        timer.enter("input_validation")
    # --- Fail-fast validation (before any I/O) ---
    has_line_range = start_line is not None or end_line is not None

    if has_line_range and (start_line is None or end_line is None):
        raise ValueError("start_line and end_line must both be provided together")

    if old_string is None:
        # Mode C: line replace — require line range, reject replace_all
        if not has_line_range:
            raise ValueError(
                "old_string is required for find/replace mode. "
                "Provide start_line/end_line for line-range replacement."
            )
        if replace_all:
            raise ValueError("replace_all is not supported with line-range replacement (Mode C)")
    else:
        # Mode A or B
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
    if timer is not None:
        timer.enter("binary_check")
    try:
        sample = (await aread_bytes(file_path, cache._io_executor))[:8192]
        if _is_binary_content(sample):
            raise ValueError(f"Binary file not supported: {path}. Edit only works with text files.")
    except OSError as e:
        raise PermissionError(f"Cannot read file: {e}") from e

    # Try to get content from cache first (huge token savings!)
    if timer is not None:
        timer.enter("cache_lookup")
    content: str
    from_cache = False
    mtime: float | None = None
    cached = await cache.get(str(file_path))

    if cached:
        mtime = (await astat(file_path, cache._io_executor)).st_mtime
        if cached.mtime >= mtime:
            content = await cache.get_content(cached)
            from_cache = True
            logger.debug(f"Using cached content for edit: {path}")
        else:
            # mtime changed — check content hash before falling back to disk
            try:
                disk_content = await aread_text(file_path, executor=cache._io_executor)
            except UnicodeDecodeError:
                disk_content = await aread_text(
                    file_path, errors="replace", executor=cache._io_executor
                )
            if hash_content(disk_content) == cached.content_hash:
                await cache.update_mtime(str(file_path), mtime)
                content = await cache.get_content(cached)
                from_cache = True
                logger.debug(f"Content hash match for edit: {path}")
            else:
                content = disk_content
    else:
        # No cache entry, read from disk
        try:
            content = await aread_text(file_path, executor=cache._io_executor)
        except UnicodeDecodeError:
            content = await aread_text(file_path, errors="replace", executor=cache._io_executor)

    # Validate content size
    if len(content) > MAX_EDIT_SIZE:
        raise ValueError(
            f"File too large for edit: {len(content):,} bytes exceeds {MAX_EDIT_SIZE:,} byte limit"
        )

    # --- Mode dispatch ---
    if timer is not None:
        timer.enter("anchor_search")
    if old_string is None:
        # Mode C: line-range replacement (start_line/end_line guaranteed non-None by validation)
        if start_line is None or end_line is None:
            raise TypeError("start_line and end_line required for line-range mode")
        _substring, char_start, char_end = _extract_line_range(content, start_line, end_line)
        # Preserve line terminator: if the replaced range ended with \n, ensure new_string does too
        if _substring.endswith("\n") and not new_string.endswith("\n"):
            new_string = new_string + "\n"
        new_content = content[:char_start] + new_string + content[char_end:]
        line_numbers = list(range(start_line, end_line + 1))
        matches_found = end_line - start_line + 1
        replacements_made = matches_found

    elif has_line_range:
        # Mode B: scoped find/replace within line range
        if start_line is None or end_line is None:
            raise TypeError("start_line and end_line required for line-range mode")
        substring, char_start, char_end = _extract_line_range(content, start_line, end_line)

        # Search within the substring only
        quick_count = substring.count(old_string)
        if quick_count > MAX_MATCHES:
            raise ValueError(
                f"Too many matches ({quick_count:,}) within lines {start_line}-{end_line}. "
                f"Maximum {MAX_MATCHES:,} occurrences allowed."
            )

        if quick_count == 0:
            hint = _format_anchor_miss_hint(substring, old_string)
            raise ValueError(
                f"old_string not found within lines {start_line}-{end_line} of {path}. "
                "Hint: Ensure exact whitespace and indentation match." + hint
            )

        if quick_count > 1 and not replace_all:
            # Find line numbers within substring, then offset to absolute
            sub_line_numbers = _find_match_line_numbers(substring, old_string)
            abs_line_numbers = [ln + start_line - 1 for ln in sub_line_numbers]
            raise ValueError(
                f"old_string found {quick_count} times at lines {abs_line_numbers} "
                f"(within range {start_line}-{end_line}) in {path}. "
                "Hint: Provide more context to make the match unique, or use replace_all=True"
            )

        # Perform replacement within substring
        if replace_all:
            new_substring = substring.replace(old_string, new_string)
            replacements_made = quick_count
        else:
            new_substring = substring.replace(old_string, new_string, 1)
            replacements_made = 1

        new_content = content[:char_start] + new_substring + content[char_end:]

        # Report absolute line numbers
        sub_line_numbers = _find_match_line_numbers(substring, old_string)
        line_numbers = [ln + start_line - 1 for ln in sub_line_numbers]
        if not replace_all:
            line_numbers = line_numbers[:1]
        matches_found = quick_count

    else:
        # Mode A: full-file find/replace (existing behavior)
        quick_count = content.count(old_string)
        if quick_count > MAX_MATCHES:
            raise ValueError(
                f"Too many matches ({quick_count:,}). "
                f"Maximum {MAX_MATCHES:,} occurrences allowed for edit operations."
            )

        line_numbers = _find_match_line_numbers(content, old_string)
        matches_found = len(line_numbers)

        if matches_found == 0:
            hint = _format_anchor_miss_hint(content, old_string)
            raise ValueError(
                f"old_string not found in {path}. "
                "Hint: Ensure exact whitespace and indentation match." + hint
            )

        if matches_found > 1 and not replace_all:
            raise ValueError(
                f"old_string found {matches_found} times at lines {line_numbers} in {path}. "
                "Hint: Provide more context to make the match unique, or use replace_all=True"
            )

        if replace_all:
            new_content = content.replace(old_string, new_string)
            replacements_made = matches_found
        else:
            new_content = content.replace(old_string, new_string, 1)
            replacements_made = 1
            line_numbers = line_numbers[:1]

    # Generate diff
    if timer is not None:
        timer.enter("diff_gen")
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
        if timer is not None:
            timer.enter("atomic_write")
        try:
            await awrite_atomic(file_path, new_content, cache._io_executor)
        except OSError as e:
            raise PermissionError(f"Cannot write file: {e}") from e

        # Auto-format if requested
        formatted = False
        if auto_format:
            if timer is not None:
                timer.enter("format_subprocess")
            formatted = await _format_file(file_path)
            if formatted:
                # Re-read formatted content
                new_content = await aread_text(file_path, executor=cache._io_executor)
                content_hash = hash_content(new_content.encode("utf-8"))

                # Re-compute diff against original (before format)
                diff_content = generate_diff(content, new_content)
                diff_stats_result = diff_stats(content, new_content)
                diff_content = _suppress_large_diff(diff_content, content_tokens) or ""

        # Update cache with final content.
        if timer is not None:
            timer.enter("cache_refresh")
        # Re-stat after the write (and any formatter pass): the pre-write
        # mtime captured for the freshness check is stale once awrite_atomic
        # bumps it, and persisting it would make the next read miss cache.
        mtime = (await astat(file_path, cache._io_executor)).st_mtime
        await cache.refresh_path(
            str(file_path),
            new_content,
            mtime,
        )
        action = f"Edited ({replacements_made} replacement(s))"
        if formatted:
            action += " and formatted"
        logger.debug(f"{action} and cached: {path}")

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
        dry_run=dry_run,
    )


async def smart_batch_edit(
    cache: SemanticCache,
    path: str,
    edits: (list[tuple[str | None, str, int | None, int | None]] | list[tuple[str, str]]),
    dry_run: bool = False,
    auto_format: bool = False,
) -> BatchEditResult:
    """Apply multiple independent edits to a file.

    Each edit is processed independently - some can succeed while others fail.
    Successful edits are applied even if some fail (partial apply).

    Each edit tuple is (old_string | None, new_string, start_line | None, end_line | None):
    - (old, new, None, None) — Mode A: full-file find/replace
    - (old, new, start, end) — Mode B: scoped search within line range
    - (None, new, start, end) — Mode C: replace entire line range

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
        sample = (await aread_bytes(file_path, cache._io_executor))[:8192]
        if _is_binary_content(sample):
            raise ValueError(f"Binary file not supported: {path}")
    except OSError as e:
        raise PermissionError(f"Cannot read file: {e}") from e

    # Get content (from cache if possible)
    content: str
    from_cache = False
    mtime: float | None = None
    cached = await cache.get(str(file_path))

    if cached:
        mtime = (await astat(file_path, cache._io_executor)).st_mtime
        if cached.mtime >= mtime:
            content = await cache.get_content(cached)
            from_cache = True
        else:
            # mtime changed — check content hash before falling back to disk
            try:
                disk_content = await aread_text(file_path, executor=cache._io_executor)
            except UnicodeDecodeError:
                disk_content = await aread_text(
                    file_path, errors="replace", executor=cache._io_executor
                )
                logger.warning(f"File {path} contains non-UTF-8 characters")
            if hash_content(disk_content) == cached.content_hash:
                await cache.update_mtime(str(file_path), mtime)
                content = await cache.get_content(cached)
                from_cache = True
            else:
                content = disk_content
    else:
        # No cache entry, read from disk
        try:
            content = await aread_text(file_path, executor=cache._io_executor)
        except UnicodeDecodeError:
            content = await aread_text(file_path, errors="replace", executor=cache._io_executor)
            logger.warning(f"File {path} contains non-UTF-8 characters")

    original_content = content

    # Normalize edits to 4-tuples: (old | None, new, start_line | None, end_line | None)
    normalized: list[tuple[str | None, str, int | None, int | None]] = []
    for edit in edits:
        if len(edit) == 2:
            normalized.append((edit[0], edit[1], None, None))
        else:
            normalized.append((edit[0], edit[1], edit[2], edit[3]))

    # Process each edit independently — validate and collect results
    outcomes: list[SingleEditOutcome] = []
    # Successful edits: (old | None, new, sort_line, start_line | None,
    # end_line | None, outcome_index)
    successful_edits: list[tuple[str | None, str, int, int | None, int | None, int]] = []

    for old_string, new_string, sl, el in normalized:
        try:
            has_range = sl is not None or el is not None

            if has_range and (sl is None or el is None):
                raise ValueError("start_line and end_line must both be provided together")

            if old_string is None:
                # Mode C: line replace
                if not has_range:
                    raise ValueError("old_string is required without line range")
                if sl is not None and el is not None:
                    _extract_line_range(content, sl, el)  # validates bounds
                    sort_line = sl
                    outcomes.append(
                        SingleEditOutcome(
                            old_string="",
                            new_string=new_string,
                            success=True,
                            line_number=sl,
                            error=None,
                        )
                    )
                    successful_edits.append(
                        (None, new_string, sort_line, sl, el, len(outcomes) - 1)
                    )

            elif not old_string:
                raise ValueError("old_string cannot be empty")

            elif old_string == new_string:
                raise ValueError("old_string and new_string are identical")

            elif has_range:
                # Mode B: scoped search
                if sl is None or el is None:
                    raise TypeError("start_line and end_line required for line-range mode")
                substring, _cs, _ce = _extract_line_range(content, sl, el)
                sub_matches = _find_match_line_numbers(substring, old_string)
                if not sub_matches:
                    raise ValueError(f"old_string not found within lines {sl}-{el}")
                abs_line = sub_matches[0] + sl - 1
                outcomes.append(
                    SingleEditOutcome(
                        old_string=old_string,
                        new_string=new_string,
                        success=True,
                        line_number=abs_line,
                        error=None,
                    )
                )
                successful_edits.append(
                    (old_string, new_string, abs_line, sl, el, len(outcomes) - 1)
                )

            else:
                # Mode A: full-file search
                line_numbers = _find_match_line_numbers(content, old_string)
                if not line_numbers:
                    raise ValueError("not found")
                outcomes.append(
                    SingleEditOutcome(
                        old_string=old_string,
                        new_string=new_string,
                        success=True,
                        line_number=line_numbers[0],
                        error=None,
                    )
                )
                successful_edits.append(
                    (old_string, new_string, line_numbers[0], None, None, len(outcomes) - 1)
                )

        except ValueError as exc:
            outcomes.append(
                SingleEditOutcome(
                    old_string=old_string or "",
                    new_string=new_string,
                    success=False,
                    line_number=None,
                    error=str(exc),
                )
            )

    # Ranged edits (Mode B/C) splice by character offsets re-derived from the
    # progressively-mutated content, so two edits over overlapping line ranges
    # would corrupt each other. Fail the later-listed one instead of applying.
    accepted_ranges: list[tuple[int, int]] = []
    surviving: list[tuple[str | None, str, int, int | None, int | None, int]] = []
    for entry in successful_edits:
        sl, el, outcome_idx = entry[3], entry[4], entry[5]
        if sl is not None and el is not None:
            if any(sl <= hi and lo <= el for lo, hi in accepted_ranges):
                outcome = outcomes[outcome_idx]
                outcome.success = False
                outcome.line_number = None
                outcome.error = f"line range {sl}-{el} overlaps another edit in this batch"
                continue
            accepted_ranges.append((sl, el))
        surviving.append(entry)
    successful_edits = surviving

    # Apply successful edits (sort by line number descending to preserve positions)
    new_content = content
    if successful_edits:
        successful_edits.sort(key=lambda x: x[2], reverse=True)

        for old_str, new_str, _sort_line, sl, el, _outcome_idx in successful_edits:
            if old_str is None and sl is not None and el is not None:
                # Mode C: splice by line range
                _sub, cs, ce = _extract_line_range(new_content, sl, el)
                if _sub.endswith("\n") and not new_str.endswith("\n"):
                    new_str = new_str + "\n"
                new_content = new_content[:cs] + new_str + new_content[ce:]
            elif old_str is not None and sl is not None and el is not None:
                # Mode B: scoped replace
                sub, cs, ce = _extract_line_range(new_content, sl, el)
                new_sub = sub.replace(old_str, new_str, 1)
                new_content = new_content[:cs] + new_sub + new_content[ce:]
            elif old_str is not None:
                # Mode A: full-file replace
                new_content = new_content.replace(old_str, new_str, 1)

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
            await awrite_atomic(file_path, new_content, cache._io_executor)
        except OSError as e:
            raise PermissionError(f"Cannot write file: {e}") from e

        # Auto-format if requested
        formatted = False
        if auto_format:
            formatted = await _format_file(file_path)
            if formatted:
                # Re-read formatted content
                new_content = await aread_text(file_path, executor=cache._io_executor)
                content_hash = hash_content(new_content.encode("utf-8"))

                # Re-compute diff against original (before format)
                diff_content = generate_diff(original_content, new_content)
                stats = diff_stats(original_content, new_content)
                diff_content = _suppress_large_diff(diff_content, original_tokens) or ""

        # Update cache with final content.
        # Re-stat after the write (and any formatter pass): the pre-write
        # mtime captured for the freshness check is stale once awrite_atomic
        # bumps it, and persisting it would make the next read miss cache.
        mtime = (await astat(file_path, cache._io_executor)).st_mtime
        await cache.refresh_path(
            str(file_path),
            new_content,
            mtime,
        )
        action = f"Multi-edit ({succeeded} succeeded, {failed} failed)"
        if formatted:
            action += " and formatted"
        logger.debug(f"{action}: {path}")

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
        dry_run=dry_run,
    )
