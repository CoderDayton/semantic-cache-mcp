"""Smart read operations."""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import time
from pathlib import Path

from ..config import MAX_CONTENT_SIZE
from ..core import count_tokens, generate_diff, summarize_semantic, truncate_semantic
from ..core.hashing import hash_content
from ..logger import log_marker
from ..types import BatchReadResult, FileReadSummary, ReadResult
from ..utils import aread_bytes, astat
from ._helpers import _diff_context_lines, _is_binary_content
from .store import SemanticCache

# Magic-byte prefixes for cheap mime sniffing when extension lookup fails.
# Intentionally small — covers the formats users most commonly try to read.
_MIME_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"II*\x00", "image/tiff"),  # little-endian TIFF
    (b"MM\x00*", "image/tiff"),  # big-endian TIFF
    (b"PK\x03\x04", "application/zip"),
    (b"%PDF-", "application/pdf"),
    (b"\x7fELF", "application/x-elf"),
    (b"\xca\xfe\xba\xbe", "application/java-vm"),
)

# RIFF containers share a prefix across WebP / WAVE / AVI — the form is at
# bytes 8..12. Disambiguated separately so WAV/AVI files aren't mis-labelled
# as image/webp.
_RIFF_FORMS: dict[bytes, str] = {
    b"WEBP": "image/webp",
    b"WAVE": "audio/wav",
    b"AVI ": "video/x-msvideo",
}

# Valid BMP DIB-header sizes (BITMAPCOREHEADER through BITMAPV5HEADER). The
# 'BM' signature is only 2 bytes, so any binary starting 0x42 0x4D matches
# it; requiring a known DIB header size at bytes 14-17 rejects that.
_BMP_DIB_HEADER_SIZES: frozenset[int] = frozenset({12, 40, 52, 56, 64, 108, 124})


def _is_bmp(raw: bytes) -> bool:
    """Validate a BMP beyond its weak 2-byte 'BM' signature."""
    return (
        raw[:2] == b"BM"
        and len(raw) >= 18
        and int.from_bytes(raw[14:18], "little") in _BMP_DIB_HEADER_SIZES
    )


def _guess_mime(path: Path, raw: bytes) -> str:
    """Best-effort mime guess. Extension first, then magic-byte sniff."""
    guess = mimetypes.guess_type(str(path))[0]
    if guess:
        return guess
    for prefix, mime in _MIME_MAGIC:
        if raw.startswith(prefix):
            return mime
    if _is_bmp(raw):
        return "image/bmp"
    if raw.startswith(b"RIFF") and len(raw) >= 12:
        return _RIFF_FORMS.get(raw[8:12], "application/octet-stream")
    return "application/octet-stream"


def _sniff_image_mime(raw: bytes) -> str | None:
    """Detect an image format from magic bytes alone, ignoring the filename.

    Returns an ``image/*`` mime when `raw` begins with a recognized image
    signature (PNG, JPEG, GIF, TIFF, BMP, WebP), else ``None``. The
    extension is deliberately not consulted: `read_image` uses this so a
    mis-named file — text saved as ``.png``, or a real image with the
    wrong/no extension — is judged by content, not by its name.
    """
    for prefix, mime in _MIME_MAGIC:
        if mime.startswith("image/") and raw.startswith(prefix):
            return mime
    if _is_bmp(raw):
        return "image/bmp"
    if raw.startswith(b"RIFF") and len(raw) >= 12 and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


logger = logging.getLogger(__name__)

# DoS limits
MAX_BATCH_FILES = 50
MAX_BATCH_TOKENS = 200_000

# Diff gate: emit a unified diff instead of full content when the diff is at
# least this much smaller than the file. Raised from 0.6 so small real edits to
# mid/large files still diff (the diff carries the changed line numbers, which is
# what the agent needs after an edit) instead of falling through to a full read.
_DIFF_MAX_RATIO = 0.9
# Below this size a diff's @@-header overhead isn't worth it — return full
# content instead. Tiny files are cheap, and a diff just adds noise.
_DIFF_MIN_TOKENS = 200


async def smart_read(
    cache: SemanticCache,
    path: str,
    max_size: int = MAX_CONTENT_SIZE,
    diff_mode: bool = True,
    force_full: bool = False,
    refresh_cache: bool = True,
    summarize: bool = True,
) -> ReadResult:
    """Read file with intelligent caching and optimization.

    Strategies (in order of token savings):
    1. File unchanged (mtime match) -> "// No changes" (99% reduction)
    2. File changed -> unified diff (80-95% reduction)
    3. Large file -> smart truncation (50-80% reduction)
    4. New file -> full content with caching for future reads
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

    # Check cache first; only stat when needed (hit comparison or cold-path
    # refresh). Reading bytes is the dominant cost on the slow path, so the
    # stat is folded in just before refresh_path() instead of up front.
    cached = await cache.get(str(file_path))
    mtime: float = 0.0
    if cached is not None:
        mtime = (await astat(file_path, cache._io_executor)).st_mtime

    if cached and diff_mode and not force_full and cached.mtime >= mtime:
        await cache.record_access(str(file_path))
        unchanged_msg = f"// File unchanged: {path} ({cached.tokens} tokens cached)"
        msg_tokens = count_tokens(unchanged_msg)
        if msg_tokens < cached.tokens:
            return ReadResult(
                content=unchanged_msg,
                from_cache=True,
                is_diff=False,
                tokens_original=cached.tokens,
                tokens_returned=msg_tokens,
                tokens_saved=cached.tokens - msg_tokens,
                truncated=False,
                compression_ratio=msg_tokens / cached.tokens if cached.tokens else 1.0,
                content_hash=cached.content_hash,
            )
        cached_content = await cache.get_content(cached)
        return ReadResult(
            content=cached_content,
            from_cache=True,
            is_diff=False,
            tokens_original=cached.tokens,
            tokens_returned=cached.tokens,
            tokens_saved=0,
            truncated=False,
            compression_ratio=1.0,
            content_hash=cached.content_hash,
        )

    # Slow path: file changed, missing from cache, or force_full requested.
    raw = await aread_bytes(file_path, cache._io_executor)
    if _is_binary_content(raw):
        # Surface structured metadata instead of raising. Tool callers
        # branch on `is_binary` and skip downstream content processing.
        return ReadResult(
            content="",
            from_cache=False,
            is_diff=False,
            tokens_original=0,
            tokens_returned=0,
            tokens_saved=0,
            truncated=False,
            compression_ratio=1.0,
            is_binary=True,
            size=len(raw),
            mime=_guess_mime(file_path, raw),
        )
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("utf-8", errors="replace")
        logger.warning(f"File {path} contains non-UTF-8 characters, using replacement")

    tokens_original = count_tokens(content)

    # Memoized hash of the just-read content: several branches below compare
    # it, and the ReadResult carries it back so callers don't need an extra
    # cache-entry lookup to learn the current hash.
    _hash: str | None = None

    def _content_hash() -> str:
        nonlocal _hash
        if _hash is None:
            _hash = hash_content(content)
        return _hash

    cache_is_fresh = False
    if cached and force_full:
        if cached.mtime >= mtime:
            cache_is_fresh = True
        elif _content_hash() == cached.content_hash:
            await cache.update_mtime(str(file_path), mtime)
            cache_is_fresh = True

    # Strategy 1 & 2: Cached file (unchanged or diff)
    if cached and diff_mode and not force_full:
        if cached.mtime >= mtime:
            # File unchanged (mtime match)
            pass
        elif _content_hash() == cached.content_hash:
            # Content identical despite mtime change — update mtime, treat as unchanged
            await cache.update_mtime(str(file_path), mtime)
        else:
            # Content actually changed — fall through to diff logic below
            _original_cached = cached  # save before nulling
            cached = None  # sentinel: forces diff path

        if cached is not None:
            # Unchanged path (either mtime match or content hash match)
            await cache.record_access(str(file_path))
            unchanged_msg = f"// File unchanged: {path} ({cached.tokens} tokens cached)"
            # NOTE: do NOT add extra text here — the JSON "unchanged":true
            # field signals the LLM that full content is already in context.
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
                    content_hash=cached.content_hash,
                )

            # Small file - return full content
            cached_content = await cache.get_content(cached)
            return ReadResult(
                content=cached_content,
                from_cache=True,
                is_diff=False,
                tokens_original=tokens_original,
                tokens_returned=tokens_original,
                tokens_saved=0,
                truncated=False,
                compression_ratio=1.0,
                content_hash=cached.content_hash,
            )

        # Restore the original entry — we nulled it as sentinel above.
        # No re-fetch needed; the entry hasn't changed in storage.
        cached = _original_cached

        # File changed - generate diff with stats
        if cached is not None:
            old_content = await cache.get_content(cached)
            diff_content = generate_diff(
                old_content, content, context_lines=_diff_context_lines(old_content)
            )
            diff_tokens = count_tokens(diff_content)

            if (
                tokens_original >= _DIFF_MIN_TOKENS
                and diff_tokens < tokens_original * _DIFF_MAX_RATIO
            ):
                # Bare diff: the `is_diff`/`diff_state` fields and the `@@` hunk
                # headers already mark it as a diff, so a prose prefix would just
                # cost tokens.
                result_content = diff_content
                await cache.refresh_path(
                    str(file_path),
                    content,
                    mtime,
                )

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
                    content_hash=_content_hash(),
                )

    # Strategy 3 & 4: Full read (semantic summarization if large)
    truncated = False
    final_content = content

    # Line-addressed reads (offset/limit) pass summarize=False: they must slice
    # literal disk lines, so summarizing here would fabricate line numbers and a
    # false total. Full-file reads keep summarization for large files.
    if summarize and len(content) > max_size:
        # Use semantic summarization to preserve important content. Runs in the
        # executor — summarizing a large file is CPU-heavy and would otherwise
        # block the event loop.
        try:
            started = time.perf_counter()
            log_marker(logger, "summarize.begin", path=path, chars=len(content))

            def _summarize() -> str:
                return summarize_semantic(content, max_size, embed_fn=None)

            loop = asyncio.get_running_loop()
            final_content = await loop.run_in_executor(cache._io_executor, _summarize)
            log_marker(
                logger,
                "summarize.end",
                path=path,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            truncated = True
        except Exception as e:
            log_marker(logger, "summarize.fail", path=path, error=type(e).__name__)
            logger.warning(f"Semantic summarization failed: {e}, using fallback truncation")
            final_content = truncate_semantic(content, max_size)
            truncated = True

    should_refresh_cache = refresh_cache or not cache_is_fresh
    if should_refresh_cache:
        if cached is None:
            mtime = (await astat(file_path, cache._io_executor)).st_mtime
        await cache.refresh_path(
            str(file_path),
            content,
            mtime,
        )

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
        content_hash=_content_hash(),
    )


async def batch_smart_read(
    cache: SemanticCache,
    paths: list[str],
    max_total_tokens: int = 50000,
    priority: list[str] | None = None,
    diff_mode: bool = True,
) -> BatchReadResult:
    """Read multiple files with token budget, priority ordering, and unchanged detection.

    priority paths are read first but do NOT override the token budget.
    diff_mode=False forces full content (use after context compression).
    """
    # DoS protection
    paths = paths[:MAX_BATCH_FILES]
    max_total_tokens = max(1, min(max_total_tokens, MAX_BATCH_TOKENS))

    # Batch-fetch all cache entries in parallel (one gather instead of N serial awaits).
    resolved_map = {p: str(Path(p).expanduser().resolve()) for p in paths}
    _entries = await asyncio.gather(*(cache.get(rp) for rp in resolved_map.values()))
    _cache_map = dict(zip(resolved_map.values(), _entries, strict=True))

    # Pre-fetch stat results through the executor so the estimation loop
    # doesn't block the event loop with synchronous syscalls (matters on
    # NFS/FUSE mounts where stat can be slow).
    import os  # noqa: PLC0415

    async def _safe_stat(p: Path) -> os.stat_result | None:
        try:
            return await astat(p, cache._io_executor)
        except (OSError, ValueError):
            return None

    _stat_results = await asyncio.gather(*(_safe_stat(Path(rp)) for rp in resolved_map.values()))
    _stat_map: dict[str, os.stat_result | None] = dict(
        zip(resolved_map.values(), _stat_results, strict=True)
    )

    # Estimate tokens for sorting and skipped-file enrichment.
    def estimate_min_tokens(p: str) -> int:
        rp = resolved_map[p]
        cached = _cache_map.get(rp)
        st = _stat_map.get(rp)
        if st is None:
            return 1
        if cached:
            if cached.mtime >= st.st_mtime:
                unchanged_msg = f"// File unchanged: {p} ({cached.tokens} tokens cached)"
                return min(cached.tokens, count_tokens(unchanged_msg))
            return cached.tokens
        return max(1, int(st.st_size / 4))

    token_estimates = {p: estimate_min_tokens(p) for p in paths}

    # Priority-aware ordering: priority paths first (in given order), then remainder smallest-first.
    if priority:
        priority_set = set(priority)
        paths_set = set(paths)
        priority_ordered = [p for p in priority if p in paths_set]
        remainder = [p for p in paths if p not in priority_set]
        remainder_sorted = sorted(remainder, key=lambda p: token_estimates[p])
        paths_sorted = priority_ordered + remainder_sorted
    else:
        paths_sorted = sorted(paths, key=lambda p: token_estimates[p])

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
            result = await smart_read(
                cache,
                path,
                diff_mode=diff_mode,
                force_full=False,
            )

            if result.is_binary:
                # Binary files are silently skipped from batch results; the
                # single-file `read` tool surfaces structured metadata instead.
                files.append(
                    FileReadSummary(path=path, tokens=0, status="skipped", from_cache=False)
                )
                files_skipped += 1
                continue

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
