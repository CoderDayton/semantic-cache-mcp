"""Smart read operations."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np

from ..config import MAX_CONTENT_SIZE
from ..core import count_tokens, diff_stats, generate_diff, summarize_semantic, truncate_semantic
from ..core.hashing import hash_content
from ..logger import log_marker
from ..types import BatchReadResult, EmbeddingVector, FileReadSummary, ReadResult
from ..utils import aread_bytes, aread_text, astat
from ._helpers import _is_binary_content
from .store import SemanticCache

logger = logging.getLogger(__name__)

# DoS limits
MAX_BATCH_FILES = 50
MAX_BATCH_TOKENS = 200_000


async def smart_read(
    cache: SemanticCache,
    path: str,
    max_size: int = MAX_CONTENT_SIZE,
    diff_mode: bool = True,
    force_full: bool = False,
    refresh_cache: bool = True,
    _embedding: EmbeddingVector | None = None,
) -> ReadResult:
    """Read file with intelligent caching and optimization.

    Strategies (in order of token savings):
    1. File unchanged (mtime match) -> "// No changes" (99% reduction)
    2. File changed -> unified diff (80-95% reduction)
    3. Similar file in cache -> reference + diff (70-90% reduction)
    4. Large file -> smart truncation (50-80% reduction)
    5. New file -> full content with caching for future reads
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

    # Single read: binary check + decode in one I/O operation
    raw = await aread_bytes(file_path, cache._io_executor)
    if _is_binary_content(raw):
        raise ValueError(
            f"Binary file not supported: {path}. Semantic cache only handles text files."
        )
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("utf-8", errors="replace")
        logger.warning(f"File {path} contains non-UTF-8 characters, using replacement")

    mtime = (await astat(file_path, cache._io_executor)).st_mtime
    tokens_original = count_tokens(content)
    computed_embedding = _embedding

    cached = await cache.get(str(file_path))
    cache_is_fresh = False
    if cached and force_full:
        if cached.mtime >= mtime:
            cache_is_fresh = True
        elif hash_content(content) == cached.content_hash:
            await cache.update_mtime(str(file_path), mtime)
            cache_is_fresh = True

    # Strategy 1 & 2: Cached file (unchanged or diff)
    if cached and diff_mode and not force_full:
        if cached.mtime >= mtime:
            # File unchanged (mtime match)
            pass
        elif hash_content(content) == cached.content_hash:
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
            )

        # Restore the original entry — we nulled it as sentinel above.
        # No re-fetch needed; the entry hasn't changed in storage.
        cached = _original_cached

        # File changed - generate diff with stats
        if cached is not None:
            old_content = await cache.get_content(cached)
            diff_content = generate_diff(old_content, content)
            stats = diff_stats(old_content, content)
            diff_tokens = count_tokens(diff_content)

            if diff_tokens < tokens_original * 0.6:
                stats_msg = (
                    f"// Stats: +{stats['insertions']} -{stats['deletions']} "
                    f"~{stats['modifications']} lines, "
                    f"{stats['compression_ratio']:.1%} size\n"
                )
                result_content = (
                    f"// Diff for {path} (changed since cache):\n{stats_msg}{diff_content}"
                )
                if computed_embedding is None:
                    computed_embedding = await cache.get_embedding(content, path)
                await cache.refresh_path(
                    str(file_path),
                    content,
                    mtime,
                    computed_embedding,
                    embedding_path=path,
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
                )

    # Strategy 3: Semantic similarity
    if not cached and diff_mode and not force_full:
        if computed_embedding is None:
            computed_embedding = await cache.get_embedding(content, path)
        if computed_embedding:
            similar_path = await cache.find_similar(computed_embedding, str(file_path))
            if similar_path:
                similar_entry = await cache.get(similar_path)
                if similar_entry:
                    similar_content = await cache.get_content(similar_entry)
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
                        await cache.refresh_path(
                            str(file_path),
                            content,
                            mtime,
                            computed_embedding,
                            embedding_path=path,
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
                            semantic_match=similar_path,
                        )

    # Strategy 4 & 5: Full read (semantic summarization if large; uses pre-fetched embedding)
    truncated = False
    final_content = content

    if len(content) > max_size:
        # Use semantic summarization to preserve important content.
        # The entire summarize_semantic call (including its embed_fn callback)
        # MUST run in the executor — ONNX is not thread-safe and embed_fn
        # calls it synchronously.
        try:
            started = time.perf_counter()
            log_marker(logger, "summarize.begin", path=path, chars=len(content))

            def _summarize() -> str:
                from . import embed as _sync_embed  # noqa: PLC0415

                def embed_fn(text: str):
                    emb = _sync_embed(text)
                    if emb is None:
                        return None
                    return np.asarray(emb, dtype=np.float32)

                return summarize_semantic(content, max_size, embed_fn=embed_fn)

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
        await cache.refresh_path(
            str(file_path),
            content,
            mtime,
            computed_embedding,
            embedding_path=path,
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
        priority_ordered = [p for p in priority if p in set(paths)]
        remainder = [p for p in paths if p not in priority_set]
        remainder_sorted = sorted(remainder, key=lambda p: token_estimates[p])
        paths_sorted = priority_ordered + remainder_sorted
    else:
        paths_sorted = sorted(paths, key=lambda p: token_estimates[p])

    # Pre-scan: batch embed all new/changed files in a single model call.
    # This amortizes ONNX Runtime inference overhead from N calls → 1.
    # Files that are unchanged (cached + mtime match) are skipped since
    # smart_read won't need an embedding for them.
    _to_embed: list[tuple[str, str, str]] = []  # (original_path, resolved_str, content)
    for _path in paths_sorted:
        _resolved = Path(_path).expanduser().resolve()
        _cached = _cache_map.get(str(_resolved))
        if _cached and _resolved.exists():
            _file_mtime = _resolved.stat().st_mtime
            if _cached.mtime >= _file_mtime:
                continue  # unchanged — no embedding needed
            # Check content hash before assuming changed
            try:
                _raw_text = await aread_text(_resolved, executor=cache._io_executor)
                if hash_content(_raw_text) == _cached.content_hash:
                    await cache.update_mtime(str(_resolved), _file_mtime)
                    continue  # content unchanged — no embedding needed
            except Exception:  # nosec B110
                pass
        if not _resolved.exists() or not _resolved.is_file():
            continue  # will error in smart_read, skip pre-scan
        try:
            _raw = await aread_bytes(_resolved, cache._io_executor)
            if _is_binary_content(_raw):
                continue  # binary file — smart_read will raise ValueError
            _content = _raw.decode("utf-8")
            _to_embed.append((_path, str(_resolved), _content))
        except Exception:  # nosec B112 — pre-scan best-effort; smart_read handles real errors
            continue

    # Batch embed all candidates; fall back to per-file if anything fails
    _prefetched: dict[str, EmbeddingVector] = {}
    if _to_embed:
        _pairs = [(_rpath, _cnt) for _, _rpath, _cnt in _to_embed]
        _results = cache.get_embeddings_batch(_pairs)
        for (_opath, _rpath, _), _emb in zip(_to_embed, _results, strict=False):
            if _emb is not None:
                _prefetched[_rpath] = _emb

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
            _resolved_key = str(Path(path).expanduser().resolve())
            result = await smart_read(
                cache,
                path,
                diff_mode=diff_mode,
                force_full=False,
                _embedding=_prefetched.get(_resolved_key),
            )

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
