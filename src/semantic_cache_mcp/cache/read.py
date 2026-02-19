"""Smart read operations for the cache package."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import MAX_CONTENT_SIZE
from ..core import count_tokens, diff_stats, generate_diff, summarize_semantic, truncate_semantic
from ..types import BatchReadResult, FileReadSummary, ReadResult
from .store import SemanticCache

logger = logging.getLogger(__name__)

# DoS limits
MAX_BATCH_FILES = 50
MAX_BATCH_TOKENS = 200_000


def _fit_content_to_max_size(content: str, max_size: int, cache: SemanticCache) -> tuple[str, bool]:
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


def batch_smart_read(
    cache: SemanticCache,
    paths: list[str],
    max_total_tokens: int = 50000,
    priority: list[str] | None = None,
    diff_mode: bool = True,
) -> BatchReadResult:
    """Read multiple files with token budget, priority ordering, and unchanged detection.

    Args:
        cache: SemanticCache instance
        paths: List of file paths
        max_total_tokens: Token budget (capped at 200K)
        priority: Paths to read first (order preserved). Does NOT override budget.
        diff_mode: When False, always return full content (use after context compression).

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
            result = smart_read(cache, path, diff_mode=diff_mode, force_full=False)

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
