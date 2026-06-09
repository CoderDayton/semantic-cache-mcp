"""Keyword (BM25) search and file comparison operations."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from pathlib import Path

from ..core import count_tokens, diff_stats, generate_diff
from ..core.hashing import hash_content
from ..types import (
    CacheEntry,
    DiffResult,
    GlobMatch,
    GlobResult,
    SearchMatch,
    SearchResult,
)
from ..utils import aread_bytes, astat
from ._helpers import _is_binary_content, _suppress_large_diff
from .store import SemanticCache

logger = logging.getLogger(__name__)

# DoS limits
MAX_SEARCH_K = 100
MAX_SEARCH_QUERY_LEN = 8000
MAX_GLOB_MATCHES = 1000
GLOB_TIMEOUT_SECONDS = 5

# FTS5 reads -, *, :, ^, parentheses and the barewords AND/OR/NOT/NEAR as query
# operators. The user-facing search tool takes plain keywords, so a token like
# "in-flight" parses as `in NOT flight` and a stray "*" raises a syntax error —
# both surfacing as an empty result. Reduce each token to its word runs and
# quote them, turning every term into an FTS5 string literal that can never be
# read as an operator.
_FTS_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _sanitize_fts_query(query: str) -> str:
    """Turn a free-text query into a safe FTS5 ``MATCH`` expression.

    Each whitespace-separated chunk is reduced to its ``\\w+`` runs and
    re-emitted as a double-quoted FTS5 string. A chunk that splits into several
    runs (``in-flight`` -> ``in``, ``flight``) becomes an adjacency phrase
    ``"in flight"`` so the compound still matches the source text; a single run
    becomes a plain quoted term. Quoting makes every term a string literal, so
    no input reaches the FTS5 parser as an operator. Returns ``""`` when the
    query has no indexable characters (the caller then yields no matches).
    """
    chunks: list[str] = []
    for raw in query.split():
        words = _FTS_WORD_RE.findall(raw)
        if words:
            chunks.append('"' + " ".join(words) + '"')
    return " ".join(chunks)


def _normalize_relevance(scores: list[float]) -> list[float]:
    """Map FTS5 ``bm25()`` scores to a 0-1 relevance, best = 1.0.

    FTS5 ``bm25()`` returns *negative* numbers where the most relevant row is
    the most negative — the search SQL sorts ascending, so ``scores[0]`` is the
    best. A row's relevance is its magnitude relative to that best score:
    dividing two negatives yields a value in ``(0, 1]`` for well-ordered input.
    Results are clamped to ``[0, 1]`` to absorb a degenerate ordering or a
    non-negative score, and a zero or empty best maps everything to 0.0.

    The previous code guarded on ``max_score > 0``; bm25 scores are negative, so
    that branch forced every similarity to 0.0.
    """
    if not scores:
        return []
    best = scores[0]
    if best == 0:
        return [0.0] * len(scores)
    return [max(0.0, min(1.0, round(score / best, 4))) for score in scores]


async def _read_text_file(cache: SemanticCache, file_path: Path, display_path: str) -> str:
    """Read and decode a text file with the standard binary/UTF-8 guards."""

    raw_bytes = await aread_bytes(file_path, cache._io_executor)
    if _is_binary_content(raw_bytes):
        raise ValueError(
            f"Binary file not supported: {display_path}. Search only works with text files."
        )
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File is not valid UTF-8: {display_path}") from exc


async def _load_diff_input(
    cache: SemanticCache, file_path: Path, display_path: str
) -> tuple[CacheEntry | None, str, bool]:
    """Load diff input, reusing cache on mtime or content-hash matches."""

    cached = await cache.get(str(file_path))
    if cached is not None:
        file_mtime = (await astat(file_path, cache._io_executor)).st_mtime
        if cached.mtime >= file_mtime:
            return cached, await cache.get_content(cached), True

    content = await _read_text_file(cache, file_path, display_path)
    if cached is not None:
        file_mtime = (await astat(file_path, cache._io_executor)).st_mtime
        if hash_content(content) == cached.content_hash:
            await cache.update_mtime(str(file_path), file_mtime)
            return cached, content, True

    return cached, content, False


_SEARCH_CACHE_MAX_ENTRIES = 32


async def semantic_search(
    cache: SemanticCache,
    query: str,
    k: int = 10,
    directory: str | None = None,
) -> SearchResult:
    """Search cached files by keyword relevance using BM25 search."""
    # DoS protection
    k = max(1, min(k, MAX_SEARCH_K))
    query = query[:MAX_SEARCH_QUERY_LEN]

    # In-session result cache: identical (query, k, directory) tuples in
    # the same session skip the full BM25 round-trip.
    # Invalidated on every cache mutation via SemanticCache._bump_search_cache.
    cache_key = (query, k, directory)
    cached_result = cache._search_cache.get(cache_key)
    if cached_result is not None:
        cache._search_cache.move_to_end(cache_key)
        cache.metrics.record("search_cache_hit", None)
        return cached_result

    # Resolve directory for post-search filtering (is_relative_to is secure
    # against prefix attacks like /project vs /project_evil)
    resolved_dir: Path | None = None
    if directory:
        resolved_dir = Path(directory).expanduser().resolve()

    # BM25 keyword search via ContentStorage.
    # Request extra results when directory filtering will reduce the set
    storage = cache._storage
    search_k = k * 3 if resolved_dir else k
    # User queries are free-text keywords, not the FTS5 query DSL; sanitize so
    # operator characters are matched literally instead of returning nothing or
    # raising (see _sanitize_fts_query). An empty sanitized query yields no rows.
    results = await storage.search_by_query(
        query=_sanitize_fts_query(query),
        k=search_k,
    )

    if not results:
        stats = await storage.get_stats()
        total = stats.get("files_cached", 0)
        empty = SearchResult(query=query, matches=[], files_searched=0, cached_files=int(total))
        _store_search_result(cache, cache_key, empty)
        return empty

    # Build matches with directory filtering
    filtered: list[tuple[str, str, float]] = []
    for path, preview, score in results:
        if len(filtered) >= k:
            break
        # Secure directory filter: is_relative_to prevents prefix attacks
        if resolved_dir and not Path(path).is_relative_to(resolved_dir):
            continue
        filtered.append((path, preview, score))

    # Map raw bm25() scores to a 0-1 relevance (best result = 1.0) so callers
    # can judge matches without knowing FTS5 score internals.
    relevances = _normalize_relevance([score for _, _, score in filtered])

    filtered_paths = [path for path, _, _ in filtered]
    entries = await asyncio.gather(*(cache.get(p) for p in filtered_paths))
    entry_map = dict(zip(filtered_paths, entries, strict=False))

    matches: list[SearchMatch] = []
    for (path, preview, _score), similarity in zip(filtered, relevances, strict=True):
        entry = entry_map[path]
        tokens = entry.tokens if entry else 0
        matches.append(
            SearchMatch(
                path=path,
                similarity=similarity,
                tokens=tokens,
                preview=preview.replace("\n", " "),
            )
        )

    stats = await storage.get_stats()
    total = stats.get("files_cached", 0)

    result = SearchResult(
        query=query,
        matches=matches,
        files_searched=int(total),
        cached_files=int(total),
    )
    _store_search_result(cache, cache_key, result)
    return result


def _store_search_result(
    cache: SemanticCache,
    key: tuple[str, int, str | None],
    result: SearchResult,
) -> None:
    sc = cache._search_cache
    sc[key] = result
    sc.move_to_end(key)
    while len(sc) > _SEARCH_CACHE_MAX_ENTRIES:
        sc.popitem(last=False)


async def compare_files(
    cache: SemanticCache,
    path1: str,
    path2: str,
    context_lines: int = 3,
) -> DiffResult:
    """Compare two files using the cache. Returns a unified diff."""
    # Cap context_lines to prevent excessive diff output
    context_lines = max(0, min(context_lines, 50))

    file1 = Path(path1).expanduser().resolve()
    file2 = Path(path2).expanduser().resolve()

    # Existence and type checks before any I/O
    if not file1.is_file():
        raise FileNotFoundError(f"File not found: {path1}")
    if not file2.is_file():
        raise FileNotFoundError(f"File not found: {path2}")

    # Get content for both files (from cache or disk)
    content1: str
    content2: str
    from_cache1 = False
    from_cache2 = False

    # File 1
    cached1, content1, from_cache1 = await _load_diff_input(cache, file1, path1)

    # File 2
    cached2, content2, from_cache2 = await _load_diff_input(cache, file2, path2)

    # Generate diff (suppress if very large to avoid blowing up response tokens)
    diff_content = generate_diff(content1, content2, context_lines=context_lines)
    stats = diff_stats(content1, content2)
    full_tokens = count_tokens(content1) + count_tokens(content2)
    diff_content = _suppress_large_diff(diff_content, full_tokens) or ""

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
        from_cache=(from_cache1, from_cache2),
    )


async def glob_with_cache_status(
    cache: SemanticCache,
    pattern: str,
    directory: str = ".",
    cached_only: bool = False,
) -> GlobResult:
    """Find files by glob pattern with cache status.

    cached_only=True reduces noise on large repos.
    """
    dir_path = Path(directory).expanduser().resolve()

    matches: list[GlobMatch] = []
    cached_count = 0
    total_cached_tokens = 0

    deadline = time.monotonic() + GLOB_TIMEOUT_SECONDS

    def _collect_paths() -> tuple[list[Path], bool]:
        """Walk the glob synchronously up to MAX_GLOB_MATCHES or the deadline.

        Path.glob() does blocking stat() calls under the hood, so the entire
        walk runs on the IO executor instead of the event loop. Returns the
        (bounded) match list and a flag indicating whether the deadline cut
        the walk short.
        """
        out: list[Path] = []
        for entry in dir_path.glob(pattern):
            if len(out) >= MAX_GLOB_MATCHES:
                break
            if time.monotonic() > deadline:
                return out, True
            out.append(entry)
        return out, False

    loop = asyncio.get_running_loop()
    candidates, deadline_hit = await loop.run_in_executor(cache._io_executor, _collect_paths)
    if deadline_hit:
        logger.warning(f"Glob timed out after {GLOB_TIMEOUT_SECONDS}s")

    count = 0
    for file_path in candidates:
        if count >= MAX_GLOB_MATCHES:
            break
        if time.monotonic() > deadline:
            logger.warning(f"Glob timed out after {GLOB_TIMEOUT_SECONDS}s")
            break
        if not file_path.is_file():
            continue
        if file_path.is_symlink() and not file_path.resolve().is_relative_to(dir_path):
            continue

        count += 1
        path_str = str(file_path)
        mtime = (await astat(file_path, cache._io_executor)).st_mtime

        # Check cache status
        cached = await cache.get(path_str)
        is_cached = cached is not None
        tokens = cached.tokens if cached else None

        # Skip uncached files when cached_only is set
        if cached_only and not is_cached:
            continue

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
