"""Semantic search and similarity operations for the cache package."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ..core import count_tokens, diff_stats, generate_diff
from ..core.embeddings import embed_query
from ..core.similarity import cosine_similarity, top_k_from_quantized
from ..types import (
    DiffResult,
    GlobMatch,
    GlobResult,
    SearchMatch,
    SearchResult,
    SimilarFile,
    SimilarFilesResult,
)
from .store import SemanticCache

logger = logging.getLogger(__name__)

# DoS limits
MAX_SEARCH_K = 100
MAX_SEARCH_QUERY_LEN = 8000
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
