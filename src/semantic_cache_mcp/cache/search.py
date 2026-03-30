"""Semantic search and similarity operations."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from ..core import count_tokens, diff_stats, generate_diff
from ..core.embeddings import embed_query
from ..core.hashing import hash_content
from ..core.similarity import cosine_similarity
from ..logger import log_marker
from ..types import (
    CacheEntry,
    DiffResult,
    EmbeddingVector,
    GlobMatch,
    GlobResult,
    SearchMatch,
    SearchResult,
    SimilarFile,
    SimilarFilesResult,
)
from ..utils import aread_bytes, astat
from ._helpers import _is_binary_content, _suppress_large_diff
from .store import SemanticCache

logger = logging.getLogger(__name__)

# DoS limits
MAX_SEARCH_K = 100
MAX_SEARCH_QUERY_LEN = 8000
MAX_SIMILAR_K = 50
MAX_GLOB_MATCHES = 1000
GLOB_TIMEOUT_SECONDS = 5


async def _read_text_file(cache: SemanticCache, file_path: Path, display_path: str) -> str:
    """Read and decode a text file with the standard binary/UTF-8 guards."""

    raw_bytes = await aread_bytes(file_path, cache._io_executor)
    if _is_binary_content(raw_bytes):
        raise ValueError(f"File is binary and cannot be processed: {display_path}")
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"File is not valid UTF-8: {display_path}") from exc


async def _load_diff_input(
    cache: SemanticCache, file_path: Path, display_path: str
) -> tuple[CacheEntry | None, str, EmbeddingVector | None, bool]:
    """Load diff input, reusing cache on mtime or content-hash matches."""

    cached = await cache.get(str(file_path))
    if cached is not None:
        file_mtime = (await astat(file_path, cache._io_executor)).st_mtime
        if cached.mtime >= file_mtime:
            return cached, await cache.get_content(cached), None, True

    content = await _read_text_file(cache, file_path, display_path)
    if cached is not None:
        file_mtime = (await astat(file_path, cache._io_executor)).st_mtime
        if hash_content(content) == cached.content_hash:
            await cache.update_mtime(str(file_path), file_mtime)
            return cached, content, None, True

    embedding = await cache.get_embedding(content, str(file_path))
    return cached, content, embedding, False


async def semantic_search(
    cache: SemanticCache,
    query: str,
    k: int = 10,
    directory: str | None = None,
) -> SearchResult:
    """Search cached files by semantic meaning using hybrid BM25+vector search."""
    # DoS protection
    k = max(1, min(k, MAX_SEARCH_K))
    query = query[:MAX_SEARCH_QUERY_LEN]

    # Embed query for vector component of hybrid search.
    # MUST go through executor — ONNX is not thread-safe.
    loop = asyncio.get_running_loop()
    started = time.perf_counter()
    log_marker(logger, "embed.query.begin", chars=len(query))
    query_embedding = await loop.run_in_executor(cache._io_executor, embed_query, query)
    log_marker(
        logger,
        "embed.query.end",
        ok=query_embedding is not None,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )

    # Resolve directory for post-search filtering (is_relative_to is secure
    # against prefix attacks like /project vs /project_evil)
    resolved_dir: Path | None = None
    if directory:
        resolved_dir = Path(directory).expanduser().resolve()

    # Use hybrid search (BM25 + vector) via VectorStorage
    # Request extra results when directory filtering will reduce the set
    storage = cache._storage
    search_k = k * 3 if resolved_dir else k
    results = await storage.search_hybrid(
        query=query,
        embedding=query_embedding,
        k=search_k,
    )

    if not results:
        stats = await storage.get_stats()
        total = stats.get("files_cached", 0)
        return SearchResult(query=query, matches=[], files_searched=0, cached_files=int(total))

    # Build matches with directory filtering
    filtered: list[tuple[str, str, float]] = []
    for path, preview, score in results:
        if len(filtered) >= k:
            break
        # Secure directory filter: is_relative_to prevents prefix attacks
        if resolved_dir and not Path(path).is_relative_to(resolved_dir):
            continue
        filtered.append((path, preview, score))

    # Normalize scores to 0–1 range (best result = 1.0) so LLMs can
    # judge relevance without knowing RRF score internals.
    max_score = filtered[0][2] if filtered else 1.0
    matches: list[SearchMatch] = []
    for path, preview, score in filtered:
        entry = await cache.get(path)
        tokens = entry.tokens if entry else 0
        normalized = round(score / max_score, 4) if max_score > 0 else 0.0
        matches.append(
            SearchMatch(
                path=path,
                similarity=normalized,
                tokens=tokens,
                preview=preview.replace("\n", " "),
            )
        )

    stats = await storage.get_stats()
    total = stats.get("files_cached", 0)

    return SearchResult(
        query=query,
        matches=matches,
        files_searched=int(total),
        cached_files=int(total),
    )


async def compare_files(
    cache: SemanticCache,
    path1: str,
    path2: str,
    context_lines: int = 3,
) -> DiffResult:
    """Compare two files using cache. Returns diff and semantic similarity score."""
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
    emb1 = None
    emb2 = None
    from_cache1 = False
    from_cache2 = False

    # File 1
    cached1, content1, emb1, from_cache1 = await _load_diff_input(cache, file1, path1)

    # File 2
    cached2, content2, emb2, from_cache2 = await _load_diff_input(cache, file2, path2)

    # Generate diff (suppress if very large to avoid blowing up response tokens)
    diff_content = generate_diff(content1, content2, context_lines=context_lines)
    stats = diff_stats(content1, content2)
    full_tokens = count_tokens(content1) + count_tokens(content2)
    diff_content = _suppress_large_diff(diff_content, full_tokens) or ""

    # Compute semantic similarity between embeddings (normalized)
    similarity = 0.0
    sim_embedding1 = cached1.embedding if cached1 and cached1.embedding else None
    sim_embedding2 = cached2.embedding if cached2 and cached2.embedding else None
    if not sim_embedding1 and not from_cache1:
        sim_embedding1 = emb1
    if not sim_embedding2 and not from_cache2:
        sim_embedding2 = emb2
    if sim_embedding1 and sim_embedding2:
        # Normalize to proper cosine similarity in [0, 1] range
        raw_sim = cosine_similarity(sim_embedding1, sim_embedding2)
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


async def find_similar_files(
    cache: SemanticCache,
    path: str,
    k: int = 5,
) -> SimilarFilesResult:
    """Find files semantically similar to the given file using HNSW search."""
    k = max(1, min(k, MAX_SIMILAR_K))
    file_path = Path(path).expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    cached = await cache.get(str(file_path))
    source_tokens = 0
    source_embedding: EmbeddingVector | None

    if cached is not None:
        file_mtime = (await astat(file_path, cache._io_executor)).st_mtime
        if cached.mtime >= file_mtime:
            source_tokens = cached.tokens
            if cached.embedding is not None:
                source_embedding = cached.embedding
            else:
                cached_content = await cache.get_content(cached)
                source_embedding = await cache.get_embedding(cached_content, str(file_path))
        else:
            content = await _read_text_file(cache, file_path, path)
            if hash_content(content) == cached.content_hash:
                await cache.update_mtime(str(file_path), file_mtime)
                source_tokens = cached.tokens
                source_embedding = cached.embedding or await cache.get_embedding(
                    content, str(file_path)
                )
            else:
                source_tokens = count_tokens(content)
                source_embedding = await cache.get_embedding(content, str(file_path))
    else:
        content = await _read_text_file(cache, file_path, path)
        source_tokens = count_tokens(content)
        source_embedding = await cache.get_embedding(content, str(file_path))

    if source_embedding is None:
        return SimilarFilesResult(
            source_path=str(file_path),
            source_tokens=source_tokens,
            similar_files=[],
            files_searched=0,
        )

    # Use VectorStorage HNSW search
    storage = cache._storage
    results = await storage.find_similar_multi(
        embedding=source_embedding,
        exclude_path=str(file_path),
        k=k,
    )

    similar_files: list[SimilarFile] = []
    for sim_path, sim_score in results:
        entry = await cache.get(sim_path)
        tokens = entry.tokens if entry else 0
        similar_files.append(
            SimilarFile(
                path=sim_path,
                similarity=round(sim_score, 4),
                tokens=tokens,
            )
        )

    stats = await storage.get_stats()
    total = stats.get("files_cached", 0)

    return SimilarFilesResult(
        source_path=str(file_path),
        source_tokens=source_tokens,
        similar_files=similar_files,
        files_searched=int(total),
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

    count = 0
    for file_path in dir_path.glob(pattern):
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
