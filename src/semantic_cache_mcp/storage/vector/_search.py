"""Similarity and query search for :class:`VectorStorage`.

Vector similarity (``find_similar``/``find_similar_multi``), BM25 keyword
search (``search_by_query``), and hybrid RRF fusion (``search_hybrid``) over
cached content — the ranked-score read side, sibling to the exact-match
``_grep`` subsystem.

Split out of the ``VectorStorage`` god-module: each function takes the storage
instance explicitly (``store``) instead of ``self``, so the whole search
subsystem lives in one place. ``VectorStorage`` keeps thin delegating methods
for the symbols its callers and tests depend on (``find_similar``,
``find_similar_multi``, ``search_by_query``, ``search_hybrid``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ...config import SIMILARITY_THRESHOLD
from . import (
    _META_CHUNK_INDEX,
    _META_HAS_EMBEDDING,
    _META_IS_PARENT,
    _META_PATH,
    _META_PREVIEW,
    _META_TOTAL_CHUNKS,
    _PREVIEW_CHARS,
)

if TYPE_CHECKING:
    from ...types import EmbeddingVector
    from . import VectorStorage

logger = logging.getLogger(__name__)


async def find_similar(
    store: VectorStorage, embedding: EmbeddingVector, exclude_path: str | None = None
) -> str | None:
    """Return the most similar cached file path above SIMILARITY_THRESHOLD, or None."""
    if store._closed:
        return None

    try:
        results = await store._collection.similarity_search(
            query=embedding,
            k=10,  # Get candidates, then filter
        )
    except Exception as e:
        logger.warning(f"Similarity search failed: {e}")
        return None

    if not results:
        return None

    for doc, distance in results:
        # COSINE distance in simplevecdb: 0 = identical, 2 = opposite
        # Convert to similarity: 1 - (distance / 2) maps [0,2] → [1,0]
        similarity = 1.0 - (distance / 2.0)
        candidate_path = doc.metadata.get(_META_PATH)
        has_emb = doc.metadata.get(_META_HAS_EMBEDDING, False)
        # Defense-in-depth: chunk children carry zero embeddings; a query
        # with near-zero magnitude could match them as false positives if
        # `has_embedding` is ever stale (schema migration, partial write).
        is_chunk_child = (
            doc.metadata.get(_META_CHUNK_INDEX, -1) >= 0
            and not doc.metadata.get(_META_IS_PARENT, False)
            and doc.metadata.get(_META_TOTAL_CHUNKS, 1) > 1
        )

        is_match = (
            has_emb
            and not is_chunk_child
            and candidate_path
            and candidate_path != exclude_path
            and similarity >= SIMILARITY_THRESHOLD
        )
        if is_match:
            logger.debug(f"Similar file found: {candidate_path} (similarity={similarity:.3f})")
            return candidate_path

    return None


async def find_similar_multi(
    store: VectorStorage,
    embedding: EmbeddingVector,
    exclude_path: str | None = None,
    k: int = 5,
    threshold: float | None = None,
) -> list[tuple[str, float]]:
    """Return up to k (path, similarity) tuples above threshold."""
    if store._closed:
        return []
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD

    try:
        results = await store._collection.similarity_search(
            query=embedding,
            k=k * 3,  # Over-fetch to account for filtering
        )
    except Exception as e:
        logger.warning(f"Similarity search failed: {e}")
        return []

    matches: list[tuple[str, float]] = []
    seen_paths: set[str] = set()

    for doc, distance in results:
        similarity = 1.0 - (distance / 2.0)
        candidate_path = doc.metadata.get(_META_PATH)
        has_emb = doc.metadata.get(_META_HAS_EMBEDDING, False)
        is_chunk_child = (
            doc.metadata.get(_META_CHUNK_INDEX, -1) >= 0
            and not doc.metadata.get(_META_IS_PARENT, False)
            and doc.metadata.get(_META_TOTAL_CHUNKS, 1) > 1
        )

        if (
            has_emb
            and not is_chunk_child
            and candidate_path
            and candidate_path != exclude_path
            and candidate_path not in seen_paths
            and similarity >= threshold
        ):
            seen_paths.add(candidate_path)
            matches.append((candidate_path, similarity))

            if len(matches) >= k:
                break

    return matches


async def search_by_query(
    store: VectorStorage,
    query: str,
    k: int = 5,
    filter: dict | None = None,
) -> list[tuple[str, str, float]]:
    """BM25 keyword search over cached content. FTS5 syntax supported."""
    if store._closed:
        return []
    try:
        results = await store._collection.keyword_search(query, k=k * 2, filter=filter)
    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")
        return []

    return dedupe_search_results(results, k)


async def search_hybrid(
    store: VectorStorage,
    query: str,
    embedding: EmbeddingVector | None = None,
    k: int = 5,
    filter: dict | None = None,
) -> list[tuple[str, str, float]]:
    """Hybrid BM25 + vector search with RRF fusion.

    Falls back to keyword-only if no embedding provided.
    """
    if store._closed:
        return []

    try:
        results = await store._collection.hybrid_search(
            query,
            k=k * 2,
            filter=filter,
            query_vector=embedding,
        )
    except Exception as e:
        logger.warning(f"Hybrid search failed: {e}")
        return []

    return dedupe_search_results(results, k)


def dedupe_search_results(
    results: list[tuple],
    k: int,
) -> list[tuple[str, str, float]]:
    """Deduplicate by path, keeping best score. Skips parent (empty) docs."""
    seen_paths: set[str] = set()
    matches: list[tuple[str, str, float]] = []
    _preview_chars = _PREVIEW_CHARS

    for doc, score in results:
        meta = doc.metadata
        if meta.get(_META_IS_PARENT, False):
            continue
        path = meta.get(_META_PATH, "")
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        preview = meta.get(_META_PREVIEW) or doc.page_content[:_preview_chars]
        matches.append((path, preview, float(score)))
        if len(matches) >= k:
            break

    return matches
