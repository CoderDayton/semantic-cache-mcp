"""BM25 query search for :class:`VectorStorage`.

BM25 keyword search (``search_by_query``) over cached content — the ranked-score
read side, sibling to the exact-match ``_grep`` subsystem.

Split out of the ``VectorStorage`` god-module: each function takes the storage
instance explicitly (``store``) instead of ``self``, so the whole search
subsystem lives in one place. ``VectorStorage`` keeps a thin delegating method
(``search_by_query``) for the symbol its callers and tests depend on.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from . import (
    _META_IS_PARENT,
    _META_PATH,
    _META_PREVIEW,
    _PREVIEW_CHARS,
)

if TYPE_CHECKING:
    from . import VectorStorage

logger = logging.getLogger(__name__)


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
