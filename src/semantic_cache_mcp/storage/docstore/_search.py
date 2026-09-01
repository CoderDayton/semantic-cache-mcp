"""BM25 query search for :class:`ContentStorage`.

BM25 keyword search (``search_by_query``) over cached content — the ranked-score
read side, sibling to the exact-match ``_grep`` subsystem.

Split out of the ``ContentStorage`` god-module: each function takes the storage
instance explicitly (``store``) instead of ``self``, so the whole search
subsystem lives in one place. ``ContentStorage`` keeps a thin delegating method
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
    from . import ContentStorage

logger = logging.getLogger(__name__)


async def search_by_query(
    store: ContentStorage,
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

    return dedupe_search_results(results, k, query=query)


def _match_preview(page_content: str, stored_preview: str | None, terms: list[str]) -> str:
    """A window around the first query term, falling back to the head.

    The stored preview is the file's first 200 characters — for source that is
    the module docstring and its imports, which does not say why the file
    matched and is text the follow-up `read` returns anyway. Centering on a
    query term spends the same budget on the part the caller is looking for.
    """
    lowered = page_content.lower()
    position = -1
    for term in terms:
        found = lowered.find(term)
        if found != -1 and (position == -1 or found < position):
            position = found

    if position == -1:
        return stored_preview or page_content[:_PREVIEW_CHARS]

    # Leave roughly a third of the window ahead of the term so the reader sees
    # what leads into it, and clamp to the content's bounds.
    start = max(0, position - _PREVIEW_CHARS // 3)
    return page_content[start : start + _PREVIEW_CHARS]


def _query_terms(query: str) -> list[str]:
    """Lowercased words worth locating, longest first so the most specific wins."""
    words = {word.strip('"*^-()').lower() for word in query.split()}
    return sorted((w for w in words if len(w) > 2), key=len, reverse=True)


def dedupe_search_results(
    results: list[tuple],
    k: int,
    query: str = "",
) -> list[tuple[str, str, float]]:
    """Deduplicate by path, keeping best score. Skips parent (empty) docs."""
    seen_paths: set[str] = set()
    matches: list[tuple[str, str, float]] = []
    terms = _query_terms(query)

    for doc, score in results:
        meta = doc.metadata
        if meta.get(_META_IS_PARENT, False):
            continue
        path = meta.get(_META_PATH, "")
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        preview = _match_preview(doc.page_content, meta.get(_META_PREVIEW), terms)
        matches.append((path, preview, float(score)))
        if len(matches) >= k:
            break

    return matches
