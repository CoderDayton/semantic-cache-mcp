"""A scoped search must still fill its result set.

The docstore is shared by every project the client has ever opened, so
``semantic_search`` restricts BM25 results to the requested directory — and
since scoping is the default, that is the common path, not an edge one. The
directory reaches SQL as a prefix clause, which is what keeps the result set
full: ``LIMIT`` applies after the ``WHERE``, so a filter applied afterwards
would let another project's files occupy slots that are then discarded,
returning a handful of matches, or none, while plenty existed.
"""

from __future__ import annotations

import time
from pathlib import Path

from semantic_cache_mcp.cache import SemanticCache
from semantic_cache_mcp.cache.search import semantic_search

_TERM = "authentication middleware token"


async def _seed(cache: SemanticCache, root: Path, count: int, marker: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        target = root / f"mod_{index}.py"
        target.write_text(f"# {_TERM} {marker}\ndef handler_{index}():\n    return {index}\n")
        await cache.put(str(target), target.read_text(), time.time())


async def test_a_scoped_search_is_not_crowded_out_by_another_project(
    tmp_path: Path,
) -> None:
    """120 out-of-scope files must not starve a 5-result search of 20 in-scope ones."""
    cache = SemanticCache(db_path=tmp_path / "cache.db")
    await _seed(cache, tmp_path / "other", 120, "other")
    await _seed(cache, tmp_path / "proj", 20, "proj")

    result = await semantic_search(cache, _TERM, k=5, directory=str(tmp_path / "proj"))

    assert len(result.matches) == 5, (
        f"asked for 5 in-scope matches, got {len(result.matches)} with 20 available"
    )
    assert all(m.path.startswith(str(tmp_path / "proj")) for m in result.matches)


async def test_a_scoped_search_returns_everything_it_has_when_short(
    tmp_path: Path,
) -> None:
    """Fewer in-scope files than k is not an error — it returns what exists."""
    cache = SemanticCache(db_path=tmp_path / "cache.db")
    await _seed(cache, tmp_path / "other", 120, "other")
    await _seed(cache, tmp_path / "proj", 3, "proj")

    result = await semantic_search(cache, _TERM, k=10, directory=str(tmp_path / "proj"))

    assert len(result.matches) == 3
    assert all(m.path.startswith(str(tmp_path / "proj")) for m in result.matches)


async def test_the_store_filters_by_prefix_before_it_ranks(tmp_path: Path) -> None:
    """The directory reaches SQL, so ``k`` in-scope rows come back in one query.

    Without a prefix clause the store can only rank globally and hand the
    caller whatever survives a Python-side filter, which is why scoping used to
    need an over-fetch at all.
    """
    cache = SemanticCache(db_path=tmp_path / "cache.db")
    await _seed(cache, tmp_path / "other", 120, "other")
    await _seed(cache, tmp_path / "proj", 20, "proj")

    rows = await cache._storage.search_by_query(
        query=_TERM, k=5, path_prefix=str(tmp_path / "proj")
    )

    assert len(rows) == 5
    assert all(path.startswith(str(tmp_path / "proj")) for path, _, _ in rows)


async def test_a_prefix_is_matched_literally_not_as_a_like_pattern(tmp_path: Path) -> None:
    """``_`` and ``%`` are LIKE wildcards; a directory named with one is not.

    ``pro_j`` would otherwise match ``proXj`` and leak another project's files
    into a scoped search — the exact leak the scoping exists to close.
    """
    cache = SemanticCache(db_path=tmp_path / "cache.db")
    await _seed(cache, tmp_path / "pro_j", 4, "underscore")
    await _seed(cache, tmp_path / "proXj", 4, "wildcard")

    rows = await cache._storage.search_by_query(
        query=_TERM, k=8, path_prefix=str(tmp_path / "pro_j")
    )

    assert rows, "the literal directory must still match its own files"
    assert all(Path(path).parent.name == "pro_j" for path, _, _ in rows)


async def test_an_unscoped_search_still_spans_the_whole_store(tmp_path: Path) -> None:
    """Passing no directory to the cache API keeps the old, global behaviour.

    The scoping default lives in the tool layer, where the client root is
    known; this function is the documented programmatic API and stays literal.
    """
    cache = SemanticCache(db_path=tmp_path / "cache.db")
    await _seed(cache, tmp_path / "other", 4, "other")
    await _seed(cache, tmp_path / "proj", 4, "proj")

    result = await semantic_search(cache, _TERM, k=8)

    roots = {Path(m.path).parent.name for m in result.matches}
    assert roots == {"other", "proj"}
