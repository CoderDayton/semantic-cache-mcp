"""Regression tests for BM25 search relevance scoring and FTS5 query safety.

Covers two fixed bugs in ``cache/search.py``:

1. ``similarity`` was always 0.0 — the normalizer guarded on ``max_score > 0``
   but FTS5 ``bm25()`` returns negative scores, so every result fell to 0.0.
2. Hyphenated / operator-bearing queries (``in-flight``, ``*``) silently
   returned nothing because the raw query was handed to FTS5 ``MATCH``.
"""

from __future__ import annotations

import time

import pytest

from semantic_cache_mcp.cache import SemanticCache
from semantic_cache_mcp.cache.search import (
    _normalize_relevance,
    _sanitize_fts_query,
    semantic_search,
)

# ---------------------------------------------------------------------------
# _normalize_relevance — bm25 scores are negative, best = most negative
# ---------------------------------------------------------------------------


class TestNormalizeRelevance:
    def test_empty(self) -> None:
        assert _normalize_relevance([]) == []

    def test_best_maps_to_one(self) -> None:
        # bm25 ascending: scores[0] is the most negative (best).
        out = _normalize_relevance([-2.5, -1.0, -0.5])
        assert out == [1.0, 0.4, 0.2]

    def test_single_result_is_one_not_zero(self) -> None:
        # The exact regression: a lone match used to normalize to 0.0.
        assert _normalize_relevance([-3.2]) == [1.0]

    def test_clamped_to_unit_interval(self) -> None:
        # A later score more negative than the first clamps at 1.0.
        out = _normalize_relevance([-1.0, -2.0])
        assert out == [1.0, 1.0]

    def test_positive_score_clamps_to_zero(self) -> None:
        # A non-negative score against a negative best would go negative; clamp.
        out = _normalize_relevance([-2.0, 1.0])
        assert out[0] == 1.0
        assert out[1] == 0.0

    def test_zero_best_is_degenerate_zero(self) -> None:
        assert _normalize_relevance([0.0, 0.0]) == [0.0, 0.0]


# ---------------------------------------------------------------------------
# _sanitize_fts_query — neutralize FTS5 operator syntax in free-text queries
# ---------------------------------------------------------------------------


class TestSanitizeFtsQuery:
    def test_plain_word(self) -> None:
        assert _sanitize_fts_query("needle") == '"needle"'

    def test_multiword_terms_are_anded(self) -> None:
        assert _sanitize_fts_query("drain shutdown") == '"drain" "shutdown"'

    def test_hyphen_becomes_adjacency_phrase(self) -> None:
        assert _sanitize_fts_query("in-flight") == '"in flight"'

    def test_underscore_identifier_preserved(self) -> None:
        assert _sanitize_fts_query("smart_read") == '"smart_read"'

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("*", ""),
            ("---", ""),
            ("   ", ""),
            ("NEAR(", '"NEAR"'),
            ('"unbalanced', '"unbalanced"'),
            ("a AND b", '"a" "AND" "b"'),
        ],
    )
    def test_operators_neutralized(self, query: str, expected: str) -> None:
        assert _sanitize_fts_query(query) == expected

    def test_unicode_word_chars_kept(self) -> None:
        assert _sanitize_fts_query("café") == '"café"'


# ---------------------------------------------------------------------------
# Integration — through semantic_search over a real SemanticCache
# ---------------------------------------------------------------------------


class TestSearchIntegration:
    async def test_top_result_similarity_is_one(self, semantic_cache: SemanticCache) -> None:
        now = time.time()
        await semantic_cache.put("/proj/dense.py", "needle needle needle filler\n", now)
        await semantic_cache.put("/proj/sparse.py", "needle filler filler filler\n", now)

        result = await semantic_search(semantic_cache, "needle", k=5)

        assert result.matches, "expected at least one match"
        assert result.matches[0].similarity == 1.0
        assert all(0.0 <= m.similarity <= 1.0 for m in result.matches)
        # Not the old always-zero behavior.
        assert any(m.similarity > 0.0 for m in result.matches)

    async def test_hyphenated_query_matches(self, semantic_cache: SemanticCache) -> None:
        now = time.time()
        await semantic_cache.put(
            "/proj/a.py", "the worker handles in-flight requests during drain\n", now
        )
        await semantic_cache.put("/proj/b.py", "unrelated content about widgets\n", now)

        result = await semantic_search(semantic_cache, "in-flight", k=5)

        paths = [m.path for m in result.matches]
        assert any(p.endswith("a.py") for p in paths), paths

    async def test_multiword_query_with_hyphen_not_poisoned(
        self, semantic_cache: SemanticCache
    ) -> None:
        now = time.time()
        await semantic_cache.put(
            "/proj/x.py", "drain shutdown waits for in-flight operations to finish\n", now
        )

        result = await semantic_search(semantic_cache, "drain shutdown in-flight", k=5)

        assert any(m.path.endswith("x.py") for m in result.matches)

    async def test_operator_only_query_is_empty_not_error(
        self, semantic_cache: SemanticCache
    ) -> None:
        now = time.time()
        await semantic_cache.put("/proj/y.py", "hello world content\n", now)

        # "*" alone sanitizes to an empty query -> no matches, no exception.
        result = await semantic_search(semantic_cache, "*", k=5)
        assert result.matches == []

        # A real term beside an operator still matches.
        result2 = await semantic_search(semantic_cache, "hello *", k=5)
        assert any(m.path.endswith("y.py") for m in result2.matches)
