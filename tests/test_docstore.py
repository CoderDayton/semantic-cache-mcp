"""Direct unit tests for the vendored SQLite + FTS5 DocStore.

The DocStore SQL (BM25 keyword_search, metadata filter, FTS rowid sync) is the
load-bearing lifted code; these exercise it directly rather than through the
ContentStorage facade.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from semantic_cache_mcp.storage.docstore._docstore import AsyncDocStore, DocStore, Document


def _store(tmp_path: Path) -> DocStore:
    return DocStore(tmp_path / "docstore.db")


# ---------------------------------------------------------------------------
# add_texts / id alignment (the RETURNING-order MAJOR fix)
# ---------------------------------------------------------------------------


class TestAddAndSearch:
    def test_add_returns_ids(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        ids = s.add_texts(["alpha", "beta"], [{"path": "/a"}, {"path": "/b"}])
        assert len(ids) == 2
        assert all(isinstance(i, int) for i in ids)
        assert s.count() == 2

    def test_fts_rowid_aligns_with_text(self, tmp_path: Path) -> None:
        """Each unique term must retrieve the document that actually contains it.

        This is the regression guard for the FTS rowid<->text sync: if add_texts
        ever zipped ids to the wrong texts, keyword_search would return the wrong
        page_content for a match.
        """
        s = _store(tmp_path)
        s.add_texts(
            ["zebra unique_aaa", "llama unique_bbb", "otter unique_ccc"],
            [{"path": "/a"}, {"path": "/b"}, {"path": "/c"}],
        )
        for term, expect in (
            ("unique_aaa", "zebra"),
            ("unique_bbb", "llama"),
            ("unique_ccc", "otter"),
        ):
            results = s.keyword_search(term, k=5)
            assert len(results) == 1, term
            doc, score = results[0]
            assert isinstance(doc, Document)
            assert expect in doc.page_content
            assert isinstance(score, float)

    def test_keyword_search_best_first(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add_texts(
            ["needle needle needle filler", "needle filler filler filler"],
            [{"path": "/dense"}, {"path": "/sparse"}],
        )
        results = s.keyword_search("needle", k=5)
        assert len(results) == 2
        scores = [sc for _, sc in results]
        assert scores == sorted(scores)  # bm25 ascending == best-first

    def test_empty_query_returns_empty(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add_texts(["x"], [{"path": "/x"}])
        assert s.keyword_search("   ", k=5) == []

    @pytest.mark.parametrize("bad", ['"unbalanced', "*", "NEAR("])
    def test_malformed_query_raises_valueerror(self, tmp_path: Path, bad: str) -> None:
        s = _store(tmp_path)
        s.add_texts(["hello world"], [{"path": "/h"}])
        with pytest.raises(ValueError, match="full-text search query"):
            s.keyword_search(bad, k=5)


# ---------------------------------------------------------------------------
# get_documents + metadata filter
# ---------------------------------------------------------------------------


class TestFilter:
    def test_no_filter_returns_all_ordered(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add_texts(["a", "b", "c"], [{"path": "/a"}, {"path": "/b"}, {"path": "/c"}])
        docs = s.get_documents()
        assert [m["path"] for _, _, m in docs] == ["/a", "/b", "/c"]
        assert [i for i, _, _ in docs] == sorted(i for i, _, _ in docs)

    def test_scalar_filter(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add_texts(["a", "b"], [{"path": "/a"}, {"path": "/b"}])
        docs = s.get_documents({"path": "/b"})
        assert [m["path"] for _, _, m in docs] == ["/b"]

    def test_list_filter_in(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add_texts(["a", "b", "c"], [{"path": "/a"}, {"path": "/b"}, {"path": "/c"}])
        docs = s.get_documents({"path": ["/a", "/c"]})
        assert sorted(m["path"] for _, _, m in docs) == ["/a", "/c"]

    def test_bool_filter(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add_texts(
            ["p", "c"], [{"path": "/x", "is_parent": True}, {"path": "/x", "is_parent": False}]
        )
        docs = s.get_documents({"is_parent": True})
        assert len(docs) == 1 and docs[0][2]["is_parent"] is True

    def test_double_quote_key_rejected(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        with pytest.raises(ValueError, match="double-quote"):
            s.build_filter_clause({'bad"key': "x"})

    def test_keyword_search_honors_filter(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add_texts(["shared term", "shared term"], [{"path": "/keep"}, {"path": "/drop"}])
        results = s.keyword_search("shared", k=5, filter_dict={"path": "/keep"})
        assert len(results) == 1
        assert results[0][0].metadata["path"] == "/keep"


# ---------------------------------------------------------------------------
# update / delete / clear
# ---------------------------------------------------------------------------


class TestMutations:
    def test_update_metadata_shallow_merges(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        ids = s.add_texts(["a"], [{"path": "/a", "tokens": 10}])
        n = s.update_metadata([(ids[0], {"mtime": 5.0})])
        assert n == 1
        meta = s.get_documents({"path": "/a"})[0][2]
        assert meta == {"path": "/a", "tokens": 10, "mtime": 5.0}  # merged, not replaced

    def test_update_metadata_skips_missing_ids(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add_texts(["a"], [{"path": "/a"}])
        assert s.update_metadata([(9999, {"x": 1})]) == 0

    def test_delete_by_ids_removes_from_fts(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        ids = s.add_texts(["findme term"], [{"path": "/a"}])
        assert s.keyword_search("findme", k=5)
        removed = s.delete_by_ids(ids)
        assert removed == ids
        assert s.keyword_search("findme", k=5) == []  # FTS row gone too
        assert s.count() == 0

    def test_delete_returns_only_existing(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        ids = s.add_texts(["a"], [{"path": "/a"}])
        assert s.delete_by_ids([ids[0], 9999]) == ids

    def test_clear(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        s.add_texts(["a", "b"], [{"path": "/a"}, {"path": "/b"}])
        assert s.clear() == 2
        assert s.count() == 0
        assert s.keyword_search("a", k=5) == []


# ---------------------------------------------------------------------------
# parent/child + async adapter
# ---------------------------------------------------------------------------


class TestStructureAndAsync:
    def test_parent_child(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        pid = s.add_texts([""], [{"path": "/big", "is_parent": True}])[0]
        s.add_texts(
            ["chunk a", "chunk b"],
            [{"path": "/big", "chunk_index": 0}, {"path": "/big", "chunk_index": 1}],
            parent_ids=[pid, pid],
        )
        docs = s.get_documents({"path": "/big"})
        assert len(docs) == 3  # parent + 2 children

    async def test_async_round_trip(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        ex = ThreadPoolExecutor(max_workers=1)
        try:
            ads = AsyncDocStore(store, ex)
            ids = await ads.add_texts(["async term"], [{"path": "/a"}])
            assert len(ids) == 1
            assert await ads.count() == 1
            results = await ads.keyword_search("async", k=5, filter={"path": "/a"})
            assert results and results[0][0].metadata["path"] == "/a"
            docs = await ads.get_documents({"path": "/a"})
            assert len(docs) == 1
            await ads.save()
        finally:
            ex.shutdown(wait=True)
            store.close()
