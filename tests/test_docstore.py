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

    def test_bad_metadata_column_rejected(self, tmp_path: Path) -> None:
        s = _store(tmp_path)
        with pytest.raises(ValueError, match="identifier"):
            s.build_filter_clause({"path": "/a"}, metadata_column="metadata; DROP TABLE x --")

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


class TestMetadataProjections:
    """Text-free reads: every caller that only inspects metadata uses these.

    Selecting the text column to read one metadata field re-materializes the
    whole cached corpus, which is what `stats`, `search` and the eviction index
    were each doing on every call.
    """

    @staticmethod
    def _seed(store: DocStore) -> None:
        store.add_texts(
            ["", "chunk one", "chunk two", "solo file", "orphan"],
            [
                {"path": "/big", "is_parent": True, "total_chunks": 2, "tokens": 90},
                {"path": "/big", "chunk_index": 0, "total_chunks": 2, "tokens": 40},
                {"path": "/big", "chunk_index": 1, "total_chunks": 2, "tokens": 50},
                {"path": "/solo", "total_chunks": 1, "tokens": 7},
                {"path": "", "total_chunks": 1, "tokens": 3},
            ],
        )

    def test_get_metadata_matches_get_documents(self, tmp_path: Path) -> None:
        store = DocStore(tmp_path / "m.db")
        try:
            self._seed(store)
            expected = [(doc_id, meta) for doc_id, _text, meta in store.get_documents()]
            assert store.get_metadata() == expected
            filtered = [
                (doc_id, meta) for doc_id, _text, meta in store.get_documents({"path": "/big"})
            ]
            assert store.get_metadata({"path": "/big"}) == filtered
        finally:
            store.close()

    def test_get_document_ids_matches_get_documents(self, tmp_path: Path) -> None:
        store = DocStore(tmp_path / "i.db")
        try:
            self._seed(store)
            assert store.get_document_ids() == [d[0] for d in store.get_documents()]
        finally:
            store.close()

    def test_file_stats_matches_the_python_accounting(self, tmp_path: Path) -> None:
        store = DocStore(tmp_path / "s.db")
        try:
            self._seed(store)
            unique: set[str] = set()
            tokens = 0
            for _doc_id, _text, meta in store.get_documents():
                if meta.get("path"):
                    unique.add(meta["path"])
                if meta.get("is_parent", False) or meta.get("total_chunks", 1) == 1:
                    tokens += meta.get("tokens", 0)
            assert store.file_stats(
                path_key="path",
                tokens_key="tokens",
                is_parent_key="is_parent",
                total_chunks_key="total_chunks",
            ) == (len(unique), tokens)
            # A chunked file is counted once, through its parent — not per chunk.
            assert tokens == 90 + 7 + 3
        finally:
            store.close()

    def test_file_stats_on_an_empty_store(self, tmp_path: Path) -> None:
        store = DocStore(tmp_path / "e.db")
        try:
            assert store.file_stats(
                path_key="path",
                tokens_key="tokens",
                is_parent_key="is_parent",
                total_chunks_key="total_chunks",
            ) == (0, 0)
        finally:
            store.close()

    def test_distinct_paths_skips_parents_and_blanks(self, tmp_path: Path) -> None:
        store = DocStore(tmp_path / "p.db")
        try:
            self._seed(store)
            assert sorted(store.distinct_paths(path_key="path", is_parent_key="is_parent")) == [
                "/big",
                "/solo",
            ]
        finally:
            store.close()


class TestFtsOptimize:
    """A delete leaves an index entry behind; only a full merge discards it.

    FTS5 records a deletion as an entry rather than removing one, so a store
    that has evicted many files carries the vocabulary of every file it ever
    held. ``documents_fts_data`` is the index's own shadow table, so its row
    count measures how many segments survive without needing ``dbstat``.
    """

    @staticmethod
    def _segments(store: DocStore) -> int:
        # Single-threaded test, so raw-connection access needs no extra locking.
        return int(store.conn.execute("SELECT count(*) FROM documents_fts_data").fetchone()[0])

    @staticmethod
    def _churn(store: DocStore) -> list[int]:
        """Add 300 docs one commit at a time, then evict the first 200."""
        ids: list[int] = []
        for i in range(300):
            text = " ".join(f"t{i}w{j}" for j in range(60))
            ids.extend(store.add_texts([text], [{"path": f"/f{i}"}]))
        store.delete_by_ids(ids[:200])
        return ids

    def test_optimize_discards_delete_markers(self, tmp_path: Path) -> None:
        store = DocStore(tmp_path / "opt.db")
        try:
            ids = self._churn(store)
            before = self._segments(store)
            store.optimize()
            after = self._segments(store)
            assert after < before, f"optimize left {after} segments (was {before})"
            # Survivors stay searchable and evicted documents stay gone.
            assert len(store.keyword_search("t250w5", k=5)) == 1
            assert store.keyword_search("t10w5", k=5) == []
            assert store.count() == len(ids) - 200
        finally:
            store.close()

    def test_optimize_preserves_bm25_ranking(self, tmp_path: Path) -> None:
        store = DocStore(tmp_path / "rank.db")
        try:
            store.add_texts(
                ["needle needle needle filler", "needle filler filler filler"],
                [{"path": "/dense"}, {"path": "/sparse"}],
            )
            before = [(d.metadata["path"], s) for d, s in store.keyword_search("needle", k=5)]
            store.optimize()
            after = [(d.metadata["path"], s) for d, s in store.keyword_search("needle", k=5)]
            assert before == after
        finally:
            store.close()

    def test_optimize_is_safe_on_an_empty_store(self, tmp_path: Path) -> None:
        store = DocStore(tmp_path / "empty.db")
        try:
            store.optimize()
            assert store.count() == 0
        finally:
            store.close()

    def test_optimize_after_close_does_not_raise(self, tmp_path: Path) -> None:
        store = DocStore(tmp_path / "closed.db")
        store.add_texts(["alpha"], [{"path": "/a"}])
        store.close()
        store.optimize()  # the connection is gone; this must be a no-op

    def test_content_storage_close_runs_it(self, tmp_path: Path, monkeypatch) -> None:
        """Shutdown is the only path that pays for the merge."""
        from semantic_cache_mcp.storage.docstore import ContentStorage

        calls: list[str] = []
        real = DocStore.optimize
        monkeypatch.setattr(
            DocStore,
            "optimize",
            lambda self: (calls.append("optimize"), real(self))[1],
        )
        storage = ContentStorage(db_path=tmp_path / "cs.db")
        storage.close()
        assert calls == ["optimize"]


class TestFileReclaim:
    """Freeing pages and giving them back are two different things.

    SQLite retains freed pages and reuses them unless the database is in an
    auto-vacuum mode, so a cache that evicts and re-indexes for weeks sits at
    its high-water mark no matter how much of its content is dead.
    """

    AUTO_VACUUM_INCREMENTAL = 2

    @staticmethod
    def _mode(store: DocStore) -> int:
        return int(store.conn.execute("PRAGMA auto_vacuum").fetchone()[0])

    @staticmethod
    def _pages(store: DocStore) -> tuple[int, int]:
        """(page_count, freelist_count) — logical size and reclaimable pages."""
        return (
            int(store.conn.execute("PRAGMA page_count").fetchone()[0]),
            int(store.conn.execute("PRAGMA freelist_count").fetchone()[0]),
        )

    def test_new_store_is_created_in_incremental_mode(self, tmp_path: Path, caplog) -> None:
        """And gets there without a rewrite: the mode is set while the file is
        still empty, before the WAL pragma writes its first header page.
        """
        import logging

        with caplog.at_level(logging.INFO, logger="semantic_cache_mcp.storage.docstore._docstore"):
            store = DocStore(tmp_path / "new.db")
        try:
            assert self._mode(store) == self.AUTO_VACUUM_INCREMENTAL
            assert "auto-vacuum" not in caplog.text
        finally:
            store.close()

    def test_legacy_store_is_migrated_without_losing_data(self, tmp_path: Path, caplog) -> None:
        """A database written before 0.5.3 opens in mode 0 and must be rewritten."""
        import logging
        import sqlite3

        db_path = tmp_path / "legacy.db"
        legacy = sqlite3.connect(str(db_path))
        legacy.execute(
            "CREATE TABLE documents (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "text TEXT NOT NULL, metadata TEXT, parent_id INTEGER)"
        )
        legacy.execute("CREATE VIRTUAL TABLE documents_fts USING fts5(text)")
        legacy.execute(
            "INSERT INTO documents(text, metadata) VALUES ('survivor term', '{\"path\":\"/keep\"}')"
        )
        legacy.execute("INSERT INTO documents_fts(rowid, text) VALUES (1, 'survivor term')")
        legacy.commit()
        assert int(legacy.execute("PRAGMA auto_vacuum").fetchone()[0]) == 0
        legacy.close()

        with caplog.at_level(logging.INFO, logger="semantic_cache_mcp.storage.docstore._docstore"):
            store = DocStore(db_path)
        try:
            assert self._mode(store) == self.AUTO_VACUUM_INCREMENTAL
            assert "incremental auto-vacuum" in caplog.text  # it really was rewritten
            results = store.keyword_search("survivor", k=5)
            assert len(results) == 1
            assert results[0][0].metadata["path"] == "/keep"
            assert store.count() == 1
        finally:
            store.close()

    def test_the_mode_persists_so_a_reopen_never_pays_again(self, tmp_path: Path) -> None:
        """The mode lives in the database header, so the check is self-describing
        and needs no migration marker beside the file.
        """
        import sqlite3

        db_path = tmp_path / "twice.db"
        first = DocStore(db_path)
        first.add_texts(["alpha"], [{"path": "/a"}])
        first.close()

        # Read the persisted header directly — this is the value the reopen path
        # sees, and reading INCREMENTAL is exactly what makes it return early.
        raw = sqlite3.connect(str(db_path))
        try:
            assert int(raw.execute("PRAGMA auto_vacuum").fetchone()[0]) == (
                self.AUTO_VACUUM_INCREMENTAL
            )
        finally:
            raw.close()

        second = DocStore(db_path)
        try:
            assert self._mode(second) == self.AUTO_VACUUM_INCREMENTAL
            assert second.count() == 1
        finally:
            second.close()

    def test_reclaim_returns_pages_after_a_large_eviction(self, tmp_path: Path) -> None:
        store = DocStore(tmp_path / "reclaim.db")
        try:
            ids: list[int] = []
            for i in range(400):
                ids.extend(
                    store.add_texts(
                        [" ".join(f"t{i}w{j}" for j in range(80))], [{"path": f"/f{i}"}]
                    )
                )
            store.delete_by_ids(ids[:350])
            store.optimize()
            before_pages, before_free = self._pages(store)
            assert before_free > 0, "nothing was freed, so this proves nothing"

            store.reclaim_free_pages()
            after_pages, after_free = self._pages(store)
            assert after_pages < before_pages
            assert after_free == 0
            # The surviving documents are untouched by the rewrite.
            assert store.count() == len(ids) - 350
            assert len(store.keyword_search("t399w5", k=5)) == 1
        finally:
            store.close()

    def test_reclaim_is_a_no_op_with_nothing_to_give_back(self, tmp_path: Path) -> None:
        store = DocStore(tmp_path / "quiet.db")
        try:
            store.add_texts(["alpha"], [{"path": "/a"}])
            before = self._pages(store)
            store.reclaim_free_pages()
            assert self._pages(store) == before
        finally:
            store.close()

    def test_reclaim_after_close_does_not_raise(self, tmp_path: Path) -> None:
        store = DocStore(tmp_path / "closed2.db")
        store.add_texts(["alpha"], [{"path": "/a"}])
        store.close()
        store.reclaim_free_pages()

    def test_content_storage_close_reclaims(self, tmp_path: Path, monkeypatch) -> None:
        from semantic_cache_mcp.storage.docstore import ContentStorage

        calls: list[str] = []
        for name in ("optimize", "reclaim_free_pages"):
            real = getattr(DocStore, name)
            monkeypatch.setattr(
                DocStore,
                name,
                lambda self, _n=name, _r=real: (calls.append(_n), _r(self))[1],
            )
        storage = ContentStorage(db_path=tmp_path / "cs2.db")
        storage.close()
        # Order matters: the merge frees the pages this then hands back.
        assert calls == ["optimize", "reclaim_free_pages"]
