"""The trigram index that narrows grep to the files that can match."""

from __future__ import annotations

from pathlib import Path

import pytest

from semantic_cache_mcp.storage.docstore import ContentStorage, _grep
from semantic_cache_mcp.storage.docstore._grams import gram_query, required_literals


@pytest.fixture
async def storage(tmp_path: Path):
    vs = ContentStorage(db_path=tmp_path / "cache.db")
    yield vs
    vs.close()


def _gram_rows(vs: ContentStorage) -> int:
    conn = vs._sync_collection.conn  # noqa: SLF001
    return conn.execute("SELECT COUNT(*) FROM gram_files").fetchone()[0]


class TestIndexBatches:
    async def test_a_large_first_grep_indexes_in_bounded_batches(
        self, storage, monkeypatch
    ) -> None:
        monkeypatch.setattr(_grep, "GREP_INDEX_BATCH_CHARS", 1_000)
        for i in range(10):
            await storage.put(f"/f{i}.py", f"file {i} " + "x" * 400 + "\n", mtime=1.0)

        batches: list[int] = []
        real = storage._collection.index_grams  # noqa: SLF001

        async def spy(entries):
            batches.append(sum(len(text) for _, _, text in entries))
            await real(entries)

        monkeypatch.setattr(storage._collection, "index_grams", spy)  # noqa: SLF001
        await storage.grep("index_all_now", fixed_string=True)
        assert len(batches) > 1
        assert all(size <= 1_000 for size in batches)
        assert _gram_rows(storage) == 10


class TestRequiredLiterals:
    def test_a_fixed_string_is_one_literal(self) -> None:
        assert required_literals("a|b*(c", fixed_string=True) == ["a|b*(c"]

    def test_an_escape_class_ends_a_literal(self) -> None:
        assert required_literals(r"\bparse\(", fixed_string=False) == ["parse("]

    def test_metacharacters_split_literals(self) -> None:
        assert required_literals(r"foo.bar", fixed_string=False) == ["foo", "bar"]
        assert required_literals(r"error: code \d+", fixed_string=False) == ["error: code "]
        assert required_literals(r"ab+cd", fixed_string=False) == ["ab", "cd"]

    def test_an_escaped_punctuation_mark_is_literal(self) -> None:
        assert required_literals(r"a\.b", fixed_string=False) == ["a.b"]

    def test_patterns_with_optional_parts_have_no_required_literal(self) -> None:
        for pattern in (r"alpha|beta", r"abc?d", r"ab*c", r"x[abc]y", r"a{2}bcd"):
            assert required_literals(pattern, fixed_string=False) is None, pattern

    def test_a_numeric_escape_gives_up(self) -> None:
        assert required_literals(r"\x41BC", fixed_string=False) is None
        assert required_literals(r"(ab)\1cd", fixed_string=False) is None


class TestGramQuery:
    def test_each_literal_is_a_required_phrase(self) -> None:
        assert gram_query(["parse(", "ab", "x = 1"], case_sensitive=True) == (
            '"parse(" AND "x = 1"'
        )

    def test_quotes_are_doubled(self) -> None:
        assert gram_query(['say "hi"'], case_sensitive=True) == '"say ""hi"""'

    def test_short_literals_give_no_query(self) -> None:
        assert gram_query(["ab", "cd"], case_sensitive=True) is None

    def test_case_insensitive_queries_use_ascii_runs_only(self) -> None:
        assert gram_query(["café au lait"], case_sensitive=False) == '"caf" AND " au lait"'
        assert gram_query(["éèê"], case_sensitive=False) is None
        assert gram_query(["éèê"], case_sensitive=True) == '"éèê"'

    def test_literals_are_prefolded(self) -> None:
        assert gram_query(["ıf x"], case_sensitive=True) == '"if x"'


class TestGrepUsesTheIndex:
    async def test_a_word_boundary_regex_finds_its_match(self, storage) -> None:
        await storage.put("/a.py", "def parse(x):\n    pass\n", mtime=1.0)
        results = await storage.grep(r"\bparse")
        assert [r["path"] for r in results] == ["/a.py"]

    async def test_a_second_grep_loads_only_the_candidates(self, storage, monkeypatch) -> None:
        await storage.put("/x.py", "the needle is here\n", mtime=1.0)
        await storage.put("/y.py", "nothing to see\n", mtime=1.0)
        await storage.put("/z.py", "nor here\n", mtime=1.0)
        assert [r["path"] for r in await storage.grep("needle")] == ["/x.py"]

        loaded: list[list[str]] = []
        real = _grep.load_files

        async def spy(store, paths):
            loaded.append(sorted(paths))
            return await real(store, paths)

        monkeypatch.setattr(_grep, "load_files", spy)
        assert [r["path"] for r in await storage.grep("needle")] == ["/x.py"]
        assert loaded == [["/x.py"]]

    async def test_a_file_changed_since_indexing_is_still_searched(self, storage) -> None:
        await storage.put("/x.py", "the needle is here\n", mtime=1.0)
        await storage.put("/y.py", "nothing to see\n", mtime=1.0)
        await storage.grep("needle")
        await storage.put("/y.py", "a needle appeared\n", mtime=2.0)
        results = await storage.grep("needle")
        assert sorted(r["path"] for r in results) == ["/x.py", "/y.py"]

    async def test_scattered_trigrams_are_not_a_candidate(self, storage, monkeypatch) -> None:
        await storage.put("/hit.py", "parse it\n", mtime=1.0)
        await storage.put("/miss.py", "par ars rse\n", mtime=1.0)
        await storage.grep("zzz_index_everything")

        loaded: list[list[str]] = []
        real = _grep.load_files

        async def spy(store, paths):
            loaded.append(sorted(paths))
            return await real(store, paths)

        monkeypatch.setattr(_grep, "load_files", spy)
        assert [r["path"] for r in await storage.grep("parse")] == ["/hit.py"]
        assert loaded == [["/hit.py"]]

    async def test_a_chunked_file_is_matched_across_chunk_boundaries(self, storage) -> None:
        body = "".join(f"line {i} filler text to pad things out\n" for i in range(600))
        await storage.put("/big.txt", body + "the needle_at_the_end\n", mtime=1.0)
        await storage.grep("needle_at")
        results = await storage.grep("needle_at_the_end")
        assert [r["path"] for r in results] == ["/big.txt"]

    async def test_dotless_i_matches_case_insensitively_once_indexed(self, storage) -> None:
        await storage.put("/t.py", "ıf x\n", mtime=1.0)
        await storage.put("/u.py", "nothing\n", mtime=1.0)
        await storage.grep("zzz_index_everything")
        results = await storage.grep("IF X", fixed_string=True, case_sensitive=False)
        assert [r["path"] for r in results] == ["/t.py"]

    async def test_a_deleted_path_leaves_no_index_row(self, storage) -> None:
        await storage.put("/x.py", "the needle is here\n", mtime=1.0)
        await storage.grep("needle")
        assert _gram_rows(storage) == 1
        await storage.delete_path("/x.py")
        assert _gram_rows(storage) == 0

    async def test_clear_drops_the_index(self, storage) -> None:
        await storage.put("/x.py", "the needle is here\n", mtime=1.0)
        await storage.grep("needle")
        await storage.clear()
        assert _gram_rows(storage) == 0

    async def test_indexing_a_path_deleted_mid_grep_leaves_no_row(self, storage) -> None:
        await storage.put("/x.py", "the needle is here\n", mtime=1.0)
        head = await storage.get("/x.py")
        await storage.delete_path("/x.py")
        await storage._collection.index_grams(  # noqa: SLF001
            [("/x.py", head.content_hash, "the needle is here\n")]
        )
        assert _gram_rows(storage) == 0

    async def test_indexing_a_superseded_hash_leaves_no_row(self, storage) -> None:
        await storage.put("/x.py", "old needle\n", mtime=1.0)
        old = (await storage.get("/x.py")).content_hash
        await storage.put("/x.py", "new needle\n", mtime=2.0)
        await storage._collection.index_grams([("/x.py", old, "old needle\n")])  # noqa: SLF001
        assert _gram_rows(storage) == 0

    async def test_many_candidates_load_in_bounded_batches(self, storage, monkeypatch) -> None:
        for i in range(3):
            await storage.put(f"/f{i}.py", "needle\n", mtime=1.0)
        monkeypatch.setattr(_grep, "GREP_LOAD_BATCH", 2)
        sizes: list[int] = []
        real = storage._collection.get_documents  # noqa: SLF001

        async def spy(filter_dict=None):
            sizes.append(len(filter_dict["path"]))
            return await real(filter_dict=filter_dict)

        monkeypatch.setattr(storage._collection, "get_documents", spy)  # noqa: SLF001
        files = await _grep.load_files(storage, ["/f0.py", "/f1.py", "/f2.py"])
        assert sorted(files) == ["/f0.py", "/f1.py", "/f2.py"]
        assert sizes == [2, 1]

    async def test_an_unreadable_gram_table_disables_the_index(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.storage.docstore._docstore import DocStore

        store = DocStore(tmp_path / "d.db")
        conn = store.conn
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "UPDATE sqlite_master SET sql = replace(sql, 'contentless_delete', 'bogus_option') "
            "WHERE name = 'grams'"
        )
        conn.commit()
        store.close()
        reopened = DocStore(tmp_path / "d.db")
        try:
            assert reopened.grams_enabled is False
            reopened.forget_paths(None)
        finally:
            reopened.close()

    async def test_grep_works_without_trigram_support(self, storage) -> None:
        storage._sync_collection.grams_enabled = False  # noqa: SLF001
        await storage.put("/x.py", "the needle is here\n", mtime=1.0)
        await storage.put("/y.py", "nothing\n", mtime=1.0)
        assert [r["path"] for r in await storage.grep("needle")] == ["/x.py"]
        assert _gram_rows(storage) == 0
