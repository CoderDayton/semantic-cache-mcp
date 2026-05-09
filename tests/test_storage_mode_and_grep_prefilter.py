"""Tests for StorageMode metadata tagging and grep BM25 prefilter.

Covers:
  * `_META_STORAGE_MODE` is set to `chunked` for chunked files,
    `single_doc` for small files, and `single_doc_fallback` is selectable.
  * `_grep_prefilter_query` extracts useful tokens from literals and regex,
    quotes them as phrases (so reserved words don't break the FTS5 query),
    and returns None when no usable tokens exist.
  * `grep()` produces the same result set whether prefilter is engaged or
    bypassed (the fallback path).
"""

from __future__ import annotations

from pathlib import Path

from semantic_cache_mcp.storage.vector import _META_STORAGE_MODE, StorageMode, VectorStorage


def _make_vs(tmp_path: Path) -> VectorStorage:
    return VectorStorage(db_path=tmp_path / "vec.db")


class TestStorageModeTagging:
    async def test_small_file_tagged_single_doc(self, tmp_path: Path) -> None:
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/small.txt", "hello\n", mtime=1.0)
            docs = await vs._find_docs_by_path("/small.txt")  # noqa: SLF001
            assert docs, "expected at least one stored doc"
            modes = {meta.get(_META_STORAGE_MODE) for _, meta, _ in docs}
            assert modes == {StorageMode.SINGLE_DOC.value}
        finally:
            vs.close()

    async def test_large_file_tagged_chunked(self, tmp_path: Path) -> None:
        vs = _make_vs(tmp_path)
        try:
            big = ("abcdefghij" * 1000) + "\n"  # ~10KB > CHUNK_THRESHOLD
            await vs.put("/big.txt", big, mtime=1.0)
            docs = await vs._find_docs_by_path("/big.txt")  # noqa: SLF001
            modes = {meta.get(_META_STORAGE_MODE) for _, meta, _ in docs}
            # Both parent and children carry CHUNKED.
            assert modes == {StorageMode.CHUNKED.value}
        finally:
            vs.close()


class TestGrepPrefilterTokenExtraction:
    def test_literal_tokens_phrase_quoted(self) -> None:
        # 5-char tokens — pass the gate (≥ 4 chars).
        q = VectorStorage._grep_prefilter_query(  # noqa: SLF001
            "hello world", fixed_string=True
        )
        assert q == '"hello" "world"'

    def test_short_reserved_words_skip_prefilter_entirely(self) -> None:
        """Short FTS5 reserved words (NOT/OR/AND ≤ 3 chars) gate out the prefilter.

        Tokens shorter than _GREP_PREFILTER_MIN_TOKEN_LEN never reach FTS5,
        so the boolean-operator interpretation risk is moot — full scan handles
        the query.
        """
        q = VectorStorage._grep_prefilter_query(  # noqa: SLF001
            "NOT OR AND", fixed_string=True
        )
        assert q is None

    def test_engaged_tokens_are_phrase_quoted(self) -> None:
        """When the prefilter engages, every token must be phrase-quoted.

        Phrase quoting protects against any FTS5 reserved word (current or
        future) that happens to land in a long-enough token, and against
        chars like ``"`` and ``*`` that are FTS5 syntax.
        """
        q = VectorStorage._grep_prefilter_query(  # noqa: SLF001
            "configure deploy", fixed_string=True
        )
        assert q == '"configure" "deploy"'

    def test_regex_with_literal_substring_extracts(self) -> None:
        # tokens "error" (5) and "code" (4) — both pass the ≥ 4 gate.
        q = VectorStorage._grep_prefilter_query(  # noqa: SLF001
            r"error: code \d+", fixed_string=False
        )
        assert q == '"error" "code"'

    def test_regex_with_only_short_tokens_returns_none(self) -> None:
        # tokens "abc" (3) and "xyz" (3) — both below ≥ 4 gate.
        q = VectorStorage._grep_prefilter_query(  # noqa: SLF001
            r"abc.*xyz", fixed_string=False
        )
        assert q is None

    def test_authoritative_empty_threshold(self) -> None:
        # max token length determines whether empty BM25 is trusted.
        assert VectorStorage._grep_empty_is_authoritative("identifier") is True  # noqa: SLF001
        assert VectorStorage._grep_empty_is_authoritative("nicel") is False  # noqa: SLF001
        assert VectorStorage._grep_empty_is_authoritative("[A-Z]+") is False  # noqa: SLF001

    def test_regex_with_no_literal_runs_returns_none(self) -> None:
        q = VectorStorage._grep_prefilter_query(  # noqa: SLF001
            r"[A-Z][a-z]+", fixed_string=False
        )
        assert q is None

    def test_short_substring_filtered(self) -> None:
        # All single chars + escapes — nothing of length >= 2.
        q = VectorStorage._grep_prefilter_query(  # noqa: SLF001
            r"a.b.c", fixed_string=False
        )
        assert q is None


class TestGrepEndToEndUsesPrefilter:
    async def test_literal_match_via_prefilter(self, tmp_path: Path) -> None:
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/x.py", "needle is here\nfoo bar\n", mtime=1.0)
            await vs.put("/y.py", "no match in here\n", mtime=1.0)
            results = await vs.grep("needle", fixed_string=True)
            paths = [r["path"] for r in results]
            assert paths == ["/x.py"]
        finally:
            vs.close()

    async def test_regex_with_literal_run_uses_prefilter(self, tmp_path: Path) -> None:
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/log.txt", "WARN: code 42\nINFO: ok\n", mtime=1.0)
            results = await vs.grep(r"WARN: code \d+")
            assert any(r["path"] == "/log.txt" for r in results)
        finally:
            vs.close()

    async def test_regex_without_literal_falls_back(self, tmp_path: Path) -> None:
        # Pattern has no usable token; full-scan fallback must still find it.
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/code.py", "Hello\n", mtime=1.0)
            results = await vs.grep(r"[A-Z][a-z]+")
            assert any(r["path"] == "/code.py" for r in results)
        finally:
            vs.close()

    async def test_no_match_short_circuits_via_prefilter(self, tmp_path: Path) -> None:
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/a.txt", "alpha beta\n", mtime=1.0)
            results = await vs.grep("zzzzzznotpresent", fixed_string=True)
            assert results == []
        finally:
            vs.close()

    async def test_grep_finds_short_substring_inside_identifier(self, tmp_path: Path) -> None:
        """3-char patterns must NOT engage the prefilter (substring-vs-token).

        Regression for the issue where grep('foo') would silently miss
        matches inside identifiers like 'needlefoo' because BM25's whole-
        token semantics doesn't index 'foo' as a sub-string.
        """
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/x.py", "needlefoo = 1\n", mtime=1.0)
            results = await vs.grep("foo", fixed_string=True)
            assert [r["path"] for r in results] == ["/x.py"]
        finally:
            vs.close()

    async def test_grep_finds_medium_substring_via_fallback(self, tmp_path: Path) -> None:
        """5-char tokens engage prefilter but empty BM25 is not authoritative.

        Tokens in the [4, 7] length range trigger the prefilter, but if BM25
        returns no candidates the dispatcher must fall back to a full scan
        because a substring may still hide inside a longer document token.
        """
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/y.py", "mynicelongword = 1\n", mtime=1.0)
            results = await vs.grep("nicel", fixed_string=True)
            assert [r["path"] for r in results] == ["/y.py"]
        finally:
            vs.close()

    async def test_grep_long_pattern_empty_short_circuits(self, tmp_path: Path) -> None:
        """Long tokens (≥ 8 chars) make BM25's empty result authoritative."""
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/z.py", "alpha beta gamma\n", mtime=1.0)
            # Pattern is long enough that no plausible document token contains
            # it as a substring; BM25 returning 0 is trusted.
            results = await vs.grep("zzznotpresent", fixed_string=True)
            assert results == []
        finally:
            vs.close()

    async def test_path_filter_combined_with_prefilter(self, tmp_path: Path) -> None:
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/repo/src/a.py", "shared = 1\n", mtime=1.0)
            await vs.put("/repo/tests/a.py", "shared = 2\n", mtime=1.0)
            results = await vs.grep("shared", fixed_string=True, path="src/*.py")
            assert [r["path"] for r in results] == ["/repo/src/a.py"]
        finally:
            vs.close()
