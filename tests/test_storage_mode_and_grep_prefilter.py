"""Tests for StorageMode metadata tagging and grep BM25 prefilter.

Covers:
  * `_META_STORAGE_MODE` is set to `chunked` for chunked files,
    `single_doc` for small files, and `single_doc_fallback` is selectable.
  * `_grep_required_tokens` extracts the mandatory literal tokens a pattern
    needs, and returns None when the pattern cannot be soundly prefiltered.
  * `grep()` produces the same result set whether the prefilter is engaged
    or bypassed (the full-scan fallback path).
"""

from __future__ import annotations

import json
from pathlib import Path

from semantic_cache_mcp.storage.docstore import (
    _META_ACCESS_HISTORY,
    _META_STORAGE_MODE,
    ContentStorage,
    StorageMode,
)


def _make_vs(tmp_path: Path) -> ContentStorage:
    return ContentStorage(db_path=tmp_path / "vec.db")


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


class TestGrepRequiredTokens:
    def test_literal_tokens_extracted(self) -> None:
        toks = ContentStorage._grep_required_tokens(  # noqa: SLF001
            "hello world", fixed_string=True
        )
        assert toks == ["hello", "world"]

    def test_short_tokens_only_returns_none(self) -> None:
        # NOT(3) OR(2) AND(3) — none reach the >= 4 length gate.
        assert (
            ContentStorage._grep_required_tokens("NOT OR AND", fixed_string=True)  # noqa: SLF001
            is None
        )

    def test_regex_with_literal_run_extracts(self) -> None:
        # '+' and ':' are safe metacharacters — the literal tokens stay mandatory.
        toks = ContentStorage._grep_required_tokens(  # noqa: SLF001
            r"error: code \d+", fixed_string=False
        )
        assert toks == ["error", "code"]

    def test_regex_alternation_returns_none(self) -> None:
        # '|' makes a token non-mandatory; the prefilter must not engage.
        assert (
            ContentStorage._grep_required_tokens(  # noqa: SLF001
                r"alpha|betagamma", fixed_string=False
            )
            is None
        )

    def test_regex_optional_quantifiers_return_none(self) -> None:
        # '*', '?' and '{' all allow a token to be absent from a match.
        for pat in (r"abcd.*wxyz", r"abcde?x", r"abcd{2,3}"):
            assert (
                ContentStorage._grep_required_tokens(pat, fixed_string=False)  # noqa: SLF001
                is None
            )

    def test_regex_no_literal_runs_returns_none(self) -> None:
        assert (
            ContentStorage._grep_required_tokens(  # noqa: SLF001
                r"[A-Z][a-z]+", fixed_string=False
            )
            is None
        )

    def test_unsafe_metachars_are_literal_in_fixed_string(self) -> None:
        # In fixed-string mode '|' is a literal char — tokens stay mandatory.
        toks = ContentStorage._grep_required_tokens(  # noqa: SLF001
            "alpha|betagamma", fixed_string=True
        )
        assert toks == ["alpha", "betagamma"]

    def test_regex_character_class_returns_none(self) -> None:
        # A character class matches one char, but a >= 4-char alphanumeric
        # run inside it ([abcd], [aeiou], [[:alpha:]]) would be extracted as
        # a bogus mandatory token — the prefilter must decline these.
        for pat in (r"x[abcd]y", r"vowel[aeiou]end", r"[[:alpha:]]word"):
            assert (
                ContentStorage._grep_required_tokens(pat, fixed_string=False)  # noqa: SLF001
                is None
            ), pat


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

    async def test_regex_character_class_does_not_miss_match(self, tmp_path: Path) -> None:
        """A regex character class must not be mistaken for a literal token.
        'xay' matches r'x[abcd]y'; the prefilter must full-scan rather than
        treat 'abcd' as mandatory and short-circuit to an empty result.
        """
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/hit.py", "xay\n", mtime=1.0)
            results = await vs.grep(r"x[abcd]y")
            assert [r["path"] for r in results] == ["/hit.py"]
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

    async def test_grep_finds_token_inside_camelcase_identifier(self, tmp_path: Path) -> None:
        """grep('function') must find a file whose only occurrence is inside a
        camelCase compound identifier. FTS5's unicode61 tokenizer keeps
        'functionHelper' whole, so a raw BM25 whole-token match would miss it.
        """
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/camel.py", "const functionHelper = 1\n", mtime=1.0)
            results = await vs.grep("function", fixed_string=True)
            assert [r["path"] for r in results] == ["/camel.py"]
        finally:
            vs.close()

    async def test_grep_regex_finds_token_inside_camelcase(self, tmp_path: Path) -> None:
        """A metacharacter-free regex token must also match inside a compound
        identifier ('register' inside 'registerHandler')."""
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/camel.py", "registerHandler()\n", mtime=1.0)
            results = await vs.grep(r"register")
            assert [r["path"] for r in results] == ["/camel.py"]
        finally:
            vs.close()

    async def test_grep_regex_alternation_is_sound(self, tmp_path: Path) -> None:
        """A regex with alternation must not require every branch — the
        prefilter must fall back to a full scan rather than AND the tokens.
        """
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/a.py", "alpha only here\n", mtime=1.0)
            results = await vs.grep(r"alpha|betagamma")
            assert [r["path"] for r in results] == ["/a.py"]
        finally:
            vs.close()

    async def test_sound_candidates_uses_vocab_not_full_scan(self, tmp_path: Path) -> None:
        """_grep_sound_candidates returns exact paths via the FTS5 vocabulary
        expansion — a non-None result proves the BM25 prefilter (not the
        full-scan fallback) resolved the camelCase substring.
        """
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/camel.py", "const functionHelper = 1\n", mtime=1.0)
            candidates = await vs._grep_sound_candidates(  # noqa: SLF001
                "function", fixed_string=True, path_filter=None
            )
            assert candidates == ["/camel.py"]
        finally:
            vs.close()


class TestHasCachedPathsUnder:
    async def test_no_cached_files_under_path(self, tmp_path: Path) -> None:
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/repo/src/a.py", "x = 1\n", mtime=1.0)
            assert await vs.has_cached_paths_under("uncached/sub/") is False
            assert await vs.has_cached_paths_under("src/a.py") is True
        finally:
            vs.close()

    async def test_none_filter_matches_anything(self, tmp_path: Path) -> None:
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/repo/src/a.py", "x = 1\n", mtime=1.0)
            assert await vs.has_cached_paths_under(None) is True
        finally:
            vs.close()

    async def test_empty_cache_returns_false(self, tmp_path: Path) -> None:
        vs = _make_vs(tmp_path)
        try:
            assert await vs.has_cached_paths_under(None) is False
            assert await vs.has_cached_paths_under("anywhere") is False
        finally:
            vs.close()


class TestRecordAccessCorruptHistory:
    """record_access is awaited bare on the cache-hit read path — a corrupt
    persisted access_history value must not crash that read.
    """

    async def test_record_access_survives_unparseable_history(self, tmp_path: Path) -> None:
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/repo/a.py", "x = 1\n", mtime=1.0)
            # Corrupt the persisted history out from under record_access.
            docs = await vs._find_docs_by_path("/repo/a.py")  # noqa: SLF001
            await vs._collection.update_metadata(  # noqa: SLF001
                [(doc_id, {_META_ACCESS_HISTORY: "}{ not json"}) for doc_id, _m, _t in docs]
            )
            # Must not raise.
            await vs.record_access("/repo/a.py")
            # History was repaired to a valid JSON list with the new access.
            docs_after = await vs._find_docs_by_path("/repo/a.py")  # noqa: SLF001
            history = json.loads(docs_after[0][1][_META_ACCESS_HISTORY])
            assert isinstance(history, list)
            assert len(history) == 1
        finally:
            vs.close()

    async def test_record_access_drops_non_numeric_entries(self, tmp_path: Path) -> None:
        vs = _make_vs(tmp_path)
        try:
            await vs.put("/repo/b.py", "y = 2\n", mtime=1.0)
            docs = await vs._find_docs_by_path("/repo/b.py")  # noqa: SLF001
            await vs._collection.update_metadata(  # noqa: SLF001
                [(doc_id, {_META_ACCESS_HISTORY: '[1.0, "bad", 3.0]'}) for doc_id, _m, _t in docs]
            )
            await vs.record_access("/repo/b.py")
            docs_after = await vs._find_docs_by_path("/repo/b.py")  # noqa: SLF001
            history = json.loads(docs_after[0][1][_META_ACCESS_HISTORY])
            # 1.0 and 3.0 kept, "bad" dropped, plus the new access timestamp.
            assert all(isinstance(t, int | float) for t in history)
            assert len(history) == 3
        finally:
            vs.close()
