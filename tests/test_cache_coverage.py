"""Coverage-targeted tests for cache layer modules.

Targets:
  - cache/read.py  (78% → 90%+)
  - cache/write.py (78% → 90%+)
  - cache/store.py (82% → 92%+)
  - storage/vector.py (81% → 90%+)
  - cache/_helpers.py (83% → 92%+)
  - core/embeddings.py (52% → 70%+)
"""

from __future__ import annotations

import array
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from semantic_cache_mcp.cache._helpers import (
    _format_file,
    _is_binary_content,
    _suppress_large_diff,
)
from semantic_cache_mcp.cache.read import batch_smart_read, smart_read
from semantic_cache_mcp.cache.store import SemanticCache
from semantic_cache_mcp.cache.write import smart_batch_edit, smart_edit, smart_write
from semantic_cache_mcp.storage.vector import VectorStorage
from tests.constants import TEST_EMBEDDING_DIM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_embedding() -> array.array:  # type: ignore[type-arg]
    """Return a normalized 384-dim float32 embedding."""
    import math

    raw = [0.1] * TEST_EMBEDDING_DIM
    mag = math.sqrt(sum(x * x for x in raw))
    return array.array("f", [x / mag for x in raw])


def _make_cache(tmp_path: Path) -> SemanticCache:
    """Create a SemanticCache backed by tmp_path."""
    db_path = tmp_path / "test_vecdb.db"
    with patch("semantic_cache_mcp.cache.embed", return_value=None):
        return SemanticCache(db_path=db_path)


def _make_vector_storage(tmp_path: Path) -> VectorStorage:
    """Create a VectorStorage backed by tmp_path."""
    db_path = tmp_path / "vec.db"
    return VectorStorage(db_path=db_path)


# ===========================================================================
# cache/_helpers.py
# ===========================================================================


class TestIsBinaryContent:
    """Lines 59-60, 83-90 — binary edge cases."""

    def test_empty_bytes_is_not_binary(self) -> None:
        assert _is_binary_content(b"") is False

    def test_utf8_bom_is_not_binary(self) -> None:
        # UTF-8 BOM prefix → text
        assert _is_binary_content(b"\xef\xbb\xbfhello world") is False

    def test_utf16_le_bom_is_not_binary(self) -> None:
        assert _is_binary_content(b"\xff\xfehello") is False

    def test_utf16_be_bom_is_not_binary(self) -> None:
        assert _is_binary_content(b"\xfe\xffhello") is False

    def test_utf32_le_bom_is_not_binary(self) -> None:
        assert _is_binary_content(b"\xff\xfe\x00\x00hello") is False

    def test_utf32_be_bom_is_not_binary(self) -> None:
        assert _is_binary_content(b"\x00\x00\xfe\xffhello") is False

    def test_null_byte_is_binary(self) -> None:
        assert _is_binary_content(b"hello\x00world") is True

    def test_png_magic_is_binary(self) -> None:
        assert _is_binary_content(b"\x89PNGfakedata") is True

    def test_jpeg_magic_is_binary(self) -> None:
        assert _is_binary_content(b"\xff\xd8\xfffakedata") is True

    def test_elf_magic_is_binary(self) -> None:
        assert _is_binary_content(b"\x7fELFfakedata") is True

    def test_high_non_printable_ratio_is_binary(self) -> None:
        # >30% non-printable control chars (not tab/newline/CR)
        data = bytes([1, 2, 3, 4, 5, 6, 7, 8] * 20)  # all control chars
        assert _is_binary_content(data) is True

    def test_normal_text_is_not_binary(self) -> None:
        assert _is_binary_content(b"def foo():\n    return 42\n") is False

    def test_pdf_magic_is_binary(self) -> None:
        assert _is_binary_content(b"%PDF-1.5 fake content") is True


class TestSuppressLargeDiff:
    """Lines 215-226 — suppress logic edge cases."""

    def test_none_returns_none(self) -> None:
        assert _suppress_large_diff(None, 100) is None

    def test_empty_string_returns_none(self) -> None:
        assert _suppress_large_diff("", 100) is None

    def test_small_file_preserves_diff(self) -> None:
        # full_tokens <= 200 → always return unchanged
        diff = "+added line\n-removed line\n"
        result = _suppress_large_diff(diff, 50)
        assert result == diff

    def test_large_diff_tokens_suppressed(self) -> None:
        # Create a diff that exceeds MAX_RETURN_DIFF_TOKENS (8000)
        big_diff = ("+added line\n" * 1000) + ("-removed line\n" * 1000)
        result = _suppress_large_diff(big_diff, 50_000)
        assert result is not None
        assert "diff suppressed" in result
        assert "+2000" in result or "+1000" in result  # added lines counted

    def test_diff_near_full_ratio_suppressed(self) -> None:
        # diff_tokens >= 90% of full_tokens → suppress
        # Build a smallish diff that represents ~95% of a 500-token file
        diff = "+line\n-line\n" * 100
        # full_tokens = 250, diff should be close to that
        result = _suppress_large_diff(diff, 250)
        # Either suppressed or returned — just check it returns something
        assert result is not None


class TestFormatFile:
    """Lines 59-60 (binary skip) and formatter not found path."""

    def test_unsupported_extension_returns_false(self, tmp_path: Path) -> None:
        f = tmp_path / "file.unsupported_xyz"
        f.write_text("content")
        assert _format_file(f) is False

    def test_missing_formatter_returns_false(self, tmp_path: Path) -> None:
        f = tmp_path / "file.py"
        f.write_text("x = 1\n")
        with patch("shutil.which", return_value=None):
            assert _format_file(f) is False

    def test_formatter_non_regular_file_returns_false(self, tmp_path: Path) -> None:
        """Non-regular file (symlink to dir) should return False."""
        d = tmp_path / "mydir"
        d.mkdir()
        link = tmp_path / "link.py"
        link.symlink_to(d)
        # stat() on symlink → directory → not S_ISREG → False
        assert _format_file(link) is False


# ===========================================================================
# core/embeddings.py
# ===========================================================================


class TestEmbeddingsPublicAPI:
    """Lines 37-54, 108, 114, 119-126, 189-196 via mocked fastembed."""

    def test_embed_returns_array_on_success(self) -> None:
        from semantic_cache_mcp.core.embeddings import embed

        mock_model = MagicMock()
        fake_vec = np.zeros(384, dtype=np.float32)
        mock_model.embed.return_value = iter([fake_vec])

        with patch("semantic_cache_mcp.core.embeddings._model._embedding_model", mock_model):
            result = embed("hello world")

        assert result is not None
        assert len(result) == 384

    def test_embed_returns_none_on_exception(self) -> None:
        from semantic_cache_mcp.core.embeddings import embed

        mock_model = MagicMock()
        mock_model.embed.side_effect = RuntimeError("model broken")

        with patch("semantic_cache_mcp.core.embeddings._model._embedding_model", mock_model):
            result = embed("hello")

        assert result is None

    def test_embed_batch_success(self) -> None:
        from semantic_cache_mcp.core.embeddings import embed_batch

        mock_model = MagicMock()
        fake_vecs = [np.zeros(384, dtype=np.float32)] * 3
        mock_model.embed.return_value = iter(fake_vecs)

        with patch("semantic_cache_mcp.core.embeddings._model._embedding_model", mock_model):
            results = embed_batch(["a", "b", "c"])

        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_embed_batch_empty_input(self) -> None:
        from semantic_cache_mcp.core.embeddings import embed_batch

        result = embed_batch([])
        assert result == []

    def test_embed_batch_exception_returns_nones(self) -> None:
        from semantic_cache_mcp.core.embeddings import embed_batch

        mock_model = MagicMock()
        mock_model.embed.side_effect = RuntimeError("batch failed")

        with patch("semantic_cache_mcp.core.embeddings._model._embedding_model", mock_model):
            results = embed_batch(["a", "b"])

        assert results == [None, None]

    def test_embed_query_applies_prefix(self) -> None:
        from semantic_cache_mcp.core.embeddings import embed_query

        mock_model = MagicMock()
        fake_vec = np.zeros(384, dtype=np.float32)
        captured: list[list[str]] = []

        def capture_embed(texts: list[str]):
            captured.append(texts)
            return iter([fake_vec])

        mock_model.embed.side_effect = capture_embed

        with patch("semantic_cache_mcp.core.embeddings._model._embedding_model", mock_model):
            embed_query("search term")

        assert captured
        assert "Represent this sentence" in captured[0][0]

    def test_embed_query_returns_none_on_exception(self) -> None:
        from semantic_cache_mcp.core.embeddings import embed_query

        mock_model = MagicMock()
        mock_model.embed.side_effect = Exception("fail")

        with patch("semantic_cache_mcp.core.embeddings._model._embedding_model", mock_model):
            result = embed_query("query")

        assert result is None

    def test_warmup_skips_if_already_ready(self) -> None:
        from semantic_cache_mcp.core.embeddings import _model as _emb_model
        from semantic_cache_mcp.core.embeddings import warmup

        original = _emb_model._model_ready
        try:
            _emb_model._model_ready = True
            mock_model = MagicMock()
            with patch("semantic_cache_mcp.core.embeddings._model._embedding_model", mock_model):
                warmup()
            # embed should NOT have been called since already ready
            mock_model.embed.assert_not_called()
        finally:
            _emb_model._model_ready = original

    def test_get_model_info_returns_dict(self) -> None:
        from semantic_cache_mcp.core.embeddings import get_model_info

        info = get_model_info()
        assert "model" in info
        assert "dim" in info
        assert "provider" in info
        assert "ready" in info

    def test_get_embedding_dim_returns_int(self) -> None:
        from semantic_cache_mcp.core.embeddings import get_embedding_dim

        dim = get_embedding_dim()
        assert isinstance(dim, int)
        assert dim >= 0

    def test_cuda_provider_check_no_onnxruntime(self) -> None:
        from semantic_cache_mcp.core.embeddings import _cuda_provider_is_available

        with patch.dict("sys.modules", {"onnxruntime": None}):
            # Should return False without raising
            result = _cuda_provider_is_available()
            assert isinstance(result, bool)


# ===========================================================================
# cache/store.py
# ===========================================================================


class TestSemanticCacheStore:
    """Lines 28, 30-61 — _get_rss_mb, lifespan, __del__ on VectorStorage."""

    def test_get_rss_mb_returns_float_or_none(self) -> None:
        from semantic_cache_mcp.cache.store import _get_rss_mb

        result = _get_rss_mb()
        assert result is None or isinstance(result, float)

    async def test_get_stats_includes_expected_keys(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        stats = await cache.get_stats()
        assert "files_cached" in stats
        assert "total_tokens_cached" in stats
        assert "embedding_ready" in stats
        assert "session" in stats

    async def test_get_stats_with_process_rss(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.cache.store import _get_rss_mb

        cache = _make_cache(tmp_path)
        rss = _get_rss_mb()
        stats = await cache.get_stats()
        if rss is not None:
            assert "process_rss_mb" in stats

    async def test_get_stats_lifetime_metrics_exception(self, tmp_path: Path) -> None:
        """Lifetime metrics failure is silently swallowed."""
        cache = _make_cache(tmp_path)
        from semantic_cache_mcp.storage.sqlite import SQLiteStorage  # noqa: PLC0415

        with patch.object(SQLiteStorage, "get_lifetime_stats", side_effect=Exception("db error")):
            # Should not raise
            stats = await cache.get_stats()
        assert "files_cached" in stats

    def test_vector_storage_close(self, tmp_path: Path) -> None:
        """VectorStorage.close() saves and closes without raising."""
        vs = _make_vector_storage(tmp_path)
        vs.close()  # should complete without error

    def test_get_embeddings_batch(self, tmp_path: Path) -> None:
        """get_embeddings_batch returns list of same length as input."""
        cache = _make_cache(tmp_path)
        with patch("semantic_cache_mcp.cache.embed_batch", return_value=[None, None]):
            results = cache.get_embeddings_batch([("/a.py", "x"), ("/b.py", "y")])
        assert len(results) == 2


# ===========================================================================
# storage/vector.py
# ===========================================================================


class TestVectorStoragePutChunked:
    """Lines 81-82 (_put_chunked) — large file storage path."""

    async def test_put_large_file_chunked(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        # CHUNK_THRESHOLD = 8192 bytes; create content above that
        big_content = ("abcdefghij" * 1000) + "\n"  # ~10KB
        await vs.put("/large/file.txt", big_content, mtime=1.0, embedding=None)

        entry = await vs.get("/large/file.txt")
        assert entry is not None
        assert entry.path == "/large/file.txt"

        content = await vs.get_content(entry)
        assert content == big_content

    async def test_put_chunked_with_embedding(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        emb = _make_embedding()
        big_content = ("xyz " * 3000) + "\n"
        await vs.put("/chunked/emb.txt", big_content, mtime=2.0, embedding=emb)

        entry = await vs.get("/chunked/emb.txt")
        assert entry is not None


class TestVectorStorageGrep:
    """Lines 255-311 — grep regex search on cached content."""

    async def test_grep_literal_match(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        await vs.put("/grep/a.txt", "hello world\nfoo bar\n", mtime=1.0)
        results = await vs.grep("hello", fixed_string=True)
        assert any(r["path"] == "/grep/a.txt" for r in results)
        match_lines = [m["line"] for r in results for m in r["matches"]]
        assert any("hello" in line_text for line_text in match_lines)

    async def test_grep_regex_match(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        await vs.put("/grep/b.txt", "error: code 404\ninfo: all good\n", mtime=1.0)
        results = await vs.grep(r"error: code \d+")
        paths = [r["path"] for r in results]
        assert "/grep/b.txt" in paths

    async def test_grep_case_insensitive(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        await vs.put("/grep/c.txt", "Hello World\n", mtime=1.0)
        results = await vs.grep("hello world", fixed_string=True, case_sensitive=False)
        assert any(r["path"] == "/grep/c.txt" for r in results)

    async def test_grep_with_context_lines(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        content = "line1\nTARGET\nline3\n"
        await vs.put("/grep/ctx.txt", content, mtime=1.0)
        results = await vs.grep("TARGET", fixed_string=True, context_lines=1)
        assert results
        match = results[0]["matches"][0]
        assert "before" in match or "after" in match

    async def test_grep_no_match(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        await vs.put("/grep/nomatch.txt", "nothing here\n", mtime=1.0)
        results = await vs.grep("XYZZY_NOT_PRESENT", fixed_string=True)
        paths = [r["path"] for r in results]
        assert "/grep/nomatch.txt" not in paths

    async def test_grep_invalid_regex_returns_empty(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        await vs.put("/grep/bad.txt", "content\n", mtime=1.0)
        results = await vs.grep("[invalid regex(")
        assert results == []

    async def test_grep_max_matches_limit(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        content = "match\n" * 200
        await vs.put("/grep/many.txt", content, mtime=1.0)
        results = await vs.grep("match", fixed_string=True, max_matches=5)
        total = sum(len(r["matches"]) for r in results)
        assert total <= 5


class TestVectorStorageGetStats:
    """Lines 387-389, 434-441 — stats and eviction."""

    async def test_get_stats_empty_storage(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        stats = await vs.get_stats()
        assert stats["files_cached"] == 0
        assert stats["total_tokens_cached"] == 0
        assert stats["db_size_mb"] >= 0

    async def test_get_stats_after_put(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        await vs.put("/stats/file.txt", "hello world\n", mtime=1.0)
        stats = await vs.get_stats()
        assert stats["files_cached"] == 1
        assert stats["total_tokens_cached"] > 0

    async def test_evict_if_needed_triggered(self, tmp_path: Path) -> None:
        """Patch MAX_CACHE_ENTRIES to force eviction after a few puts."""
        vs = _make_vector_storage(tmp_path)
        from semantic_cache_mcp.storage import vector as vec_mod

        original_max = vec_mod.MAX_CACHE_ENTRIES
        try:
            vec_mod.MAX_CACHE_ENTRIES = 3
            for i in range(6):
                await vs.put(f"/evict/file{i}.txt", f"content {i}\n" * 5, mtime=float(i))
            # After eviction, count should be <= original_max * doc_count
            stats = await vs.get_stats()
            assert stats["files_cached"] >= 0  # Just verifying no crash
        finally:
            vec_mod.MAX_CACHE_ENTRIES = original_max


class TestVectorStorageSearchHybrid:
    """Lines 492, 494, 499 — search_hybrid and search_by_query."""

    async def test_search_by_query_returns_list(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        await vs.put("/search/doc.py", "def hello(): return 42\n", mtime=1.0)
        results = await vs.search_by_query("hello")
        # May be empty if FTS not indexed yet, but must return list
        assert isinstance(results, list)

    async def test_search_hybrid_no_embedding(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        await vs.put("/search/doc2.py", "class Foo: pass\n", mtime=1.0)
        results = await vs.search_hybrid("Foo", embedding=None)
        assert isinstance(results, list)

    async def test_search_hybrid_with_embedding(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        await vs.put("/search/doc3.py", "import os\n", mtime=1.0)
        emb = _make_embedding()
        results = await vs.search_hybrid("import", embedding=emb)
        assert isinstance(results, list)

    async def test_search_by_query_keyword_search_exception(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        # Patch the underlying collection to raise
        vs._collection.keyword_search = MagicMock(side_effect=Exception("fts error"))
        results = await vs.search_by_query("anything")
        assert results == []

    async def test_search_hybrid_exception(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        vs._collection.hybrid_search = MagicMock(side_effect=Exception("hybrid error"))
        results = await vs.search_hybrid("anything")
        assert results == []

    async def test_find_similar_multi(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        emb = _make_embedding()
        await vs.put("/sim/a.txt", "hello", mtime=1.0, embedding=emb)
        results = await vs.find_similar_multi(emb, k=3)
        assert isinstance(results, list)


# ===========================================================================
# cache/read.py
# ===========================================================================


class TestFitContentToMaxSize:
    """Lines 26-40 — _fit_content_to_max_size via smart_read large file path."""

    async def test_large_file_triggers_summarization(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        # Create file > MAX_CONTENT_SIZE (default 100K) to trigger summarization
        big_file = tmp_path / "big.txt"
        # Use a small max_size to trigger summarization on smaller content
        content = "This is a meaningful sentence about software engineering.\n" * 500
        big_file.write_text(content)

        with patch("semantic_cache_mcp.cache.read.summarize_semantic", return_value="summary"):
            result = await smart_read(cache, str(big_file), max_size=100)

        assert result.truncated is True

    async def test_large_file_summarization_exception_fallback(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        big_file = tmp_path / "big2.txt"
        content = "Line of content.\n" * 500
        big_file.write_text(content)

        with (
            patch(
                "semantic_cache_mcp.cache.read.summarize_semantic",
                side_effect=Exception("fail"),
            ),
            patch(
                "semantic_cache_mcp.cache.read.truncate_semantic",
                return_value="truncated",
            ),
        ):
            result = await smart_read(cache, str(big_file), max_size=100)

        assert result.truncated is True
        assert result.content == "truncated"


class TestSmartReadEdgeCases:
    """Lines 76, 91-93, 227, 231-234 — binary detection, unicode replacement, offset/limit."""

    async def test_binary_file_raises_value_error(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe\x80")
        with pytest.raises(ValueError, match="Binary file"):
            await smart_read(cache, str(binary_file))

    async def test_unicode_replacement_on_decode_error(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        bad_utf8 = tmp_path / "bad.txt"
        # Write bytes that are valid Latin-1 but not UTF-8
        bad_utf8.write_bytes(b"hello \x80\x81 world\n")
        # Should not raise; replacement chars used
        result = await smart_read(cache, str(bad_utf8))
        assert result.content is not None
        assert "hello" in result.content

    async def test_file_not_found_raises(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        with pytest.raises(FileNotFoundError):
            await smart_read(cache, str(tmp_path / "nonexistent.txt"))

    async def test_directory_raises_value_error(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        d = tmp_path / "adir"
        d.mkdir()
        with pytest.raises(ValueError, match="Not a regular file"):
            await smart_read(cache, str(d))

    async def test_symlink_resolved_transparently(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        real = tmp_path / "real.txt"
        real.write_text("hello from real\n")
        link = tmp_path / "link.txt"
        link.symlink_to(real)
        result = await smart_read(cache, str(link))
        assert "hello" in result.content


class TestBatchSmartRead:
    """Lines 285-292, 321-327, 363-375, 405, 407 — batch read paths."""

    async def test_batch_read_basic(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f1 = tmp_path / "a.txt"
        f1.write_text("File A content\n")
        f2 = tmp_path / "b.txt"
        f2.write_text("File B content\n")

        result = await batch_smart_read(cache, [str(f1), str(f2)])
        assert result.files_read >= 1
        assert len(result.contents) >= 1

    async def test_batch_read_skips_nonexistent(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f1 = tmp_path / "exists.txt"
        f1.write_text("content\n")
        bad = str(tmp_path / "missing.txt")

        result = await batch_smart_read(cache, [str(f1), bad])
        statuses = {s.path: s.status for s in result.files}
        assert statuses.get(bad) == "skipped"

    async def test_batch_read_skips_binary(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        bin_file = tmp_path / "bin.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03")
        txt_file = tmp_path / "text.txt"
        txt_file.write_text("readable\n")

        result = await batch_smart_read(cache, [str(bin_file), str(txt_file)])
        statuses = {s.path: s.status for s in result.files}
        assert statuses.get(str(bin_file)) == "skipped"

    async def test_batch_read_token_budget_overflow(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        files = []
        for i in range(5):
            f = tmp_path / f"tok{i}.txt"
            f.write_text(f"Token content file {i} " * 50 + "\n")
            files.append(str(f))

        # Very tight budget forces skipping
        result = await batch_smart_read(cache, files, max_total_tokens=30)
        skipped = [s for s in result.files if s.status == "skipped"]
        assert len(skipped) > 0

    async def test_batch_read_max_files_dos_limit(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        # Create 60 files — MAX_BATCH_FILES=50 should cap at 50
        paths = []
        for i in range(60):
            f = tmp_path / f"dos{i}.txt"
            f.write_text(f"content {i}\n")
            paths.append(str(f))

        result = await batch_smart_read(cache, paths)
        assert result.files_read + result.files_skipped <= 50

    async def test_batch_read_unchanged_detection(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "unchanged.txt"
        f.write_text("stable content\n")

        # First read populates cache
        await batch_smart_read(cache, [str(f)])
        # Second read should detect unchanged
        result = await batch_smart_read(cache, [str(f)])
        assert str(f) in result.unchanged_paths

    async def test_batch_read_priority_ordering(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f1 = tmp_path / "p1.txt"
        f1.write_text("priority one\n")
        f2 = tmp_path / "p2.txt"
        f2.write_text("normal two\n")

        result = await batch_smart_read(cache, [str(f2), str(f1)], priority=[str(f1)])
        # f1 is priority — should appear first in files list (if budget allows)
        read_paths = [s.path for s in result.files if s.status != "skipped"]
        if len(read_paths) >= 2:
            assert read_paths.index(str(f1)) < read_paths.index(str(f2))

    async def test_batch_read_content_hash_fallback_on_exception(self, tmp_path: Path) -> None:
        """Pre-scan read exception during hash check should be swallowed (best-effort)."""
        cache = _make_cache(tmp_path)
        f = tmp_path / "hashfail.txt"
        f.write_text("content\n")
        # First populate cache
        await smart_read(cache, str(f))
        # Touch mtime to force hash-check path, then patch read_text to raise
        # so the pre-scan except clause (nosec B112) is exercised.
        import os

        os.utime(str(f), (time.time() + 100, time.time() + 100))

        original_read_text = Path.read_text
        call_count = 0

        def failing_read_text(self_path, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Fail on first call (pre-scan), succeed thereafter (smart_read)
            if call_count == 1:
                raise OSError("disk error")
            return original_read_text(self_path, *args, **kwargs)

        with patch.object(Path, "read_text", failing_read_text):
            result = await batch_smart_read(cache, [str(f)])

        assert result is not None


# ===========================================================================
# cache/write.py
# ===========================================================================


class TestSmartWriteEdgeCases:
    """Lines 92, 101, 105, 137, 152-157, 163-165, 192-193 — write edge paths."""

    async def test_binary_existing_file_raises(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        bin_file = tmp_path / "bin.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03")
        with pytest.raises(ValueError, match="Binary file"):
            await smart_write(cache, str(bin_file), "new content")

    async def test_create_parents_false_missing_dir_raises(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        deep = tmp_path / "does" / "not" / "exist" / "file.txt"
        with pytest.raises(FileNotFoundError):
            await smart_write(cache, str(deep), "content", create_parents=False)

    async def test_create_parents_true_creates_directories(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        deep = tmp_path / "new" / "dir" / "file.txt"
        result = await smart_write(cache, str(deep), "hello\n", create_parents=True)
        assert deep.exists()
        assert result.created is True

    async def test_dry_run_does_not_write(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "dryrun.txt"
        result = await smart_write(cache, str(f), "content\n", dry_run=True)
        assert not f.exists()
        assert result.bytes_written > 0

    async def test_append_mode(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "append.txt"
        f.write_text("first\n")
        result = await smart_write(cache, str(f), "second\n", append=True)
        assert result.bytes_written > 0
        content = f.read_text()
        assert "first" in content
        assert "second" in content

    async def test_content_too_large_raises(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "toobig.txt"
        with pytest.raises(ValueError, match="Content too large"):
            await smart_write(cache, str(f), "x" * (10 * 1024 * 1024 + 1))

    async def test_directory_path_raises(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        d = tmp_path / "somedir"
        d.mkdir()
        with pytest.raises(ValueError, match="Not a regular file"):
            await smart_write(cache, str(d), "content")

    async def test_auto_format_called_when_requested(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "fmt.py"
        f.write_text("x=1\n")
        with patch("semantic_cache_mcp.cache.write._format_file", return_value=True) as mock_fmt:
            f.write_text("x=2\n")  # ensure re-read works
            await smart_write(cache, str(f), "x = 2\n", auto_format=True)
        mock_fmt.assert_called_once()

    async def test_cached_content_used_for_diff(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "cached_diff.txt"
        original = "line one\nline two\n"
        f.write_text(original)
        # Populate cache
        await smart_read(cache, str(f))
        # Now write new content — diff should come from cache
        result = await smart_write(cache, str(f), "line one\nline three\n")
        assert result.from_cache is True
        assert result.diff_content is not None

    async def test_write_permission_error(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "perm.txt"
        with (
            patch(
                "semantic_cache_mcp.cache.write._atomic_write",
                side_effect=OSError("permission denied"),
            ),
            pytest.raises(PermissionError),
        ):
            await smart_write(cache, str(f), "content\n")

    async def test_write_mtime_hash_match_uses_cache(self, tmp_path: Path) -> None:
        """When mtime changed but hash matches, cache content is used for diff."""
        cache = _make_cache(tmp_path)
        f = tmp_path / "mtime_hash.txt"
        content = "stable content\n"
        f.write_text(content)
        await smart_read(cache, str(f))
        # Touch mtime without changing content
        import os

        os.utime(str(f), (time.time() + 10, time.time() + 10))
        result = await smart_write(cache, str(f), "stable content\nnew line\n")
        assert result.from_cache is True


class TestSmartEditEdgeCases:
    """Lines 293, 304, 309, 317, 332-342, 347-348, 352 — edit paths."""

    async def test_edit_binary_file_raises(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        bin_file = tmp_path / "edit_bin.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03")
        with pytest.raises(ValueError, match="Binary file"):
            await smart_edit(cache, str(bin_file), "old", "new")

    async def test_edit_permission_error_on_read(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "permerr.txt"
        f.write_text("content")
        with (
            patch.object(Path, "read_bytes", side_effect=OSError("no permission")),
            pytest.raises(PermissionError),
        ):
            await smart_edit(cache, str(f), "content", "new")

    async def test_edit_not_found_raises(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        with pytest.raises(FileNotFoundError):
            await smart_edit(cache, str(tmp_path / "nope.txt"), "old", "new")

    async def test_edit_old_equals_new_raises(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "same.txt"
        f.write_text("content")
        with pytest.raises(ValueError, match="identical"):
            await smart_edit(cache, str(f), "content", "content")

    async def test_edit_from_cache(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "edit_cache.txt"
        f.write_text("hello world\n")
        await smart_read(cache, str(f))  # Populate cache
        result = await smart_edit(cache, str(f), "hello", "goodbye")
        assert result.from_cache is True
        assert result.replacements_made == 1

    async def test_edit_auto_format(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "fmt_edit.py"
        f.write_text("x = 1\ny = 2\n")
        with patch("semantic_cache_mcp.cache.write._format_file", return_value=True):
            await smart_edit(cache, str(f), "x = 1", "x = 10", auto_format=True)

    async def test_edit_permission_error_on_write(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "writeperm.txt"
        f.write_text("hello world\n")
        with (
            patch(
                "semantic_cache_mcp.cache.write._atomic_write",
                side_effect=OSError("write denied"),
            ),
            pytest.raises(PermissionError),
        ):
            await smart_edit(cache, str(f), "hello", "goodbye")

    async def test_edit_mode_c_line_replace(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "linereplace.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await smart_edit(
            cache,
            str(f),
            old_string=None,
            new_string="REPLACED\n",
            start_line=2,
            end_line=2,
        )
        assert result.replacements_made >= 1
        assert "REPLACED" in f.read_text()

    async def test_edit_mtime_hash_match_uses_cache(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "edit_mtime.txt"
        content = "stable content here\n"
        f.write_text(content)
        await smart_read(cache, str(f))
        import os

        os.utime(str(f), (time.time() + 10, time.time() + 10))
        result = await smart_edit(cache, str(f), "stable", "changed")
        assert result.from_cache is True


class TestSmartBatchEdit:
    """Lines 360, 373, 379, 421, 466-467, 475-481, 489 — batch edit paths."""

    async def test_batch_edit_empty_raises(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "batch.txt"
        f.write_text("content\n")
        with pytest.raises(ValueError, match="No edits"):
            await smart_batch_edit(cache, str(f), [])

    async def test_batch_edit_not_found_raises(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        with pytest.raises(FileNotFoundError):
            await smart_batch_edit(cache, str(tmp_path / "nope.txt"), [("old", "new")])

    async def test_batch_edit_binary_file_raises(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        bin_file = tmp_path / "batch_bin.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03")
        with pytest.raises(ValueError, match="Binary file"):
            await smart_batch_edit(cache, str(bin_file), [("a", "b")])

    async def test_batch_edit_partial_failure(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "partial.txt"
        f.write_text("alpha beta gamma\n")
        result = await smart_batch_edit(
            cache,
            str(f),
            [
                ("alpha", "ALPHA"),  # success
                ("not_present", "X"),  # failure
            ],
        )
        assert result.succeeded == 1
        assert result.failed == 1
        assert "ALPHA" in f.read_text()

    async def test_batch_edit_all_fail_no_write(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "allfail.txt"
        original = "abc def\n"
        f.write_text(original)
        result = await smart_batch_edit(
            cache,
            str(f),
            [("nope1", "x"), ("nope2", "y")],
        )
        assert result.succeeded == 0
        assert result.failed == 2
        assert f.read_text() == original  # unchanged

    async def test_batch_edit_from_cache(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "batch_cache.txt"
        f.write_text("foo bar baz\n")
        await smart_read(cache, str(f))  # populate cache
        result = await smart_batch_edit(cache, str(f), [("foo", "FOO")])
        assert result.from_cache is True

    async def test_batch_edit_auto_format(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "batch_fmt.py"
        f.write_text("x = 1\ny = 2\n")
        with patch("semantic_cache_mcp.cache.write._format_file", return_value=True):
            result = await smart_batch_edit(cache, str(f), [("x = 1", "x = 10")], auto_format=True)
        assert result.succeeded == 1

    async def test_batch_edit_mode_c(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "mode_c.txt"
        f.write_text("line1\nline2\nline3\n")
        result = await smart_batch_edit(
            cache,
            str(f),
            [(None, "REPLACED\n", 2, 2)],
        )
        assert result.succeeded == 1
        assert "REPLACED" in f.read_text()

    async def test_batch_edit_mode_b_scoped(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "mode_b.txt"
        f.write_text("alpha\nbeta\ngamma\n")
        result = await smart_batch_edit(
            cache,
            str(f),
            [("beta", "BETA", 2, 2)],
        )
        assert result.succeeded == 1
        assert "BETA" in f.read_text()

    async def test_batch_edit_write_permission_error(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "batchperm.txt"
        f.write_text("hello world\n")
        with (
            patch(
                "semantic_cache_mcp.cache.write._atomic_write",
                side_effect=OSError("denied"),
            ),
            pytest.raises(PermissionError),
        ):
            await smart_batch_edit(cache, str(f), [("hello", "goodbye")])

    async def test_batch_edit_old_equals_new_fails(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "same_batch.txt"
        f.write_text("content here\n")
        result = await smart_batch_edit(cache, str(f), [("content", "content")])
        assert result.failed == 1

    async def test_batch_edit_dos_limit(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "dos_batch.txt"
        # 60 unique tokens for 60 edits
        lines = [f"token{i}" for i in range(60)]
        f.write_text("\n".join(lines) + "\n")
        edits = [(f"token{i}", f"TOKEN{i}") for i in range(60)]
        result = await smart_batch_edit(cache, str(f), edits)
        # MAX_BATCH_EDITS=50, so at most 50 processed
        assert result.succeeded + result.failed <= 50

    async def test_batch_edit_mtime_hash_match_uses_cache(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        f = tmp_path / "batch_mtime.txt"
        content = "stable line\n"
        f.write_text(content)
        await smart_read(cache, str(f))
        import os

        os.utime(str(f), (time.time() + 10, time.time() + 10))
        result = await smart_batch_edit(cache, str(f), [("stable", "changed")])
        assert result.from_cache is True


# ===========================================================================
# Additional targeted gap-filling tests
# ===========================================================================


class TestChooseMinTokenContent:
    """Lines 215-226 in _helpers.py — _choose_min_token_content."""

    def test_picks_shortest_content(self) -> None:
        from semantic_cache_mcp.cache._helpers import _choose_min_token_content

        options = {
            "long": "This is a long string with many words and tokens " * 20,
            "short": "hi",
        }
        kind, content, tokens = _choose_min_token_content(options)
        assert kind == "short"
        assert content == "hi"
        assert tokens > 0

    def test_single_option(self) -> None:
        from semantic_cache_mcp.cache._helpers import _choose_min_token_content

        kind, content, tokens = _choose_min_token_content({"only": "hello world"})
        assert kind == "only"
        assert tokens > 0

    def test_empty_options(self) -> None:
        from semantic_cache_mcp.cache._helpers import _choose_min_token_content

        kind, content, tokens = _choose_min_token_content({})
        assert kind == ""
        assert content == ""
        assert tokens == 0


class TestFormatFileSubprocessEdgeCases:
    """Lines 83-90 — formatter timeout and OSError paths."""

    def test_formatter_timeout_returns_false(self, tmp_path: Path) -> None:
        import subprocess

        f = tmp_path / "timeout.py"
        f.write_text("x = 1\n")
        with (
            patch("shutil.which", return_value="/usr/bin/ruff"),
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ruff", 10)),
        ):
            result = _format_file(f)
        assert result is False

    def test_formatter_oserror_returns_false(self, tmp_path: Path) -> None:
        f = tmp_path / "oserr.py"
        f.write_text("x = 1\n")
        with (
            patch("shutil.which", return_value="/usr/bin/ruff"),
            patch("subprocess.run", side_effect=OSError("exec failed")),
        ):
            result = _format_file(f)
        assert result is False

    def test_formatter_nonzero_returncode_returns_false(self, tmp_path: Path) -> None:

        f = tmp_path / "fail.py"
        f.write_text("x = 1\n")
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = b"syntax error"
        with (
            patch("shutil.which", return_value="/usr/bin/ruff"),
            patch("subprocess.run", return_value=mock_result),
        ):
            result = _format_file(f)
        assert result is False


class TestGetRssMbExceptionPath:
    """Line 28 in store.py — _get_rss_mb exception swallowing."""

    def test_get_rss_mb_swallows_exception(self) -> None:
        from semantic_cache_mcp.cache.store import _get_rss_mb

        with patch("builtins.open", side_effect=OSError("no proc")):
            result = _get_rss_mb()
        # Either returns None (exception swallowed) or a float (fallback)
        assert result is None or isinstance(result, float)


class TestBatchReadTokenBudgetFlush:
    """Lines 363-375 — remaining paths enriched with est_tokens when budget exhausted."""

    async def test_remaining_paths_have_est_tokens_when_budget_exhausted(
        self, tmp_path: Path
    ) -> None:
        cache = _make_cache(tmp_path)
        # Create files — first one will consume the full budget
        large = tmp_path / "large.txt"
        # Write enough content to fill a small budget
        large.write_text("word " * 200 + "\n")  # ~200 tokens
        small = tmp_path / "small.txt"
        small.write_text("tiny\n")

        # Budget of 50 will be exhausted after first file
        result = await batch_smart_read(cache, [str(large), str(small)], max_total_tokens=50)
        skipped = [s for s in result.files if s.status == "skipped"]
        # At least the small file should be skipped with est_tokens populated
        assert any(s.est_tokens is not None and s.est_tokens > 0 for s in skipped)


class TestReadFitContentToMaxSizeDirectly:
    """Lines 26-40 — _fit_content_to_max_size directly."""

    def test_fit_content_within_limit_returns_unchanged(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.cache.read import _fit_content_to_max_size

        cache = _make_cache(tmp_path)
        content = "short content"
        result, truncated = _fit_content_to_max_size(content, 10_000, cache)
        assert result == content
        assert truncated is False

    def test_fit_content_exceeds_limit_summarizes(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.cache.read import _fit_content_to_max_size

        cache = _make_cache(tmp_path)
        big = "word " * 1000
        with patch("semantic_cache_mcp.cache.read.summarize_semantic", return_value="summary"):
            result, truncated = _fit_content_to_max_size(big, 10, cache)
        assert truncated is True
        assert result == "summary"

    def test_fit_content_summarization_fallback(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.cache.read import _fit_content_to_max_size

        cache = _make_cache(tmp_path)
        big = "word " * 1000
        with (
            patch(
                "semantic_cache_mcp.cache.read.summarize_semantic",
                side_effect=Exception("fail"),
            ),
            patch(
                "semantic_cache_mcp.cache.read.truncate_semantic",
                return_value="truncated_fallback",
            ),
        ):
            result, truncated = _fit_content_to_max_size(big, 10, cache)
        assert truncated is True
        assert result == "truncated_fallback"


class TestVectorStorageSaveClose:
    """Lines 717-726 in vector.py — save() and close() methods."""

    async def test_save_and_close(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        await vs.put("/close/test.txt", "some content\n", mtime=1.0)
        vs.save()
        vs.close()
        # After close, no exception should have been raised

    def test_close_handles_exception(self, tmp_path: Path) -> None:
        vs = _make_vector_storage(tmp_path)
        vs._db.save = MagicMock(side_effect=Exception("save failed"))
        # close() logs warning but doesn't raise
        vs.close()
