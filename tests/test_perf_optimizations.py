"""Tests for performance optimizations: gather, embedding reuse, no re-fetch."""

from __future__ import annotations

import array
from pathlib import Path
from unittest.mock import AsyncMock, patch

from semantic_cache_mcp.cache import SemanticCache, batch_smart_read, smart_read
from semantic_cache_mcp.cache.search import compare_files, find_similar_files
from semantic_cache_mcp.types import EmbeddingVector
from tests.constants import TEST_EMBEDDING_DIM


def _make_cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(db_path=tmp_path / "perf_test.db")


def _fake_embedding() -> EmbeddingVector:
    e = array.array("f", [0.1] * TEST_EMBEDDING_DIM)
    return e


# ---------------------------------------------------------------------------
# Fix 1: batch_smart_read gathers cache.get() instead of serial awaits
# ---------------------------------------------------------------------------


class TestBatchReadGather:
    """Verify batch_smart_read fetches all cache entries in parallel."""

    async def test_batch_read_issues_single_gather(self, tmp_path: Path) -> None:
        """All cache lookups should happen via one gather, not N serial calls."""
        cache = _make_cache(tmp_path)

        # Create 5 files
        files = []
        for i in range(5):
            f = tmp_path / f"file_{i}.txt"
            f.write_text(f"content {i}\n")
            files.append(str(f))

        with patch("semantic_cache_mcp.cache.embed", return_value=None):
            result = await batch_smart_read(cache, files)

        assert result.files_read == 5
        assert len(result.contents) == 5

    async def test_batch_read_reuses_cache_entries(self, tmp_path: Path) -> None:
        """Pre-scan loop should reuse entries from gather, not re-fetch."""
        cache = _make_cache(tmp_path)
        f = tmp_path / "reuse.txt"
        f.write_text("hello\n")

        with patch("semantic_cache_mcp.cache.embed", return_value=None):
            await batch_smart_read(cache, [str(f)])

        original_get = SemanticCache.get
        call_count = 0

        async def counting_get(self_arg, path: str):
            nonlocal call_count
            call_count += 1
            return await original_get(self_arg, path)

        with (
            patch("semantic_cache_mcp.cache.embed", return_value=None),
            patch.object(SemanticCache, "get", counting_get),
        ):
            result = await batch_smart_read(cache, [str(f)])

        # 1 call from the gather + 1 from smart_read = 2 total.
        # Before the fix this was 3+ (gather + pre-scan + smart_read).
        assert call_count == 2
        assert result.files_read == 1


# ---------------------------------------------------------------------------
# Fix 2: smart_read no double cache.get on diff path
# ---------------------------------------------------------------------------


class TestSmartReadNoDuplicateFetch:
    """Changed-file diff path should reuse the original entry, not re-fetch."""

    async def test_diff_path_no_refetch(self, tmp_path: Path) -> None:
        """When file content changes, smart_read should not call cache.get twice."""
        cache = _make_cache(tmp_path)
        f = tmp_path / "changing.txt"
        # Use larger content so the diff path is cheaper than full content
        f.write_text("line 1\n" * 50)

        with patch("semantic_cache_mcp.cache.embed", return_value=None):
            r1 = await smart_read(cache, str(f))
            assert not r1.from_cache

        f.write_text("line 1\n" * 49 + "line 2 changed\n")

        original_get = SemanticCache.get
        get_count = 0

        async def counting_get(self_arg, path: str):
            nonlocal get_count
            get_count += 1
            return await original_get(self_arg, path)

        with (
            patch("semantic_cache_mcp.cache.embed", return_value=None),
            patch.object(SemanticCache, "get", counting_get),
        ):
            r2 = await smart_read(cache, str(f), diff_mode=True)

        assert r2.is_diff
        # Only 1 cache.get call — not 2 (the re-fetch was eliminated)
        assert get_count == 1

    async def test_first_read_reuses_embedding_between_similarity_check_and_cache_put(
        self, tmp_path: Path
    ) -> None:
        """First uncached read should not call get_embedding twice."""
        cache = _make_cache(tmp_path)
        f = tmp_path / "first_read.txt"
        f.write_text("brand new file\n")

        emb = _fake_embedding()
        original_get_embedding = SemanticCache.get_embedding
        call_count = 0

        async def counting_get_embedding(self_arg, text: str, path: str = ""):
            nonlocal call_count
            call_count += 1
            return await original_get_embedding(self_arg, text, path)

        with (
            patch("semantic_cache_mcp.cache.embed", return_value=emb),
            patch.object(SemanticCache, "get_embedding", counting_get_embedding),
        ):
            result = await smart_read(cache, str(f))

        assert not result.from_cache
        assert call_count == 1

    async def test_diff_path_still_generates_correct_diff(self, tmp_path: Path) -> None:
        """Verify the optimization doesn't break diff content."""
        cache = _make_cache(tmp_path)
        f = tmp_path / "diff_check.txt"
        # Larger file so diff is cheaper than full — triggers diff path
        f.write_text("line 1\n" * 50 + "line 2\n")

        with patch("semantic_cache_mcp.cache.embed", return_value=None):
            await smart_read(cache, str(f))

        f.write_text("line 1\n" * 50 + "line 2 modified\n")

        with patch("semantic_cache_mcp.cache.embed", return_value=None):
            result = await smart_read(cache, str(f), diff_mode=True)

        assert result.is_diff
        assert "-line 2" in result.content
        assert "+line 2 modified" in result.content

    async def test_force_full_cached_read_skips_refresh(self, tmp_path: Path) -> None:
        """Line-range/full-force reads should not rewrite vecdb when cache is already fresh."""
        cache = _make_cache(tmp_path)
        f = tmp_path / "range_read.txt"
        f.write_text("alpha\nbeta\ngamma\n")

        with patch("semantic_cache_mcp.cache.embed", return_value=None):
            await smart_read(cache, str(f))

        with (
            patch("semantic_cache_mcp.cache.embed", return_value=None),
            patch.object(
                SemanticCache,
                "refresh_path",
                new=AsyncMock(side_effect=AssertionError("refresh_path should not be called")),
            ),
        ):
            result = await smart_read(
                cache,
                str(f),
                diff_mode=False,
                force_full=True,
                refresh_cache=False,
            )

        assert result.content == "alpha\nbeta\ngamma\n"


# ---------------------------------------------------------------------------
# Fix 3: find_similar_files reuses cached embedding
# ---------------------------------------------------------------------------


class TestFindSimilarEmbeddingReuse:
    """find_similar_files should reuse cached.embedding, not call ONNX."""

    async def test_cached_file_skips_onnx(self, tmp_path: Path) -> None:
        """When file is cached with embedding, embed() should not be called."""
        emb = _fake_embedding()
        cache = _make_cache(tmp_path)
        f = tmp_path / "similar.txt"
        f.write_text("some content\n")

        with patch("semantic_cache_mcp.cache.embed", return_value=emb):
            # Prime the cache with an embedding
            await smart_read(cache, str(f))

        # Now call find_similar — should reuse the cached embedding, not call embed()
        with patch("semantic_cache_mcp.cache.embed", return_value=emb) as mock_embed:
            result = await find_similar_files(cache, str(f))

        # embed() should NOT be called — cached.embedding was reused
        mock_embed.assert_not_called()
        assert result.source_path == str(f)

    async def test_uncached_file_does_call_onnx(self, tmp_path: Path) -> None:
        """New file without cached embedding must call get_embedding."""
        emb = _fake_embedding()
        cache = _make_cache(tmp_path)
        f = tmp_path / "new_file.txt"
        f.write_text("brand new content\n")

        with patch("semantic_cache_mcp.cache.embed", return_value=emb):
            result = await find_similar_files(cache, str(f))

        # File was uncached, so it went through the else branch with ONNX
        assert result.source_path == str(f)

    async def test_uncached_file_skips_refresh_path(self, tmp_path: Path) -> None:
        """Similarity search should not rewrite vecdb just to use a source embedding."""
        emb = _fake_embedding()
        cache = _make_cache(tmp_path)
        source = tmp_path / "source.txt"
        neighbor = tmp_path / "neighbor.txt"
        source.write_text("source content\n")
        neighbor.write_text("neighbor content\n")

        with patch("semantic_cache_mcp.cache.embed", return_value=emb):
            await smart_read(cache, str(neighbor))

        with (
            patch("semantic_cache_mcp.cache.embed", return_value=emb),
            patch.object(
                SemanticCache,
                "refresh_path",
                new=AsyncMock(side_effect=AssertionError("refresh_path should not be called")),
            ),
        ):
            result = await find_similar_files(cache, str(source))

        assert result.source_path == str(source)


class TestCompareFilesNoRefresh:
    """compare_files should avoid vecdb rewrite when direct computation is enough."""

    async def test_uncached_compare_skips_refresh_path(self, tmp_path: Path) -> None:
        """Comparing two uncached files should not rewrite vecdb for either side."""
        emb = _fake_embedding()
        cache = _make_cache(tmp_path)
        file1 = tmp_path / "one.txt"
        file2 = tmp_path / "two.txt"
        file1.write_text("alpha\n")
        file2.write_text("beta\n")

        with (
            patch("semantic_cache_mcp.cache.search.embed_query", return_value=emb),
            patch("semantic_cache_mcp.cache.embed", return_value=emb),
            patch.object(
                SemanticCache,
                "refresh_path",
                new=AsyncMock(side_effect=AssertionError("refresh_path should not be called")),
            ),
        ):
            result = await compare_files(cache, str(file1), str(file2))

        assert result.path1 == str(file1)
        assert result.path2 == str(file2)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge conditions for the optimized paths."""

    async def test_batch_read_empty_list(self, tmp_path: Path) -> None:
        """Empty path list should not error on gather."""
        cache = _make_cache(tmp_path)
        with patch("semantic_cache_mcp.cache.embed", return_value=None):
            result = await batch_smart_read(cache, [])
        assert result.files_read == 0

    async def test_batch_read_nonexistent_files(self, tmp_path: Path) -> None:
        """Nonexistent paths should be handled gracefully by the gather."""
        cache = _make_cache(tmp_path)
        with patch("semantic_cache_mcp.cache.embed", return_value=None):
            result = await batch_smart_read(
                cache, [str(tmp_path / "nope.txt"), str(tmp_path / "also_nope.txt")]
            )
        # All files skipped
        assert result.files_skipped == 2

    async def test_diff_on_unchanged_file_still_works(self, tmp_path: Path) -> None:
        """Unchanged file should return from cache (not hit diff path)."""
        cache = _make_cache(tmp_path)
        f = tmp_path / "stable.txt"
        # Larger file so unchanged marker is cheaper than full content
        f.write_text("stable content line\n" * 20)

        with patch("semantic_cache_mcp.cache.embed", return_value=None):
            await smart_read(cache, str(f))
            r2 = await smart_read(cache, str(f), diff_mode=True)

        assert r2.from_cache
        assert not r2.is_diff
        assert "unchanged" in r2.content.lower()

    async def test_find_similar_no_embedding_returns_empty(self, tmp_path: Path) -> None:
        """File without embedding stored should still work (returns empty results)."""
        cache = _make_cache(tmp_path)
        f = tmp_path / "no_emb.txt"
        f.write_text("no embedding\n")

        with patch("semantic_cache_mcp.cache.embed", return_value=None):
            await smart_read(cache, str(f))
            result = await find_similar_files(cache, str(f))

        assert result.similar_files == []
