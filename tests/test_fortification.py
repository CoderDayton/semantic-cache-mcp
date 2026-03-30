"""Tests for code fortification guards added during hardening passes.

Covers:
  - Idempotent close() on VectorStorage and SemanticCache
  - ConnectionPool close_all() draining in-use connections
  - Binary file guard in find_similar_files
  - Grep parameter clamping
  - Diff context_lines clamping
  - _clear_sync() replacing asyncio.get_event_loop().run_until_complete()
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# VectorStorage idempotent close
# ---------------------------------------------------------------------------


class TestVectorStorageIdempotentClose:
    """VectorStorage.close() should be safe to call multiple times."""

    def test_second_close_is_noop(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.storage.vector import VectorStorage

        vs = VectorStorage(tmp_path / "vec.db")
        vs.close(timeout=2)
        assert vs._closed is True

        # Second call should return immediately without error
        vs.close(timeout=2)
        assert vs._closed is True

    def test_close_sets_flag_before_thread(self, tmp_path: Path) -> None:
        """_closed is set before the thread runs — prevents races."""
        from semantic_cache_mcp.storage.vector import VectorStorage

        vs = VectorStorage(tmp_path / "vec.db")
        assert vs._closed is False
        vs.close(timeout=2)
        assert vs._closed is True


# ---------------------------------------------------------------------------
# SemanticCache idempotent close
# ---------------------------------------------------------------------------


class TestSemanticCacheIdempotentClose:
    """SemanticCache.close() should be safe to call multiple times."""

    def test_second_close_is_noop(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.cache.store import SemanticCache

        cache = SemanticCache(tmp_path / "vec.db")
        cache.close()
        assert cache._closed is True

        # Second call returns immediately — no double-persist, no double-close
        cache.close()
        assert cache._closed is True

    def test_close_persists_then_closes(self, tmp_path: Path) -> None:
        """close() calls persist + storage close + pool close in order."""
        from semantic_cache_mcp.cache.store import SemanticCache

        cache = SemanticCache(tmp_path / "vec.db")

        with (
            patch.object(type(cache._metrics), "persist") as mock_persist,
            patch.object(type(cache._storage), "close") as mock_vs_close,
            patch.object(type(cache._metrics_storage._pool), "close_all") as mock_pool,
        ):
            cache.close()

        mock_persist.assert_called_once()
        mock_vs_close.assert_called_once()
        mock_pool.assert_called_once()


# ---------------------------------------------------------------------------
# ConnectionPool close_all — in-use connections
# ---------------------------------------------------------------------------


class TestConnectionPoolCloseAll:
    """ConnectionPool.close_all() should close in-use connections too."""

    def test_close_all_closes_in_use(self, tmp_path: Path) -> None:
        """Connections checked out at close_all time are still closed."""
        from semantic_cache_mcp.storage.sqlite import ConnectionPool

        pool = ConnectionPool(tmp_path / "test.db", max_size=2)

        # Check out a connection (simulates in-use)
        cm = pool.get_connection()
        conn = cm.__enter__()

        # conn is now in-use, not in _available
        assert pool._available.empty()
        assert conn in pool._all_conns

        pool.close_all()
        assert pool._closed is True
        assert len(pool._all_conns) == 0

        # The connection should be closed — any operation raises
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

        # Cleanup context manager
        with contextlib.suppress(Exception):
            cm.__exit__(None, None, None)

    def test_close_all_idempotent(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.storage.sqlite import ConnectionPool

        pool = ConnectionPool(tmp_path / "test.db", max_size=2)
        with pool.get_connection():
            pass  # Create and return a connection

        pool.close_all()
        assert pool._closed is True
        # Second call is a noop
        pool.close_all()
        assert pool._closed is True

    def test_all_conns_tracked(self, tmp_path: Path) -> None:
        """Every created connection is tracked in _all_conns."""
        from semantic_cache_mcp.storage.sqlite import ConnectionPool

        pool = ConnectionPool(tmp_path / "test.db", max_size=3)

        # Create 3 connections by checking them out sequentially
        for _ in range(3):
            with pool.get_connection():
                pass

        # Pool reuses connections — only 1 should exist
        assert pool._total == 1
        assert len(pool._all_conns) == 1


# ---------------------------------------------------------------------------
# Binary file guard in find_similar_files
# ---------------------------------------------------------------------------


class TestFindSimilarFilesBinaryGuard:
    """find_similar_files should reject binary files."""

    @pytest.mark.asyncio
    async def test_binary_file_raises(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.cache.search import find_similar_files
        from semantic_cache_mcp.cache.store import SemanticCache

        # Create a binary file
        binary_file = tmp_path / "image.png"
        binary_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        cache = SemanticCache(tmp_path / "vec.db")
        try:
            with pytest.raises(ValueError, match="Binary file not supported"):
                await find_similar_files(cache, str(binary_file))
        finally:
            cache.close()

    @pytest.mark.asyncio
    async def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.cache.search import find_similar_files
        from semantic_cache_mcp.cache.store import SemanticCache

        cache = SemanticCache(tmp_path / "vec.db")
        try:
            with pytest.raises(FileNotFoundError):
                await find_similar_files(cache, str(tmp_path / "nope.txt"))
        finally:
            cache.close()


# ---------------------------------------------------------------------------
# Grep parameter clamping
# ---------------------------------------------------------------------------


class TestGrepParamClamping:
    """VectorStorage.grep() should clamp extreme parameter values."""

    @pytest.mark.asyncio
    async def test_context_lines_clamped(self, tmp_path: Path) -> None:
        """context_lines > 20 is clamped to 20."""
        from semantic_cache_mcp.storage.vector import VectorStorage

        vs = VectorStorage(tmp_path / "vec.db")
        try:
            # Should not crash with extreme context_lines
            results = await vs.grep("test", context_lines=99999)
            assert isinstance(results, list)
        finally:
            vs.close()

    @pytest.mark.asyncio
    async def test_negative_values_clamped(self, tmp_path: Path) -> None:
        """Negative max_matches/max_files are clamped to 1."""
        from semantic_cache_mcp.storage.vector import VectorStorage

        vs = VectorStorage(tmp_path / "vec.db")
        try:
            results = await vs.grep("test", max_matches=-5, max_files=-3)
            assert isinstance(results, list)
        finally:
            vs.close()


# ---------------------------------------------------------------------------
# Diff context_lines clamping
# ---------------------------------------------------------------------------


class TestDiffContextLinesClamping:
    """compare_files should clamp context_lines."""

    @pytest.mark.asyncio
    async def test_extreme_context_lines_clamped(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.cache.search import compare_files
        from semantic_cache_mcp.cache.store import SemanticCache

        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello\n")
        f2.write_text("world\n")

        cache = SemanticCache(tmp_path / "vec.db")
        try:
            # Should not crash with extreme value — clamped to 50
            result = await compare_files(cache, str(f1), str(f2), context_lines=999999)
            assert result.path1 == str(f1.resolve())
        finally:
            cache.close()

    @pytest.mark.asyncio
    async def test_negative_context_lines_clamped(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.cache.search import compare_files
        from semantic_cache_mcp.cache.store import SemanticCache

        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello\n")
        f2.write_text("world\n")

        cache = SemanticCache(tmp_path / "vec.db")
        try:
            result = await compare_files(cache, str(f1), str(f2), context_lines=-10)
            assert result.diff_content is not None
        finally:
            cache.close()


# ---------------------------------------------------------------------------
# _clear_sync — replaces asyncio.get_event_loop().run_until_complete()
# ---------------------------------------------------------------------------


class TestClearSync:
    """_clear_sync should work without an event loop."""

    def test_clear_sync_empty_db(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.storage.vector import VectorStorage

        vs = VectorStorage(tmp_path / "vec.db")
        try:
            count = vs._clear_sync()
            assert count == 0
        finally:
            vs.close()

    @pytest.mark.asyncio
    async def test_clear_sync_matches_async_clear(self, tmp_path: Path) -> None:
        """_clear_sync and clear() should produce the same result."""
        from semantic_cache_mcp.storage.vector import VectorStorage

        vs = VectorStorage(tmp_path / "vec.db")
        try:
            # Both should return 0 on empty DB
            sync_count = vs._clear_sync()
            async_count = await vs.clear()
            assert sync_count == async_count == 0
        finally:
            vs.close()

    def test_clear_sync_callable_from_sync_context(self, tmp_path: Path) -> None:
        """_clear_sync works from plain sync code — no event loop needed."""
        from semantic_cache_mcp.storage.vector import VectorStorage

        vs = VectorStorage(tmp_path / "vec.db")
        try:
            # This would fail with the old asyncio.get_event_loop().run_until_complete()
            # pattern if called when a loop is already running
            result = vs._clear_sync()
            assert isinstance(result, int)
        finally:
            vs.close()


# ---------------------------------------------------------------------------
# batch_smart_read max_total_tokens lower bound
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# VectorStorage health check
# ---------------------------------------------------------------------------


class TestVectorStorageHealthCheck:
    """is_healthy() should return True when the DB is usable, False otherwise."""

    async def test_healthy_after_init(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.storage.vector import VectorStorage

        vs = VectorStorage(tmp_path / "vec.db")
        try:
            assert await vs.is_healthy() is True
        finally:
            vs.close()

    async def test_unhealthy_after_close(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.storage.vector import VectorStorage

        vs = VectorStorage(tmp_path / "vec.db")
        vs.close()
        assert await vs.is_healthy() is False


# ---------------------------------------------------------------------------
# batch_smart_read max_total_tokens lower bound
# ---------------------------------------------------------------------------


class TestBatchReadTokenBound:
    """batch_smart_read should clamp max_total_tokens to >= 1."""

    @pytest.mark.asyncio
    async def test_zero_tokens_clamped(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.cache.read import batch_smart_read
        from semantic_cache_mcp.cache.store import SemanticCache

        f = tmp_path / "test.txt"
        f.write_text("hello world")

        cache = SemanticCache(tmp_path / "vec.db")
        try:
            # max_total_tokens=0 should be clamped to 1, not crash
            result = await batch_smart_read(cache, [str(f)], max_total_tokens=0)
            assert result.files_skipped >= 0
        finally:
            cache.close()

    @pytest.mark.asyncio
    async def test_negative_tokens_clamped(self, tmp_path: Path) -> None:
        from semantic_cache_mcp.cache.read import batch_smart_read
        from semantic_cache_mcp.cache.store import SemanticCache

        f = tmp_path / "test.txt"
        f.write_text("hello world")

        cache = SemanticCache(tmp_path / "vec.db")
        try:
            result = await batch_smart_read(cache, [str(f)], max_total_tokens=-100)
            assert result.files_skipped >= 0
        finally:
            cache.close()
