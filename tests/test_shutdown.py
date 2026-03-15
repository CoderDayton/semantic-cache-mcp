"""Tests for graceful shutdown: async_close, drain, begin/end operation, shielded writes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

from semantic_cache_mcp.cache import SemanticCache


def _make_cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(db_path=tmp_path / "shutdown_test.db")


# ---------------------------------------------------------------------------
# begin_operation / end_operation / request_shutdown
# ---------------------------------------------------------------------------


class TestOperationTracking:
    """Cover begin_operation, end_operation, request_shutdown."""

    async def test_begin_operation_returns_true_normally(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        assert cache.begin_operation() is True
        cache.end_operation()

    async def test_begin_operation_returns_false_after_shutdown(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        cache.request_shutdown()
        assert cache.begin_operation() is False

    async def test_end_operation_sets_drained(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        cache.begin_operation()
        assert not cache._drained.is_set()
        cache.end_operation()
        assert cache._drained.is_set()

    async def test_end_operation_clamps_at_zero(self, tmp_path: Path) -> None:
        """end_operation called without begin should not go negative."""
        cache = _make_cache(tmp_path)
        cache.end_operation()  # no begin — should clamp
        assert cache._inflight == 0
        assert cache._drained.is_set()

    async def test_multiple_operations_drain(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        cache.begin_operation()
        cache.begin_operation()
        assert cache._inflight == 2
        cache.end_operation()
        assert cache._inflight == 1
        assert not cache._drained.is_set()
        cache.end_operation()
        assert cache._inflight == 0
        assert cache._drained.is_set()


# ---------------------------------------------------------------------------
# async_close
# ---------------------------------------------------------------------------


class TestAsyncClose:
    """Cover async_close drain, timeout, CancelledError, and idempotency."""

    async def test_async_close_no_inflight(self, tmp_path: Path) -> None:
        """Clean close with nothing in-flight."""
        cache = _make_cache(tmp_path)
        await cache.async_close()
        assert cache._closed is True
        assert cache._shutting_down is True

    async def test_async_close_idempotent(self, tmp_path: Path) -> None:
        """Second call is a no-op."""
        cache = _make_cache(tmp_path)
        await cache.async_close()
        await cache.async_close()  # should not raise
        assert cache._closed is True

    async def test_async_close_waits_for_drain(self, tmp_path: Path) -> None:
        """async_close waits for in-flight operations to complete."""
        cache = _make_cache(tmp_path)
        cache.begin_operation()

        async def finish_later():
            await asyncio.sleep(0.1)
            cache.end_operation()

        asyncio.create_task(finish_later())
        await cache.async_close()
        assert cache._closed is True
        assert cache._inflight == 0

    async def test_async_close_timeout_forces_close(self, tmp_path: Path) -> None:
        """If drain times out, close proceeds anyway."""
        cache = _make_cache(tmp_path)
        cache.begin_operation()
        # Never end the operation — drain should timeout (use short timeout)
        with patch.object(SemanticCache, "_DRAIN_TIMEOUT", 0.1):
            await cache.async_close()
        assert cache._closed is True
        assert cache._inflight == 1  # never drained

    async def test_async_close_handles_cancelled_error(self, tmp_path: Path) -> None:
        """CancelledError during drain should not prevent close."""
        cache = _make_cache(tmp_path)
        cache.begin_operation()

        # Make the drain wait raise CancelledError
        with patch.object(cache._drained, "wait", side_effect=asyncio.CancelledError):
            await cache.async_close()

        assert cache._closed is True

    async def test_async_close_persists_metrics(self, tmp_path: Path) -> None:
        """Metrics should be persisted during close."""
        cache = _make_cache(tmp_path)
        with patch.object(type(cache._metrics), "persist", wraps=cache._metrics.persist) as mock_p:
            await cache.async_close()
        mock_p.assert_called_once()

    async def test_async_close_survives_metrics_failure(self, tmp_path: Path) -> None:
        """If metrics.persist raises, storage should still close."""
        cache = _make_cache(tmp_path)
        with patch.object(type(cache._metrics), "persist", side_effect=RuntimeError("db locked")):
            await cache.async_close()
        assert cache._closed is True


# ---------------------------------------------------------------------------
# sync close
# ---------------------------------------------------------------------------


class TestSyncClose:
    """Cover the synchronous close() fallback."""

    async def test_sync_close(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        cache.close()
        assert cache._closed is True
        assert cache._shutting_down is True

    async def test_sync_close_idempotent(self, tmp_path: Path) -> None:
        cache = _make_cache(tmp_path)
        cache.close()
        cache.close()  # second call should not raise
        assert cache._closed is True

    async def test_sync_close_survives_storage_failure(self, tmp_path: Path) -> None:
        """If VectorStorage.close raises, metrics pool should still close."""
        from semantic_cache_mcp.storage.vector import VectorStorage

        cache = _make_cache(tmp_path)
        with patch.object(VectorStorage, "close", side_effect=RuntimeError("boom")):
            cache.close()
        assert cache._closed is True


# ---------------------------------------------------------------------------
# _shielded_write
# ---------------------------------------------------------------------------


class TestShieldedWrite:
    """Cover _shielded_write: normal, shutdown rejection, and task naming."""

    async def test_shielded_write_normal(self, tmp_path: Path) -> None:
        """Normal execution returns the coroutine result."""
        from semantic_cache_mcp.server.tools import _shielded_write

        cache = _make_cache(tmp_path)

        async def fake_write():
            return "done"

        result = await _shielded_write(cache, fake_write())
        assert result == "done"
        assert cache._inflight == 0  # end_operation was called

    async def test_shielded_write_rejected_during_shutdown(self, tmp_path: Path) -> None:
        """_shielded_write raises RuntimeError when shutting down."""
        from semantic_cache_mcp.server.tools import _shielded_write

        cache = _make_cache(tmp_path)
        cache.request_shutdown()

        async def fake_write():
            return "should not run"

        coro = fake_write()
        try:
            await _shielded_write(cache, coro)
            msg = "Expected RuntimeError"
            raise AssertionError(msg)
        except RuntimeError as e:
            assert "shutting down" in str(e)
        finally:
            coro.close()  # suppress "coroutine was never awaited" warning

    async def test_shielded_write_task_is_named(self, tmp_path: Path) -> None:
        """The inner task should be named 'shielded-write'."""
        from semantic_cache_mcp.server.tools import _shielded_write

        cache = _make_cache(tmp_path)
        task_name = None

        async def capture_name():
            nonlocal task_name
            task_name = asyncio.current_task().get_name()  # type: ignore[union-attr]
            return "ok"

        await _shielded_write(cache, capture_name())
        assert task_name == "shielded-write"

    async def test_shielded_write_completes_on_outer_cancel(self, tmp_path: Path) -> None:
        """When the outer task is cancelled, the shielded write still finishes."""
        from semantic_cache_mcp.server.tools import _shielded_write

        cache = _make_cache(tmp_path)
        write_completed = False

        async def slow_write():
            nonlocal write_completed
            await asyncio.sleep(0.2)
            write_completed = True
            return "written"

        async def run_shielded():
            return await _shielded_write(cache, slow_write())

        task = asyncio.create_task(run_shielded())
        await asyncio.sleep(0.05)
        task.cancel()

        # Let the shielded write finish
        await asyncio.sleep(0.3)
        assert write_completed is True
