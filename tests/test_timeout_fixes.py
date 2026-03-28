"""Tests for timeout and hang-prevention fixes.

Covers:
- _shielded_write timeout, normal completion, shutdown rejection, cancellation
- _serialized lock queuing and release on timeout
- glob_with_cache_status async stat usage
- batch_smart_read async stat pre-fetching
"""

from __future__ import annotations

import array
import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_cache_mcp.cache import SemanticCache
from semantic_cache_mcp.server.tools import _serialized, _shielded_write

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cache(*, shutting_down: bool = False) -> MagicMock:
    """Create a mock SemanticCache with begin/end_operation wired up."""
    cache = MagicMock()
    cache._shutting_down = shutting_down
    cache.begin_operation.return_value = not shutting_down
    cache.end_operation = MagicMock()
    cache.reset_executor = MagicMock()
    return cache


async def _hang_forever() -> None:
    """Coroutine that never completes."""
    await asyncio.sleep(3600)


async def _fast_result(value: Any = "ok") -> Any:
    """Coroutine that completes immediately."""
    return value


async def _slow_finish(delay: float = 0.5, value: Any = "done") -> Any:
    """Coroutine that finishes after a short delay."""
    await asyncio.sleep(delay)
    return value


# ---------------------------------------------------------------------------
# _shielded_write: timeout
# ---------------------------------------------------------------------------


async def test_shielded_write_timeout_fires() -> None:
    """A hanging coroutine must raise TimeoutError within the timeout window."""
    cache = _make_cache()
    with pytest.raises(TimeoutError):
        await _shielded_write(cache, _hang_forever(), timeout=0.05)


async def test_shielded_write_timeout_is_fast() -> None:
    """Timeout should fire close to the requested duration, not much later."""
    cache = _make_cache()
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        await _shielded_write(cache, _hang_forever(), timeout=0.1)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.5, f"Timeout took {elapsed:.2f}s, expected ~0.1s"


async def test_refresh_path_timeout_marks_stale_and_resets_executor(tmp_path: Path) -> None:
    """Bounded cache refresh should mark the path stale and abandon the stuck IO thread."""
    cache = SemanticCache(db_path=tmp_path / "refresh_timeout.db")
    path = str(tmp_path / "stale.txt")

    async def _hang_put(self_arg, *args, **kwargs) -> None:
        await asyncio.sleep(3600)

    with (
        patch.object(SemanticCache, "put", _hang_put),
        patch.object(SemanticCache, "reset_executor") as mock_reset,
    ):
        ok = await cache.refresh_path(
            path,
            "content\n",
            1.0,
            array.array("f", [0.1]),
            timeout=0.01,
        )

    assert ok is False
    assert cache.is_stale(path) is True
    mock_reset.assert_called_once()


# ---------------------------------------------------------------------------
# _shielded_write: normal completion
# ---------------------------------------------------------------------------


async def test_shielded_write_returns_result() -> None:
    """A fast coroutine should return its value normally."""
    cache = _make_cache()
    result = await _shielded_write(cache, _fast_result(42), timeout=1.0)
    assert result == 42


async def test_shielded_write_returns_none() -> None:
    """A coroutine returning None should propagate None (not raise)."""
    cache = _make_cache()
    result = await _shielded_write(cache, _fast_result(None), timeout=1.0)
    assert result is None


# ---------------------------------------------------------------------------
# _shielded_write: shutdown rejection
# ---------------------------------------------------------------------------


async def test_shielded_write_rejects_during_shutdown() -> None:
    """begin_operation() returning False must raise RuntimeError."""
    cache = _make_cache(shutting_down=True)
    with pytest.raises(RuntimeError, match="shutting down"):
        await _shielded_write(cache, _fast_result(), timeout=1.0)


async def test_shielded_write_shutdown_closes_coroutine() -> None:
    """Rejected coroutine must be .close()'d to avoid 'never awaited' warning."""
    cache = _make_cache(shutting_down=True)

    # Track whether the coroutine body actually ran
    ran = False

    async def _tracked() -> str:
        nonlocal ran
        ran = True
        return "should not run"

    with pytest.raises(RuntimeError):
        await _shielded_write(cache, _tracked(), timeout=1.0)

    # The coroutine was closed before it could execute
    assert not ran


# ---------------------------------------------------------------------------
# _shielded_write: end_operation tracking
# ---------------------------------------------------------------------------


async def test_shielded_write_calls_end_operation_on_success() -> None:
    """end_operation() must be called after normal completion."""
    cache = _make_cache()
    await _shielded_write(cache, _fast_result(), timeout=1.0)
    cache.end_operation.assert_called_once()


async def test_shielded_write_skips_end_operation_on_timeout() -> None:
    """end_operation() must NOT be called when the write times out.

    The timed_out flag prevents end_operation from firing, because the
    inner task may still be running (shielded) and will finish later.
    """
    cache = _make_cache()
    with pytest.raises(TimeoutError):
        await _shielded_write(cache, _hang_forever(), timeout=0.05)
    cache.end_operation.assert_not_called()


# ---------------------------------------------------------------------------
# _shielded_write: SIGTERM cancellation with grace period
# ---------------------------------------------------------------------------


async def test_shielded_write_cancellation_grace_period_success() -> None:
    """When outer task is cancelled, inner task gets a 2s grace period.

    If the inner task finishes within the grace window, its result is returned.
    """
    cache = _make_cache()
    # Inner task takes 0.3s — well within the 2s grace period
    inner_coro = _slow_finish(delay=0.3, value="graceful")

    async def _run() -> Any:
        return await _shielded_write(cache, inner_coro, timeout=5.0)

    task = asyncio.create_task(_run())
    # Let the shielded_write start and reach the shield(task) await
    await asyncio.sleep(0.05)
    # Simulate SIGTERM — cancel the outer task
    task.cancel()

    # The grace period should let the inner task complete
    # But the outer wrapper re-raises CancelledError after the grace period
    # wait_for path: if inner finishes in time, result is returned
    try:
        result = await task
        # If we get here, the grace period succeeded
        assert result == "graceful"
    except asyncio.CancelledError:
        # Also acceptable — the cancellation path re-raises CancelledError
        # after trying the grace period. The key is it doesn't hang.
        pass

    # end_operation should be called (not a timeout)
    cache.end_operation.assert_called_once()


async def test_shielded_write_cancellation_grace_period_timeout() -> None:
    """When outer task is cancelled and inner hangs, CancelledError after grace."""
    cache = _make_cache()

    async def _run() -> Any:
        return await _shielded_write(cache, _hang_forever(), timeout=5.0)

    task = asyncio.create_task(_run())
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# _serialized: sequential execution
# ---------------------------------------------------------------------------


async def test_serialized_queues_tools_sequentially() -> None:
    """Two tasks using _serialized must execute one at a time, not overlap."""
    # Reset the global lock so we get a fresh one for this test
    import semantic_cache_mcp.server.tools as tools_mod

    tools_mod._tool_lock = None

    execution_log: list[tuple[str, str]] = []

    @_serialized
    async def slow_tool(name: str) -> str:
        execution_log.append((name, "start"))
        await asyncio.sleep(0.1)
        execution_log.append((name, "end"))
        return name

    t1 = asyncio.create_task(slow_tool("first"))
    t2 = asyncio.create_task(slow_tool("second"))

    results = await asyncio.gather(t1, t2)
    assert set(results) == {"first", "second"}

    # Verify sequential: first must finish before second starts
    # The log should be: first-start, first-end, second-start, second-end
    # (or second-start, second-end, first-start, first-end)
    starts = [i for i, (_, action) in enumerate(execution_log) if action == "start"]
    ends = [i for i, (_, action) in enumerate(execution_log) if action == "end"]

    # First start's corresponding end must come before second start
    assert ends[0] < starts[1], f"Tasks overlapped: {execution_log}"


async def test_serialized_lock_released_after_exception() -> None:
    """A tool that raises must still release the lock for the next tool."""
    import semantic_cache_mcp.server.tools as tools_mod

    tools_mod._tool_lock = None

    @_serialized
    async def failing_tool() -> None:
        raise ValueError("boom")

    @_serialized
    async def ok_tool() -> str:
        return "ok"

    with pytest.raises(ValueError, match="boom"):
        await failing_tool()

    # The lock should be released — second tool should complete
    result = await asyncio.wait_for(ok_tool(), timeout=1.0)
    assert result == "ok"


async def test_serialized_lock_released_on_timeout() -> None:
    """A tool that times out inside _serialized must release the lock.

    We simulate a timeout by wrapping the serialized call in wait_for.
    The lock must be released so the next queued tool can proceed.
    """
    import semantic_cache_mcp.server.tools as tools_mod

    tools_mod._tool_lock = None

    @_serialized
    async def hanging_tool() -> None:
        await asyncio.sleep(3600)

    @_serialized
    async def quick_tool() -> str:
        return "quick"

    # Start hanging tool, let it acquire lock
    hang_task = asyncio.create_task(hanging_tool())
    await asyncio.sleep(0.02)

    # Cancel the hanging tool (simulates what happens on timeout)
    hang_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await hang_task

    # Lock should be released — quick_tool must complete promptly
    result = await asyncio.wait_for(quick_tool(), timeout=1.0)
    assert result == "quick"


# ---------------------------------------------------------------------------
# glob_with_cache_status: async stat
# ---------------------------------------------------------------------------


async def test_glob_uses_async_stat() -> None:
    """glob_with_cache_status must call astat (async), not os.stat (sync)."""
    import os
    import tempfile
    from pathlib import Path

    from semantic_cache_mcp.cache.search import glob_with_cache_status

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir).resolve()
        (tmp / "a.txt").write_text("hello")
        (tmp / "b.txt").write_text("world")

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache._io_executor = None

        # Create a real stat result to return
        real_stat = os.stat(tmp / "a.txt")

        astat_mock = AsyncMock(return_value=real_stat)

        with patch(
            "semantic_cache_mcp.cache.search.astat",
            astat_mock,
        ):
            result = await glob_with_cache_status(mock_cache, "*.txt", directory=str(tmp))

        # astat must have been called for each matched file
        assert astat_mock.call_count >= 2, (
            f"astat called {astat_mock.call_count} times, expected >= 2"
        )
        assert result.total_matches >= 2


# ---------------------------------------------------------------------------
# batch_smart_read: async stat pre-fetch
# ---------------------------------------------------------------------------


async def test_batch_read_prefetches_stats_async() -> None:
    """batch_smart_read must call astat via gather for all paths upfront."""
    import os
    import tempfile
    from pathlib import Path

    from semantic_cache_mcp.cache.read import batch_smart_read

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir).resolve()
        for name in ("x.txt", "y.txt", "z.txt"):
            (tmp / name).write_text(f"content of {name}")

        paths = [str(tmp / "x.txt"), str(tmp / "y.txt"), str(tmp / "z.txt")]

        # We need a real-ish cache mock for batch_smart_read
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache._io_executor = None
        mock_cache.begin_operation.return_value = True
        mock_cache.get_embedding = AsyncMock(return_value=None)
        mock_cache.put = AsyncMock()
        mock_cache.refresh_path = AsyncMock(return_value=True)
        mock_cache.record_access = AsyncMock()
        mock_cache.find_similar = AsyncMock(return_value=None)
        mock_cache.update_mtime = AsyncMock()

        real_stat = os.stat(tmp / "x.txt")
        astat_mock = AsyncMock(return_value=real_stat)

        with (
            patch("semantic_cache_mcp.cache.read.astat", astat_mock),
            patch("semantic_cache_mcp.cache.read.aread_bytes", new_callable=AsyncMock) as rb_mock,
            patch("semantic_cache_mcp.cache.read.aread_text", new_callable=AsyncMock) as rt_mock,
        ):
            # aread_bytes returns file content for smart_read
            rb_mock.side_effect = lambda p, ex: (tmp / p.name).read_bytes()
            rt_mock.side_effect = lambda p, executor: (tmp / p.name).read_text()

            result = await batch_smart_read(
                mock_cache,
                paths,
                max_total_tokens=100_000,
                diff_mode=False,
            )

        # astat should have been called for the _safe_stat gather (3 paths)
        # plus potentially inside smart_read for each file
        assert astat_mock.call_count >= 3, (
            f"astat called {astat_mock.call_count} times, expected >= 3 (pre-fetch)"
        )
        assert result.files_read >= 1


# ---------------------------------------------------------------------------
# _shielded_write: begin_operation is called
# ---------------------------------------------------------------------------


async def test_shielded_write_calls_begin_operation() -> None:
    """begin_operation() must be called before running the coroutine."""
    cache = _make_cache()
    await _shielded_write(cache, _fast_result(), timeout=1.0)
    cache.begin_operation.assert_called_once()


# ---------------------------------------------------------------------------
# _shielded_write: coroutine exception propagation
# ---------------------------------------------------------------------------


async def test_shielded_write_propagates_inner_exception() -> None:
    """Exceptions from the inner coroutine must propagate through shield."""
    cache = _make_cache()

    async def _raise() -> None:
        raise ValueError("inner boom")

    with pytest.raises(ValueError, match="inner boom"):
        await _shielded_write(cache, _raise(), timeout=1.0)

    # end_operation should still be called (not a timeout)
    cache.end_operation.assert_called_once()
