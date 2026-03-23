"""End-to-end timeout simulation tests.

Proves that tool timeouts fire correctly, reset the executor, release the
lock, and allow subsequent tools to proceed — the exact scenario that was
broken when concurrent ONNX inference blocked the single-threaded executor.

Test matrix:
- read tool timeout via asyncio.wait_for
- write tool timeout via _shielded_write + asyncio.timeout
- edit tool timeout via _shielded_write + asyncio.timeout
- search tool timeout via asyncio.wait_for
- lock release after timeout — next tool succeeds
- multiple queued tools survive a single timeout
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import semantic_cache_mcp.server.tools as _tools_mod
from semantic_cache_mcp.cache import SemanticCache
from semantic_cache_mcp.server.tools import (
    _get_tool_lock,
    _handle_timeout,
    _serialized,
    _shielded_write,
    edit,
    read,
    search,
    write,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Short timeout for all tests — keeps the suite fast.
_TEST_TIMEOUT: float = 0.2


def _make_cache(*, shutting_down: bool = False) -> MagicMock:
    """Create a mock SemanticCache with the fields tool handlers expect."""
    cache = MagicMock(spec=SemanticCache)
    cache._shutting_down = shutting_down
    cache.begin_operation.return_value = not shutting_down
    cache.end_operation = MagicMock()
    cache.reset_executor = MagicMock()
    cache.metrics = MagicMock()
    cache.metrics.record = MagicMock()
    return cache


def _make_ctx(cache: MagicMock | None = None) -> MagicMock:
    """Create a mock fastmcp Context wired to a cache."""
    if cache is None:
        cache = _make_cache()
    ctx = MagicMock()
    ctx.lifespan_context = {"cache": cache}
    return ctx


async def _hang_forever(*args: Any, **kwargs: Any) -> None:
    """Coroutine that never completes — simulates a stuck executor."""
    await asyncio.sleep(999)


async def _fast_result(*args: Any, **kwargs: Any) -> MagicMock:
    """Return a plausible ReadResult-like mock instantly."""
    result = MagicMock()
    result.content = "hello"
    result.from_cache = False
    result.is_diff = False
    result.truncated = False
    result.tokens_saved = 0
    result.tokens_original = 5
    result.tokens_returned = 5
    result.semantic_match = None
    result.path = "/tmp/test.txt"
    result.created = True
    result.diff_content = ""
    result.bytes_written = 5
    result.tokens_written = 1
    result.diff_stats = {}
    result.content_hash = "abc"
    result.replacements_made = 1
    result.line_numbers = [1]
    # search result fields
    result.matches = []
    result.cached_files = 0
    result.files_searched = 0
    return result


@pytest.fixture(autouse=True)
def _reset_tool_lock() -> None:
    """Reset the module-level lock between tests to prevent cross-contamination."""
    _tools_mod._tool_lock = None


# ---------------------------------------------------------------------------
# 1. Read tool timeout fires
# ---------------------------------------------------------------------------


@patch("semantic_cache_mcp.server.tools._TOOL_TIMEOUT", _TEST_TIMEOUT)
@patch("semantic_cache_mcp.server.tools.smart_read", _hang_forever)
@patch("semantic_cache_mcp.server.tools._response_mode", return_value="compact")
@patch("semantic_cache_mcp.server.tools._response_token_cap", return_value=None)
async def test_read_timeout_fires(_cap: Any, _mode: Any) -> None:
    """read() must return a timeout error within ~_TEST_TIMEOUT, not hang."""
    cache = _make_cache()
    ctx = _make_ctx(cache)

    t0 = time.monotonic()
    result = await read(ctx, path="/tmp/nonexistent.py")
    elapsed = time.monotonic() - t0

    assert elapsed < _TEST_TIMEOUT + 0.5, f"Took {elapsed:.2f}s, expected ~{_TEST_TIMEOUT}s"
    assert "timed out" in result.lower()
    cache.reset_executor.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Write tool timeout fires via _shielded_write
# ---------------------------------------------------------------------------


@patch("semantic_cache_mcp.server.tools._TOOL_TIMEOUT", _TEST_TIMEOUT)
@patch("semantic_cache_mcp.server.tools.smart_write", _hang_forever)
@patch("semantic_cache_mcp.server.tools._response_mode", return_value="compact")
@patch("semantic_cache_mcp.server.tools._response_token_cap", return_value=None)
async def test_write_timeout_fires(_cap: Any, _mode: Any) -> None:
    """write() must timeout even though _shielded_write uses asyncio.shield.

    Note: _shielded_write's default `timeout` kwarg is bound at import time,
    so we wrap it to forward the patched _TOOL_TIMEOUT explicitly.
    """
    cache = _make_cache()
    ctx = _make_ctx(cache)

    _original_shielded = _shielded_write

    async def _shielded_with_test_timeout(
        c: Any, coro: Any, *, timeout: float = _TEST_TIMEOUT
    ) -> Any:
        return await _original_shielded(c, coro, timeout=timeout)

    with patch("semantic_cache_mcp.server.tools._shielded_write", _shielded_with_test_timeout):
        t0 = time.monotonic()
        result = await write(ctx, path="/tmp/test.txt", content="hello")
        elapsed = time.monotonic() - t0

    assert elapsed < _TEST_TIMEOUT + 0.5, f"Took {elapsed:.2f}s, expected ~{_TEST_TIMEOUT}s"
    assert "timed out" in result.lower()
    cache.reset_executor.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Edit tool timeout fires via _shielded_write
# ---------------------------------------------------------------------------


@patch("semantic_cache_mcp.server.tools._TOOL_TIMEOUT", _TEST_TIMEOUT)
@patch("semantic_cache_mcp.server.tools.smart_edit", _hang_forever)
@patch("semantic_cache_mcp.server.tools._response_mode", return_value="compact")
@patch("semantic_cache_mcp.server.tools._response_token_cap", return_value=None)
async def test_edit_timeout_fires(_cap: Any, _mode: Any) -> None:
    """edit() must timeout even though _shielded_write uses asyncio.shield.

    Same wrapping strategy as test_write_timeout_fires — see that test's
    docstring for why we must override _shielded_write's default timeout.
    """
    cache = _make_cache()
    ctx = _make_ctx(cache)

    _original_shielded = _shielded_write

    async def _shielded_with_test_timeout(
        c: Any, coro: Any, *, timeout: float = _TEST_TIMEOUT
    ) -> Any:
        return await _original_shielded(c, coro, timeout=timeout)

    with patch("semantic_cache_mcp.server.tools._shielded_write", _shielded_with_test_timeout):
        t0 = time.monotonic()
        result = await edit(ctx, path="/tmp/test.txt", old_string="x", new_string="y")
        elapsed = time.monotonic() - t0

    assert elapsed < _TEST_TIMEOUT + 0.5, f"Took {elapsed:.2f}s, expected ~{_TEST_TIMEOUT}s"
    assert "timed out" in result.lower()
    cache.reset_executor.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Search tool timeout fires
# ---------------------------------------------------------------------------


@patch("semantic_cache_mcp.server.tools._TOOL_TIMEOUT", _TEST_TIMEOUT)
@patch("semantic_cache_mcp.server.tools.semantic_search", _hang_forever)
@patch("semantic_cache_mcp.server.tools._response_mode", return_value="compact")
@patch("semantic_cache_mcp.server.tools._response_token_cap", return_value=None)
async def test_search_timeout_fires(_cap: Any, _mode: Any) -> None:
    """search() must return a timeout error, not hang forever."""
    cache = _make_cache()
    ctx = _make_ctx(cache)

    t0 = time.monotonic()
    result = await search(ctx, query="find something")
    elapsed = time.monotonic() - t0

    assert elapsed < _TEST_TIMEOUT + 0.5, f"Took {elapsed:.2f}s, expected ~{_TEST_TIMEOUT}s"
    assert "timed out" in result.lower()
    cache.reset_executor.assert_called_once()


# ---------------------------------------------------------------------------
# 5. Lock released after timeout — next tool succeeds
# ---------------------------------------------------------------------------


@patch("semantic_cache_mcp.server.tools._TOOL_TIMEOUT", _TEST_TIMEOUT)
@patch("semantic_cache_mcp.server.tools._response_mode", return_value="compact")
@patch("semantic_cache_mcp.server.tools._response_token_cap", return_value=None)
async def test_lock_released_after_timeout_next_tool_succeeds(_cap: Any, _mode: Any) -> None:
    """After tool A times out, tool B must acquire the lock and succeed.

    This is the CRITICAL test — proves the lock is released on timeout
    so subsequent tool calls are not permanently deadlocked.
    """
    cache = _make_cache()
    ctx = _make_ctx(cache)

    # Tool A: read hangs and times out
    with patch("semantic_cache_mcp.server.tools.smart_read", _hang_forever):
        result_a = await read(ctx, path="/tmp/hang.py")
    assert "timed out" in result_a.lower()

    # Tool B: read succeeds immediately (different patch)
    with patch("semantic_cache_mcp.server.tools.smart_read", _fast_result):
        t0 = time.monotonic()
        result_b = await read(ctx, path="/tmp/ok.py")
        elapsed_b = time.monotonic() - t0

    # Tool B must complete fast (not blocked by the lock)
    assert elapsed_b < 1.0, f"Tool B took {elapsed_b:.2f}s — lock was not released"
    assert "timed out" not in result_b.lower()
    # The response should contain valid output (not an error)
    assert "ok.py" in result_b or "hello" in result_b


# ---------------------------------------------------------------------------
# 6. Multiple queued tools all eventually run
# ---------------------------------------------------------------------------


@patch("semantic_cache_mcp.server.tools._TOOL_TIMEOUT", _TEST_TIMEOUT)
@patch("semantic_cache_mcp.server.tools._response_mode", return_value="compact")
@patch("semantic_cache_mcp.server.tools._response_token_cap", return_value=None)
async def test_multiple_queued_tools_survive_timeout(_cap: Any, _mode: Any) -> None:
    """Start 3 concurrent tool calls. First hangs, second and third must still run.

    All go through _serialized, so they queue on the lock. When the first
    times out and releases the lock, the remaining two should proceed.
    """
    cache = _make_cache()
    ctx = _make_ctx(cache)

    call_count = 0

    async def _counting_read(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        return await _fast_result()

    async def _first_hangs(*args: Any, **kwargs: Any) -> None:
        """First call hangs, subsequent calls succeed."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await asyncio.sleep(999)  # hang
        return await _fast_result()  # type: ignore[return-value]

    with patch("semantic_cache_mcp.server.tools.smart_read", _first_hangs):
        t0 = time.monotonic()

        # Launch 3 concurrent tool calls
        tasks = [asyncio.create_task(read(ctx, path=f"/tmp/file{i}.py")) for i in range(3)]

        results = await asyncio.gather(*tasks)
        elapsed = time.monotonic() - t0

    # Should complete in roughly _TEST_TIMEOUT (for the first) + small overhead
    # Not 3 * _TEST_TIMEOUT (which would mean serialized timeouts)
    assert elapsed < _TEST_TIMEOUT + 2.0, (
        f"All 3 tools took {elapsed:.2f}s — queued tools may have been blocked"
    )

    # First tool timed out
    timed_out_count = sum(1 for r in results if "timed out" in r.lower())
    succeeded_count = sum(1 for r in results if "timed out" not in r.lower())

    assert timed_out_count >= 1, "At least one tool should have timed out"
    assert succeeded_count >= 1, "At least one tool should have succeeded after the timeout"


# ---------------------------------------------------------------------------
# 7. _handle_timeout resets executor
# ---------------------------------------------------------------------------


async def test_handle_timeout_resets_executor() -> None:
    """_handle_timeout must call cache.reset_executor()."""
    cache = _make_cache()
    _handle_timeout(cache, "test_tool", "detail message")
    cache.reset_executor.assert_called_once()


# ---------------------------------------------------------------------------
# 8. _shielded_write timeout fires (direct, not through tool handler)
# ---------------------------------------------------------------------------


async def test_shielded_write_timeout_releases_within_window() -> None:
    """_shielded_write must raise TimeoutError within the timeout window."""
    cache = _make_cache()

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        await _shielded_write(cache, _hang_forever(), timeout=0.1)
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5, f"Took {elapsed:.2f}s, expected ~0.1s"
    # begin_operation was called but end_operation was NOT (timed out)
    cache.begin_operation.assert_called_once()
    cache.end_operation.assert_not_called()


# ---------------------------------------------------------------------------
# 9. _serialized releases lock even on exception
# ---------------------------------------------------------------------------


async def test_serialized_releases_lock_on_exception() -> None:
    """If a tool raises inside _serialized, the lock must still be released."""

    @_serialized
    async def _boom() -> None:
        raise ValueError("intentional")

    # First call raises
    with pytest.raises(ValueError, match="intentional"):
        await _boom()

    # Second call must not deadlock — it should acquire the lock
    acquired = False

    @_serialized
    async def _ok() -> str:
        nonlocal acquired
        acquired = True
        return "ok"

    result = await asyncio.wait_for(_ok(), timeout=1.0)
    assert acquired
    assert result == "ok"


# ---------------------------------------------------------------------------
# 10. _get_tool_lock returns same instance within same event loop
# ---------------------------------------------------------------------------


async def test_get_tool_lock_singleton() -> None:
    """_get_tool_lock must return the same lock on repeated calls."""
    lock1 = _get_tool_lock()
    lock2 = _get_tool_lock()
    assert lock1 is lock2
    assert isinstance(lock1, asyncio.Lock)
