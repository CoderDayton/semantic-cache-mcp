"""Tests for the supervised tool worker subprocess."""

from __future__ import annotations

import os
import time
from multiprocessing.connection import Connection
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.exceptions import ToolError

from semantic_cache_mcp.server._tool_worker import (
    ToolProcessSupervisor,
    _tool_worker_main_async,
    _worker_result,
)
from semantic_cache_mcp.server.tools import read


def _echo_worker(conn: Connection) -> None:
    conn.send({"op": "ready"})
    while True:
        request = conn.recv()
        if request.get("op") == "shutdown":
            conn.close()
            return
        conn.send({"op": "result", "result": f"echo:{request['tool']}:{request['kwargs']}"})


def _restart_probe_worker(conn: Connection) -> None:
    state_file = Path(os.environ["SEMANTIC_CACHE_WORKER_STATE"])
    spawn_count = int(state_file.read_text()) if state_file.exists() else 0
    state_file.write_text(str(spawn_count + 1))

    conn.send({"op": "ready"})
    request = conn.recv()
    if request.get("op") == "shutdown":
        conn.close()
        return

    if spawn_count == 0:
        time.sleep(60)
        return

    conn.send({"op": "result", "result": "ok-after-restart"})
    while True:
        request = conn.recv()
        if request.get("op") == "shutdown":
            conn.close()
            return
        conn.send({"op": "result", "result": "ok-after-restart"})


def _tool_error_worker(conn: Connection) -> None:
    conn.send({"op": "ready"})
    while True:
        request = conn.recv()
        if request.get("op") == "shutdown":
            conn.close()
            return
        if request["kwargs"].get("fail"):
            conn.send({"op": "tool_error", "error": "read: synthetic failure"})
            continue
        conn.send({"op": "result", "result": "ok-after-tool-error"})


def _slow_restart_startup_worker(conn: Connection) -> None:
    state_file = Path(os.environ["SEMANTIC_CACHE_SLOW_RESTART_STATE"])
    spawn_count = int(state_file.read_text()) if state_file.exists() else 0
    state_file.write_text(str(spawn_count + 1))

    if spawn_count > 0:
        time.sleep(1.0)

    conn.send({"op": "ready"})
    request = conn.recv()
    if request.get("op") == "shutdown":
        conn.close()
        return

    if spawn_count == 0:
        time.sleep(60)
        return

    conn.send({"op": "result", "result": "ok-after-slow-restart"})
    while True:
        request = conn.recv()
        if request.get("op") == "shutdown":
            conn.close()
            return
        conn.send({"op": "result", "result": "ok-after-slow-restart"})


def _malformed_protocol_worker(conn: Connection) -> None:
    state_file = Path(os.environ["SEMANTIC_CACHE_PROTOCOL_STATE"])
    spawn_count = int(state_file.read_text()) if state_file.exists() else 0
    state_file.write_text(str(spawn_count + 1))

    conn.send({"op": "ready"})
    while True:
        request = conn.recv()
        if request.get("op") == "shutdown":
            conn.close()
            return
        if spawn_count == 0:
            conn.send({"op": "result"})
            continue
        conn.send({"op": "result", "result": "ok-after-protocol-restart"})


@pytest.mark.asyncio
async def test_tool_process_supervisor_round_trip() -> None:
    supervisor = ToolProcessSupervisor(worker_target=_echo_worker, startup_timeout=2.0)
    await supervisor.start()
    try:
        result = await supervisor.call_tool(
            "read",
            {"path": "/tmp/x"},
            output_mode="compact",
            max_response_tokens=None,
            timeout=1.0,
        )
    finally:
        await supervisor.async_close()

    assert "echo:read" in result


@pytest.mark.asyncio
async def test_tool_process_supervisor_restarts_after_timeout(tmp_path: Path) -> None:
    state_file = tmp_path / "worker-state.txt"
    old = os.environ.get("SEMANTIC_CACHE_WORKER_STATE")
    os.environ["SEMANTIC_CACHE_WORKER_STATE"] = str(state_file)

    supervisor = ToolProcessSupervisor(worker_target=_restart_probe_worker, startup_timeout=2.0)
    await supervisor.start()
    try:
        with pytest.raises(TimeoutError):
            await supervisor.call_tool(
                "read",
                {"path": "/tmp/hang"},
                output_mode="compact",
                max_response_tokens=None,
                timeout=0.1,
            )

        result = await supervisor.call_tool(
            "read",
            {"path": "/tmp/recovered"},
            output_mode="compact",
            max_response_tokens=None,
            timeout=1.0,
        )
    finally:
        await supervisor.async_close()
        if old is None:
            os.environ.pop("SEMANTIC_CACHE_WORKER_STATE", None)
        else:
            os.environ["SEMANTIC_CACHE_WORKER_STATE"] = old

    assert result == "ok-after-restart"
    assert state_file.read_text() == "2"


@pytest.mark.asyncio
async def test_tool_process_supervisor_timeout_does_not_wait_for_restart_startup(
    tmp_path: Path,
) -> None:
    state_file = tmp_path / "slow-restart-state.txt"
    old = os.environ.get("SEMANTIC_CACHE_SLOW_RESTART_STATE")
    os.environ["SEMANTIC_CACHE_SLOW_RESTART_STATE"] = str(state_file)

    supervisor = ToolProcessSupervisor(
        worker_target=_slow_restart_startup_worker,
        startup_timeout=2.0,
    )
    await supervisor.start()
    try:
        t0 = time.monotonic()
        with pytest.raises(TimeoutError):
            await supervisor.call_tool(
                "read",
                {"path": "/tmp/hang"},
                output_mode="compact",
                max_response_tokens=None,
                timeout=0.1,
            )
        elapsed = time.monotonic() - t0

        result = await supervisor.call_tool(
            "read",
            {"path": "/tmp/recovered"},
            output_mode="compact",
            max_response_tokens=None,
            timeout=2.0,
        )
    finally:
        await supervisor.async_close()
        if old is None:
            os.environ.pop("SEMANTIC_CACHE_SLOW_RESTART_STATE", None)
        else:
            os.environ["SEMANTIC_CACHE_SLOW_RESTART_STATE"] = old

    assert elapsed < 0.5
    assert result == "ok-after-slow-restart"
    assert state_file.read_text() == "2"


@pytest.mark.asyncio
async def test_tool_process_supervisor_preserves_tool_errors() -> None:
    supervisor = ToolProcessSupervisor(worker_target=_tool_error_worker, startup_timeout=2.0)
    await supervisor.start()
    try:
        with pytest.raises(ToolError, match="read: synthetic failure"):
            await supervisor.call_tool(
                "read",
                {"path": "/tmp/fail", "fail": True},
                output_mode="compact",
                max_response_tokens=None,
                timeout=1.0,
            )

        result = await supervisor.call_tool(
            "read",
            {"path": "/tmp/recovered"},
            output_mode="compact",
            max_response_tokens=None,
            timeout=1.0,
        )
    finally:
        await supervisor.async_close()

    assert result == "ok-after-tool-error"


@pytest.mark.asyncio
async def test_tool_process_supervisor_restarts_after_protocol_error(tmp_path: Path) -> None:
    state_file = tmp_path / "protocol-state.txt"
    old = os.environ.get("SEMANTIC_CACHE_PROTOCOL_STATE")
    os.environ["SEMANTIC_CACHE_PROTOCOL_STATE"] = str(state_file)

    supervisor = ToolProcessSupervisor(
        worker_target=_malformed_protocol_worker, startup_timeout=2.0
    )
    await supervisor.start()
    try:
        with pytest.raises(RuntimeError, match="result response missing 'result'"):
            await supervisor.call_tool(
                "read",
                {"path": "/tmp/fail"},
                output_mode="compact",
                max_response_tokens=None,
                timeout=1.0,
            )

        result = await supervisor.call_tool(
            "read",
            {"path": "/tmp/recovered"},
            output_mode="compact",
            max_response_tokens=None,
            timeout=1.0,
        )
    finally:
        await supervisor.async_close()
        if old is None:
            os.environ.pop("SEMANTIC_CACHE_PROTOCOL_STATE", None)
        else:
            os.environ["SEMANTIC_CACHE_PROTOCOL_STATE"] = old

    assert result == "ok-after-protocol-restart"
    assert state_file.read_text() == "2"


def test_worker_result_rejects_non_mapping_protocol_frames() -> None:
    with pytest.raises(RuntimeError, match="expected dict response"):
        _worker_result("bad-frame")


@pytest.mark.asyncio
async def test_tool_worker_startup_uses_embedding_metadata_without_warmup() -> None:
    conn = MagicMock()
    conn.recv.side_effect = [{"op": "shutdown"}]
    cache = MagicMock()
    cache._storage.clear_if_model_changed = MagicMock()
    cache.async_close = AsyncMock()

    with (
        patch("semantic_cache_mcp.server._tool_worker.get_tokenizer"),
        patch("semantic_cache_mcp.server._tool_worker.SemanticCache", return_value=cache),
        patch("semantic_cache_mcp.server._tool_worker.get_embedding_dim", return_value=768),
        patch(
            "semantic_cache_mcp.server._tool_worker.get_model_info",
            return_value={
                "model": "test-model",
                "dim": 0,
                "cache_dir": "/tmp/cache",
                "provider": "unknown",
                "ready": False,
            },
        ),
    ):
        await _tool_worker_main_async(conn)

    cache._storage.clear_if_model_changed.assert_called_once_with("test-model", 768)
    conn.send.assert_any_call({"op": "ready"})


@pytest.mark.asyncio
@patch("semantic_cache_mcp.server.tools._response_mode", return_value="compact")
@patch("semantic_cache_mcp.server.tools._response_token_cap", return_value=None)
async def test_read_uses_remote_runtime(_cap: MagicMock, _mode: MagicMock) -> None:
    remote = MagicMock()
    remote._is_tool_process_supervisor = True
    remote.call_tool = AsyncMock(
        return_value={"path": "/tmp/demo.py", "content": "remote-response"}
    )
    ctx = MagicMock()
    ctx.lifespan_context = {"cache": remote}

    result = await read(ctx, path="/tmp/demo.py")

    assert result == {"path": "/tmp/demo.py", "content": "remote-response"}
    remote.call_tool.assert_awaited_once()


@pytest.mark.asyncio
@patch("semantic_cache_mcp.server.tools._response_mode", return_value="compact")
@patch("semantic_cache_mcp.server.tools._response_token_cap", return_value=None)
async def test_read_remote_timeout_returns_error(_cap: MagicMock, _mode: MagicMock) -> None:
    remote = MagicMock()
    remote._is_tool_process_supervisor = True
    remote.call_tool = AsyncMock(side_effect=TimeoutError())
    ctx = MagicMock()
    ctx.lifespan_context = {"cache": remote}

    with pytest.raises(ToolError, match="read: timed out"):
        await read(ctx, path="/tmp/demo.py")
