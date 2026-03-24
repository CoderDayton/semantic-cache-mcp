"""Tests for the supervised tool worker subprocess."""

from __future__ import annotations

import os
import time
from multiprocessing.connection import Connection
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from semantic_cache_mcp.server._tool_worker import ToolProcessSupervisor
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
@patch("semantic_cache_mcp.server.tools._response_mode", return_value="compact")
@patch("semantic_cache_mcp.server.tools._response_token_cap", return_value=None)
async def test_read_uses_remote_runtime(_cap: MagicMock, _mode: MagicMock) -> None:
    remote = MagicMock()
    remote._is_tool_process_supervisor = True
    remote.call_tool = AsyncMock(return_value="remote-response")
    ctx = MagicMock()
    ctx.lifespan_context = {"cache": remote}

    result = await read(ctx, path="/tmp/demo.py")

    assert result == "remote-response"
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

    result = await read(ctx, path="/tmp/demo.py")

    assert "timed out" in result.lower()
