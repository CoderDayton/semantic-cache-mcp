"""Real stdio MCP transport regression tests."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from fastmcp.client import Client
from fastmcp.client.transports.stdio import StdioTransport
from fastmcp.exceptions import ToolError

from semantic_cache_mcp.config import CACHE_DIR


def _flatten_tool_result(result: object) -> str:
    if hasattr(result, "content"):
        blocks: list[str] = []
        for item in result.content:  # type: ignore[attr-defined]
            text = getattr(item, "text", None)
            blocks.append(text if text is not None else str(item))
        return "\n".join(blocks)
    return str(result)


def _link_or_copy_tree(src: Path, dst: Path) -> None:
    try:
        dst.symlink_to(src, target_is_directory=True)
    except OSError:
        shutil.copytree(src, dst)


def _prepare_runtime_cache(temp_cache: Path) -> None:
    tokenizer_src = CACHE_DIR / "tokenizer"
    models_src = CACHE_DIR / "models"
    tokenizer_file = tokenizer_src / "o200k_base.tiktoken"

    if not tokenizer_file.exists():
        pytest.skip("stdio integration test requires cached tokenizer assets")
    if not models_src.exists():
        pytest.skip("stdio integration test requires cached embedding model assets")

    temp_cache.mkdir(parents=True, exist_ok=True)
    _link_or_copy_tree(tokenizer_src, temp_cache / "tokenizer")
    _link_or_copy_tree(models_src, temp_cache / "models")


@pytest.mark.asyncio
async def test_stdio_timeout_returns_error_and_next_call_succeeds(tmp_path: Path) -> None:
    """A timed-out stdio tool call must not kill the session for the next call."""
    runtime_cache = tmp_path / "runtime-cache"
    _prepare_runtime_cache(runtime_cache)

    files_dir = tmp_path / "files"
    files_dir.mkdir()

    paths: list[str] = []
    payload = "alpha beta gamma delta epsilon\n" * 4000
    for i in range(20):
        p = files_dir / f"big_{i}.txt"
        p.write_text(payload + f"file={i}\n")
        paths.append(str(p))

    env = os.environ.copy()
    env["SEMANTIC_CACHE_DIR"] = str(runtime_cache)
    env["TOOL_TIMEOUT"] = "0.2"
    env["TOOL_OUTPUT_MODE"] = "debug"
    env["EMBEDDING_DEVICE"] = "cpu"
    env["LOG_LEVEL"] = "WARNING"

    server_log = tmp_path / "server.stderr.log"
    transport = StdioTransport(
        command="uv",
        args=["run", "semantic-cache-mcp"],
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
        keep_alive=False,
        log_file=server_log,
    )

    async with Client(transport, timeout=20, init_timeout=30) as client:
        first = await client.call_tool(
            "batch_read",
            {"paths": json.dumps(paths), "max_total_tokens": 200_000},
            raise_on_error=False,
        )
        with pytest.raises(ToolError, match="batch_read: timed out"):
            await client.call_tool(
                "batch_read",
                {"paths": json.dumps(paths), "max_total_tokens": 200_000},
            )
        second = await client.call_tool("stats", {}, raise_on_error=False)

    first_body = _flatten_tool_result(first)
    second_body = _flatten_tool_result(second)

    assert first.is_error is True
    assert "timed out" in first_body
    assert second.is_error is False
    assert second.data is not None
    assert type(second.data).__name__ == "StatsResponse"
    assert second.data.storage is not None
    assert second.data.session is not None
    assert "files_cached" in second_body
