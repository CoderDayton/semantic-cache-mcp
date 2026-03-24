"""FastMCP protocol contract tests for the server surface."""

from __future__ import annotations

import pytest
from fastmcp import Client

from semantic_cache_mcp.server import mcp


@pytest.mark.asyncio
async def test_all_tools_expose_output_schema() -> None:
    async with Client(mcp, timeout=20, init_timeout=30) as client:
        tools = await client.list_tools()

    missing = [tool.name for tool in tools if tool.outputSchema is None]
    assert missing == []


@pytest.mark.asyncio
async def test_stats_returns_named_typed_output() -> None:
    async with Client(mcp, timeout=20, init_timeout=30) as client:
        result = await client.call_tool("stats", {}, raise_on_error=False)

    assert result.is_error is False
    assert result.data is not None
    assert type(result.data).__name__ == "StatsResponse"
    assert result.data.mode == "compact"
    assert result.data.storage is not None
