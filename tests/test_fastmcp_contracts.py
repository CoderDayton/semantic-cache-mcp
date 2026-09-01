"""FastMCP protocol contract tests for the server surface."""

from __future__ import annotations

import pytest
from fastmcp import Client

from semantic_cache_mcp import __version__
from semantic_cache_mcp.server import mcp
from semantic_cache_mcp.server._tool_models import TOOL_RESPONSE_MODELS


@pytest.mark.asyncio
async def test_initialize_reports_this_package_version() -> None:
    """`serverInfo.version` names this server, not whatever fastmcp is installed.

    Only the handshake era carries an initialize result at all, so the wire check
    runs in legacy mode; the sessionless era reads the same value off the server.
    """
    async with Client(mcp, mode="legacy", timeout=20, init_timeout=30) as client:
        info = client.initialize_result.server_info

    assert info.name == "semantic-cache-mcp"
    assert info.version == __version__
    assert mcp.version == __version__


@pytest.mark.asyncio
async def test_no_tool_publishes_an_output_schema() -> None:
    """Output schemas stay off the wire; the models remain the contract.

    See `tests/test_wire_footprint.py` for why, and for the guard that every
    tool still declares a response model in `TOOL_RESPONSE_MODELS`.
    """
    async with Client(mcp, timeout=20, init_timeout=30) as client:
        tools = await client.list_tools()

    assert [tool.name for tool in tools if tool.output_schema is not None] == []


# Fields a caller must keep and echo back for the cache to save anything.
_POSSESSION_FIELDS = ("content_hash", "file_hash", "coverage_token")


@pytest.mark.asyncio
async def test_every_tool_and_parameter_is_described() -> None:
    """FastMCP folds the `Args:` block into the input schema.

    A parameter with no description means that block lost an entry — or the
    whole block — and the tool ships something the model has to guess at.
    """
    async with Client(mcp, timeout=20, init_timeout=30) as client:
        tools = await client.list_tools()

    undescribed: list[str] = []
    for tool in tools:
        if not (tool.description or "").strip():
            undescribed.append(tool.name)
        props = (tool.input_schema or {}).get("properties") or {}
        undescribed += [
            f"{tool.name}.{name}"
            for name, schema in props.items()
            if not (schema.get("description") or "").strip()
        ]

    assert undescribed == []


@pytest.mark.asyncio
async def test_descriptions_open_with_a_one_line_summary() -> None:
    """The first line is what the model scans when choosing between tools."""
    async with Client(mcp, timeout=20, init_timeout=30) as client:
        tools = await client.list_tools()

    bad: list[str] = []
    for tool in tools:
        lines = (tool.description or "").splitlines()
        first = lines[0] if lines else ""
        if not first.endswith(".") or len(first) > 100:
            bad.append(f"{tool.name}: bad summary line {first!r}")
        if len(lines) > 1 and lines[1].strip():
            bad.append(f"{tool.name}: summary is not followed by a blank line")

    assert bad == []


@pytest.mark.asyncio
async def test_args_block_never_leaks_into_the_description() -> None:
    """A malformed `Args:` block stays in the prose instead of becoming schema."""
    async with Client(mcp, timeout=20, init_timeout=30) as client:
        tools = await client.list_tools()

    leaked = [
        tool.name
        for tool in tools
        if any(line.strip() == "Args:" for line in (tool.description or "").splitlines())
    ]

    assert leaked == []


@pytest.mark.asyncio
async def test_possession_fields_are_explained_where_they_can_be_returned() -> None:
    """A tool that can hand back a possession field must say so in its prose.

    These fields only save tokens if the caller knows to keep and echo them,
    so one that goes unmentioned is a saving nobody will ever collect.
    """
    async with Client(mcp, timeout=20, init_timeout=30) as client:
        tools = await client.list_tools()

    # Read the shape from the declared models, not the wire: the schemas are
    # deliberately unpublished, so `tool.output_schema` is None for every tool.
    schemas = {name: set(model.model_fields) for name, model in TOOL_RESPONSE_MODELS.items()}
    # Self-check: if the model shape ever changes, this guard would silently
    # inspect nothing and pass for the wrong reason.
    assert "coverage_token" in schemas["read"], "response model shape changed; guard is blind"

    missing = [
        f"{tool.name} can return `{field}` but never mentions it"
        for tool in tools
        for field in _POSSESSION_FIELDS
        if field in schemas[tool.name] and field not in (tool.description or "")
    ]

    assert missing == []


@pytest.mark.asyncio
async def test_stats_returns_one_readable_report() -> None:
    """`stats` answers in a single representation, and it carries the figures."""
    async with Client(mcp, timeout=20, init_timeout=30) as client:
        result = await client.call_tool("stats", {}, raise_on_error=False)

    assert result.is_error is False
    assert result.structured_content is None

    text = "".join(getattr(block, "text", "") for block in result.content)
    assert "Semantic Cache" in text
    assert "Storage:" in text
    assert "Session:" in text
