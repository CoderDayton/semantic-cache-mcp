"""Tests for the param-alias / did-you-mean middleware (item 11)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client

from semantic_cache_mcp.server._mcp import mcp


@pytest.fixture
async def mcp_client():
    """Connect a Client to the in-process FastMCP server."""
    async with Client(mcp) as client:
        yield client


async def _expect_tool_error(client, tool_name: str, args: dict, match: str) -> None:
    """Call a tool and assert the error contains `match`."""
    from fastmcp.exceptions import ClientError, ToolError

    try:
        await client.call_tool(tool_name, args)
    except (ToolError, ClientError) as e:
        assert match in str(e), f"expected {match!r} in {e!s}"
        return
    except Exception as e:
        # FastMCP may wrap into a generic Exception when transport raises.
        assert match in str(e), f"unexpected error type: {type(e).__name__}: {e}"
        return
    pytest.fail(f"expected error containing {match!r}; tool succeeded")


def _payload(result) -> dict:  # noqa: ANN001
    """Extract dict payload from a FastMCP tool result.

    fastmcp may return: a Pydantic response model, a dict, or wrap the
    payload in `structured_content` on the raw protocol. Normalize all
    shapes to a plain dict.
    """
    data = getattr(result, "data", None)
    if data is None:
        # Fall back to the raw structured content if `data` parsing failed.
        sc = getattr(result, "structured_content", None)
        if isinstance(sc, dict):
            return sc
    if hasattr(data, "model_dump"):
        return data.model_dump(exclude_none=True)
    if isinstance(data, dict):
        return data
    # Last resort: inspect attributes via vars().
    return vars(data) if data is not None else {}


async def test_abs_path_alias_rewritten_to_path(mcp_client, tmp_path: Path) -> None:
    """`abs_path=` is silently rewritten to `path=`."""
    f = tmp_path / "alias_test.txt"
    f.write_text("hello\n")
    result = await mcp_client.call_tool("read", {"abs_path": str(f)})
    payload = _payload(result)
    assert "hello" in payload["content"]


async def test_query_alias_rewritten_to_pattern(mcp_client, tmp_path: Path) -> None:
    """`query=` on grep is rewritten to `pattern=`."""
    f = tmp_path / "query_alias.py"
    f.write_text("def foo(): pass\n")
    # Seed cache via read first.
    await mcp_client.call_tool("read", {"path": str(f)})
    result = await mcp_client.call_tool("grep", {"query": "foo", "fixed_string": True})
    payload = _payload(result)
    assert payload["total_matches"] >= 1


async def test_paths_not_aliased_on_read(mcp_client, tmp_path: Path) -> None:
    """`paths` is intentionally NOT aliased to `path` — `batch_read` owns
    `paths` as a real parameter and a global rename would collapse a
    list-of-files call. On single-path tools it surfaces as did-you-mean.
    """
    f = tmp_path / "paths_alias.txt"
    f.write_text("contents\n")
    await _expect_tool_error(
        mcp_client,
        "read",
        {"paths": str(f)},
        "did you mean 'path'",
    )


async def test_unknown_param_gets_close_match_hint(mcp_client, tmp_path: Path) -> None:
    """A misspelled known param yields a `did you mean` hint."""
    f = tmp_path / "fuzz.txt"
    f.write_text("x\n")
    # `offsett` is one char off from `offset`.
    await _expect_tool_error(
        mcp_client,
        "read",
        {"path": str(f), "offsett": 1},
        "did you mean 'offset'",
    )


async def test_unknown_param_no_close_match(mcp_client, tmp_path: Path) -> None:
    """A completely unknown param errors cleanly, no false suggestions."""
    f = tmp_path / "no_match.txt"
    f.write_text("x\n")
    await _expect_tool_error(
        mcp_client,
        "read",
        {"path": str(f), "totally_unrelated_xyz": True},
        "unknown parameter 'totally_unrelated_xyz'",
    )


async def test_file_alias_to_path(mcp_client, tmp_path: Path) -> None:
    f = tmp_path / "file_alias.txt"
    f.write_text("data\n")
    result = await mcp_client.call_tool("read", {"file": str(f)})
    assert "data" in _payload(result)["content"]
