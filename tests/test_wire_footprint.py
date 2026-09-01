"""What the server puts on the wire, and how much of it the model pays for.

Two costs are checked here, because both are paid on every single turn and
neither shows up in any per-call metric:

  * the advertised tool surface (`tools/list`), which sits in the prompt
    prefix of every request;
  * the tool result, which MCP lets a server send twice — once as a text
    block and again as `structuredContent`. Clients disagree about which they
    forward, so a server that emits both is gambling with the caller's budget.

Neither is free to get wrong in the other direction: dropping the published
output schema must not drop the *contract*, so the response models stay and
this file proves every tool still has one.
"""

from __future__ import annotations

import json

import pytest
from fastmcp import Client

from semantic_cache_mcp.config import resolve_output_policy
from semantic_cache_mcp.core import count_tokens
from semantic_cache_mcp.server import mcp
from semantic_cache_mcp.server._tool_models import TOOL_RESPONSE_MODELS

# The advertised surface is prompt-prefix cost on every request. Claude Code
# demotes an MCP server behind a tool-search step once its descriptions pass
# ~10% of the context window, so this ceiling is a real behavioural cliff and
# not a style preference. Raise it only with a measurement that says why.
_MAX_ADVERTISED_TOKENS = 11_000


async def _tools():
    async with Client(mcp, timeout=20, init_timeout=30) as client:
        return await client.list_tools()


class TestOutputSchemasAreNotPublished:
    async def test_no_tool_advertises_an_output_schema(self) -> None:
        tools = await _tools()
        published = [t.name for t in tools if t.output_schema is not None]

        assert published == [], (
            "output schemas are ~58% of this server's advertised bytes and the "
            "Anthropic Messages API has no field to put them in"
        )

    async def test_every_tool_still_has_a_declared_response_model(self) -> None:
        """Dropping the published schema must not drop the drift guard."""
        tools = await _tools()
        missing = [t.name for t in tools if t.name not in TOOL_RESPONSE_MODELS]

        assert missing == [], f"tools with no response model: {missing}"

    async def test_registry_has_no_entries_for_tools_that_do_not_exist(self) -> None:
        tools = {t.name for t in await _tools()}
        stale = sorted(set(TOOL_RESPONSE_MODELS) - tools)

        assert stale == [], f"response models for absent tools: {stale}"


class TestResultsAreSentOnce:
    async def test_a_tool_result_carries_no_structured_duplicate(self, tmp_path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("def hello():\n    return 'world'\n")

        async with Client(mcp, timeout=20, init_timeout=30) as client:
            result = await client.call_tool("read", {"path": str(target)})

        assert result.structured_content is None, (
            "the response body is on the wire twice; every file read costs double"
        )
        assert result.content, "stripping the duplicate must not strip the answer"

    async def test_the_surviving_text_block_is_the_whole_payload(self, tmp_path) -> None:
        target = tmp_path / "mod.py"
        target.write_text("def hello():\n    return 'world'\n")

        async with Client(mcp, timeout=20, init_timeout=30) as client:
            result = await client.call_tool("read", {"path": str(target)})

        text = "".join(getattr(block, "text", "") for block in result.content)
        payload = json.loads(text)

        assert payload["content"] == "def hello():\n    return 'world'\n"
        assert payload["content_hash"]

    async def test_an_error_still_reports_as_an_error(self, tmp_path) -> None:
        async with Client(mcp, timeout=20, init_timeout=30) as client:
            result = await client.call_tool(
                "read", {"path": str(tmp_path / "absent.py")}, raise_on_error=False
            )

        assert result.is_error is True

    async def test_an_image_result_keeps_its_image_block(self, tmp_path) -> None:
        png = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000a49444154789c6360000002000100ffff03000006000557bfabd400"
            "00000049454e44ae426082"
        )
        target = tmp_path / "dot.png"
        target.write_bytes(png)

        async with Client(mcp, timeout=20, init_timeout=30) as client:
            result = await client.call_tool("read_image", {"path": str(target)})

        kinds = {getattr(block, "type", "") for block in result.content}
        assert "image" in kinds, "stripping structured content must not drop content blocks"
        assert result.structured_content is None


class TestAdvertisedSurfaceStaysSmall:
    async def test_total_advertised_tokens_are_under_the_ceiling(self) -> None:
        tools = await _tools()
        blob = json.dumps(
            [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                }
                for t in tools
            ]
        )
        total = count_tokens(blob)

        assert total <= _MAX_ADVERTISED_TOKENS, (
            f"advertised tool surface is {total} tokens, over the {_MAX_ADVERTISED_TOKENS} ceiling"
        )


class TestOutputPolicyResolution:
    """MCP requires `structuredContent` from any tool that declares a schema.

    Publishing schemas while suppressing structured content would emit a
    protocol violation on every call, so the policy resolves that combination
    rather than letting an operator configure it.
    """

    @pytest.mark.parametrize(
        ("publish", "structured", "expected"),
        [
            (False, False, (False, False)),
            (False, True, (False, True)),
            (True, True, (True, True)),
            (True, False, (True, True)),
        ],
    )
    def test_schema_publication_forces_structured_content(
        self, publish: bool, structured: bool, expected: tuple[bool, bool]
    ) -> None:
        assert resolve_output_policy(publish, structured) == expected
