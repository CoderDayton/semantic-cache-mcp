"""`warm`: make files searchable without paying to read them.

`grep` and `search` only see cached files, and the only way to cache a file was
`batch_read`, which returns every byte. Seeding this repo's `src/` that way
costs ~212k tokens to enable a 300-token grep. `warm` indexes the same files
and returns counts.

The invariant that matters is negative: no path through this tool may put file
content in the response. Everything else here guards the second rule — that a
file which does not get indexed is *reported*, never silently missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError

from semantic_cache_mcp.cache import SemanticCache
from semantic_cache_mcp.core import count_tokens
from semantic_cache_mcp.server.tools import grep, warm

_SECRET = "distinctive_marker_token"


@pytest.fixture
def cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(db_path=tmp_path / "cache.db")


@pytest.fixture
def ctx(cache: SemanticCache) -> MagicMock:
    context = MagicMock(spec=Context)
    context.lifespan_context = {"cache": cache}
    return context


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    for i in range(5):
        (root / "pkg" / f"mod_{i}.py").write_text(
            f"def function_{i}():\n    return {i}  # {_SECRET}\n"
        )
    (root / "README.md").write_text(f"# Project\n\n{_SECRET}\n")
    return root


def _payload(result) -> dict:  # noqa: ANN001
    assert isinstance(result, dict), f"expected a payload dict, got {type(result).__name__}"
    return result


def _all_strings(value) -> list[str]:  # noqa: ANN001
    """Every string anywhere in a nested payload."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _all_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _all_strings(v)]
    return []


class TestNoContentEverLeaves:
    async def test_response_contains_no_file_text(self, ctx: MagicMock, project: Path) -> None:
        payload = _payload(await warm(ctx, str(project / "pkg" / "*.py")))

        assert all(_SECRET not in s for s in _all_strings(payload)), (
            "warm leaked file content into its response"
        )

    async def test_response_has_no_content_field(self, ctx: MagicMock, project: Path) -> None:
        payload = _payload(await warm(ctx, str(project / "pkg" / "*.py")))
        assert "content" not in payload
        assert "contents" not in payload
        assert "files" not in payload

    async def test_response_stays_tiny(self, ctx: MagicMock, project: Path) -> None:
        payload = _payload(await warm(ctx, str(project / "**" / "*")))
        rendered = json.dumps(payload, separators=(",", ":"))

        assert count_tokens(rendered) < 200, f"warm response is not cheap: {rendered}"


class TestItActuallyWarms:
    async def test_grep_finds_warmed_files(self, ctx: MagicMock, project: Path) -> None:
        before = _payload(await grep(ctx, pattern=_SECRET))
        assert before["total_matches"] == 0

        await warm(ctx, str(project / "pkg" / "*.py"))

        after = _payload(await grep(ctx, pattern=_SECRET))
        assert after["total_matches"] >= 5

    async def test_counts_what_it_indexed(self, ctx: MagicMock, project: Path) -> None:
        payload = _payload(await warm(ctx, str(project / "pkg" / "*.py")))

        assert payload["warmed"] == 5
        assert payload["already_current"] == 0
        assert payload["tokens_indexed"] > 0

    async def test_a_second_warm_reindexes_nothing(self, ctx: MagicMock, project: Path) -> None:
        pattern = str(project / "pkg" / "*.py")
        await warm(ctx, pattern)
        payload = _payload(await warm(ctx, pattern))

        assert payload["warmed"] == 0
        assert payload["already_current"] == 5

    async def test_a_changed_file_is_reindexed(self, ctx: MagicMock, project: Path) -> None:
        target = project / "pkg" / "mod_0.py"
        pattern = str(project / "pkg" / "*.py")
        await warm(ctx, pattern)

        target.write_text("def replaced():\n    return 'new_body_marker'\n")
        payload = _payload(await warm(ctx, pattern))

        assert payload["warmed"] == 1
        matches = _payload(await grep(ctx, pattern="new_body_marker"))
        assert matches["total_matches"] == 1


class TestInputHandling:
    async def test_accepts_a_comma_separated_list(self, ctx: MagicMock, project: Path) -> None:
        paths = f"{project / 'pkg' / 'mod_0.py'},{project / 'pkg' / 'mod_1.py'}"
        payload = _payload(await warm(ctx, paths))
        assert payload["warmed"] == 2

    async def test_accepts_a_json_array(self, ctx: MagicMock, project: Path) -> None:
        paths = json.dumps([str(project / "pkg" / "mod_0.py")])
        payload = _payload(await warm(ctx, paths))
        assert payload["warmed"] == 1

    async def test_empty_paths_is_a_clear_error(self, ctx: MagicMock) -> None:
        with pytest.raises(ToolError, match="warm: "):
            await warm(ctx, "   ")

    async def test_malformed_json_is_a_clear_error(self, ctx: MagicMock) -> None:
        with pytest.raises(ToolError, match="warm: "):
            await warm(ctx, '["unclosed')

    @pytest.mark.parametrize("bad", [0, -1, 100_000])
    async def test_out_of_range_max_files_is_rejected(
        self, ctx: MagicMock, project: Path, bad: int
    ) -> None:
        with pytest.raises(ToolError, match="max_files"):
            await warm(ctx, str(project / "pkg" / "*.py"), max_files=bad)


class TestNothingFailsSilently:
    async def test_a_missing_file_is_reported_not_raised(
        self, ctx: MagicMock, project: Path
    ) -> None:
        payload = _payload(await warm(ctx, str(project / "pkg" / "absent.py")))

        assert payload["skipped"] == 1
        reasons = {entry["reason"] for entry in payload["failures"]}
        assert reasons == {"not_found"}

    async def test_a_directory_is_reported_not_indexed(self, ctx: MagicMock, project: Path) -> None:
        payload = _payload(await warm(ctx, str(project / "pkg")))

        assert payload["skipped"] == 1
        assert payload["failures"][0]["reason"] == "not_a_file"

    async def test_a_binary_file_is_reported(self, ctx: MagicMock, project: Path) -> None:
        blob = project / "blob.bin"
        blob.write_bytes(b"\x00\x01\x02\xff" * 64)

        payload = _payload(await warm(ctx, str(blob)))

        assert payload["skipped"] == 1
        assert payload["failures"][0]["reason"] == "binary"

    async def test_an_oversized_file_is_reported(self, ctx: MagicMock, project: Path) -> None:
        from semantic_cache_mcp.server.tools import _WARM_MAX_FILE_BYTES

        huge = project / "huge.txt"
        huge.write_text("x" * (_WARM_MAX_FILE_BYTES + 1))

        payload = _payload(await warm(ctx, str(huge)))

        assert payload["skipped"] == 1
        assert payload["failures"][0]["reason"] == "too_large"

    async def test_partial_success_reports_both_halves(self, ctx: MagicMock, project: Path) -> None:
        paths = f"{project / 'pkg' / 'mod_0.py'},{project / 'pkg' / 'absent.py'}"
        payload = _payload(await warm(ctx, paths))

        assert payload["warmed"] == 1
        assert payload["skipped"] == 1

    async def test_more_files_than_the_cap_says_so(self, ctx: MagicMock, project: Path) -> None:
        payload = _payload(await warm(ctx, str(project / "pkg" / "*.py"), max_files=2))

        assert payload["warmed"] == 2
        assert payload["truncated"] is True
        assert payload["hint"]

    async def test_exactly_the_cap_is_not_called_truncated(
        self, ctx: MagicMock, project: Path
    ) -> None:
        """A glob matching exactly ``max_files`` files left nothing behind."""
        payload = _payload(await warm(ctx, str(project / "pkg" / "*.py"), max_files=5))

        assert payload["warmed"] == 5
        assert "truncated" not in payload

    async def test_the_failure_list_is_bounded(self, ctx: MagicMock, tmp_path: Path) -> None:
        from semantic_cache_mcp.server.tools import _WARM_MAX_FAILURES_REPORTED

        missing = ",".join(str(tmp_path / f"absent_{i}.py") for i in range(40))
        payload = _payload(await warm(ctx, missing))

        assert payload["skipped"] == 40
        assert len(payload["failures"]) == _WARM_MAX_FAILURES_REPORTED


class TestOverTheWire:
    """The same flow through a real MCP client, worker subprocess included."""

    async def test_warm_then_grep_then_outline(self, project: Path) -> None:
        from fastmcp import Client

        from semantic_cache_mcp.server import mcp

        async with Client(mcp, timeout=30, init_timeout=30) as client:
            warmed = await client.call_tool("warm", {"paths": str(project / "pkg" / "*.py")})
            body = json.loads("".join(getattr(b, "text", "") for b in warmed.content))
            assert body["warmed"] == 5
            assert warmed.structured_content is None
            assert all(_SECRET not in s for s in _all_strings(body))

            found = await client.call_tool("grep", {"pattern": _SECRET, "fixed_string": True})
            matches = json.loads("".join(getattr(b, "text", "") for b in found.content))
            assert matches["total_matches"] >= 5

            outlined = await client.call_tool(
                "read", {"path": str(project / "pkg" / "mod_0.py"), "outline": True}
            )
            mapped = json.loads("".join(getattr(b, "text", "") for b in outlined.content))
            assert mapped["outline"] is True
            assert "def function_0():" in mapped["content"]


class TestPayloadShape:
    async def test_payload_is_json_serializable(self, ctx: MagicMock, project: Path) -> None:
        payload = _payload(await warm(ctx, str(project / "pkg" / "*.py")))
        assert json.loads(json.dumps(payload)) == payload

    async def test_the_counts_are_always_present(self, ctx: MagicMock, project: Path) -> None:
        """A caller must be able to tell what happened without parsing prose."""
        payload = _payload(await warm(ctx, str(project / "pkg" / "*.py")))
        assert {"warmed", "already_current", "skipped"} <= set(payload)
