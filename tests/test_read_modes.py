"""Ranged reads without a line-number tax, and the structural outline mode.

Two token costs are settled here:

  * A ranged read used to prefix every line with a six-column gutter, measured
    at ~17% of the window on real source. The window's line range is already in
    `lines`, so the gutter is now opt-in.
  * A large file's only cheap answer used to be a summary — thousands of tokens
    of selected segments. `outline=true` returns one line per definition with
    the line number a follow-up ranged read needs.
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
from semantic_cache_mcp.server.tools import read

SOURCE = '''"""Module."""


class Widget:
    def __init__(self, name):
        self.name = name

    def render(self):
        return self.name


def helper(a, b):
    return a + b
'''


@pytest.fixture
def cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(db_path=tmp_path / "cache.db")


@pytest.fixture
def ctx(cache: SemanticCache) -> MagicMock:
    context = MagicMock(spec=Context)
    context.lifespan_context = {"cache": cache}
    return context


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    target = tmp_path / "mod.py"
    target.write_text(SOURCE)
    return target


def _payload(result) -> dict:  # noqa: ANN001
    assert isinstance(result, dict), f"expected a payload dict, got {type(result).__name__}"
    return result


class TestRangedReadLineNumbers:
    async def test_default_ranged_read_returns_literal_lines(
        self, ctx: MagicMock, source_file: Path
    ) -> None:
        payload = _payload(await read(ctx, str(source_file), offset=4, limit=3))
        expected = "\n".join(SOURCE.splitlines()[3:6])

        assert payload["content"] == expected

    async def test_default_ranged_read_has_no_gutter(
        self, ctx: MagicMock, source_file: Path
    ) -> None:
        payload = _payload(await read(ctx, str(source_file), offset=1, limit=5))
        assert "\t" not in payload["content"]

    async def test_line_numbers_true_restores_the_gutter(
        self, ctx: MagicMock, source_file: Path
    ) -> None:
        payload = _payload(await read(ctx, str(source_file), offset=4, limit=1, line_numbers=True))
        assert payload["content"].startswith("     4\t")

    async def test_the_range_is_reported_either_way(
        self, ctx: MagicMock, source_file: Path
    ) -> None:
        """Dropping the gutter must not drop the caller's ability to locate the window."""
        for line_numbers in (False, True):
            payload = _payload(
                await read(ctx, str(source_file), offset=4, limit=3, line_numbers=line_numbers)
            )
            assert payload["lines"] == {"start": 4, "end": 6, "total": len(SOURCE.splitlines())}

    async def test_the_gutter_is_what_costs_the_tokens(
        self, ctx: MagicMock, source_file: Path
    ) -> None:
        plain = _payload(await read(ctx, str(source_file), offset=1, limit=12))["content"]
        numbered = _payload(
            await read(ctx, str(source_file), offset=1, limit=12, line_numbers=True)
        )["content"]

        assert count_tokens(numbered) > count_tokens(plain)

    async def test_coverage_token_still_redeems_a_held_window(
        self, ctx: MagicMock, source_file: Path
    ) -> None:
        first = _payload(await read(ctx, str(source_file), offset=1, limit=4))
        token = first["coverage_token"]

        second = _payload(await read(ctx, str(source_file), offset=1, limit=4, known_hash=token))
        assert second["unchanged"] is True

    async def test_a_whole_file_window_still_mints_a_claimable_hash(
        self, ctx: MagicMock, source_file: Path
    ) -> None:
        payload = _payload(await read(ctx, str(source_file), offset=1, limit=1000))
        assert "content_hash" in payload
        assert not payload["content_hash"].startswith("partial:")


class TestOutlineMode:
    async def test_outline_lists_definitions_with_line_numbers(
        self, ctx: MagicMock, source_file: Path
    ) -> None:
        payload = _payload(await read(ctx, str(source_file), outline=True))

        assert payload["outline"] is True
        assert payload["symbols"] >= 3
        lines = payload["content"].splitlines()
        assert any(line.endswith("class Widget:") for line in lines)
        assert all(line.split(":", 1)[0].isdigit() for line in lines)

    async def test_outline_line_numbers_point_at_the_real_lines(
        self, ctx: MagicMock, source_file: Path
    ) -> None:
        payload = _payload(await read(ctx, str(source_file), outline=True))
        disk = SOURCE.splitlines()

        for entry in payload["content"].splitlines():
            number, _, text = entry.partition(":")
            assert text.strip()[:20] in disk[int(number) - 1]

    async def test_outline_is_far_cheaper_than_the_file(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        big = tmp_path / "big.py"
        big.write_text(
            "".join(
                f"def function_{i}(argument):\n"
                f'    """Do the work for case {i}."""\n'
                f"    total = argument * {i}\n"
                f"    for step in range({i} + 1):\n"
                f"        total += step * argument\n"
                f"    if total < 0:\n"
                f"        raise ValueError('negative total')\n"
                f"    return total\n\n"
                for i in range(400)
            )
        )
        outline = _payload(await read(ctx, str(big), outline=True))["content"]

        assert count_tokens(outline) < count_tokens(big.read_text()) / 5

    async def test_outline_is_not_possession(self, ctx: MagicMock, source_file: Path) -> None:
        """A map of a file is not the file, so it earns no claimable hash."""
        payload = _payload(await read(ctx, str(source_file), outline=True))

        assert "content_hash" not in payload
        assert payload["file_hash"].startswith("partial:")

    async def test_an_outline_hash_is_refused_as_a_possession_claim(
        self, ctx: MagicMock, source_file: Path
    ) -> None:
        outlined = _payload(await read(ctx, str(source_file), outline=True))
        payload = _payload(await read(ctx, str(source_file), known_hash=outlined["file_hash"]))

        assert payload.get("unchanged") is not True
        assert payload["content"] == SOURCE

    async def test_a_file_with_no_definitions_says_so(self, ctx: MagicMock, tmp_path: Path) -> None:
        flat = tmp_path / "data.txt"
        flat.write_text("alpha\nbeta\ngamma\n")

        payload = _payload(await read(ctx, str(flat), outline=True))

        assert payload["symbols"] == 0
        assert payload["reason"] == "no_definitions_found"
        assert payload["hint"]
        assert "content" not in payload

    async def test_outline_reports_the_file_length(self, ctx: MagicMock, source_file: Path) -> None:
        payload = _payload(await read(ctx, str(source_file), outline=True))
        assert payload["total_lines"] == len(SOURCE.splitlines())

    async def test_outline_of_a_binary_file_returns_binary_metadata(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        blob = tmp_path / "blob.bin"
        blob.write_bytes(b"\x00\x01\x02\xff" * 64)

        payload = _payload(await read(ctx, str(blob), outline=True))

        assert payload["is_binary"] is True
        assert "content" not in payload

    async def test_outline_of_a_missing_file_is_an_error(self, ctx: MagicMock) -> None:
        with pytest.raises(ToolError, match="read: "):
            await read(ctx, "/nonexistent/never.py", outline=True)

    async def test_outline_seeds_the_cache_like_a_read(
        self, ctx: MagicMock, cache: SemanticCache, source_file: Path
    ) -> None:
        await read(ctx, str(source_file), outline=True)
        assert await cache.get(str(source_file.resolve())) is not None


class TestConflictingModes:
    @pytest.mark.parametrize("window", [{"offset": 2}, {"limit": 3}, {"offset": 2, "limit": 3}])
    async def test_outline_with_a_line_window_is_refused(
        self, ctx: MagicMock, source_file: Path, window: dict
    ) -> None:
        """Silently picking one of two requested modes is how a caller reads the wrong thing."""
        with pytest.raises(ToolError, match="outline"):
            await read(ctx, str(source_file), outline=True, **window)

    async def test_line_numbers_on_a_whole_file_read_is_refused(
        self, ctx: MagicMock, source_file: Path
    ) -> None:
        with pytest.raises(ToolError, match="line_numbers"):
            await read(ctx, str(source_file), line_numbers=True)


class TestRemoteForwarding:
    async def test_new_parameters_reach_the_worker(self) -> None:
        """`_forward_kwargs` reads the signature, so a new param must forward."""
        import inspect

        params = set(inspect.signature(read).parameters)
        assert {"outline", "line_numbers"} <= params


class TestPayloadShape:
    async def test_outline_payload_is_json_serializable(
        self, ctx: MagicMock, source_file: Path
    ) -> None:
        payload = _payload(await read(ctx, str(source_file), outline=True))
        assert json.loads(json.dumps(payload)) == payload
