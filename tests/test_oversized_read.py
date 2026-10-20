"""A full read of a file over `max_size` returns a map, never chosen chunks.

The map is the file's outline (`<line>: <signature>`) when it has definitions,
or a fixed head/tail sample with `// L<a>-<b>` anchors when it has none. Either
way every line number is a real `offset` into the file on disk, the body of the
file is not delivered, and the hash stays `partial:`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastmcp import Context

from semantic_cache_mcp.cache import SemanticCache, smart_read
from semantic_cache_mcp.server.response import _response_mode_override
from semantic_cache_mcp.server.tools import read

_BUDGET = 4000
_OUTLINE_LINE = re.compile(r"^(\d+): (.+)$")


def _code(n: int = 400) -> str:
    parts = ['"""Module."""\n\n']
    for i in range(n):
        parts.append(
            f"def handler_{i:03d}(payload: dict) -> int:\n"
            f'    """Handle event {i}."""\n'
            f'    return payload.get("count", 0) * {i}\n\n\n'
        )
    return "".join(parts)


def _log(n: int = 1000) -> str:
    return "".join(f"log entry {i:05d} status=ok\n" for i in range(n))


def _assert_outline_points_at_disk(body: str, source: str) -> None:
    disk = source.splitlines()
    lines = body.splitlines()
    assert lines, "outline is empty"
    for line in lines:
        if re.fullmatch(r"// \d+ more symbols omitted", line):
            continue
        match = _OUTLINE_LINE.match(line)
        assert match, f"not an outline line: {line!r}"
        number, signature = int(match.group(1)), match.group(2).strip()
        assert disk[number - 1].strip().startswith(signature), (
            f"line {number} is {disk[number - 1]!r}, outline says {signature!r}"
        )


def large_file_map(content: str, *, filename: str) -> str:
    from semantic_cache_mcp.core import large_file_map as fn

    return fn(content, filename=filename)


@pytest.fixture()
def ctx(tmp_path: Path) -> MagicMock:
    c = MagicMock(spec=Context)
    c.lifespan_context = {"cache": SemanticCache(db_path=tmp_path / "cache.db")}
    return c


class TestLargeFileMap:
    def test_code_maps_to_its_outline(self) -> None:
        source = _code()
        body = large_file_map(source, filename="big.py")

        _assert_outline_points_at_disk(body, source)
        assert body.splitlines()[0] == "3: def handler_000(payload: dict) -> int:"
        assert "return payload.get" not in body

    def test_unstructured_text_maps_to_anchored_head_and_tail(self) -> None:
        source = _log()
        lines = source.splitlines()
        body = large_file_map(source, filename="app.log")

        assert "// L1-20\n" + "\n".join(lines[:20]) in body
        assert "// L981-1000\n" + "\n".join(lines[980:]) in body
        assert "entry 00500" not in body

    def test_one_huge_line_is_clipped(self) -> None:
        body = large_file_map("x" * 200_000, filename="bundle.min.js")
        assert len(body) < 5000

    def test_short_unstructured_file_is_not_sampled_twice(self) -> None:
        source = "a=1 " * 50_000 + "\nlast line\n"
        body = large_file_map(source, filename="data.txt")
        assert body.count("// L") == 1


class TestSmartReadOversized:
    async def test_full_read_returns_the_map(self, tmp_path: Path) -> None:
        cache = SemanticCache(db_path=tmp_path / "cache.db")
        p = tmp_path / "big.py"
        p.write_text(_code())

        result = await smart_read(cache, str(p), max_size=_BUDGET, force_full=True)

        assert result.truncated is True
        _assert_outline_points_at_disk(result.content, p.read_text())


class TestReadToolOversized:
    async def _read(self, ctx: MagicMock, path: Path, **kw: Any) -> dict[str, Any]:
        return await read(ctx, str(path), **kw)

    async def test_code_file_returns_outline_and_partial_hash(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        p = tmp_path / "big.py"
        p.write_text(_code())

        d = await self._read(ctx, p, max_size=_BUDGET)

        assert d["truncated"] is True
        assert "content_hash" not in d
        assert d["file_hash"].startswith("partial:")
        assert d["total_lines"] == len(p.read_text().splitlines())
        _assert_outline_points_at_disk(d["content"], p.read_text())

    async def test_hint_points_at_ranged_reads_not_a_summary(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        p = tmp_path / "big.py"
        p.write_text(_code())

        token = _response_mode_override.set("normal")
        try:
            d = await self._read(ctx, p, max_size=_BUDGET)
        finally:
            _response_mode_override.reset(token)

        assert "offset" in d["hint"]
        assert "summarized" not in d["hint"]

    async def test_log_file_returns_head_and_tail(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "app.log"
        p.write_text(_log())

        d = await self._read(ctx, p, max_size=_BUDGET)

        assert d["truncated"] is True
        assert d["total_lines"] == 1000
        assert "log entry 00000" in d["content"]
        assert "log entry 00999" in d["content"]
        assert "log entry 00500" not in d["content"]
