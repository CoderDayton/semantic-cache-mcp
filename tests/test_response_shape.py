"""What grep, glob and search spend their tokens on.

The findings behind this file, all measured on real responses:

  * every match carried a `{"line_number":N,"line":"..."}` envelope — ~16
    tokens of braces and repeated key names, ~1.6k on a capped grep;
  * every path was absolute and fully repeated, ~25 tokens each against ~12
    for the repo-relative form, and `glob` returns up to 1000 of them;
  * `context_lines` sliced a fresh window per match with no overlap merging,
    so clustered matches re-sent the same source line up to four times;
  * "which files mention X" paid for every matching line;
  * prose hints restated what `complete`/`limit_reached` already encoded.

None of that may be fixed by making the answer vaguer: a compact response
still has to name the exact line number of every line it shows, and a path it
shortens has to be reconstructable from the `root` it reports.
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
from semantic_cache_mcp.server._paths import relativize, shared_root
from semantic_cache_mcp.server.response import _minimal_payload, _response_overrides
from semantic_cache_mcp.server.tools import glob, grep, search, warm

_BODY = """import os


def alpha():
    marker_token = 1
    return marker_token


def beta():
    marker_token = 2
    return marker_token
"""


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
    for name in ("one", "two", "three"):
        (root / "pkg" / f"{name}.py").write_text(_BODY)
    return root


async def _warmed(ctx: MagicMock, project: Path) -> None:
    await warm(ctx, str(project / "pkg" / "*.py"))


def _absolute(payload: dict, relative: str) -> str:
    root = payload.get("root")
    return f"{root}/{relative}" if root else relative


class TestSharedRoot:
    def test_a_common_directory_is_found(self) -> None:
        assert (
            shared_root(["/home/dev/project/pkg/one.py", "/home/dev/project/pkg/two.py"])
            == "/home/dev/project/pkg"
        )

    def test_the_prefix_is_component_wise_not_textual(self) -> None:
        """`/home/dev/bc` and `/home/dev/bd` share `/home/dev`, never `/home/dev/b`."""
        assert (
            shared_root(["/home/dev/project/bc/one.py", "/home/dev/project/bd/two.py"])
            == "/home/dev/project"
        )

    def test_a_prefix_too_short_to_pay_for_itself_is_skipped(self) -> None:
        """Naming `/a/b` costs more than the four characters it removes twice."""
        assert shared_root(["/a/b/one.py", "/a/b/two.py"]) is None

    def test_a_single_path_earns_no_root(self) -> None:
        """Naming a root and then a relative path costs more than one path."""
        assert shared_root(["/some/deep/path/one.py"]) is None

    def test_a_useless_root_is_not_reported(self) -> None:
        assert shared_root(["/a.py", "/b.py"]) is None

    def test_no_paths_is_no_root(self) -> None:
        assert shared_root([]) is None

    def test_mixed_relative_and_absolute_does_not_raise(self) -> None:
        assert shared_root(["relative/one.py", "/absolute/two.py"]) is None

    def test_relativize_round_trips(self) -> None:
        paths = ["/home/dev/project/one.py", "/home/dev/project/pkg/two.py"]
        root = shared_root(paths)
        assert root == "/home/dev/project"
        for path in paths:
            assert f"{root}/{relativize(path, root)}" == path

    def test_a_path_outside_the_root_is_left_alone(self) -> None:
        assert relativize("/elsewhere/x.py", "/a/b") == "/elsewhere/x.py"

    def test_a_sibling_with_a_shared_prefix_is_not_relativized(self) -> None:
        """`/a/bc` must not be read as living under `/a/b`."""
        assert relativize("/a/bc/x.py", "/a/b") == "/a/bc/x.py"


class TestGrepPaths:
    async def test_a_shared_root_is_reported_once(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        payload = await grep(ctx, pattern="marker_token")

        assert payload["root"] == str(project / "pkg")
        assert all(not entry["path"].startswith("/") for entry in payload["files"])

    async def test_paths_rebuild_into_the_real_files(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        payload = await grep(ctx, pattern="marker_token")

        for entry in payload["files"]:
            assert Path(_absolute(payload, entry["path"])).is_file()

    async def test_relative_paths_are_cheaper(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        payload = await grep(ctx, pattern="marker_token")

        shortened = count_tokens(json.dumps(payload))
        absolute = count_tokens(
            json.dumps(
                {
                    **payload,
                    "files": [
                        {**f, "path": _absolute(payload, f["path"])} for f in payload["files"]
                    ],
                }
            )
        )
        assert shortened < absolute


class TestGrepMatchShape:
    async def test_matches_are_number_prefixed_strings(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        payload = await grep(ctx, pattern="marker_token")

        entry = payload["files"][0]
        assert all(isinstance(line, str) for line in entry["lines"])
        assert entry["lines"][0].startswith("5:")

    async def test_the_per_match_envelope_is_gone(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        payload = await grep(ctx, pattern="marker_token")

        rendered = json.dumps(payload)
        assert "line_number" not in rendered
        assert '"count"' not in rendered, "per-file count sits next to the list it counts"

    async def test_every_line_number_is_truthful(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        payload = await grep(ctx, pattern="marker_token", context_lines=2)

        for entry in payload["files"]:
            disk = Path(_absolute(payload, entry["path"])).read_text().splitlines()
            for rendered in entry["lines"]:
                number, text = _split_line(rendered)
                assert disk[number - 1] == text, f"line {number} is not what the file holds"

    async def test_a_match_line_and_a_context_line_are_distinguishable(
        self, ctx: MagicMock, project: Path
    ) -> None:
        await _warmed(ctx, project)
        payload = await grep(ctx, pattern="marker_token", context_lines=1)

        lines = payload["files"][0]["lines"]
        separators = {_separator(line) for line in lines}

        assert ":" in separators, "no match line"
        assert "-" in separators, "no context line"
        assert all("marker_token" in text for text in _texts(lines, ":"))


class TestGrepContextIsMergedNotRepeated:
    async def test_overlapping_windows_emit_each_line_once(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        target = tmp_path / "dense.py"
        target.write_text("a\nb\nhit\nc\nhit\nd\ne\n")
        await warm(ctx, str(target))

        payload = await grep(ctx, pattern="hit", fixed_string=True, context_lines=3)
        lines = payload["files"][0]["lines"]
        numbers = [_split_line(line)[0] for line in lines]

        assert numbers == sorted(numbers)
        assert len(numbers) == len(set(numbers)), f"a line was sent twice: {lines}"

    async def test_a_match_wins_over_the_same_line_as_context(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        target = tmp_path / "adjacent.py"
        target.write_text("hit\nhit\n")
        await warm(ctx, str(target))

        payload = await grep(ctx, pattern="hit", fixed_string=True, context_lines=1)
        lines = payload["files"][0]["lines"]

        assert lines == ["1:hit", "2:hit"]

    async def test_merging_actually_saves_tokens(self, ctx: MagicMock, tmp_path: Path) -> None:
        target = tmp_path / "clustered.py"
        target.write_text("".join(f"line_{i} hit\n" if i % 2 else f"line_{i}\n" for i in range(20)))
        await warm(ctx, str(target))

        merged = await grep(ctx, pattern="hit", fixed_string=True, context_lines=3)
        lines = merged["files"][0]["lines"]
        assert len(lines) <= 20, "merged context cannot exceed the file's line count"


class TestGrepOutputModes:
    async def test_paths_mode_drops_the_lines(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        payload = await grep(ctx, pattern="marker_token", output="paths")

        assert payload["files"]
        assert all("lines" not in entry for entry in payload["files"])
        assert payload["total_matches"] == 12

    async def test_count_mode_drops_the_files(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        payload = await grep(ctx, pattern="marker_token", output="count")

        assert "files" not in payload
        assert payload["total_matches"] == 12
        assert payload["files_matched"] == 3

    async def test_count_mode_is_drastically_cheaper(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        full = await grep(ctx, pattern="marker_token")
        counted = await grep(ctx, pattern="marker_token", output="count")

        assert count_tokens(json.dumps(counted)) < count_tokens(json.dumps(full)) / 2

    @pytest.mark.parametrize("bad", ["lines", "", "MATCHES", "all"])
    async def test_an_unknown_output_mode_is_refused(
        self, ctx: MagicMock, project: Path, bad: str
    ) -> None:
        with pytest.raises(ToolError, match="output"):
            await grep(ctx, pattern="marker_token", output=bad)


class TestGrepEchoAndHints:
    async def test_compact_mode_does_not_echo_the_pattern(
        self, ctx: MagicMock, project: Path
    ) -> None:
        await _warmed(ctx, project)
        with _response_overrides("compact", None):
            payload = await grep(ctx, pattern="marker_token")

        assert "pattern" not in payload
        assert "path" not in payload

    async def test_normal_mode_echoes_for_traceability(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        with _response_overrides("normal", None):
            payload = await grep(ctx, pattern="marker_token")

        assert payload["pattern"] == "marker_token"

    async def test_a_capped_scan_says_so_without_prose(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        payload = await grep(ctx, pattern="marker_token", max_matches=2)

        assert payload["complete"] is False
        assert payload["limit_reached"] == "max_matches"
        assert "hint" not in payload, "complete/limit_reached already carry this"

    async def test_the_cache_miss_hint_survives(self, ctx: MagicMock, tmp_path: Path) -> None:
        """This one names a tool the caller should call, so it earns its tokens."""
        payload = await grep(ctx, pattern="anything", path="nowhere/absent.py")

        assert payload["reason"] == "no_files_cached_under_path"
        assert "warm" in payload["hint"]


class TestGlobPaths:
    async def test_glob_reports_a_root(self, ctx: MagicMock, project: Path) -> None:
        payload = await glob(ctx, pattern="*.py", directory=str(project / "pkg"))

        assert payload["root"] == str(project / "pkg")
        assert all(not m["path"].startswith("/") for m in payload["matches"])

    async def test_glob_paths_rebuild(self, ctx: MagicMock, project: Path) -> None:
        payload = await glob(ctx, pattern="*.py", directory=str(project / "pkg"))

        for match in payload["matches"]:
            assert Path(_absolute(payload, match["path"])).is_file()


class TestSearch:
    async def test_similarity_is_two_decimals(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        payload = await search(ctx, query="marker_token")

        for match in payload["matches"]:
            assert match["similarity"] == round(match["similarity"], 2)

    async def test_paths_are_relative_to_a_reported_root(
        self, ctx: MagicMock, project: Path
    ) -> None:
        await _warmed(ctx, project)
        payload = await search(ctx, query="marker_token")

        assert payload["root"] == str(project / "pkg")
        for match in payload["matches"]:
            assert Path(_absolute(payload, match["path"])).is_file()

    async def test_the_match_count_is_not_repeated(self, ctx: MagicMock, project: Path) -> None:
        await _warmed(ctx, project)
        with _response_overrides("normal", None):
            payload = await search(ctx, query="marker_token")

        assert "count" not in payload, "the matches array is right there"

    async def test_the_preview_shows_why_the_file_matched(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        target = tmp_path / "buried.py"
        target.write_text(
            '"""Module docstring."""\n\nimport os\nimport sys\n\n'
            + "filler = 0\n" * 40
            + "def unmistakable_symbol():\n    return 1\n"
        )
        await warm(ctx, str(target))

        payload = await search(ctx, query="unmistakable_symbol", show_preview=True)
        preview = payload["matches"][0]["preview"]

        assert "unmistakable_symbol" in preview, (
            f"preview shows the file head, not the match: {preview!r}"
        )


class TestGracefulTruncation:
    async def test_an_over_budget_grep_keeps_some_matches(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        """Dropping every match forces a second call — a net token loss."""
        target = tmp_path / "many.py"
        target.write_text("hit\n" * 400)
        await warm(ctx, str(target))

        with _response_overrides("compact", 400):
            payload = await grep(ctx, pattern="hit", fixed_string=True, max_matches=400)

        assert payload["files"][0]["lines"], "a capped grep still has room for some matches"
        assert payload["truncated_matches"] > 0, "and it must say how many it cut"
        assert payload["total_matches"] == 400

    def test_the_refit_keeps_lines_that_fit(self) -> None:
        """`_minimal_payload` used to drop every match, forcing a second call."""
        payload = {
            "ok": True,
            "tool": "grep",
            "total_matches": 3,
            "files": [{"path": "a.py", "lines": ["1:hit", "2:hit", "3:hit"]}],
        }
        minimal = _minimal_payload(payload, 200)

        assert minimal["files"][0]["lines"] == ["1:hit", "2:hit", "3:hit"]
        assert "truncated" not in minimal

    def test_the_refit_reports_what_it_could_not_keep(self) -> None:
        payload = {
            "ok": True,
            "tool": "grep",
            "total_matches": 3,
            "files": [{"path": "a.py", "lines": ["1:" + "x" * 400, "2:" + "x" * 400]}],
        }
        minimal = _minimal_payload(payload, 60)

        assert minimal["truncated"] is True
        assert len(minimal.get("files", [{}])[0].get("lines", [])) <= 1

    def test_without_a_budget_the_old_stripping_still_applies(self) -> None:
        minimal = _minimal_payload({"ok": True, "tool": "grep", "files": [{"path": "a.py"}]})
        assert minimal["truncated"] is True
        assert "files" not in minimal


def _prefix_end(rendered: str) -> int:
    """Index of the separator that closes the line-number prefix."""
    for index, char in enumerate(rendered):
        if char in ":-":
            return index
    raise AssertionError(f"no line-number prefix in {rendered!r}")


def _split_line(rendered: str) -> tuple[int, str]:
    """Parse `"<n>:<text>"` (a match) or `"<n>-<text>"` (a context line)."""
    index = _prefix_end(rendered)
    return int(rendered[:index]), rendered[index + 1 :]


def _separator(rendered: str) -> str:
    return rendered[_prefix_end(rendered)]


def _texts(lines: list[str], separator: str) -> list[str]:
    return [_split_line(line)[1] for line in lines if _separator(line) == separator]
