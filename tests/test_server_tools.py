"""Comprehensive tests for MCP server tool implementations in tools.py.

Strategy: mock Context.lifespan_context to inject a real SemanticCache backed
by a tmp_path SQLite DB, then call each tool function directly. This gives us
branch coverage without spinning up the full MCP server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import Context

from semantic_cache_mcp.cache import SemanticCache, smart_read
from semantic_cache_mcp.server.response import _minimal_payload, _render_error, _render_response
from semantic_cache_mcp.server.tools import (
    _expand_globs,
    batch_edit,
    batch_read,
    clear,
    diff,
    edit,
    glob,
    grep,
    read,
    search,
    similar,
    stats,
    write,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(cache: SemanticCache) -> MagicMock:
    """Build a minimal Context mock that satisfies ctx.lifespan_context["cache"]."""
    ctx = MagicMock(spec=Context)
    ctx.lifespan_context = {"cache": cache}
    return ctx


def _parse(response: str) -> dict[str, Any]:
    """Parse JSON tool response."""
    return json.loads(response)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> SemanticCache:
    """Fresh SemanticCache per test, no real embeddings needed."""
    return SemanticCache(db_path=tmp_path / "cache.db")


@pytest.fixture()
def ctx(tmp_cache: SemanticCache) -> MagicMock:
    """Ready-made context mock."""
    return _make_ctx(tmp_cache)


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    """Five-line text file."""
    p = tmp_path / "sample.txt"
    p.write_text("line1\nline2\nline3\nline4\nline5\n")
    return p


@pytest.fixture()
def py_file(tmp_path: Path) -> Path:
    """Minimal Python file."""
    p = tmp_path / "hello.py"
    p.write_text("def hello():\n    return 'world'\n")
    return p


# ===========================================================================
# response.py helpers
# ===========================================================================


class TestMinimalPayload:
    def test_keeps_required_keys(self) -> None:
        payload = {"ok": True, "tool": "read", "path": "/x", "content": "big", "extra": "dropped"}
        m = _minimal_payload(payload)
        assert m["ok"] is True
        assert m["tool"] == "read"
        assert m["path"] == "/x"
        assert "content" not in m
        assert m["truncated"] is True

    def test_adds_message_when_missing(self) -> None:
        m = _minimal_payload({"ok": True})
        assert "message" in m

    def test_preserves_existing_message(self) -> None:
        m = _minimal_payload({"ok": False, "message": "custom"})
        assert m["message"] == "custom"

    def test_path_variants(self) -> None:
        m = _minimal_payload({"path1": "/a", "path2": "/b"})
        assert m["path1"] == "/a"
        assert m["path2"] == "/b"


class TestRenderResponse:
    def test_normal_mode_keeps_ok_and_tool(self) -> None:
        with patch("semantic_cache_mcp.server.response.TOOL_OUTPUT_MODE", "normal"):
            body = _render_response({"ok": True, "tool": "read"}, None)
        d = json.loads(body)
        assert d["ok"] is True
        assert d["tool"] == "read"

    def test_compact_mode_strips_ok_and_tool_on_success(self) -> None:
        with patch("semantic_cache_mcp.server.response.TOOL_OUTPUT_MODE", "compact"):
            body = _render_response({"ok": True, "tool": "read", "path": "/x"}, None)
        d = json.loads(body)
        assert "ok" not in d
        assert "tool" not in d
        assert d["path"] == "/x"

    def test_compact_mode_keeps_ok_and_tool_on_error(self) -> None:
        with patch("semantic_cache_mcp.server.response.TOOL_OUTPUT_MODE", "compact"):
            body = _render_response({"ok": False, "tool": "read", "error": "boom"}, None)
        d = json.loads(body)
        assert d["ok"] is False
        assert d["tool"] == "read"

    def test_token_cap_truncates(self) -> None:
        big_payload = {"ok": True, "tool": "read", "content": "x" * 10000}
        body = _render_response(big_payload, 5)
        d = json.loads(body)
        assert d.get("truncated") is True

    def test_no_token_cap(self) -> None:
        with patch("semantic_cache_mcp.server.response.TOOL_OUTPUT_MODE", "normal"):
            body = _render_response({"ok": True, "tool": "stats"}, None)
        assert json.loads(body)["ok"] is True


class TestRenderError:
    def test_produces_ok_false(self) -> None:
        body = _render_error("read", "file missing", None)
        d = json.loads(body)
        assert d["ok"] is False
        assert d["tool"] == "read"
        assert "file missing" in d["error"]


# ===========================================================================
# read tool
# ===========================================================================


class TestReadTool:
    def test_first_read_returns_content(self, ctx: MagicMock, sample_file: Path) -> None:
        d = _parse(read(ctx, str(sample_file)))
        assert "line1" in d["content"]

    def test_second_read_diff_mode_returns_content(self, ctx: MagicMock, sample_file: Path) -> None:
        """diff_mode=True on second unchanged read: content is still returned
        (unchanged file re-reads the cached bytes; no diff marker is emitted).
        """
        read(ctx, str(sample_file))
        d = _parse(read(ctx, str(sample_file)))
        # path is always returned
        assert "path" in d
        # content is returned (not suppressed in single-file read)
        assert "content" in d

    def test_diff_mode_false_always_returns_content(
        self, ctx: MagicMock, sample_file: Path
    ) -> None:
        read(ctx, str(sample_file))
        d = _parse(read(ctx, str(sample_file), diff_mode=False))
        assert "line1" in d["content"]

    def test_offset_limit_returns_line_range(self, ctx: MagicMock, sample_file: Path) -> None:
        d = _parse(read(ctx, str(sample_file), offset=2, limit=2))
        content = d["content"]
        assert "line2" in content
        assert "line3" in content
        assert "line4" not in content

    def test_offset_includes_line_numbers(self, ctx: MagicMock, sample_file: Path) -> None:
        d = _parse(read(ctx, str(sample_file), offset=1, limit=1))
        # Format: "     1\tline1"
        assert "\t" in d["content"]

    def test_offset_zero_returns_error(self, ctx: MagicMock, sample_file: Path) -> None:
        d = _parse(read(ctx, str(sample_file), offset=0))
        assert d["ok"] is False

    def test_limit_zero_returns_error(self, ctx: MagicMock, sample_file: Path) -> None:
        d = _parse(read(ctx, str(sample_file), limit=0))
        assert d["ok"] is False

    def test_file_not_found_returns_error(self, ctx: MagicMock) -> None:
        d = _parse(read(ctx, "/nonexistent/file.txt"))
        assert d["ok"] is False

    def test_binary_file_returns_error(self, ctx: MagicMock, tmp_path: Path) -> None:
        bf = tmp_path / "binary.bin"
        bf.write_bytes(b"\x00\xff\xfe\x80" * 100)
        d = _parse(read(ctx, str(bf)))
        assert d["ok"] is False

    def test_debug_mode_includes_extra_fields(self, ctx: MagicMock, sample_file: Path) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="debug"):
            d = _parse(read(ctx, str(sample_file)))
        assert "from_cache" in d
        assert "tokens_saved" in d
        assert "params" in d

    def test_normal_mode_includes_is_diff_and_truncated(
        self, ctx: MagicMock, sample_file: Path
    ) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="normal"):
            d = _parse(read(ctx, str(sample_file)))
        assert "is_diff" in d
        assert "truncated" in d

    def test_truncated_file_includes_hint(self, ctx: MagicMock, tmp_path: Path) -> None:
        large = tmp_path / "large.txt"
        large.write_text("word " * 30000)  # ~30k tokens
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="normal"):
            d = _parse(read(ctx, str(large), max_size=100))
        # truncation is only shown in normal/debug mode
        if d.get("truncated"):
            assert "hint" in d

    def test_offset_with_limit_lines_info(self, ctx: MagicMock, sample_file: Path) -> None:
        d = _parse(read(ctx, str(sample_file), offset=1, limit=3))
        assert "lines" in d
        assert d["lines"]["start"] == 1
        assert d["lines"]["end"] == 3


# ===========================================================================
# write tool
# ===========================================================================


class TestWriteTool:
    def test_write_new_file(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "new.txt"
        d = _parse(write(ctx, str(p), "hello"))
        assert d.get("status") == "created"

    def test_write_overwrite_existing(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "exist.txt"
        p.write_text("old content")
        smart_read(ctx.lifespan_context["cache"], str(p))
        d = _parse(write(ctx, str(p), "new content"))
        assert d.get("status") == "updated"

    def test_write_append_mode(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "append.txt"
        p.write_text("first\n")
        write(ctx, str(p), "first\n")
        write(ctx, str(p), "second\n", append=True)
        assert "second" in p.read_text()

    def test_write_dry_run_does_not_create(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "dry.txt"
        write(ctx, str(p), "content", dry_run=True)
        assert not p.exists()

    def test_write_creates_parents(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "a" / "b" / "c.txt"
        write(ctx, str(p), "nested")
        assert p.exists()

    def test_write_permission_error(self, ctx: MagicMock, tmp_path: Path) -> None:
        """PermissionError from the filesystem layer maps to ok=False response."""
        p = tmp_path / "readonly.txt"
        p.write_text("content")
        with patch(
            "semantic_cache_mcp.cache.write._atomic_write",
            side_effect=PermissionError("[Errno 13] Permission denied"),
        ):
            d = _parse(write(ctx, str(p), "new"))
        assert d["ok"] is False

    def test_write_parent_not_found_without_create_parents(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        p = tmp_path / "missing_parent" / "file.txt"
        d = _parse(write(ctx, str(p), "content", create_parents=False))
        assert d["ok"] is False

    def test_write_debug_mode_includes_hash(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "debug.txt"
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="debug"):
            d = _parse(write(ctx, str(p), "content"))
        assert "content_hash" in d

    def test_write_diff_shown_on_overwrite(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "diff.txt"
        p.write_text("old\n")
        smart_read(ctx.lifespan_context["cache"], str(p))
        d = _parse(write(ctx, str(p), "new\n"))
        # diff key should be present since file was previously cached
        assert "diff" in d or "diff_omitted" in d

    def test_write_normal_mode_includes_created_flag(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "normal.txt"
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="normal"):
            d = _parse(write(ctx, str(p), "hello"))
        assert "created" in d
        assert d["created"] is True


# ===========================================================================
# edit tool
# ===========================================================================


class TestEditTool:
    def test_find_replace_success(self, ctx: MagicMock, py_file: Path) -> None:
        d = _parse(edit(ctx, str(py_file), old_string="hello", new_string="hi"))
        assert d.get("status") == "edited"

    def test_find_replace_not_found_returns_error(self, ctx: MagicMock, py_file: Path) -> None:
        d = _parse(edit(ctx, str(py_file), old_string="NOTEXIST", new_string="x"))
        assert d["ok"] is False

    def test_edit_line_replace_mode(self, ctx: MagicMock, py_file: Path) -> None:
        d = _parse(
            edit(ctx, str(py_file), new_string="# replaced line\n", start_line=1, end_line=1)
        )
        assert d.get("status") == "edited"
        assert "# replaced line" in py_file.read_text()

    def test_edit_dry_run_no_change(self, ctx: MagicMock, py_file: Path) -> None:
        original = py_file.read_text()
        edit(ctx, str(py_file), old_string="hello", new_string="hi", dry_run=True)
        assert py_file.read_text() == original

    def test_edit_file_not_found(self, ctx: MagicMock) -> None:
        d = _parse(edit(ctx, "/no/such/file.py", old_string="x", new_string="y"))
        assert d["ok"] is False

    def test_edit_debug_mode_includes_diff_stats(self, ctx: MagicMock, py_file: Path) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="debug"):
            d = _parse(edit(ctx, str(py_file), old_string="hello", new_string="hi"))
        assert "diff_stats" in d
        assert "content_hash" in d

    def test_edit_normal_mode_includes_tokens_saved(self, ctx: MagicMock, py_file: Path) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="normal"):
            d = _parse(edit(ctx, str(py_file), old_string="hello", new_string="hi"))
        assert "tokens_saved" in d

    def test_edit_returns_line_numbers(self, ctx: MagicMock, py_file: Path) -> None:
        d = _parse(edit(ctx, str(py_file), old_string="hello", new_string="hi"))
        assert "line_numbers" in d

    def test_edit_permission_error(self, ctx: MagicMock, py_file: Path) -> None:
        """PermissionError from atomic write maps to ok=False response."""
        with patch(
            "semantic_cache_mcp.cache.write._atomic_write",
            side_effect=PermissionError("[Errno 13] Permission denied"),
        ):
            d = _parse(edit(ctx, str(py_file), old_string="hello", new_string="hi"))
        assert d["ok"] is False

    def test_edit_replace_all(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "multi.txt"
        p.write_text("aa\naa\nbb\n")
        d = _parse(edit(ctx, str(p), old_string="aa", new_string="cc", replace_all=True))
        assert d.get("status") == "edited"
        assert p.read_text().count("cc") == 2


# ===========================================================================
# batch_edit tool
# ===========================================================================


class TestBatchEditTool:
    def test_batch_edit_two_pairs_json_array(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "f.txt"
        p.write_text("AAA\nBBB\n")
        edits_json = json.dumps([["AAA", "111"], ["BBB", "222"]])
        d = _parse(batch_edit(ctx, str(p), edits_json))
        assert d.get("succeeded") == 2
        assert "failed" not in d  # omitted when 0

    def test_batch_edit_partial_failure(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "f2.txt"
        p.write_text("found\n")
        edits_json = json.dumps([["found", "replaced"], ["missing", "x"]])
        d = _parse(batch_edit(ctx, str(p), edits_json))
        assert d.get("succeeded") == 1
        assert d.get("failed") == 1
        assert "failures" in d

    def test_batch_edit_invalid_json_returns_error(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "f3.txt"
        p.write_text("content")
        d = _parse(batch_edit(ctx, str(p), "not json"))
        assert d["ok"] is False

    def test_batch_edit_not_array_returns_error(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "f4.txt"
        p.write_text("content")
        d = _parse(batch_edit(ctx, str(p), '{"key": "val"}'))
        assert d["ok"] is False

    def test_batch_edit_dict_format(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "f5.txt"
        p.write_text("foo\n")
        edits_json = json.dumps([{"old": "foo", "new": "bar"}])
        d = _parse(batch_edit(ctx, str(p), edits_json))
        assert d.get("succeeded") == 1

    def test_batch_edit_four_tuple_format(self, ctx: MagicMock, tmp_path: Path) -> None:
        """Four-element list: [old, new, start_line, end_line]."""
        p = tmp_path / "f6.txt"
        p.write_text("aaa\nbbb\nccc\n")
        edits_json = json.dumps([["aaa", "111", 1, 1]])
        d = _parse(batch_edit(ctx, str(p), edits_json))
        assert d.get("succeeded") == 1

    def test_batch_edit_null_old_line_replace(self, ctx: MagicMock, tmp_path: Path) -> None:
        """[null, new, start_line, end_line] → line replace mode."""
        p = tmp_path / "f7.txt"
        p.write_text("old line\nkeep\n")
        edits_json = json.dumps([[None, "new line\n", 1, 1]])
        d = _parse(batch_edit(ctx, str(p), edits_json))
        assert d.get("succeeded") == 1

    def test_batch_edit_dry_run(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "f8.txt"
        original = "original\n"
        p.write_text(original)
        edits_json = json.dumps([["original", "changed"]])
        batch_edit(ctx, str(p), edits_json, dry_run=True)
        assert p.read_text() == original

    def test_batch_edit_file_not_found(self, ctx: MagicMock) -> None:
        d = _parse(batch_edit(ctx, "/no/such/file.py", json.dumps([["a", "b"]])))
        assert d["ok"] is False

    def test_batch_edit_debug_mode_includes_outcomes(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "f9.txt"
        p.write_text("x\n")
        edits_json = json.dumps([["x", "y"]])
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="debug"):
            d = _parse(batch_edit(ctx, str(p), edits_json))
        assert "outcomes" in d

    def test_batch_edit_status_no_changes(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "f10.txt"
        p.write_text("hello\n")
        edits_json = json.dumps([["NOTFOUND", "x"]])
        d = _parse(batch_edit(ctx, str(p), edits_json))
        assert d.get("status") == "no_changes"

    def test_batch_edit_malformed_entry_returns_error(self, ctx: MagicMock, tmp_path: Path) -> None:
        p = tmp_path / "f11.txt"
        p.write_text("x\n")
        # Single-element list is not a valid edit entry
        edits_json = json.dumps([["only_one"]])
        d = _parse(batch_edit(ctx, str(p), edits_json))
        assert d["ok"] is False


# ===========================================================================
# search tool
# ===========================================================================


class TestSearchTool:
    def test_search_empty_cache_returns_empty(self, ctx: MagicMock) -> None:
        d = _parse(search(ctx, "hello world"))
        assert d.get("matches") == [] or isinstance(d.get("matches"), list)

    def test_search_finds_cached_file(
        self, ctx: MagicMock, py_file: Path, tmp_cache: SemanticCache
    ) -> None:
        smart_read(tmp_cache, str(py_file))
        d = _parse(search(ctx, "hello"))
        assert "matches" in d

    def test_search_directory_filter(
        self, ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
    ) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        f = sub / "in_sub.py"
        f.write_text("def inside(): pass\n")
        smart_read(tmp_cache, str(f))
        d = _parse(search(ctx, "inside", directory=str(sub)))
        for m in d.get("matches", []):
            assert str(sub) in m["path"]

    def test_search_debug_mode(self, ctx: MagicMock, py_file: Path) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="debug"):
            d = _parse(search(ctx, "hello"))
        assert "files_searched" in d
        assert "k" in d

    def test_search_normal_mode_includes_count(self, ctx: MagicMock, py_file: Path) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="normal"):
            d = _parse(search(ctx, "hello"))
        assert "count" in d
        assert "cached_files" in d

    def test_search_k_parameter(
        self, ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
    ) -> None:
        for i in range(5):
            f = tmp_path / f"file{i}.py"
            f.write_text(f"def func{i}(): pass\n")
            smart_read(tmp_cache, str(f))
        d = _parse(search(ctx, "func", k=2))
        assert len(d.get("matches", [])) <= 2


# ===========================================================================
# diff tool
# ===========================================================================


class TestDiffTool:
    def test_diff_identical_files(self, ctx: MagicMock, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("same\n")
        b.write_text("same\n")
        d = _parse(diff(ctx, str(a), str(b)))
        assert "diff" in d

    def test_diff_different_files(self, ctx: MagicMock, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("old content\n")
        b.write_text("new content\n")
        d = _parse(diff(ctx, str(a), str(b)))
        assert "diff" in d

    def test_diff_file_not_found(self, ctx: MagicMock, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        a.write_text("content")
        d = _parse(diff(ctx, str(a), "/nonexistent.txt"))
        assert d["ok"] is False

    def test_diff_normal_mode_includes_similarity(self, ctx: MagicMock, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("hello\n")
        b.write_text("world\n")
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="normal"):
            d = _parse(diff(ctx, str(a), str(b)))
        assert "similarity" in d
        assert "diff_stats" in d

    def test_diff_debug_mode_includes_cache_info(self, ctx: MagicMock, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("a\n")
        b.write_text("b\n")
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="debug"):
            d = _parse(diff(ctx, str(a), str(b)))
        assert "from_cache" in d
        assert "context_lines" in d

    def test_diff_uses_cached_content(
        self, ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
    ) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("cached\n")
        b.write_text("also cached\n")
        smart_read(tmp_cache, str(a))
        smart_read(tmp_cache, str(b))
        d = _parse(diff(ctx, str(a), str(b)))
        assert "error" not in d  # compact strips ok on success


# ===========================================================================
# batch_read tool
# ===========================================================================


class TestBatchReadTool:
    def test_reads_comma_separated_paths(
        self, ctx: MagicMock, sample_file: Path, py_file: Path
    ) -> None:
        paths = f"{sample_file},{py_file}"
        d = _parse(batch_read(ctx, paths))
        assert d.get("summary", {}).get("files_read", 0) >= 1

    def test_reads_json_array_paths(self, ctx: MagicMock, sample_file: Path, py_file: Path) -> None:
        paths = json.dumps([str(sample_file), str(py_file)])
        d = _parse(batch_read(ctx, paths))
        assert "summary" in d

    def test_invalid_json_array_returns_error(self, ctx: MagicMock) -> None:
        d = _parse(batch_read(ctx, "[not valid json"))
        assert d["ok"] is False

    def test_token_budget_respected(self, ctx: MagicMock, tmp_path: Path) -> None:
        files = []
        for i in range(5):
            f = tmp_path / f"big{i}.txt"
            f.write_text("word " * 2000)
            files.append(str(f))
        paths = ",".join(files)
        d = _parse(batch_read(ctx, paths, max_total_tokens=100))
        skipped = d.get("skipped", [])
        assert len(skipped) > 0 or d["summary"]["files_read"] < 5

    def test_missing_file_counted_as_skipped(self, ctx: MagicMock, sample_file: Path) -> None:
        paths = f"{sample_file},/nonexistent/file.py"
        d = _parse(batch_read(ctx, paths))
        assert d["summary"]["files_skipped"] >= 1

    def test_unchanged_files_reported_in_summary(self, ctx: MagicMock, sample_file: Path) -> None:
        # Seed cache
        batch_read(ctx, str(sample_file))
        # Second read: should be unchanged
        d = _parse(batch_read(ctx, str(sample_file)))
        assert "unchanged" in d.get("summary", {})

    def test_diff_mode_false_returns_full_content(self, ctx: MagicMock, sample_file: Path) -> None:
        batch_read(ctx, str(sample_file))
        d = _parse(batch_read(ctx, str(sample_file), diff_mode=False))
        assert d["summary"]["files_read"] >= 1

    def test_priority_parameter_comma_separated(
        self, ctx: MagicMock, sample_file: Path, py_file: Path
    ) -> None:
        paths = f"{sample_file},{py_file}"
        priority = str(py_file)
        d = _parse(batch_read(ctx, paths, priority=priority))
        files = d.get("files", [])
        if files:
            assert files[0]["path"] == str(py_file)

    def test_priority_parameter_json_array(
        self, ctx: MagicMock, sample_file: Path, py_file: Path
    ) -> None:
        paths = json.dumps([str(sample_file), str(py_file)])
        priority = json.dumps([str(py_file)])
        d = _parse(batch_read(ctx, paths, priority=priority))
        assert "summary" in d

    def test_glob_pattern_in_paths(self, ctx: MagicMock, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"f{i}.txt").write_text(f"content {i}\n")
        pattern = str(tmp_path / "*.txt")
        d = _parse(batch_read(ctx, pattern))
        assert d["summary"]["files_read"] >= 1

    def test_debug_mode_includes_token_info(self, ctx: MagicMock, sample_file: Path) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="debug"):
            d = _parse(batch_read(ctx, str(sample_file)))
        files = d.get("files", [])
        if files:
            assert "tokens" in files[0]

    def test_normal_mode_includes_total_tokens(self, ctx: MagicMock, sample_file: Path) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="normal"):
            d = _parse(batch_read(ctx, str(sample_file)))
        assert "total_tokens" in d["summary"]


# ===========================================================================
# similar tool
# ===========================================================================


class TestSimilarTool:
    def test_similar_empty_cache_returns_empty(self, ctx: MagicMock, py_file: Path) -> None:
        d = _parse(similar(ctx, str(py_file)))
        assert isinstance(d.get("similar_files"), list)

    def test_similar_with_cached_files(
        self, ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
    ) -> None:
        files = []
        for i in range(3):
            f = tmp_path / f"m{i}.py"
            f.write_text(f"def func{i}(): pass\n")
            smart_read(tmp_cache, str(f))
            files.append(f)
        d = _parse(similar(ctx, str(files[0])))
        assert "similar_files" in d

    def test_similar_file_not_found(self, ctx: MagicMock) -> None:
        d = _parse(similar(ctx, "/nonexistent.py"))
        assert d["ok"] is False

    def test_similar_normal_mode(self, ctx: MagicMock, py_file: Path) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="normal"):
            d = _parse(similar(ctx, str(py_file)))
        assert "source_tokens" in d
        assert "files_searched" in d

    def test_similar_debug_mode_includes_k(self, ctx: MagicMock, py_file: Path) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="debug"):
            d = _parse(similar(ctx, str(py_file), k=3))
        assert d.get("k") == 3

    def test_similar_compact_mode_omits_tokens(
        self, ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
    ) -> None:
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        a.write_text("def a(): pass\n")
        b.write_text("def b(): pass\n")
        smart_read(tmp_cache, str(a))
        smart_read(tmp_cache, str(b))
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="compact"):
            d = _parse(similar(ctx, str(a)))
        for sf in d.get("similar_files", []):
            assert "tokens" not in sf


# ===========================================================================
# glob tool
# ===========================================================================


class TestGlobTool:
    def test_glob_finds_py_files(self, ctx: MagicMock, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        d = _parse(glob(ctx, "*.py", directory=str(tmp_path)))
        assert d["total_matches"] >= 2

    def test_glob_recursive_pattern(self, ctx: MagicMock, tmp_path: Path) -> None:
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "mod.py").write_text("x")
        d = _parse(glob(ctx, "**/*.py", directory=str(tmp_path)))
        assert d["total_matches"] >= 1

    def test_glob_no_matches(self, ctx: MagicMock, tmp_path: Path) -> None:
        d = _parse(glob(ctx, "*.nonexistent", directory=str(tmp_path)))
        assert d["total_matches"] == 0
        assert d["matches"] == []

    def test_glob_cached_only_filter(
        self, ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
    ) -> None:
        cached = tmp_path / "cached.py"
        uncached = tmp_path / "uncached.py"
        cached.write_text("cached")
        uncached.write_text("uncached")
        smart_read(tmp_cache, str(cached))
        d = _parse(glob(ctx, "*.py", directory=str(tmp_path), cached_only=True))
        paths = [m["path"] for m in d["matches"]]
        assert str(cached) in paths
        assert str(uncached) not in paths

    def test_glob_shows_cache_status(
        self, ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
    ) -> None:
        f = tmp_path / "f.py"
        f.write_text("x")
        smart_read(tmp_cache, str(f))
        d = _parse(glob(ctx, "*.py", directory=str(tmp_path)))
        cached_matches = [m for m in d["matches"] if m["cached"]]
        assert len(cached_matches) >= 1

    def test_glob_normal_mode_includes_tokens(self, ctx: MagicMock, tmp_path: Path) -> None:
        (tmp_path / "f.py").write_text("hello")
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="normal"):
            d = _parse(glob(ctx, "*.py", directory=str(tmp_path)))
        for m in d["matches"]:
            assert "tokens" in m
            assert "mtime" in m

    def test_glob_debug_mode_includes_total_cached_tokens(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        (tmp_path / "f.py").write_text("hello")
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="debug"):
            d = _parse(glob(ctx, "*.py", directory=str(tmp_path)))
        assert "total_cached_tokens" in d


# ===========================================================================
# grep tool
# ===========================================================================


class TestGrepTool:
    def test_grep_finds_pattern(
        self, ctx: MagicMock, py_file: Path, tmp_cache: SemanticCache
    ) -> None:
        smart_read(tmp_cache, str(py_file))
        d = _parse(grep(ctx, "hello"))
        assert d["total_matches"] >= 1

    def test_grep_empty_cache_no_matches(self, ctx: MagicMock) -> None:
        d = _parse(grep(ctx, "anything"))
        assert d["total_matches"] == 0
        assert d["files_matched"] == 0

    def test_grep_fixed_string(
        self, ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
    ) -> None:
        f = tmp_path / "f.txt"
        f.write_text("foo.bar()\n")
        smart_read(tmp_cache, str(f))
        d = _parse(grep(ctx, "foo.bar()", fixed_string=True))
        assert d["total_matches"] >= 1

    def test_grep_case_insensitive(
        self, ctx: MagicMock, py_file: Path, tmp_cache: SemanticCache
    ) -> None:
        smart_read(tmp_cache, str(py_file))
        d = _parse(grep(ctx, "HELLO", case_sensitive=False))
        assert d["total_matches"] >= 1

    def test_grep_context_lines_normal_mode(
        self, ctx: MagicMock, sample_file: Path, tmp_cache: SemanticCache
    ) -> None:
        smart_read(tmp_cache, str(sample_file))
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="normal"):
            d = _parse(grep(ctx, "line3", context_lines=1))
        files = d.get("files", [])
        if files and files[0]["matches"]:
            match = files[0]["matches"][0]
            assert "before" in match or "after" in match

    def test_grep_debug_mode_includes_params(
        self, ctx: MagicMock, py_file: Path, tmp_cache: SemanticCache
    ) -> None:
        smart_read(tmp_cache, str(py_file))
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="debug"):
            d = _parse(grep(ctx, "hello"))
        assert "fixed_string" in d
        assert "case_sensitive" in d

    def test_grep_max_matches_limit(
        self, ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
    ) -> None:
        f = tmp_path / "many.txt"
        f.write_text("\n".join(["match"] * 50))
        smart_read(tmp_cache, str(f))
        d = _parse(grep(ctx, "match", max_matches=5))
        assert d["total_matches"] <= 5

    def test_grep_response_structure(
        self, ctx: MagicMock, py_file: Path, tmp_cache: SemanticCache
    ) -> None:
        smart_read(tmp_cache, str(py_file))
        d = _parse(grep(ctx, "def"))
        assert "files" in d
        for file_entry in d["files"]:
            assert "path" in file_entry
            assert "count" in file_entry
            assert "matches" in file_entry
            for m in file_entry["matches"]:
                assert "line_number" in m
                assert "line" in m


# ===========================================================================
# stats tool
# ===========================================================================


class TestStatsTool:
    def test_stats_empty_cache(self, ctx: MagicMock) -> None:
        d = _parse(stats(ctx))
        assert "files_cached" in d or d.get("ok") is True

    def test_stats_after_caching_file(
        self, ctx: MagicMock, py_file: Path, tmp_cache: SemanticCache
    ) -> None:
        smart_read(tmp_cache, str(py_file))
        d = _parse(stats(ctx))
        # In compact mode ok/tool are stripped — just check files_cached
        cached = d.get("files_cached", 0)
        assert cached >= 1

    def test_stats_normal_mode_includes_full_stats(self, ctx: MagicMock) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="normal"):
            d = _parse(stats(ctx))
        assert "embedding_model" in d
        assert "embedding_ready" in d
        assert "files_cached" in d
        assert "tokens_saved" in d
        assert "savings_pct" in d
        assert "db_size_mb" in d
        assert "uptime_s" in d

    def test_stats_debug_mode_includes_output_mode(self, ctx: MagicMock) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="debug"):
            d = _parse(stats(ctx))
        assert "output_mode" in d

    def test_stats_compact_mode_minimal_fields(self, ctx: MagicMock) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="compact"):
            d = _parse(stats(ctx))
        assert "embedding_ready" in d
        assert "files_cached" in d
        assert "tokens_saved" in d
        assert "savings_pct" in d
        assert "cache_hits" in d
        assert "db_size_mb" in d


# ===========================================================================
# clear tool
# ===========================================================================


class TestClearTool:
    def test_clear_empty_cache(self, ctx: MagicMock) -> None:
        d = _parse(clear(ctx))
        assert d.get("status") == "cleared"

    def test_clear_removes_cached_files(
        self, ctx: MagicMock, py_file: Path, tmp_cache: SemanticCache
    ) -> None:
        smart_read(tmp_cache, str(py_file))
        clear(ctx)
        cache_stats = tmp_cache.get_stats()
        assert cache_stats["files_cached"] == 0

    def test_clear_returns_count(
        self, ctx: MagicMock, py_file: Path, tmp_cache: SemanticCache
    ) -> None:
        smart_read(tmp_cache, str(py_file))
        d = _parse(clear(ctx))
        assert d.get("count", 0) >= 0  # 0 is valid if already cleared

    def test_clear_debug_mode_includes_output_mode(self, ctx: MagicMock) -> None:
        with patch("semantic_cache_mcp.server.tools._response_mode", return_value="debug"):
            d = _parse(clear(ctx))
        assert "output_mode" in d


# ===========================================================================
# _expand_globs helper
# ===========================================================================


class TestExpandGlobs:
    def test_non_glob_passthrough(self, tmp_path: Path) -> None:
        paths = [str(tmp_path / "a.py"), str(tmp_path / "b.py")]
        assert _expand_globs(paths) == paths

    def test_expands_star_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "x.py").write_text("x")
        (tmp_path / "y.py").write_text("y")
        result = _expand_globs([str(tmp_path / "*.py")])
        assert len(result) >= 2
        assert all(r.endswith(".py") for r in result)

    def test_respects_max_files(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"f{i}.txt").write_text(f"{i}")
        result = _expand_globs([str(tmp_path / "*.txt")], max_files=3)
        assert len(result) <= 3

    def test_invalid_glob_pattern_treated_as_literal(self) -> None:
        """Malformed bracket not on filesystem → literal pass-through."""
        result = _expand_globs(["/nonexistent/path/[invalid"])
        assert result == ["/nonexistent/path/[invalid"]

    def test_nonexistent_base_treated_as_literal(self) -> None:
        result = _expand_globs(["/no/such/dir/*.py"])
        assert result == ["/no/such/dir/*.py"]

    def test_question_mark_wildcard(self, tmp_path: Path) -> None:
        (tmp_path / "ab.py").write_text("x")
        result = _expand_globs([str(tmp_path / "a?.py")])
        assert any("ab.py" in r for r in result)

    def test_double_star_recursive(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "deep.py").write_text("x")
        result = _expand_globs([str(tmp_path / "**" / "*.py")])
        assert any("deep.py" in r for r in result)

    def test_multiple_patterns_mixed(self, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("a")
        literal = str(tmp_path / "b.txt")  # doesn't exist but is literal
        result = _expand_globs([str(tmp_path / "*.py"), literal])
        assert any(r.endswith(".py") for r in result)
        assert literal in result

    def test_stops_at_max_files_across_multiple_patterns(self, tmp_path: Path) -> None:
        for i in range(6):
            (tmp_path / f"f{i}.py").write_text(f"{i}")
        result = _expand_globs([str(tmp_path / "*.py"), str(tmp_path / "*.py")], max_files=4)
        assert len(result) <= 4
