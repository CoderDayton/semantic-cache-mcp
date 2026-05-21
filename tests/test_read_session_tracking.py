"""Tests for the per-session unchanged tracker (item 7)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastmcp import Context

from semantic_cache_mcp.cache import SemanticCache
from semantic_cache_mcp.server._read_session import _ReadSessionTracker, get_tracker
from semantic_cache_mcp.server.tools import edit, read


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(db_path=tmp_path / "cache.db")


@pytest.fixture()
def ctx(tmp_cache: SemanticCache) -> MagicMock:
    c = MagicMock(spec=Context)
    c.lifespan_context = {"cache": tmp_cache}
    # session_id deliberately unset so the tracker uses its process fallback —
    # the tests then exercise the single-session path end to end.
    c.session_id = "test_session"
    c.client_id = "test_client"
    return c


@pytest.fixture()
def sample_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text("line1\nline2\nline3\nline4\nline5\n")
    return p


def _parse(response) -> dict:  # noqa: ANN001
    import json

    if isinstance(response, dict):
        return response
    return json.loads(response)


@pytest.fixture(autouse=True)
def _reset_tracker():
    """Tracker is process-global; reset between tests."""
    get_tracker().clear()
    yield
    get_tracker().clear()


# ---------------------------------------------------------------------------
# _ReadSessionTracker unit tests
# ---------------------------------------------------------------------------


def test_seen_returns_false_for_new_path() -> None:
    t = _ReadSessionTracker()
    assert t.seen("session_a", "/a/b.py") is False


def test_mark_then_seen_returns_true() -> None:
    t = _ReadSessionTracker()
    t.mark("session_a", "/a/b.py")
    assert t.seen("session_a", "/a/b.py") is True


def test_different_session_does_not_share() -> None:
    t = _ReadSessionTracker()
    t.mark("session_a", "/a/b.py")
    assert t.seen("session_b", "/a/b.py") is False


def test_invalidate_drops_across_all_sessions() -> None:
    t = _ReadSessionTracker()
    t.mark("session_a", "/shared.py")
    t.mark("session_b", "/shared.py")
    t.invalidate("/shared.py")
    assert t.seen("session_a", "/shared.py") is False
    assert t.seen("session_b", "/shared.py") is False


def test_clear_resets_everything() -> None:
    t = _ReadSessionTracker()
    t.mark("s", "/a")
    t.mark("s", "/b")
    t.clear()
    assert len(t) == 0


def test_lru_eviction_at_cap() -> None:
    t = _ReadSessionTracker()
    # Cap is 256 in module; mark 260 distinct paths and assert the oldest fell out.
    for i in range(260):
        t.mark("s", f"/path/{i}.py")
    assert t.seen("s", "/path/0.py") is False
    assert t.seen("s", "/path/259.py") is True


def test_none_session_falls_back_to_proc_key() -> None:
    t = _ReadSessionTracker()
    t.mark(None, "/x.py")
    assert t.seen(None, "/x.py") is True
    # Falls back to the same key regardless of how it's reached.
    assert t.seen("", "/x.py") is True


# ---------------------------------------------------------------------------
# read tool integration
# ---------------------------------------------------------------------------


async def test_first_read_returns_full_content(ctx: MagicMock, sample_file: Path) -> None:
    d = _parse(await read(ctx, str(sample_file)))
    assert "content" in d
    assert "unchanged" not in d


async def test_second_read_returns_unchanged_with_hash_and_lines(
    ctx: MagicMock, sample_file: Path
) -> None:
    # First read marks the session as having seen the file.
    await read(ctx, str(sample_file))
    d = _parse(await read(ctx, str(sample_file)))
    assert d.get("unchanged") is True
    assert "content_hash" in d
    assert "total_lines" in d
    assert d["total_lines"] > 0
    # Content body is suppressed on the unchanged response.
    assert "content" not in d


async def test_edit_invalidates_session_entry(
    ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
) -> None:
    f = tmp_path / "invalidate.py"
    f.write_text("def foo():\n    return 1\n")

    # First read: full content.
    await read(ctx, str(f))
    # Second read: would be unchanged…
    d2 = _parse(await read(ctx, str(f)))
    assert d2.get("unchanged") is True

    # Edit invalidates the tracker entry.
    await edit(ctx, str(f), "return 1", "return 2")

    # Next read: full content again, since the client hasn't seen the new bytes.
    d3 = _parse(await read(ctx, str(f)))
    assert "content" in d3
    assert "unchanged" not in d3
    assert "return 2" in d3["content"]


# ---------------------------------------------------------------------------
# regression: the first-read branch must deliver real content, never the
# "// File unchanged" marker or a truncated summary marked as fully seen
# ---------------------------------------------------------------------------


async def test_first_session_read_of_warm_cache_returns_content(
    ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
) -> None:
    """A file already warm in the cache must be delivered in full on the
    first read of a new session — not the '// File unchanged' marker."""
    f = tmp_path / "warm.py"
    f.write_text("\n".join(f"value_{i} = {i}" for i in range(60)) + "\n")

    # Session A populates the cache.
    await read(ctx, str(f))

    # Session B reads it for the first time.
    ctx_b = MagicMock(spec=Context)
    ctx_b.lifespan_context = {"cache": tmp_cache}
    ctx_b.session_id = "other_session"
    ctx_b.client_id = "other_client"
    d = _parse(await read(ctx_b, str(f)))

    assert d.get("unchanged") is not True
    assert "content" in d
    assert "value_0 = 0" in d["content"]
    assert not d["content"].startswith("// File unchanged")


async def test_edit_then_read_largish_file_returns_full_content(
    ctx: MagicMock, tmp_path: Path
) -> None:
    """After editing a file big enough to hit the cache fast path, the next
    read must return the full post-edit content, not the marker."""
    f = tmp_path / "edited.py"
    f.write_text("\n".join(f"item_{i} = {i}" for i in range(60)) + "\n")

    await read(ctx, str(f))
    await edit(ctx, str(f), "item_0 = 0", "item_0 = 999")

    d = _parse(await read(ctx, str(f)))
    assert d.get("unchanged") is not True
    assert "content" in d
    assert "item_0 = 999" in d["content"]
    assert not d["content"].startswith("// File unchanged")


async def test_truncated_read_does_not_block_reread(ctx: MagicMock, tmp_path: Path) -> None:
    """A truncated read must not mark the file as fully seen — a follow-up
    read must still deliver content, never a bare unchanged:true."""
    big = tmp_path / "big.py"
    body = "\n".join(f"row_{i} = {i}" for i in range(150)) + "\n"
    big.write_text(body)

    d1 = _parse(await read(ctx, str(big), max_size=200))
    assert "content" in d1
    assert len(d1["content"]) < len(body)  # summarized, not the whole file

    d2 = _parse(await read(ctx, str(big), max_size=200))
    assert d2.get("unchanged") is not True
    assert "content" in d2
