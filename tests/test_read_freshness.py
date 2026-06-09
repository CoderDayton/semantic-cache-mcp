"""Read-tool freshness: hash-driven `unchanged`, diffs, and ranged reads.

The read tool sends full content on a first read and a `content_hash` with it.
A re-read returns `unchanged: true` only when the caller passes back a
matching `known_hash`; otherwise it sends content. A changed file returns a
diff. There is no server-side session state.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastmcp import Context

from semantic_cache_mcp.cache import SemanticCache
from semantic_cache_mcp.cache.read import smart_read
from semantic_cache_mcp.server.tools import edit, read


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(db_path=tmp_path / "cache.db")


@pytest.fixture()
def ctx(tmp_cache: SemanticCache) -> MagicMock:
    c = MagicMock(spec=Context)
    c.lifespan_context = {"cache": tmp_cache}
    return c


def _new_ctx(tmp_cache: SemanticCache) -> MagicMock:
    """A second context over the same cache (a different client/connection)."""
    c = MagicMock(spec=Context)
    c.lifespan_context = {"cache": tmp_cache}
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


# ---------------------------------------------------------------------------
# Baseline read behavior
# ---------------------------------------------------------------------------


async def test_first_read_returns_full_content(ctx: MagicMock, sample_file: Path) -> None:
    d = _parse(await read(ctx, str(sample_file)))
    assert "content" in d
    assert "unchanged" not in d


async def test_full_read_returns_content_hash(ctx: MagicMock, sample_file: Path) -> None:
    """Every full read surfaces content_hash so the caller can echo it back."""
    d = _parse(await read(ctx, str(sample_file)))
    assert "content" in d
    assert d.get("content_hash")


async def test_reread_without_known_hash_returns_full_content(
    ctx: MagicMock, sample_file: Path
) -> None:
    """With no session state, a re-read that omits known_hash gets full content."""
    await read(ctx, str(sample_file))
    d = _parse(await read(ctx, str(sample_file)))
    assert d.get("unchanged") is not True
    assert "content" in d
    assert "content_hash" in d


async def test_warm_cache_read_returns_content(
    ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
) -> None:
    """A file already warm in the cache is delivered in full, not as the
    '// File unchanged' marker, when read without a known_hash."""
    f = tmp_path / "warm.py"
    f.write_text("\n".join(f"value_{i} = {i}" for i in range(60)) + "\n")

    await smart_read(tmp_cache, str(f))  # warm the cache directly
    d = _parse(await read(ctx, str(f)))

    assert d.get("unchanged") is not True
    assert "content" in d
    assert "value_0 = 0" in d["content"]
    assert not d["content"].startswith("// File unchanged")


# ---------------------------------------------------------------------------
# Hash-driven freshness: the caller asserts it still holds the file
# ---------------------------------------------------------------------------


async def test_known_hash_returns_unchanged(
    ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
) -> None:
    f = tmp_path / "hashed.py"
    f.write_text("\n".join(f"v_{i} = {i}" for i in range(60)) + "\n")

    d1 = _parse(await read(ctx, str(f)))
    h = d1["content_hash"]

    # A different context (never read this file) still gets unchanged by
    # asserting the hash it holds.
    ctx_b = _new_ctx(tmp_cache)
    d = _parse(await read(ctx_b, str(f), known_hash=h))
    assert d.get("unchanged") is True
    assert d["content_hash"] == h
    assert "content" not in d
    assert d.get("total_lines", 0) > 0


async def test_wrong_known_hash_returns_full_content(
    ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
) -> None:
    f = tmp_path / "hashed2.py"
    f.write_text("\n".join(f"v_{i} = {i}" for i in range(60)) + "\n")
    await read(ctx, str(f))

    ctx_b = _new_ctx(tmp_cache)
    d = _parse(await read(ctx_b, str(f), known_hash="deadbeefdeadbeef"))
    assert d.get("unchanged") is not True
    assert "content" in d


async def test_known_hash_stale_after_change_never_false_unchanged(
    ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
) -> None:
    """A matching-but-stale hash must never mask a real on-disk change."""
    f = tmp_path / "changed.py"
    f.write_text("\n".join(f"v_{i} = {i}" for i in range(120)) + "\n")
    d1 = _parse(await read(ctx, str(f)))
    h = d1["content_hash"]

    # Change the file on disk; the caller still holds the OLD hash.
    f.write_text("\n".join(f"v_{i} = {i + 1}" for i in range(120)) + "\n")
    ctx_b = _new_ctx(tmp_cache)
    d = _parse(await read(ctx_b, str(f), known_hash=h))
    assert d.get("unchanged") is not True
    assert "content" in d


# ---------------------------------------------------------------------------
# Mutations always return fresh content on the next read
# ---------------------------------------------------------------------------


async def test_edit_then_read_returns_full_content(ctx: MagicMock, tmp_path: Path) -> None:
    """After an edit, the next read (no known_hash) returns the new content."""
    f = tmp_path / "edited.py"
    f.write_text("\n".join(f"item_{i} = {i}" for i in range(60)) + "\n")

    await read(ctx, str(f))
    await edit(ctx, str(f), "item_0 = 0", "item_0 = 999")

    d = _parse(await read(ctx, str(f)))
    assert d.get("unchanged") is not True
    assert "content" in d
    assert "item_0 = 999" in d["content"]
    assert not d["content"].startswith("// File unchanged")


async def test_truncated_read_then_reread_returns_content(ctx: MagicMock, tmp_path: Path) -> None:
    """A truncated read followed by another read still delivers content."""
    big = tmp_path / "big.py"
    body = "\n".join(f"row_{i} = {i}" for i in range(150)) + "\n"
    big.write_text(body)

    d1 = _parse(await read(ctx, str(big), max_size=200))
    assert "content" in d1
    assert len(d1["content"]) < len(body)  # summarized, not the whole file

    d2 = _parse(await read(ctx, str(big), max_size=200))
    assert d2.get("unchanged") is not True
    assert "content" in d2


# ---------------------------------------------------------------------------
# Diff gate: small real changes to mid/large files still diff; tiny files
# return full content (a diff's @@-header overhead isn't worth it)
# ---------------------------------------------------------------------------


async def test_small_change_to_midsize_file_returns_diff(
    tmp_cache: SemanticCache, tmp_path: Path
) -> None:
    f = tmp_path / "mid.py"
    f.write_text("\n".join(f"item_{i} = {i}" for i in range(150)) + "\n")
    await smart_read(tmp_cache, str(f))  # cache it

    body = f.read_text().replace("item_0 = 0", "item_0 = 999")
    f.write_text(body)
    r = await smart_read(tmp_cache, str(f))
    assert r.is_diff is True


async def test_small_change_to_tiny_file_returns_full(
    tmp_cache: SemanticCache, tmp_path: Path
) -> None:
    f = tmp_path / "tiny.py"
    f.write_text("a = 1\nb = 2\n")
    await smart_read(tmp_cache, str(f))

    f.write_text("a = 1\nb = 3\n")
    r = await smart_read(tmp_cache, str(f))
    assert r.is_diff is False
    assert "b = 3" in r.content


# ---------------------------------------------------------------------------
# Cache-aware ranged reads: known_hash short-circuits a ranged re-read
# ---------------------------------------------------------------------------


async def test_ranged_read_known_hash_short_circuits(
    ctx: MagicMock, tmp_path: Path, tmp_cache: SemanticCache
) -> None:
    f = tmp_path / "ranged.py"
    f.write_text("\n".join(f"row_{i} = {i}" for i in range(40)) + "\n")
    d1 = _parse(await read(ctx, str(f)))
    h = d1["content_hash"]

    d = _parse(await read(ctx, str(f), offset=2, limit=3, known_hash=h))
    assert d.get("unchanged") is True
    assert d["content_hash"] == h
    assert "content" not in d
    assert d["lines"]["total"] == 40


async def test_ranged_read_known_hash_mtime_bump_falls_through(
    ctx: MagicMock, tmp_path: Path
) -> None:
    """An mtime bump (even with identical content) must defeat the
    short-circuit and return literal lines, never a false unchanged."""
    f = tmp_path / "ranged2.py"
    f.write_text("\n".join(f"row_{i} = {i}" for i in range(40)) + "\n")
    d1 = _parse(await read(ctx, str(f)))
    h = d1["content_hash"]

    future = time.time() + 60
    os.utime(f, (future, future))
    d = _parse(await read(ctx, str(f), offset=1, limit=3, known_hash=h))
    assert d.get("unchanged") is not True
    assert "content" in d
