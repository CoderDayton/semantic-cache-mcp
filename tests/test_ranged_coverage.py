"""Windowed possession for ranged reads.

A ranged read earns no claimable `content_hash` — the caller never saw the rest
of the file — but it does earn a signed `coverage_token` naming the lines it
was sent. Echoing that token back buys `unchanged` for a window already held,
widens coverage for a new one, and upgrades to a real `content_hash` once the
windows account for every line.

Every rejection path here must fail the same way: send the bytes.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastmcp import Context

from semantic_cache_mcp.cache import SemanticCache
from semantic_cache_mcp.server import _coverage
from semantic_cache_mcp.server._coverage import (
    EMPTY_SPANS,
    MAX_TRACKED_SPANS,
    LineSpans,
    decode_coverage_token,
    encode_coverage_token,
)
from semantic_cache_mcp.server.tools import read


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(db_path=tmp_path / "cache.db")


@pytest.fixture()
def ctx(tmp_cache: SemanticCache) -> MagicMock:
    c = MagicMock(spec=Context)
    c.lifespan_context = {"cache": tmp_cache}
    return c


@pytest.fixture()
def wide_file(tmp_path: Path) -> Path:
    p = tmp_path / "wide.py"
    p.write_text("\n".join(f"row_{i} = {i}" for i in range(40)) + "\n")
    return p


def _parse(response) -> dict:  # noqa: ANN001
    import json

    if isinstance(response, dict):
        return response
    return json.loads(response)


# ---------------------------------------------------------------------------
# LineSpans: the pure coverage algebra
# ---------------------------------------------------------------------------


def test_merge_coalesces_overlapping_and_adjacent() -> None:
    spans = EMPTY_SPANS.merge(0, 10).merge(10, 20).merge(5, 8)
    assert spans.spans == ((0, 20),)


def test_merge_keeps_disjoint_spans_sorted() -> None:
    spans = EMPTY_SPANS.merge(30, 40).merge(0, 10)
    assert spans.spans == ((0, 10), (30, 40))


def test_merge_ignores_empty_window() -> None:
    assert EMPTY_SPANS.merge(5, 5) is EMPTY_SPANS
    assert EMPTY_SPANS.merge(9, 3) is EMPTY_SPANS


def test_covers_requires_the_whole_window() -> None:
    spans = EMPTY_SPANS.merge(10, 20)
    assert spans.covers(10, 20)
    assert spans.covers(12, 15)  # a sub-window is held
    assert not spans.covers(9, 20)  # one line short at the front
    assert not spans.covers(10, 21)  # one line short at the back
    assert not spans.covers(0, 5)  # disjoint


def test_covers_across_a_gap_is_not_coverage() -> None:
    spans = EMPTY_SPANS.merge(0, 10).merge(20, 30)
    assert not spans.covers(5, 25)


def test_covers_all_and_empty_file() -> None:
    assert EMPTY_SPANS.merge(0, 40).covers_all(40)
    assert not EMPTY_SPANS.merge(0, 39).covers_all(40)
    # A file with no lines is trivially fully covered.
    assert EMPTY_SPANS.covers_all(0)


def test_span_cap_drops_the_narrowest_and_never_over_claims() -> None:
    """Overflow must under-claim: forgetting a span costs a re-read only."""
    spans = EMPTY_SPANS
    # Disjoint windows of strictly increasing width, widest last.
    for i in range(MAX_TRACKED_SPANS + 4):
        start = i * 1000
        spans = spans.merge(start, start + i + 1)

    assert len(spans.spans) == MAX_TRACKED_SPANS
    # The narrowest windows were dropped, the widest kept.
    assert not spans.covers(0, 1)
    widest_start = (MAX_TRACKED_SPANS + 3) * 1000
    assert spans.covers(widest_start, widest_start + MAX_TRACKED_SPANS + 4)


def test_spans_encode_decode_roundtrip() -> None:
    spans = EMPTY_SPANS.merge(0, 10).merge(30, 42)
    assert LineSpans.decode(spans.encode()) == spans
    assert LineSpans.decode("") == EMPTY_SPANS


@pytest.mark.parametrize("bad", ["10", "abc-20", "10-abc", "20-10", "-5-10", "10-10"])
def test_malformed_spans_decode_to_none(bad: str) -> None:
    assert LineSpans.decode(bad) is None


# ---------------------------------------------------------------------------
# Token signing: only tokens this process minted are honored
# ---------------------------------------------------------------------------


def test_token_roundtrip() -> None:
    spans = EMPTY_SPANS.merge(399, 739)
    token = encode_coverage_token("a" * 64, spans)
    assert decode_coverage_token(token) == ("a" * 64, spans)


def test_tampered_span_is_rejected() -> None:
    """Widening the ranges by hand must not widen the claim."""
    token = encode_coverage_token("a" * 64, EMPTY_SPANS.merge(0, 10))
    forged = token.replace("0-10", "0-9999", 1)
    assert forged != token
    assert decode_coverage_token(forged) is None


def test_unsigned_values_are_rejected() -> None:
    """Everything else a caller might pass as known_hash claims nothing."""
    assert decode_coverage_token(None) is None
    assert decode_coverage_token("") is None
    assert decode_coverage_token("a" * 64) is None  # a bare content_hash
    assert decode_coverage_token(f"partial:{'a' * 64}") is None  # a file_hash
    assert decode_coverage_token("pcov1:aaaa:0-10:deadbeefdeadbeef") is None


def test_token_from_another_process_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A restarted worker signs with a fresh key, so old tokens fail closed."""
    token = encode_coverage_token("a" * 64, EMPTY_SPANS.merge(0, 10))
    assert decode_coverage_token(token) is not None

    monkeypatch.setattr(_coverage, "_SIGNING_KEY", b"\x00" * 32)
    assert decode_coverage_token(token) is None


# ---------------------------------------------------------------------------
# The read tool: minting, redeeming, and refusing
# ---------------------------------------------------------------------------


async def test_ranged_read_mints_a_token_but_no_content_hash(
    ctx: MagicMock, wide_file: Path
) -> None:
    d = _parse(await read(ctx, str(wide_file), offset=5, limit=10))
    assert "content_hash" not in d  # it saw ten lines, not the file
    assert d["file_hash"].startswith("partial:")
    assert decode_coverage_token(d["coverage_token"]) is not None


async def test_token_buys_unchanged_for_the_same_window(ctx: MagicMock, wide_file: Path) -> None:
    first = _parse(await read(ctx, str(wide_file), offset=5, limit=10))
    again = _parse(
        await read(ctx, str(wide_file), offset=5, limit=10, known_hash=first["coverage_token"])
    )
    assert again.get("unchanged") is True
    assert "content" not in again
    assert again["lines"]["total"] == 40


async def test_token_buys_unchanged_for_a_sub_window(ctx: MagicMock, wide_file: Path) -> None:
    """Holding lines 5-14 means holding lines 7-9."""
    first = _parse(await read(ctx, str(wide_file), offset=5, limit=10))
    inner = _parse(
        await read(ctx, str(wide_file), offset=7, limit=3, known_hash=first["coverage_token"])
    )
    assert inner.get("unchanged") is True
    assert "content" not in inner


async def test_new_window_is_delivered_and_widens_coverage(ctx: MagicMock, wide_file: Path) -> None:
    first = _parse(await read(ctx, str(wide_file), offset=1, limit=10))
    second = _parse(
        await read(ctx, str(wide_file), offset=21, limit=10, known_hash=first["coverage_token"])
    )
    assert second.get("unchanged") is not True
    assert "row_20 = 20" in second["content"]  # line 21 really is delivered

    decoded = decode_coverage_token(second["coverage_token"])
    assert decoded is not None
    assert decoded[1].spans == ((0, 10), (20, 30))


async def test_full_coverage_upgrades_to_a_claimable_content_hash(
    ctx: MagicMock, wide_file: Path
) -> None:
    """Windows that add up to the whole file leave the caller holding it."""
    first = _parse(await read(ctx, str(wide_file), offset=1, limit=20))
    assert "content_hash" not in first

    second = _parse(
        await read(ctx, str(wide_file), offset=21, limit=20, known_hash=first["coverage_token"])
    )
    assert second["content_hash"]  # every line has now been delivered
    assert "coverage_token" not in second  # nothing partial left to record

    # And the minted hash is genuinely redeemable on a normal read.
    d = _parse(await read(ctx, str(wide_file), known_hash=second["content_hash"]))
    assert d.get("unchanged") is True


async def test_changed_file_defeats_a_held_token(ctx: MagicMock, wide_file: Path) -> None:
    """A token names the version it came from; new bytes retire it."""
    first = _parse(await read(ctx, str(wide_file), offset=5, limit=10))
    wide_file.write_text("\n".join(f"row_{i} = {i + 1}" for i in range(40)) + "\n")

    d = _parse(
        await read(ctx, str(wide_file), offset=5, limit=10, known_hash=first["coverage_token"])
    )
    assert d.get("unchanged") is not True
    assert "content" in d


async def test_tampered_token_gets_content_not_a_claim(ctx: MagicMock, wide_file: Path) -> None:
    first = _parse(await read(ctx, str(wide_file), offset=5, limit=2))
    forged = first["coverage_token"].replace("4-6", "0-40", 1)

    d = _parse(await read(ctx, str(wide_file), offset=1, limit=40, known_hash=forged))
    assert d.get("unchanged") is not True
    assert "content" in d


async def test_whole_file_hash_still_never_claims_a_narrow_window(
    ctx: MagicMock, wide_file: Path
) -> None:
    """Regression guard: holding the file must not silence a range request.

    A caller asking for a window it has never been sent must get the bytes,
    even while holding the file's `content_hash`. Only a coverage token —
    which names the lines actually delivered — redeems a window.
    """
    full = _parse(await read(ctx, str(wide_file)))
    d = _parse(await read(ctx, str(wide_file), offset=2, limit=3, known_hash=full["content_hash"]))

    assert d.get("unchanged") is not True
    assert "row_1 = 1" in d["content"]
    assert decode_coverage_token(d["coverage_token"]) is not None


async def test_empty_window_past_eof_claims_nothing(ctx: MagicMock, wide_file: Path) -> None:
    """An out-of-range window delivers no lines, so it cannot answer unchanged."""
    first = _parse(await read(ctx, str(wide_file), offset=5, limit=10))
    d = _parse(
        await read(ctx, str(wide_file), offset=500, limit=5, known_hash=first["coverage_token"])
    )
    assert d.get("unchanged") is not True


# ---------------------------------------------------------------------------
# Window diffs: a holder of the superseded window gets only what moved in it
# ---------------------------------------------------------------------------


@pytest.fixture()
def tall_file(tmp_path: Path) -> Path:
    """Wide enough lines that a 100-line window clears the diff floor."""
    p = tmp_path / "tall.py"
    p.write_text(
        "\n".join(f"CONSTANT_VALUE_{i} = compute_something({i}, scale={i * 2})" for i in range(200))
        + "\n"
    )
    return p


def _touch_future(path: Path) -> None:
    """Force the cache to read as stale regardless of clock granularity."""
    future = time.time() + 60
    os.utime(path, (future, future))


def _edit_line_five(path: Path) -> str:
    body = path.read_text().replace(
        "CONSTANT_VALUE_5 = compute_something(5, scale=10)",
        "CONSTANT_VALUE_5 = compute_something(5, scale=999)",
    )
    path.write_text(body)
    _touch_future(path)
    return body


async def test_changed_window_returns_a_diff_to_a_holder(ctx: MagicMock, tall_file: Path) -> None:
    first = _parse(await read(ctx, str(tall_file), offset=1, limit=100))
    body = _edit_line_five(tall_file)

    d = _parse(
        await read(ctx, str(tall_file), offset=1, limit=100, known_hash=first["coverage_token"])
    )
    assert d.get("is_diff") is True
    assert "@@" in d["content"]
    assert "scale=999" in d["content"]  # the new line
    assert "scale=10)" in d["content"]  # and the one it replaced
    assert len(d["content"]) < len(body)

    # Coverage resets to this window: outside it the caller still holds the
    # old file, so nothing wider may be claimed.
    decoded = decode_coverage_token(d["coverage_token"])
    assert decoded is not None
    assert decoded[1].spans == ((0, 100),)


async def test_changed_window_without_a_token_gets_full_content(
    ctx: MagicMock, tall_file: Path
) -> None:
    """A diff is unusable without its base, so an unproven caller gets lines."""
    await read(ctx, str(tall_file), offset=1, limit=100)
    _edit_line_five(tall_file)

    d = _parse(await read(ctx, str(tall_file), offset=1, limit=100))
    assert d.get("is_diff") is not True
    assert "CONSTANT_VALUE_0" in d["content"]


async def test_window_diff_hunks_use_file_line_numbers(ctx: MagicMock, tall_file: Path) -> None:
    """A window diff must speak the coordinates every other diff speaks.

    Numbering hunks from the window's own first line would make a caller
    reading `@@ -48` go and re-read file line 48 — a wasted round trip that
    costs far more than the header ever saved.
    """
    first = _parse(await read(ctx, str(tall_file), offset=101, limit=100))
    body = tall_file.read_text().replace(
        "CONSTANT_VALUE_150 = compute_something(150, scale=300)",
        "CONSTANT_VALUE_150 = compute_something(150, scale=999)",
    )
    tall_file.write_text(body)
    _touch_future(tall_file)

    d = _parse(
        await read(ctx, str(tall_file), offset=101, limit=100, known_hash=first["coverage_token"])
    )
    assert d.get("is_diff") is True
    header = re.search(r"@@ -(\d+)", d["content"])
    assert header is not None
    # The edit is at file line 151; the hunk opens a few context lines above it
    # and must land inside the requested window, not near its start.
    assert 101 <= int(header.group(1)) <= 151


async def test_diff_requires_holding_the_whole_window(ctx: MagicMock, tall_file: Path) -> None:
    """Holding half the window proves nothing about the half it did not see."""
    first = _parse(await read(ctx, str(tall_file), offset=1, limit=50))
    _edit_line_five(tall_file)

    d = _parse(
        await read(ctx, str(tall_file), offset=1, limit=100, known_hash=first["coverage_token"])
    )
    assert d.get("is_diff") is not True
    assert "CONSTANT_VALUE_99" in d["content"]  # the unheld half really is sent
