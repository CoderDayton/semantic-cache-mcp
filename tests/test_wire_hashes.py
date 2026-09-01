"""Possession hashes on the wire are short, and short is not the same as loose.

A full 64-hex BLAKE3 digest costs ~18 tokens, and one goes out with every file
delivered — ~500 on a 30-file `batch_read`, plus ~38 per ranged read inside a
`coverage_token`. Sixteen hex characters is ample to tell two versions of one
file apart, since a claim is only ever checked against the entry for the path
it names.

The danger in accepting a prefix is a claim that matches everything. So the
acceptance rule is exact-length, not "starts with": a caller may echo the
16-character wire form or the full digest, and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastmcp import Context

from semantic_cache_mcp.cache import SemanticCache
from semantic_cache_mcp.core.hashing import WIRE_HASH_LENGTH, hash_matches, short_hash
from semantic_cache_mcp.server._coverage import decode_coverage_token
from semantic_cache_mcp.server.tools import batch_read, edit, read, write

_FULL = "a" * 64
_OTHER = "b" * 64


@pytest.fixture
def cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(db_path=tmp_path / "cache.db")


@pytest.fixture
def ctx(cache: SemanticCache) -> MagicMock:
    context = MagicMock(spec=Context)
    context.lifespan_context = {"cache": cache}
    return context


@pytest.fixture
def source(tmp_path: Path) -> Path:
    target = tmp_path / "mod.py"
    target.write_text("def hello():\n    return 'world'\n")
    return target


class TestTheMatcher:
    def test_the_wire_form_is_accepted(self) -> None:
        assert hash_matches(short_hash(_FULL), _FULL) is True

    def test_the_full_digest_is_accepted(self) -> None:
        assert hash_matches(_FULL, _FULL) is True

    @pytest.mark.parametrize("length", [1, 2, 8, WIRE_HASH_LENGTH - 1, WIRE_HASH_LENGTH + 1, 63])
    def test_any_other_length_is_refused(self, length: int) -> None:
        """A shorter prefix would match every version of the file at once."""
        assert hash_matches(_FULL[:length], _FULL) is False

    def test_a_different_hash_is_refused(self) -> None:
        assert hash_matches(short_hash(_OTHER), _FULL) is False

    @pytest.mark.parametrize("claim", ["", None, "   ", "partial:" + "a" * 16, "A" * 16])
    def test_non_hex_and_empty_claims_are_refused(self, claim: str | None) -> None:
        assert hash_matches(claim, _FULL) is False

    def test_a_missing_stored_hash_matches_nothing(self) -> None:
        assert hash_matches(short_hash(_FULL), None) is False
        assert hash_matches(short_hash(_FULL), "") is False

    def test_short_hash_is_stable_and_the_right_length(self) -> None:
        assert short_hash(_FULL) == _FULL[:WIRE_HASH_LENGTH]
        assert len(short_hash(_FULL)) == WIRE_HASH_LENGTH

    def test_short_hash_leaves_an_already_short_value_alone(self) -> None:
        assert short_hash(short_hash(_FULL)) == short_hash(_FULL)


class TestReadEmitsTheWireForm:
    async def test_content_hash_is_short(self, ctx: MagicMock, source: Path) -> None:
        payload = await read(ctx, str(source))
        assert len(payload["content_hash"]) == WIRE_HASH_LENGTH

    async def test_the_short_hash_redeems_for_unchanged(self, ctx: MagicMock, source: Path) -> None:
        first = await read(ctx, str(source))
        second = await read(ctx, str(source), known_hash=first["content_hash"])
        assert second["unchanged"] is True

    async def test_a_truncated_claim_is_refused(self, ctx: MagicMock, source: Path) -> None:
        first = await read(ctx, str(source))
        again = await read(ctx, str(source), known_hash=first["content_hash"][:8])

        assert again.get("unchanged") is not True
        assert "content" in again

    async def test_a_partial_hash_is_short_and_still_prefixed(
        self, ctx: MagicMock, source: Path
    ) -> None:
        payload = await read(ctx, str(source), outline=True)
        value = payload["file_hash"]

        assert value.startswith("partial:")
        assert len(value.removeprefix("partial:")) == WIRE_HASH_LENGTH

    async def test_a_partial_hash_still_buys_nothing(self, ctx: MagicMock, source: Path) -> None:
        outlined = await read(ctx, str(source), outline=True)
        payload = await read(ctx, str(source), known_hash=outlined["file_hash"])
        assert payload.get("unchanged") is not True


class TestCoverageTokens:
    async def test_the_token_carries_the_wire_form(self, ctx: MagicMock, source: Path) -> None:
        payload = await read(ctx, str(source), offset=1, limit=1)
        decoded = decode_coverage_token(payload["coverage_token"])

        assert decoded is not None
        assert len(decoded[0]) == WIRE_HASH_LENGTH

    async def test_the_token_still_redeems(self, ctx: MagicMock, source: Path) -> None:
        first = await read(ctx, str(source), offset=1, limit=1)
        second = await read(ctx, str(source), offset=1, limit=1, known_hash=first["coverage_token"])
        assert second["unchanged"] is True

    async def test_widening_windows_still_mint_a_claimable_hash(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        target = tmp_path / "tall.py"
        target.write_text("".join(f"line_{i}\n" for i in range(10)))

        first = await read(ctx, str(target), offset=1, limit=5)
        second = await read(ctx, str(target), offset=6, limit=5, known_hash=first["coverage_token"])

        assert "content_hash" in second
        assert len(second["content_hash"]) == WIRE_HASH_LENGTH


class TestMutationsAcceptTheWireForm:
    async def test_edit_takes_a_short_known_hash(self, ctx: MagicMock, source: Path) -> None:
        first = await read(ctx, str(source))
        result = await edit(
            ctx,
            str(source),
            old_string="world",
            new_string="planet",
            known_hash=first["content_hash"],
        )

        assert "content_hash" in result, "a proven caller should get a claimable hash back"
        assert len(result["content_hash"]) == WIRE_HASH_LENGTH

    async def test_edit_without_the_hash_still_earns_nothing(
        self, ctx: MagicMock, source: Path
    ) -> None:
        result = await edit(ctx, str(source), old_string="world", new_string="planet")
        assert "content_hash" not in result
        assert result["file_hash"].startswith("partial:")

    async def test_the_hash_an_edit_returns_redeems_on_the_next_read(
        self, ctx: MagicMock, source: Path
    ) -> None:
        first = await read(ctx, str(source))
        edited = await edit(
            ctx,
            str(source),
            old_string="world",
            new_string="planet",
            known_hash=first["content_hash"],
        )
        payload = await read(ctx, str(source), known_hash=edited["content_hash"])

        assert payload["unchanged"] is True

    async def test_write_returns_the_wire_form(self, ctx: MagicMock, tmp_path: Path) -> None:
        target = tmp_path / "new.py"
        result = await write(ctx, str(target), content="x = 1\n")

        assert len(result["content_hash"]) == WIRE_HASH_LENGTH


class TestBatchRead:
    async def test_short_hashes_round_trip_through_known_hashes(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        first_file = tmp_path / "a.py"
        second_file = tmp_path / "b.py"
        first_file.write_text("alpha = 1\n")
        second_file.write_text("beta = 2\n")
        paths = f"{first_file},{second_file}"

        first = await batch_read(ctx, paths)
        claims = {item["path"]: item["content_hash"] for item in first["files"]}
        assert all(len(h) == WIRE_HASH_LENGTH for h in claims.values())

        second = await batch_read(ctx, paths, known_hashes=json.dumps(claims))
        assert second["summary"]["unchanged_count"] == 2

    async def test_a_truncated_claim_does_not_suppress_a_file(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        target = tmp_path / "a.py"
        target.write_text("alpha = 1\n")

        first = await batch_read(ctx, str(target))
        claim = first["files"][0]["content_hash"][:8]

        second = await batch_read(ctx, str(target), known_hashes=json.dumps({str(target): claim}))
        assert second["summary"].get("unchanged_count") is None
        assert second["files"][0]["content"] == "alpha = 1\n"
