"""An operation must never report work it did not do.

Every case here is a path that previously answered confidently and wrongly:
a batch edit counted as applied when the file never received it, a grep whose
capped count read as a complete total, an anchor probe that searched a summary
instead of the file. The shared property is honesty about scope — a tool may
do less than asked, but it has to say so.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from semantic_cache_mcp.cache import SemanticCache, find_edit_anchors
from semantic_cache_mcp.cache.read import smart_read
from semantic_cache_mcp.cache.write import smart_batch_edit

# Bigger than MAX_CONTENT_SIZE * 10 (the ceiling edit_preview reads under), so
# the summarizer would engage if the read did not opt out of it.
_OVER_SUMMARIZE_THRESHOLD_LINES = 20_000


async def _no_roots() -> list[str]:
    """Stand-in for Context.list_roots — these tests pass absolute paths."""
    return []


class TestBatchEditReportsWhatItDid:
    """smart_batch_edit outcomes must match what actually reached the file."""

    async def test_anchor_consumed_by_earlier_edit_is_reported_failed(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """An edit whose anchor an earlier edit deleted must not count as applied."""
        target = temp_dir / "consumed.txt"
        target.write_text("alpha\nbravo\ncharlie\n")

        result = await smart_batch_edit(
            semantic_cache,
            str(target),
            [("alpha\nbravo\n", "X\n"), ("bravo", "ZZZ")],
        )

        # The "bravo" edit sorts later by line and applies first, taking the
        # text the multi-line anchor needed.
        assert target.read_text() == "alpha\nZZZ\ncharlie\n"
        assert result.outcomes[0].success is False
        assert "no longer present" in (result.outcomes[0].error or "")
        assert result.outcomes[1].success is True
        assert result.succeeded == 1
        assert result.failed == 1

    async def test_anchor_inside_accepted_range_is_rejected(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """A find/replace landing inside another edit's line range must not be silently wiped."""
        target = temp_dir / "clobber.txt"
        target.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")

        result = await smart_batch_edit(
            semantic_cache,
            str(target),
            [("line5", "EDITED5"), (None, "REPLACED", 4, 6)],
        )

        text = target.read_text()
        assert "EDITED5" in text
        assert result.outcomes[0].success is True
        assert result.outcomes[1].success is False
        assert "line range 4-6" in (result.outcomes[1].error or "")

    async def test_range_swallowing_earlier_anchor_is_rejected(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Order does not matter: the later-listed conflicting edit is the one that fails."""
        target = temp_dir / "clobber2.txt"
        target.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")

        result = await smart_batch_edit(
            semantic_cache,
            str(target),
            [(None, "REPLACED", 4, 6), ("line5", "EDITED5")],
        )

        assert result.outcomes[0].success is True
        assert result.outcomes[1].success is False
        assert "inside another edit's line range" in (result.outcomes[1].error or "")
        assert "REPLACED" in target.read_text()

    async def test_ambiguous_anchor_is_rejected_not_silently_narrowed(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """A non-unique anchor must fail, as it does in `edit`, not edit one of N."""
        target = temp_dir / "ambiguous.txt"
        target.write_text("dup\ndup\ndup\n")

        result = await smart_batch_edit(semantic_cache, str(target), [("dup", "one")])

        assert target.read_text() == "dup\ndup\ndup\n"
        assert result.succeeded == 0
        assert result.failed == 1
        error = result.outcomes[0].error or ""
        assert "found 3 times" in error
        assert "start_line/end_line" in error

    async def test_oversized_file_is_refused_like_edit(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """batch_edit was the one mutation path with no ceiling on what it read."""
        from semantic_cache_mcp.cache._helpers import MAX_EDIT_SIZE

        target = temp_dir / "oversized.txt"
        target.write_text("x" * (MAX_EDIT_SIZE + 1))

        with pytest.raises(ValueError, match="too large"):
            await smart_batch_edit(semantic_cache, str(target), [("x", "y")])

    async def test_disjoint_edits_still_all_apply(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """The new guards must not reject edits that genuinely do not interact."""
        target = temp_dir / "disjoint.txt"
        target.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n")

        result = await smart_batch_edit(
            semantic_cache,
            str(target),
            [("line2", "TWO"), ("line8", "EIGHT"), (None, "FIVE", 5, 5)],
        )

        text = target.read_text()
        assert result.succeeded == 3
        assert result.failed == 0
        assert "TWO" in text
        assert "EIGHT" in text
        assert "FIVE" in text


class TestGrepReportsTruncation:
    """A capped scan reports a floor, and says that it is one."""

    @staticmethod
    async def _seed(cache: SemanticCache, temp_dir: Path, files: int, per_file: int) -> None:
        for index in range(files):
            path = temp_dir / f"needles{index}.txt"
            path.write_text("\n".join("needle here" for _ in range(per_file)) + "\n")
            await smart_read(cache, str(path), force_full=True)

    async def test_match_cap_marks_result_incomplete(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        await self._seed(semantic_cache, temp_dir, files=3, per_file=60)

        results = await semantic_cache._storage.grep("needle", max_matches=100)

        assert sum(len(r["matches"]) for r in results) == 100
        assert results.complete is False
        assert results.truncated_matches is True
        assert results.files_not_searched >= 1

    async def test_file_cap_marks_result_incomplete(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        await self._seed(semantic_cache, temp_dir, files=4, per_file=2)

        results = await semantic_cache._storage.grep("needle", max_files=2, max_matches=10_000)

        assert len(results) == 2
        assert results.complete is False
        assert results.truncated_files is True

    async def test_uncapped_scan_claims_completeness(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        await self._seed(semantic_cache, temp_dir, files=3, per_file=60)

        results = await semantic_cache._storage.grep("needle", max_matches=10_000)

        assert sum(len(r["matches"]) for r in results) == 180
        assert results.complete is True
        assert results.files_not_searched == 0

    async def test_empty_result_is_complete(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Zero matches from a full scan is a real zero, not a truncation."""
        await self._seed(semantic_cache, temp_dir, files=1, per_file=2)

        results = await semantic_cache._storage.grep("zzz_absent_token", fixed_string=True)

        assert list(results) == []
        assert results.complete is True


class TestAnchorProbeReadsTheFile:
    """edit_preview locates anchors in the literal file, never in a summary."""

    async def test_anchor_past_the_summarize_threshold_is_found_at_its_real_line(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        anchor = "UNIQUE_ANCHOR_TOKEN = 1"
        head = "\n".join(
            f"# head filler {i} pad pad pad pad" for i in range(_OVER_SUMMARIZE_THRESHOLD_LINES)
        )
        tail = "\n".join(
            f"# tail filler {i} pad pad pad pad" for i in range(_OVER_SUMMARIZE_THRESHOLD_LINES)
        )
        target = temp_dir / "huge.py"
        target.write_text(f"{head}\n{anchor}\n{tail}\n")
        expected_line = target.read_text().splitlines().index(anchor) + 1

        result = await smart_read(
            semantic_cache,
            str(target),
            max_size=100_000 * 10,
            diff_mode=False,
            force_full=True,
            refresh_cache=False,
            summarize=False,
        )

        assert result.truncated is False
        count, line_numbers = find_edit_anchors(result.content, anchor, max_results=50)
        assert count == 1
        assert line_numbers == [expected_line]

    async def test_edit_preview_tool_passes_summarize_false(self) -> None:
        """Guard the wiring itself — the bug was one default, not the algorithm."""
        import inspect

        from semantic_cache_mcp.server import tools

        source = inspect.getsource(tools.edit_preview)
        assert "summarize=False" in source


class TestRangedReadHashIsHonest:
    """A claimable hash must vouch for bytes the caller was actually sent."""

    async def test_whole_file_window_delivers_reconstructable_content(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """The rendering must be losslessly invertible, or the hash is a lie.

        A window covering every line mints a claimable `content_hash`, which the
        caller can redeem for `unchanged`. Stripping trailing whitespace out of
        the delivered lines certified possession of content never sent.
        """
        from semantic_cache_mcp.server import tools

        source = "alpha   \nbeta\t\ngamma\n"
        target = temp_dir / "trailing.txt"
        target.write_text(source)

        ctx = SimpleNamespace(lifespan_context={"cache": semantic_cache})
        ctx.list_roots = _no_roots
        payload = await tools.read(ctx, str(target), offset=1)

        assert "content_hash" in payload, "a whole-file window should mint a claimable hash"
        body = payload["content"]
        # Strip the "%6d\t" gutter back off and the original must return.
        recovered = "".join(line.split("\t", 1)[1] + "\n" for line in body.split("\n"))
        assert recovered == source

    async def test_partial_window_still_withholds_a_claimable_hash(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        from semantic_cache_mcp.server import tools

        target = temp_dir / "partial.txt"
        target.write_text("\n".join(f"line{i}" for i in range(1, 21)) + "\n")

        ctx = SimpleNamespace(lifespan_context={"cache": semantic_cache})
        ctx.list_roots = _no_roots
        payload = await tools.read(ctx, str(target), offset=1, limit=5)

        assert "content_hash" not in payload
        assert payload["file_hash"].startswith("partial:")
        assert "coverage_token" in payload


@pytest.mark.parametrize("mode", ["match_cap", "file_cap"])
async def test_grep_results_behaves_like_a_list(
    semantic_cache: SemanticCache, temp_dir: Path, mode: str
) -> None:
    """Callers that treat the result as a plain list keep working."""
    path = temp_dir / "listlike.txt"
    path.write_text("needle\nneedle\n")
    await smart_read(semantic_cache, str(path), force_full=True)

    kwargs = {"max_matches": 1} if mode == "match_cap" else {"max_files": 1}
    results = await semantic_cache._storage.grep("needle", **kwargs)  # type: ignore[arg-type]

    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0]["path"] == str(path)
    assert [r["path"] for r in results] == [str(path)]


class TestGrepAnswersAboutTheRightFiles:
    """A path filter that names a directory must find what is under it."""

    async def test_directory_filter_matches_files_beneath_it(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """A bare directory is a filter, not a typo — it names every file below."""
        pkg = temp_dir / "pkg"
        pkg.mkdir()
        for i in range(3):
            leaf = pkg / f"m{i}.py"
            leaf.write_text("needle here\n")
            await smart_read(semantic_cache, str(leaf), force_full=True)

        results = await semantic_cache._storage.grep("needle", path=str(pkg))

        assert len(results) == 3
        assert all(r["path"].startswith(str(pkg)) for r in results)

    async def test_directory_filter_matches_whole_components_only(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """`src` is a directory of `src/a.py`, never of `srclib/a.py`."""
        for name in ("src", "srclib"):
            d = temp_dir / name
            d.mkdir()
            leaf = d / "a.py"
            leaf.write_text("needle here\n")
            await smart_read(semantic_cache, str(leaf), force_full=True)

        results = await semantic_cache._storage.grep("needle", path=str(temp_dir / "src"))

        assert [r["path"] for r in results] == [str(temp_dir / "src" / "a.py")]

    async def test_seeded_directory_is_not_reported_as_an_unseeded_cache(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """The empty-result explanation must not blame a cache that is warm."""
        from semantic_cache_mcp.server import tools

        pkg = temp_dir / "warm"
        pkg.mkdir()
        leaf = pkg / "a.py"
        leaf.write_text("needle here\n")
        await smart_read(semantic_cache, str(leaf), force_full=True)

        ctx = SimpleNamespace(lifespan_context={"cache": semantic_cache})
        ctx.list_roots = _no_roots
        payload = await tools.grep(ctx, "needle", path=str(pkg))

        assert payload["total_matches"] == 1
        assert "reason" not in payload

    async def test_genuinely_unseeded_directory_still_says_so(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """The cache-miss explanation has to survive the directory fix."""
        from semantic_cache_mcp.server import tools

        cold = temp_dir / "cold"
        cold.mkdir()
        (cold / "a.py").write_text("needle here\n")

        ctx = SimpleNamespace(lifespan_context={"cache": semantic_cache})
        ctx.list_roots = _no_roots
        payload = await tools.grep(ctx, "needle", path=str(cold))

        assert payload["total_matches"] == 0
        assert payload["reason"] == "no_files_cached_under_path"


class TestTruncationKeepsItsExplanation:
    """A response cut down for size must still say what it found."""

    async def test_capped_grep_keeps_its_counts_when_trimmed(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Trimming drops match lines, never the scalars that explain them."""
        from semantic_cache_mcp.server import tools
        from semantic_cache_mcp.server.response import _response_overrides

        path = temp_dir / "many.txt"
        path.write_text(
            "\n".join(
                f"needle the quick brown fox number {i} jumps over the lazy dog" for i in range(400)
            )
            + "\n"
        )
        await smart_read(semantic_cache, str(path), force_full=True)

        ctx = SimpleNamespace(lifespan_context={"cache": semantic_cache})
        ctx.list_roots = _no_roots

        # Sweep the whole band between "one match fits" and "everything fits".
        # A response that reports nothing is the failure being guarded against,
        # so no cap in the range may produce one.
        for cap in (2_000, 4_000, 6_000, 8_000, 10_000, 12_000, 15_000):
            with _response_overrides("compact", cap):
                payload = await tools.grep(ctx, "needle", path=str(path), max_matches=400)

            assert payload.get("total_matches") == 400, f"cap={cap} lost the total: {payload}"
            assert "message" not in payload or "files" in payload, (
                f"cap={cap} was cut to a bare truncation notice: {payload}"
            )

    def test_minimal_payload_retains_the_explanatory_scalars(self) -> None:
        """The keep list is what makes a trimmed response readable."""
        from semantic_cache_mcp.server.response import _minimal_payload

        trimmed = _minimal_payload(
            {
                "ok": True,
                "tool": "grep",
                "pattern": "needle",
                "path": "/x",
                "total_matches": 400,
                "files_matched": 3,
                "complete": False,
                "limit_reached": "max_matches",
                "files_not_searched": 9,
                "hint": "raise max_matches",
                "files": [{"path": "/x/a", "matches": ["..."] * 500}],
            }
        )

        assert trimmed["total_matches"] == 400
        assert trimmed["complete"] is False
        assert trimmed["limit_reached"] == "max_matches"
        assert trimmed["files_not_searched"] == 9
        assert trimmed["truncated"] is True
        assert "files" not in trimmed, "the bulky field is what trimming is for"
