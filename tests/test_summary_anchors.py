"""A summary must say where its pieces came from.

`summarize_semantic` returns non-contiguous segments. Without line anchors the
reader is told to "use offset=<line>" while holding no line it can name, so the
follow-up is a guess and usually a full re-read — which costs more than the
summary saved. Each anchor must name the exact range of the segment it precedes.
"""

from __future__ import annotations

import re

import pytest

from semantic_cache_mcp.core.text import (
    DEFAULT_SUMMARIZATION_CONFIG,
    SummarizationConfig,
    summarize_semantic,
)

_ANCHOR_RE = re.compile(r"^// L(\d+)-(\d+)$")


def _source(n: int = 300) -> str:
    return "\n".join(f"def helper_{i}(arg):\n    return arg + {i}\n" for i in range(n))


def _anchored_blocks(summary: str) -> list[tuple[int, int, str]]:
    """Split a summary into (start, end, body) triples, one per anchor."""
    blocks: list[tuple[int, int, str]] = []
    current: tuple[int, int] | None = None
    body: list[str] = []
    for line in summary.splitlines():
        match = _ANCHOR_RE.match(line)
        if match:
            if current is not None:
                blocks.append((current[0], current[1], "\n".join(body)))
            current = (int(match.group(1)), int(match.group(2)))
            body = []
        elif current is not None:
            body.append(line)
    if current is not None:
        blocks.append((current[0], current[1], "\n".join(body)))
    return blocks


class TestAnchorsAreEmitted:
    def test_summary_carries_at_least_one_anchor(self) -> None:
        content = _source()
        summary = summarize_semantic(content, max_size=2000)

        assert _anchored_blocks(summary), f"no line anchors in summary: {summary[:200]!r}"

    def test_anchors_are_on_by_default(self) -> None:
        assert DEFAULT_SUMMARIZATION_CONFIG.include_line_anchors is True


class TestAnchorsAreTruthful:
    def test_each_anchor_names_the_lines_that_follow_it(self) -> None:
        content = _source()
        lines = content.splitlines()
        summary = summarize_semantic(content, max_size=3000)

        blocks = _anchored_blocks(summary)
        assert blocks

        for start, end, body in blocks:
            assert 1 <= start <= end <= len(lines), f"anchor L{start}-{end} out of range"
            expected = "\n".join(lines[start - 1 : end])
            # The final block can be cut by the max_size trim, so compare the
            # part that survived rather than requiring an exact match.
            assert expected.startswith(body.rstrip("\n")) or body.rstrip("\n") in expected, (
                f"anchor L{start}-{end} does not describe the text under it"
            )

    def test_anchors_are_in_increasing_line_order(self) -> None:
        summary = summarize_semantic(_source(), max_size=3000)
        starts = [start for start, _, _ in _anchored_blocks(summary)]
        assert starts == sorted(starts)


class TestAnchorsCanBeDisabled:
    def test_disabled_config_emits_none(self) -> None:
        config = SummarizationConfig(include_line_anchors=False)
        summary = summarize_semantic(_source(), max_size=2000, config=config)

        assert not _anchored_blocks(summary)


class TestBudgetIsStillRespected:
    @pytest.mark.parametrize("max_size", [600, 2000, 8000])
    def test_anchors_never_push_the_summary_over_max_size(self, max_size: int) -> None:
        summary = summarize_semantic(_source(), max_size=max_size)
        assert len(summary) <= max_size


class TestDegenerateInput:
    def test_empty_content_does_not_raise(self) -> None:
        assert summarize_semantic("", max_size=100) == ""

    def test_content_under_budget_is_returned_unchanged(self) -> None:
        content = "def f():\n    return 1\n"
        assert summarize_semantic(content, max_size=10_000) == content
