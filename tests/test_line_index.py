"""The line-start index must agree exactly with the splitlines it replaced.

``build_line_starts`` exists because rebuilding a line index per edit made
``batch_edit`` quadratic in (edits x file size). It is a pure-arithmetic
substitution on the hot path of every line-addressed operation, so its whole
value depends on producing byte-identical answers to the code it replaced —
which is what these tests pin, exhaustively over short inputs and randomly
over longer ones.
"""

from __future__ import annotations

import itertools
import random

import pytest

from semantic_cache_mcp.cache._helpers import (
    _extract_line_range,
    _find_match_line_numbers,
    build_line_starts,
)

# Newline-dense alphabet: the edge cases all live around \n placement
# (leading, trailing, doubled, absent), so short exhaustive strings over
# this alphabet cover them far better than long realistic ones.
_ALPHABET = "a\nb"
_EXHAUSTIVE_MAX_LEN = 6
_NEEDLES = ("a", "b", "\n", "a\n", "\nb", "ab", "\n\n")


def _reference_line_starts(content: str) -> list[int]:
    """The original implementation: materialize lines, accumulate lengths."""
    starts: list[int] = []
    position = 0
    for line in content.splitlines(keepends=True):
        starts.append(position)
        position += len(line)
    return starts


def _reference_extract(content: str, start_line: int, end_line: int) -> tuple[str, int, int]:
    """The original implementation of ``_extract_line_range``."""
    lines = content.splitlines(keepends=True)
    total = len(lines)
    if total == 0:
        raise ValueError("Cannot extract line range from empty file")
    if start_line < 1 or end_line < start_line or start_line > total or end_line > total:
        raise ValueError("out of bounds")
    char_start = sum(len(lines[i]) for i in range(start_line - 1))
    char_end = char_start + sum(len(lines[i]) for i in range(start_line - 1, end_line))
    return content[char_start:char_end], char_start, char_end


def _exhaustive_inputs() -> list[str]:
    out: list[str] = []
    for length in range(_EXHAUSTIVE_MAX_LEN + 1):
        out.extend("".join(t) for t in itertools.product(_ALPHABET, repeat=length))
    return out


def _random_inputs(count: int = 200) -> list[str]:
    rng = random.Random(20260727)
    return [
        "".join(rng.choice("abc \n\n\t") for _ in range(rng.randint(0, 300))) for _ in range(count)
    ]


class TestBuildLineStarts:
    @pytest.mark.parametrize("content", _exhaustive_inputs())
    def test_matches_reference_exhaustively(self, content: str) -> None:
        assert build_line_starts(content) == _reference_line_starts(content)

    @pytest.mark.parametrize("content", _random_inputs())
    def test_matches_reference_on_random_text(self, content: str) -> None:
        assert build_line_starts(content) == _reference_line_starts(content)

    def test_counts_lines_like_splitlines(self) -> None:
        for content in ("", "a", "\n", "a\n", "a\nb", "a\nb\n", "\n\n\n"):
            assert len(build_line_starts(content)) == len(content.splitlines(keepends=True))


class TestFindMatchLineNumbers:
    @pytest.mark.parametrize("content", _exhaustive_inputs())
    def test_matches_reference_for_every_needle(self, content: str) -> None:
        for needle in _NEEDLES:
            assert _find_match_line_numbers(content, needle) == _find_match_line_numbers(
                content, needle, line_starts=build_line_starts(content)
            )

    def test_precomputed_index_gives_identical_answers(self) -> None:
        content = "alpha\nbravo\nalpha\ncharlie\nalpha\n"
        index = build_line_starts(content)
        assert _find_match_line_numbers(content, "alpha") == [1, 3, 5]
        assert _find_match_line_numbers(content, "alpha", line_starts=index) == [1, 3, 5]

    def test_absent_needle_yields_nothing(self) -> None:
        assert _find_match_line_numbers("a\nb\n", "zzz") == []

    def test_empty_content_yields_nothing(self) -> None:
        assert _find_match_line_numbers("", "a") == []


class TestExtractLineRange:
    @pytest.mark.parametrize("content", _exhaustive_inputs())
    def test_every_valid_range_matches_reference(self, content: str) -> None:
        total = len(content.splitlines(keepends=True))
        for start in range(1, total + 1):
            for end in range(start, total + 1):
                assert _extract_line_range(content, start, end) == _reference_extract(
                    content, start, end
                )

    @pytest.mark.parametrize("content", _random_inputs(60))
    def test_random_ranges_match_reference(self, content: str) -> None:
        total = len(content.splitlines(keepends=True))
        if total == 0:
            return
        rng = random.Random(len(content))
        for _ in range(8):
            start = rng.randint(1, total)
            end = rng.randint(start, total)
            assert _extract_line_range(content, start, end) == _reference_extract(
                content, start, end
            )

    def test_offsets_splice_back_to_the_original(self) -> None:
        content = "one\ntwo\nthree\nfour\n"
        substring, char_start, char_end = _extract_line_range(content, 2, 3)
        assert substring == "two\nthree\n"
        assert content[:char_start] + substring + content[char_end:] == content

    def test_precomputed_index_gives_identical_answers(self) -> None:
        content = "one\ntwo\nthree\n"
        index = build_line_starts(content)
        assert _extract_line_range(content, 2, 3, line_starts=index) == _extract_line_range(
            content, 2, 3
        )

    @pytest.mark.parametrize(
        ("content", "start", "end"),
        [
            ("", 1, 1),
            ("a\nb\n", 0, 1),
            ("a\nb\n", 2, 1),
            ("a\nb\n", 3, 3),
            ("a\nb\n", 1, 9),
        ],
    )
    def test_rejects_what_the_reference_rejected(self, content: str, start: int, end: int) -> None:
        with pytest.raises(ValueError):
            _extract_line_range(content, start, end)
        with pytest.raises(ValueError):
            _reference_extract(content, start, end)
