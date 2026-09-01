"""Structural outline extraction: signatures and their line numbers.

The outline exists so a large file can be located without being delivered.
Every entry must therefore point at a line that really holds that text —
a wrong line number sends the reader to the wrong place and costs more than
the outline saved.
"""

from __future__ import annotations

import pytest

from semantic_cache_mcp.core.text import Outline, extract_outline, render_outline

PY_SOURCE = '''"""Module docstring."""

import os


CONSTANT = 1


class Widget:
    """A widget."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def render(
        self,
        width: int,
    ) -> str:
        return self.name


def helper(a, b):
    return a + b
'''


def _line_of(source: str, text: str) -> int:
    """1-based line number of the first line containing *text*."""
    for i, line in enumerate(source.splitlines(), start=1):
        if text in line:
            return i
    raise AssertionError(f"{text!r} not in source")


class TestLineNumbersAreTruthful:
    def test_every_entry_line_holds_its_text(self) -> None:
        outline = extract_outline(PY_SOURCE, filename="mod.py")
        lines = PY_SOURCE.splitlines()

        assert outline.entries, "no symbols found in an obviously symbolic file"
        for entry in outline.entries:
            assert 1 <= entry.line <= len(lines)
            source_line = lines[entry.line - 1]
            # The rendered text may join a wrapped signature, so compare the
            # first token rather than the whole string.
            first_token = entry.text.strip().split("(")[0].strip()
            assert first_token in source_line, (
                f"entry {entry.text!r} claims line {entry.line}, which holds {source_line!r}"
            )

    def test_finds_classes_and_functions_at_the_right_lines(self) -> None:
        outline = extract_outline(PY_SOURCE, filename="mod.py")
        by_line = {e.line: e.text.strip() for e in outline.entries}

        assert by_line[_line_of(PY_SOURCE, "class Widget")].startswith("class Widget")
        assert by_line[_line_of(PY_SOURCE, "def helper")].startswith("def helper")
        assert _line_of(PY_SOURCE, "def __init__") in by_line

    def test_total_lines_matches_the_file(self) -> None:
        outline = extract_outline(PY_SOURCE, filename="mod.py")
        assert outline.total_lines == len(PY_SOURCE.splitlines())


class TestWrappedSignatures:
    def test_multiline_signature_is_joined_and_closed(self) -> None:
        outline = extract_outline(PY_SOURCE, filename="mod.py")
        render = next(e for e in outline.entries if "render" in e.text)

        assert render.line == _line_of(PY_SOURCE, "async def render")
        assert "width" in render.text, "wrapped parameters were dropped"
        assert render.text.count("\n") == 0, "an entry must stay on one line"

    def test_runaway_signature_stops_and_says_so(self) -> None:
        source = "def broken(\n" + "".join(f"    arg{i},\n" for i in range(50))
        outline = extract_outline(source, filename="mod.py")

        entry = outline.entries[0]
        assert entry.line == 1
        assert entry.text.endswith("...)"), f"unterminated signature not marked: {entry.text!r}"


class TestNesting:
    def test_depth_reflects_indentation(self) -> None:
        outline = extract_outline(PY_SOURCE, filename="mod.py")
        by_text = {e.text.strip().split("(")[0].strip().rstrip(":"): e for e in outline.entries}

        assert by_text["class Widget"].depth == 0
        assert by_text["def __init__"].depth > 0


class TestOtherLanguages:
    @pytest.mark.parametrize(
        ("filename", "source", "expected"),
        [
            ("a.ts", "export function go(x: number) {\n  return x\n}\n", "function go"),
            ("a.ts", "export interface Shape {\n  kind: string\n}\n", "interface Shape"),
            ("a.go", "func Handle(w http.ResponseWriter) {\n}\n", "func Handle"),
            ("a.go", "type Server struct {\n}\n", "type Server"),
            ("a.rs", "pub fn run(x: u32) -> u32 {\n    x\n}\n", "fn run"),
            ("a.rs", "impl Widget {\n}\n", "impl Widget"),
            ("a.sh", "deploy() {\n  echo hi\n}\n", "deploy"),
            ("a.md", "# Title\n\nbody\n", "# Title"),
        ],
    )
    def test_recognizes_common_definition_forms(
        self, filename: str, source: str, expected: str
    ) -> None:
        outline = extract_outline(source, filename=filename)
        assert any(expected in e.text for e in outline.entries), (
            f"{filename}: {expected!r} missing from {[e.text for e in outline.entries]}"
        )


class TestCommentsAreNotSymbols:
    def test_python_comment_mentioning_def_is_ignored(self) -> None:
        source = "# def ghost():\n#     pass\ndef real():\n    pass\n"
        outline = extract_outline(source, filename="mod.py")

        assert [e.text.strip() for e in outline.entries] == ["def real():"]

    def test_c_style_comment_is_ignored(self) -> None:
        source = "// function ghost() {}\nfunction real() {}\n"
        outline = extract_outline(source, filename="a.js")

        assert all("ghost" not in e.text for e in outline.entries)

    def test_hash_heading_is_a_symbol_only_in_markdown(self) -> None:
        assert extract_outline("# Title\n", filename="notes.md").entries
        assert not extract_outline("# Title\n", filename="mod.py").entries


class TestDegenerateInput:
    @pytest.mark.parametrize("source", ["", "\n", "   \n\t\n", "no symbols here at all\n"])
    def test_symbol_free_input_returns_an_empty_outline_not_an_error(self, source: str) -> None:
        outline = extract_outline(source, filename="x.txt")

        assert isinstance(outline, Outline)
        assert outline.entries == ()
        assert outline.truncated is False
        assert render_outline(outline) == ""

    def test_content_must_be_a_string(self) -> None:
        with pytest.raises(TypeError):
            extract_outline(b"def x(): pass", filename="mod.py")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_non_positive_max_entries_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError):
            extract_outline(PY_SOURCE, filename="mod.py", max_entries=bad)

    @pytest.mark.parametrize("bad", [0, -5])
    def test_non_positive_max_tokens_is_rejected(self, bad: int) -> None:
        with pytest.raises(ValueError):
            extract_outline(PY_SOURCE, filename="mod.py", max_tokens=bad)

    def test_no_filename_still_works(self) -> None:
        outline = extract_outline(PY_SOURCE)
        assert any("class Widget" in e.text for e in outline.entries)


class TestBudget:
    def test_over_max_entries_drops_deepest_first_and_reports_it(self) -> None:
        source = "".join(f"class C{i}:\n    def m{i}(self):\n        pass\n" for i in range(20))
        outline = extract_outline(source, filename="mod.py", max_entries=20)

        assert len(outline.entries) == 20
        assert outline.truncated is True
        assert outline.dropped == 20
        # Top-level classes survive; nested methods are the ones dropped.
        assert all(e.depth == 0 for e in outline.entries)

    def test_token_budget_is_honoured_with_an_injected_counter(self) -> None:
        source = "".join(f"def function_number_{i}():\n    pass\n" for i in range(200))
        counted = render_outline(extract_outline(source, filename="mod.py"))
        full_tokens = len(counted.split())

        budget = full_tokens // 4
        outline = extract_outline(
            source,
            filename="mod.py",
            max_tokens=budget,
            count_fn=lambda text: len(text.split()),
        )

        assert outline.truncated is True
        assert len(render_outline(outline).split()) <= budget

    def test_budget_never_returns_more_than_it_was_asked_for(self) -> None:
        source = "".join(f"def f{i}():\n    pass\n" for i in range(100))
        outline = extract_outline(source, filename="mod.py", max_entries=5)
        assert len(outline.entries) == 5


class TestRendering:
    def test_render_puts_the_line_number_first(self) -> None:
        outline = extract_outline(PY_SOURCE, filename="mod.py")
        text = render_outline(outline)

        first = text.splitlines()[0]
        line_no, _, rest = first.partition(":")
        assert line_no.strip().isdigit()
        assert rest.strip()

    def test_render_preserves_entry_order_by_line(self) -> None:
        outline = extract_outline(PY_SOURCE, filename="mod.py")
        numbers = [int(line.split(":", 1)[0]) for line in render_outline(outline).splitlines()]
        assert numbers == sorted(numbers)

    def test_render_marks_a_truncated_outline(self) -> None:
        source = "".join(f"def f{i}():\n    pass\n" for i in range(100))
        outline = extract_outline(source, filename="mod.py", max_entries=5)
        assert "omitted" in render_outline(outline)


class TestPathological:
    def test_very_long_single_line_does_not_hang(self) -> None:
        source = "def f(" + "a" * 200_000 + "):\n    pass\n"
        outline = extract_outline(source, filename="mod.py")
        assert outline.entries[0].line == 1

    def test_deeply_nested_parens_are_bounded(self) -> None:
        source = "def f(" + "(" * 5000 + "\n" * 10
        outline = extract_outline(source, filename="mod.py")
        assert outline.entries[0].text.endswith("...)")
