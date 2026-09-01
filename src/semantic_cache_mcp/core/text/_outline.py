"""Structural outline: a file's definition lines and where they live.

The point is to answer "what is in this file and where" without delivering the
file. A summary selects whole segments and still costs thousands of tokens; an
outline costs a line per symbol and hands back the line numbers a ranged read
needs.

This is a heuristic scan, not a parse. It reads one line at a time against a
set of anchored patterns covering the definition forms of the languages this
server sees most. It does not track string or comment state beyond skipping
lines that *begin* as comments, so a definition quoted inside a docstring can
appear. That is the deliberate trade: a wrong entry costs one ranged read,
while a real parser costs a dependency per language. Nothing downstream may
treat an outline as proof of the file's contents — it is a map, and a map is
never redeemable as possession.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# A wrapped signature is joined across at most this many following lines. Past
# that the definition is reported open (`...)`) rather than swallowing the body.
_MAX_JOIN_LINES = 4

# Hard cap on one rendered signature. A minified bundle or a generated file can
# put a hundred kilobytes on one line; the outline must stay an index.
_MAX_ENTRY_CHARS = 400

# Cap on lines scanned. Bounds the walk over a pathological input rather than
# trusting the caller to have bounded it.
_MAX_SCAN_LINES = 500_000

_DEFAULT_MAX_ENTRIES = 2000

_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".mdx"})

# Line prefixes that open a comment in the languages handled here. `#` is
# deliberately absent: it opens a comment in Python/shell/Ruby but a heading in
# Markdown, so it is decided per filename in `_is_comment`.
_COMMENT_PREFIXES = ("//", "/*", "*/", "--", "<!--", ";;")

# One anchored pattern per definition form. Every quantifier is bounded: these
# run against attacker-shaped input (any file the caller names), and an
# unbounded repeat inside a group is how a scan becomes a hang.
_PATTERNS: tuple[str, ...] = (
    # Python / Ruby / generic
    r"(?:async\s+)?def\s+\w+",
    r"class\s+\w+",
    r"module\s+\w+",
    # JavaScript / TypeScript
    r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\b",
    r"(?:export\s+)?(?:declare\s+)?(?:abstract\s+)?(?:interface|enum|namespace)\s+\w+",
    r"(?:export\s+)?type\s+\w+\s*(?:<[^>]{0,80}>)?\s*=",
    r"(?:export\s+)?(?:const|let|var)\s+\w+\s*(?::[^=]{0,80})?=\s*"
    r"(?:async\s+)?(?:function\b|\([^)]{0,120}\)\s*(?::[^=]{0,60})?=>|\w+\s*=>)",
    # Go
    r"func\s+(?:\([^)]{0,120}\)\s*)?\w+",
    r"type\s+\w+\s+(?:struct|interface)\b",
    # Rust
    r"(?:pub(?:\([^)]{0,40}\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern\s+\"[^\"]{0,20}\"\s+)?fn\s+\w+",
    r"(?:pub(?:\([^)]{0,40}\))?\s+)?(?:struct|enum|trait|impl|mod)\s+",
    # Java / C# / C++ member declarations
    r"(?:public|private|protected|internal)\s+[\w<>\[\],.\s]{0,80}\w+\s*\(",
    # Shell function
    r"\w+\s*\(\s*\)\s*\{",
    # Clojure / Elixir
    r"(?:defn?|defmodule|defmacro|defp)\s+\S",
)

_SIGNATURE_RE = re.compile("|".join(f"(?:{p})" for p in _PATTERNS))
_MARKDOWN_HEADING_RE = re.compile(r"#{1,6}\s+\S")


@dataclass(frozen=True, slots=True)
class OutlineEntry:
    """One definition and the line it starts on.

    ``line`` is 1-based and always points at the line the definition *opens*
    on, even when ``text`` joins a signature wrapped over several lines.
    """

    line: int
    text: str
    depth: int


@dataclass(frozen=True, slots=True)
class Outline:
    """Every definition found, in file order, plus what was left out."""

    entries: tuple[OutlineEntry, ...]
    total_lines: int
    truncated: bool = False
    dropped: int = 0


def _is_markdown(filename: str | None) -> bool:
    if not filename:
        return False
    lowered = filename.lower()
    return any(lowered.endswith(suffix) for suffix in _MARKDOWN_SUFFIXES)


def _is_comment(stripped: str, *, markdown: bool) -> bool:
    if stripped.startswith(_COMMENT_PREFIXES):
        return True
    # In Markdown `#` opens a heading, which is the one structure that file
    # type has. Everywhere else it opens a comment, and a commented-out
    # definition is not a definition.
    return stripped.startswith("#") and not markdown


def _indent_columns(line: str) -> int:
    """Leading whitespace width, counting a tab as four columns."""
    width = 0
    for char in line:
        if char == " ":
            width += 1
        elif char == "\t":
            width += 4
        else:
            break
    return width


def _paren_balance(text: str) -> int:
    return text.count("(") - text.count(")")


def _build_entry(lines: list[str], index: int) -> OutlineEntry:
    """Assemble the entry opening at ``lines[index]``, joining a wrapped signature."""
    first = lines[index].rstrip()
    parts = [first]
    balance = _paren_balance(first)

    joined = 0
    while balance > 0 and joined < _MAX_JOIN_LINES and index + joined + 1 < len(lines):
        joined += 1
        follow = lines[index + joined].strip()
        parts.append(follow)
        balance += _paren_balance(follow)

    text = " ".join(part for part in parts if part)
    over_long = len(text) > _MAX_ENTRY_CHARS
    if over_long:
        text = text[:_MAX_ENTRY_CHARS].rstrip()

    if balance > 0:
        # The signature never closed inside the join window. Say so in the text
        # rather than presenting a fragment as if it were the whole thing.
        text += "...)"
    elif over_long:
        text += "..."

    return OutlineEntry(line=index + 1, text=text, depth=_indent_columns(first))


def _scan(content: str, *, markdown: bool) -> tuple[list[OutlineEntry], int]:
    lines = content.splitlines()
    total_lines = len(lines)
    entries: list[OutlineEntry] = []

    for index, line in enumerate(lines[:_MAX_SCAN_LINES]):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_comment(stripped, markdown=markdown):
            continue
        if markdown and _MARKDOWN_HEADING_RE.match(stripped):
            entries.append(OutlineEntry(line=index + 1, text=stripped, depth=0))
            continue
        if _SIGNATURE_RE.match(stripped):
            entries.append(_build_entry(lines, index))

    return entries, total_lines


def _drop_order(entries: list[OutlineEntry]) -> list[int]:
    """Indices in the order they should be sacrificed to fit a budget.

    Deepest first: a nested helper is the least useful line in a map of a file,
    and dropping it leaves the enclosing class or module still findable. Ties
    break on position so the tail goes before the head — the top of a file
    carries the imports and the primary definition.
    """
    return sorted(range(len(entries)), key=lambda i: (-entries[i].depth, -entries[i].line))


def _render_lines(entries: tuple[OutlineEntry, ...] | list[OutlineEntry]) -> list[str]:
    return [f"{entry.line}: {entry.text}" for entry in entries]


def _omitted_marker(dropped: int) -> str:
    return f"// {dropped} more symbols omitted"


def _fit_token_budget(
    entries: list[OutlineEntry],
    max_tokens: int,
    count_fn: Callable[[str], int],
    already_dropped: int,
) -> tuple[list[OutlineEntry], int]:
    """Drop entries until the rendered outline fits ``max_tokens``.

    The omitted-count marker is charged against the budget too: it is part of
    what the caller receives, and leaving it out is how a "fits" answer becomes
    a lie by exactly the size of the thing that says the answer is incomplete.
    """
    costs = [count_fn(text) for text in _render_lines(entries)]
    dropped = already_dropped
    kept = set(range(len(entries)))
    total = sum(costs)

    def _budget() -> int:
        return max_tokens - (count_fn(_omitted_marker(dropped)) if dropped else 0)

    if total <= _budget():
        return entries, dropped

    for index in _drop_order(entries):
        if not kept:
            break
        kept.discard(index)
        total -= costs[index]
        dropped += 1
        if total <= _budget():
            break

    return [entry for i, entry in enumerate(entries) if i in kept], dropped


def extract_outline(
    content: str,
    *,
    filename: str | None = None,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    max_tokens: int | None = None,
    count_fn: Callable[[str], int] | None = None,
) -> Outline:
    """Map ``content``'s definitions to their line numbers.

    Args:
        content: The file text. Must already be decoded.
        filename: Used only to decide whether `#` opens a heading or a comment.
        max_entries: Hard cap on entries returned. Excess is dropped deepest
            first and counted in ``Outline.dropped``.
        max_tokens: Optional budget for the *rendered* outline.
        count_fn: Token counter for ``max_tokens``. Defaults to a conservative
            four-characters-per-token estimate.

    Raises:
        TypeError: ``content`` is not a string.
        ValueError: ``max_entries`` or ``max_tokens`` is not positive.
    """
    if not isinstance(content, str):
        raise TypeError(f"content must be str, got {type(content).__name__}")
    if max_entries <= 0:
        raise ValueError(f"max_entries must be > 0, got {max_entries}")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError(f"max_tokens must be > 0, got {max_tokens}")

    entries, total_lines = _scan(content, markdown=_is_markdown(filename))
    dropped = 0

    if len(entries) > max_entries:
        keep = set(_drop_order(entries)[len(entries) - max_entries :])
        dropped = len(entries) - max_entries
        entries = [entry for i, entry in enumerate(entries) if i in keep]

    if max_tokens is not None:
        counter = count_fn or (lambda text: max(1, len(text) // 4))
        entries, dropped = _fit_token_budget(entries, max_tokens, counter, dropped)

    return Outline(
        entries=tuple(sorted(entries, key=lambda e: e.line)),
        total_lines=total_lines,
        truncated=dropped > 0,
        dropped=dropped,
    )


def render_outline(outline: Outline) -> str:
    """Render an outline as `line: signature`, one symbol per line.

    An empty, untruncated outline renders empty — the caller must be able to
    tell "this file has no definitions" from "some were left out", so a
    truncated outline always carries its omitted count even with no entries.
    """
    lines = _render_lines(outline.entries)
    if outline.truncated:
        lines.append(_omitted_marker(outline.dropped))
    return "\n".join(lines)


__all__ = ["Outline", "OutlineEntry", "extract_outline", "render_outline"]
