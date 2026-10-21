"""Unified diffs and diff statistics."""

from __future__ import annotations

import re
from difflib import SequenceMatcher, unified_diff

_NO_NEWLINE_MARK = "\\ No newline at end of file\n"


def _terminate(line: str) -> str:
    """Return a diff line ending in a line break, adding the marker if it had none."""
    if line.splitlines()[0] == line:
        return line + "\n" + _NO_NEWLINE_MARK
    return line


def generate_diff(old: str, new: str, context_lines: int = 3) -> str:
    """Return unified diff string, or '// No changes' if identical."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    diff = list(unified_diff(old_lines, new_lines, n=context_lines))
    # Drop difflib's boilerplate file headers ("--- \n", "+++ \n"): the path is
    # known from the response envelope and the @@ hunk headers carry the line
    # numbers, so the headers are pure token overhead on every diff.
    if len(diff) >= 2 and diff[0].startswith("--- ") and diff[1].startswith("+++ "):
        diff = diff[2:]
    result = "".join(_terminate(line) for line in diff)
    return result if result else "// No changes"


# Unified-diff hunk header: "@@ -<start>[,<len>] +<start>[,<len>] @@".
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(,\d+)? \+(\d+)(,\d+)? @@", re.MULTILINE)


def rebase_diff_hunks(diff: str, line_offset: int) -> str:
    """Shift a diff's hunk line numbers by *line_offset*.

    A diff taken over a slice of a file numbers its hunks from the slice's own
    first line, so a reader thinking in file coordinates misreads every one of
    them. Rebasing is a uniform shift of both sides' starting line; the hunk
    lengths describe the hunk itself and are unaffected. The shift stays
    correct for difflib's zero-length ranges, which decrement the start before
    formatting — moving the origin moves that result by the same amount.
    """
    if not line_offset:
        return diff

    def _shift(match: re.Match[str]) -> str:
        old_start = int(match.group(1)) + line_offset
        new_start = int(match.group(3)) + line_offset
        return f"@@ -{old_start}{match.group(2) or ''} +{new_start}{match.group(4) or ''} @@"

    return _HUNK_HEADER_RE.sub(_shift, diff)


def _format_range_unified(start: int, stop: int) -> str:
    """Format a unified-diff hunk range (mirrors difflib's private helper)."""
    beginning = start + 1
    length = stop - start
    if length == 1:
        return str(beginning)
    if not length:
        beginning -= 1
    return f"{beginning},{length}"


_DELTA_HEADER_BYTES = 32


def diff_with_stats(old: str, new: str, context_lines: int = 3) -> tuple[str, dict]:
    """Return ``generate_diff()``'s text and change statistics from one matcher pass.

    The diff text is identical to ``generate_diff(old, new, context_lines)``.
    Stats count changed lines: an equal-length replace is ``modifications``,
    any other change is ``insertions``/``deletions``. ``delta_size_bytes`` is
    the changed text's size plus a fixed header; ``compression_ratio`` is that
    over ``original_size``, or 0 when ``old`` is empty.
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    sm = SequenceMatcher(None, old_lines, new_lines)

    insertions = deletions = modifications = 0
    content_bytes = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            if i2 - i1 == j2 - j1:
                modifications += i2 - i1
                content_bytes += sum(len(s.rstrip()) for s in old_lines[i1:i2])
                content_bytes += sum(len(s.rstrip()) for s in new_lines[j1:j2])
            else:
                deletions += i2 - i1
                insertions += j2 - j1
                content_bytes += sum(len(s) for s in old_lines[i1:i2])
                content_bytes += sum(len(s) for s in new_lines[j1:j2])
        elif tag == "delete":
            deletions += i2 - i1
            content_bytes += sum(len(s) for s in old_lines[i1:i2])
        elif tag == "insert":
            insertions += j2 - j1
            content_bytes += sum(len(s) for s in new_lines[j1:j2])

    size_bytes = _DELTA_HEADER_BYTES + content_bytes
    original_size = len(old.encode()) if old else 0
    stats = {
        "insertions": insertions,
        "deletions": deletions,
        "modifications": modifications,
        "delta_size_bytes": size_bytes,
        "original_size": original_size,
        "compression_ratio": size_bytes / original_size if original_size else 0,
    }

    # Unified diff from the same matcher (opcodes are memoized by difflib).
    # Formatting mirrors difflib.unified_diff minus the ---/+++ file headers
    # that generate_diff strips anyway.
    parts: list[str] = []
    for group in sm.get_grouped_opcodes(context_lines):
        first, last = group[0], group[-1]
        range1 = _format_range_unified(first[1], last[2])
        range2 = _format_range_unified(first[3], last[4])
        parts.append(f"@@ -{range1} +{range2} @@\n")
        for gtag, gi1, gi2, gj1, gj2 in group:
            if gtag == "equal":
                parts.extend(_terminate(" " + line) for line in old_lines[gi1:gi2])
                continue
            if gtag in ("replace", "delete"):
                parts.extend(_terminate("-" + line) for line in old_lines[gi1:gi2])
            if gtag in ("replace", "insert"):
                parts.extend(_terminate("+" + line) for line in new_lines[gj1:gj2])
    diff_text = "".join(parts)

    return (diff_text if diff_text else "// No changes", stats)
