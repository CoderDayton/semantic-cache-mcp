"""Text processing utilities for diffs and truncation."""

from __future__ import annotations

from difflib import unified_diff


def generate_diff(old: str, new: str, context_lines: int = 3) -> str:
    """Generate unified diff between two texts.

    Args:
        old: Original text
        new: Updated text
        context_lines: Lines of context around changes

    Returns:
        Unified diff string or "// No changes"
    """
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    diff = unified_diff(
        old_lines, new_lines, fromfile="cached", tofile="current", n=context_lines
    )

    result = "".join(diff)
    return result if result else "// No changes"


def truncate_smart(
    content: str, max_size: int, keep_top: int = 80, keep_bottom: int = 40
) -> str:
    """Smart truncation preserving file structure.

    Keeps the beginning and end of the file, which typically contain
    the most important information (imports, exports, class definitions).

    Args:
        content: Text to truncate
        max_size: Maximum output size in characters
        keep_top: Lines to keep from start
        keep_bottom: Lines to keep from end

    Returns:
        Truncated content with indicator
    """
    if len(content) <= max_size:
        return content

    lines = content.splitlines(keepends=True)
    n_lines = len(lines)

    if n_lines <= keep_top + keep_bottom:
        return content[: max_size - 20] + "\n// [TRUNCATED]"

    top_content = "".join(lines[:keep_top])
    bottom_content = "".join(lines[-keep_bottom:]) if keep_bottom > 0 else ""
    truncation_msg = f"\n\n// ... [{n_lines - keep_top - keep_bottom} lines truncated] ...\n\n"

    total = len(top_content) + len(truncation_msg) + len(bottom_content)
    if total > max_size:
        return content[: max_size - 20] + "\n// [TRUNCATED]"

    return f"{top_content}{truncation_msg}{bottom_content}"
