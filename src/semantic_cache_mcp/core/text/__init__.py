from __future__ import annotations

from ._diff import (
    diff_with_stats,
    generate_diff,
    rebase_diff_hunks,
)
from ._outline import (
    Outline,
    OutlineEntry,
    extract_outline,
    large_file_map,
    render_outline,
)

__all__ = [
    "generate_diff",
    "rebase_diff_hunks",
    "diff_with_stats",
    # structural outline
    "Outline",
    "OutlineEntry",
    "extract_outline",
    "large_file_map",
    "render_outline",
]
