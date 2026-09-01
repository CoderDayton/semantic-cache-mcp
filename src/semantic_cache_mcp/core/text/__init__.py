from __future__ import annotations

from ._diff import (
    DiffDelta,
    compute_delta,
    diff_stats,
    diff_with_stats,
    generate_diff,
    rebase_diff_hunks,
    truncate_semantic,
    truncate_smart,
)
from ._outline import (
    Outline,
    OutlineEntry,
    extract_outline,
    render_outline,
)
from ._summarize import (
    DEFAULT_SUMMARIZATION_CONFIG,
    Segment,
    SummarizationConfig,
    extract_segments,
    score_segments,
    summarize_semantic,
    truncate_with_summarization,
)

__all__ = [
    # diff / delta / truncation
    "DiffDelta",
    "generate_diff",
    "rebase_diff_hunks",
    "diff_stats",
    "diff_with_stats",
    "compute_delta",
    "truncate_smart",
    "truncate_semantic",
    # structural outline
    "Outline",
    "OutlineEntry",
    "extract_outline",
    "render_outline",
    # summarization
    "SummarizationConfig",
    "DEFAULT_SUMMARIZATION_CONFIG",
    "Segment",
    "extract_segments",
    "score_segments",
    "summarize_semantic",
    "truncate_with_summarization",
]
