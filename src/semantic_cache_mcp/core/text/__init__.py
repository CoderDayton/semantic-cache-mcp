"""Text sub-package: diff, delta, truncation, and semantic summarization."""

from __future__ import annotations

from ._diff import (
    DiffDelta,
    apply_delta,
    compute_delta,
    diff_stats,
    generate_diff,
    truncate_semantic,
    truncate_smart,
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
    "diff_stats",
    "compute_delta",
    "apply_delta",
    "truncate_smart",
    "truncate_semantic",
    # summarization
    "SummarizationConfig",
    "DEFAULT_SUMMARIZATION_CONFIG",
    "Segment",
    "extract_segments",
    "score_segments",
    "summarize_semantic",
    "truncate_with_summarization",
]
