"""
Semantic summarization for large files exceeding context limits.

Based on TCRA-LLM approach (arXiv:2310.15556): preserve semantically important
content instead of simple positional truncation.

Algorithm:
1. Segment content into semantic chunks (functions, paragraphs, etc.)
2. Embed each segment using lightweight model
3. Score segments by: position, information density, diversity
4. Select highest-scoring segments fitting within size limit
5. Reassemble in original order with markers

This preserves the "skeleton" of the file - the most informative parts.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SummarizationConfig:
    """Configuration for semantic summarization."""

    # Segment size bounds
    min_segment_lines: int = 3
    max_segment_lines: int = 50

    # Scoring weights
    position_weight: float = 0.3  # Favor start/end of file
    density_weight: float = 0.4  # Favor information-dense segments
    diversity_weight: float = 0.3  # Penalize redundancy

    # Output
    include_markers: bool = True  # Include "[X lines omitted]" markers
    preserve_structure: bool = True  # Keep class/function headers


DEFAULT_SUMMARIZATION_CONFIG = SummarizationConfig()


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------


# Patterns that indicate segment boundaries (language-agnostic)
_BOUNDARY_PATTERNS = [
    r"^\s*class\s+\w+",  # Class definition
    r"^\s*def\s+\w+",  # Python function
    r"^\s*(async\s+)?function\s+\w+",  # JS/TS function
    r"^\s*func\s+\w+",  # Go function
    r"^\s*fn\s+\w+",  # Rust function
    r"^\s*pub\s+(fn|struct|enum)\s+",  # Rust public items
    r"^\s*export\s+(class|function|const|interface)",  # TS exports
    r"^#{1,3}\s+",  # Markdown headers
    r"^\s*##\s+",  # Alternative header
    r"^---+$",  # Horizontal rule
    r"^\s*@\w+",  # Decorators
]

_BOUNDARY_REGEX = re.compile("|".join(f"({p})" for p in _BOUNDARY_PATTERNS), re.MULTILINE)


@dataclass
class Segment:
    """A semantic segment of the file."""

    start_line: int
    end_line: int
    content: str
    is_header: bool = False  # Is this a class/function definition header?


def extract_segments(
    content: str,
    config: SummarizationConfig = DEFAULT_SUMMARIZATION_CONFIG,
) -> list[Segment]:
    """
    Extract semantic segments from content.

    Tries to split at natural boundaries (function/class definitions, headers).
    Falls back to paragraph/line-based splitting if no boundaries found.

    Args:
        content: File content
        config: Segmentation configuration

    Returns:
        List of Segment objects
    """
    lines = content.splitlines(keepends=True)
    n_lines = len(lines)

    if n_lines == 0:
        return []

    # Find boundary lines
    boundary_lines = [0]  # Always start at line 0

    for i, line in enumerate(lines):
        if _BOUNDARY_REGEX.match(line) and i not in boundary_lines:
            boundary_lines.append(i)

    boundary_lines.append(n_lines)  # End marker
    boundary_lines = sorted(set(boundary_lines))

    # Create segments from boundaries
    segments = []
    for i in range(len(boundary_lines) - 1):
        start = boundary_lines[i]
        end = boundary_lines[i + 1]

        # Skip very small segments (merge with next), but never skip first segment
        # (often contains docstrings, imports, headers)
        if end - start < config.min_segment_lines and i > 0 and i < len(boundary_lines) - 2:
            continue

        # Split large segments
        while end - start > config.max_segment_lines:
            seg_content = "".join(lines[start : start + config.max_segment_lines])
            is_header = _BOUNDARY_REGEX.match(lines[start]) is not None
            segments.append(
                Segment(
                    start_line=start,
                    end_line=start + config.max_segment_lines,
                    content=seg_content,
                    is_header=is_header,
                )
            )
            start = start + config.max_segment_lines

        # Final segment
        if end > start:
            seg_content = "".join(lines[start:end])
            is_header = _BOUNDARY_REGEX.match(lines[start]) is not None
            segments.append(
                Segment(
                    start_line=start,
                    end_line=end,
                    content=seg_content,
                    is_header=is_header,
                )
            )

    # If no boundaries found, split by paragraphs or fixed lines
    if len(segments) <= 1 and n_lines > config.max_segment_lines:
        segments = []
        # Try paragraph splitting (blank lines)
        para_boundaries = [0]
        for i, line in enumerate(lines):
            if line.strip() == "" and i > 0:
                para_boundaries.append(i + 1)
        para_boundaries.append(n_lines)

        for i in range(len(para_boundaries) - 1):
            start = para_boundaries[i]
            end = min(para_boundaries[i + 1], start + config.max_segment_lines)
            if end > start:
                seg_content = "".join(lines[start:end])
                segments.append(Segment(start_line=start, end_line=end, content=seg_content))

    return segments if segments else [Segment(start_line=0, end_line=n_lines, content=content)]


# ---------------------------------------------------------------------------
# Segment scoring
# ---------------------------------------------------------------------------


def _position_score(segment: Segment, total_lines: int) -> float:
    """
    Score based on position: favor start and end of file.

    Files often have important info at the top (imports, config) and
    bottom (main blocks, exports).
    """
    if total_lines == 0:
        return 0.5

    # Relative position [0, 1]
    rel_pos = segment.start_line / total_lines

    # U-shaped curve: high at start (x=0) and end (x=1), low in middle (x=0.5)
    # 4*(x-0.5)^2 gives 1 at x=0, 0 at x=0.5, 1 at x=1
    score = 4.0 * (rel_pos - 0.5) ** 2

    # Boost headers (class/function definitions)
    if segment.is_header:
        score = min(1.0, score + 0.2)

    return max(0.0, min(1.0, score))


def _information_density(segment: Segment) -> float:
    """
    Estimate information density of a segment.

    Higher density = more unique tokens, less repetition, more code-like.
    """
    content = segment.content

    if not content.strip():
        return 0.0

    # Count unique words (normalized)
    words = re.findall(r"\b\w+\b", content.lower())
    if not words:
        return 0.0

    unique_ratio = len(set(words)) / len(words)

    # Code density: presence of syntax characters
    syntax_chars = sum(1 for c in content if c in "{}[]()=;:,.<>+-*/&|!")
    syntax_ratio = min(1.0, syntax_chars / (len(content) + 1) * 10)

    # Penalize pure whitespace or comments
    non_whitespace = len(content.replace(" ", "").replace("\n", "").replace("\t", ""))
    content_ratio = non_whitespace / (len(content) + 1)

    return 0.4 * unique_ratio + 0.3 * syntax_ratio + 0.3 * content_ratio


def _compute_diversity_penalty(
    segment_idx: int,
    embeddings: list[NDArray[np.float32]],
    selected_indices: set[int],
    threshold: float = 0.85,
) -> float:
    """
    Penalize segments too similar to already-selected ones.

    Encourages diversity in selected content.
    """
    if not selected_indices or segment_idx >= len(embeddings):
        return 0.0

    seg_emb = embeddings[segment_idx]
    max_similarity = 0.0

    for idx in selected_indices:
        if idx < len(embeddings):
            sim = np.dot(seg_emb, embeddings[idx])
            max_similarity = max(max_similarity, sim)

    # Penalty ramps up as similarity exceeds threshold
    if max_similarity > threshold:
        return (max_similarity - threshold) / (1.0 - threshold)

    return 0.0


def score_segments(
    segments: list[Segment],
    embeddings: list[NDArray[np.float32]],
    total_lines: int,
    config: SummarizationConfig = DEFAULT_SUMMARIZATION_CONFIG,
) -> list[tuple[int, float]]:
    """
    Score segments by importance.

    Returns list of (segment_index, score) tuples, sorted by score descending.
    """
    scores = []

    for i, segment in enumerate(segments):
        pos_score = _position_score(segment, total_lines)
        density_score = _information_density(segment)

        # Combine scores (diversity applied later during selection)
        combined = (config.position_weight * pos_score + config.density_weight * density_score) / (
            config.position_weight + config.density_weight
        )

        scores.append((i, combined))

    return sorted(scores, key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Semantic summarization
# ---------------------------------------------------------------------------


def summarize_semantic(
    content: str,
    max_size: int,
    config: SummarizationConfig = DEFAULT_SUMMARIZATION_CONFIG,
    embed_fn: Callable[[str], NDArray[np.float32]] | None = None,
) -> str:
    """
    Semantic summarization: select most important segments.

    Unlike simple truncation, this preserves semantically important content
    from throughout the file, not just the beginning/end.

    Args:
        content: Full file content
        max_size: Maximum output size in characters
        config: Summarization configuration
        embed_fn: Optional embedding function (uses default if None)

    Returns:
        Summarized content with omission markers
    """
    if len(content) <= max_size:
        return content

    # Extract segments
    segments = extract_segments(content, config)

    if len(segments) <= 1:
        # Can't summarize single segment - fall back to truncation
        marker = "\n\n// [TRUNCATED - file too large]\n"
        return content[: max_size - len(marker)] + marker

    total_lines = content.count("\n") + 1

    # Get embeddings for diversity scoring
    embeddings: list[NDArray[np.float32]] = []

    if embed_fn is not None:
        # Use provided embedding function
        for seg in segments:
            try:
                emb = embed_fn(seg.content)
                if emb is not None:
                    # Normalize
                    norm = np.linalg.norm(emb)
                    embeddings.append(emb / norm if norm > 0 else emb)
                else:
                    embeddings.append(np.zeros(384, dtype=np.float32))
            except Exception:
                embeddings.append(np.zeros(384, dtype=np.float32))
    else:
        # Fallback: simple bag-of-words embedding
        for seg in segments:
            embeddings.append(_simple_embedding(seg.content))

    # Score and rank segments
    scored = score_segments(segments, embeddings, total_lines, config)

    # Greedy selection: pick highest-scoring segments that fit
    # Use 80% of max_size for content, leaving 20% buffer for markers
    content_budget = int(max_size * 0.80)
    selected_indices: set[int] = set()
    current_size = 0

    # Always include first segment if it fits (contains docstrings, imports, headers)
    if segments and len(segments[0].content) < content_budget:
        selected_indices.add(0)
        current_size += len(segments[0].content)

    for seg_idx, base_score in scored:
        # Skip if already selected
        if seg_idx in selected_indices:
            continue

        segment = segments[seg_idx]
        seg_size = len(segment.content)

        # Check if adding this segment would exceed budget
        if current_size + seg_size > content_budget:
            continue

        # Apply diversity penalty
        diversity_penalty = _compute_diversity_penalty(seg_idx, embeddings, selected_indices)
        adjusted_score = base_score * (1.0 - config.diversity_weight * diversity_penalty)

        # Accept if score is still reasonable
        if adjusted_score > 0.1 or config.preserve_structure and segment.is_header:
            selected_indices.add(seg_idx)
            current_size += seg_size

    # If we selected nothing, at least include first and last
    if not selected_indices:
        selected_indices = {0, len(segments) - 1}

    # Reassemble in original order
    result_parts: list[str] = []
    last_end = 0

    for seg_idx in sorted(selected_indices):
        segment = segments[seg_idx]

        # Add omission marker if there's a gap
        if config.include_markers and segment.start_line > last_end:
            omitted_lines = segment.start_line - last_end
            if omitted_lines > 0:
                result_parts.append(f"\n// [...{omitted_lines} lines omitted...]\n")

        result_parts.append(segment.content)
        last_end = segment.end_line

    # Final omission marker
    if config.include_markers and last_end < total_lines:
        omitted_lines = total_lines - last_end
        if omitted_lines > 0:
            result_parts.append(f"\n// [...{omitted_lines} lines omitted...]\n")

    result = "".join(result_parts)

    # Final size check - strictly enforce max_size
    if len(result) > max_size:
        # Ensure we have room for the truncation marker
        marker = "\n// [TRUNCATED]\n"
        cutoff = max(0, max_size - len(marker))
        return result[:cutoff] + marker

    return result


def _simple_embedding(text: str, dim: int = 384) -> NDArray[np.float32]:
    """
    Simple bag-of-words embedding (fallback when no model available).

    Uses hashed word frequencies as a crude semantic representation.
    """
    words = re.findall(r"\b\w+\b", text.lower())

    if not words:
        return np.zeros(dim, dtype=np.float32)

    # Hash words to dimensions
    vec = np.zeros(dim, dtype=np.float32)
    for word in words:
        idx = hash(word) % dim
        vec[idx] += 1.0

    # Normalize
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------------
# Integration with truncate_smart
# ---------------------------------------------------------------------------


def truncate_with_summarization(
    content: str,
    max_size: int,
    use_semantic: bool = True,
    embed_fn: Callable[[str], NDArray[np.float32]] | None = None,
) -> str:
    """
    Smart truncation with optional semantic summarization.

    Uses semantic summarization for better content preservation when possible,
    falls back to simple truncation for very small limits.

    Args:
        content: Full file content
        max_size: Maximum output size
        use_semantic: Use semantic summarization
        embed_fn: Optional embedding function

    Returns:
        Truncated/summarized content
    """
    if len(content) <= max_size:
        return content

    if not use_semantic or max_size < 500:
        # Too small for meaningful summarization
        return content[: max_size - 20] + "\n// [TRUNCATED]\n"

    return summarize_semantic(content, max_size, embed_fn=embed_fn)


__all__ = [
    "SummarizationConfig",
    "DEFAULT_SUMMARIZATION_CONFIG",
    "Segment",
    "extract_segments",
    "score_segments",
    "summarize_semantic",
    "truncate_with_summarization",
]
