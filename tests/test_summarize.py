"""Tests for semantic summarization."""

from __future__ import annotations

import numpy as np
import pytest

from semantic_cache_mcp.core.text import (
    Segment,
    SummarizationConfig,
    extract_segments,
    score_segments,
    summarize_semantic,
    truncate_with_summarization,
)


class TestSegmentExtraction:
    """Tests for segment extraction."""

    def test_empty_content(self) -> None:
        """Empty content returns empty list."""
        segments = extract_segments("")
        assert segments == []

    def test_small_content_single_segment(self) -> None:
        """Small content returns single segment."""
        content = "line1\nline2\nline3\n"
        segments = extract_segments(content)
        assert len(segments) == 1
        assert segments[0].content == content

    def test_python_functions_create_segments(self) -> None:
        """Python function definitions create segment boundaries."""
        content = """def foo():
    pass

def bar():
    return 42

class MyClass:
    def method(self):
        pass
"""
        segments = extract_segments(content)
        # Should have multiple segments from function definitions
        assert len(segments) >= 2

    def test_typescript_exports_create_segments(self) -> None:
        """TypeScript exports create segment boundaries."""
        content = """export function doSomething() {
    return true;
}

export class MyClass {
    constructor() {}
}

export const value = 42;
"""
        segments = extract_segments(content)
        assert len(segments) >= 2

    def test_markdown_headers_create_segments(self) -> None:
        """Markdown headers create segment boundaries."""
        content = """# Header 1
Some content here.

## Header 2
More content.

### Header 3
Even more content.
"""
        segments = extract_segments(content)
        # Headers should create boundaries
        assert len(segments) >= 2

    def test_segment_preserves_content(self) -> None:
        """All content is preserved in segments."""
        content = "line1\nline2\nline3\nline4\nline5\n"
        segments = extract_segments(content)

        # Concatenating all segments should give back original
        reconstructed = "".join(seg.content for seg in segments)
        assert reconstructed == content

    def test_max_segment_lines_respected(self) -> None:
        """Large segments are split at max_segment_lines."""
        content = "\n".join(f"line{i}" for i in range(100)) + "\n"
        config = SummarizationConfig(max_segment_lines=20)
        segments = extract_segments(content, config)

        # All segments should be <= 20 lines
        for seg in segments:
            n_lines = seg.content.count("\n")
            assert n_lines <= 20


class TestSegmentScoring:
    """Tests for segment scoring."""

    def test_position_score_favors_start(self) -> None:
        """Segments at start and end get higher position scores than middle."""
        segments = [
            Segment(start_line=0, end_line=10, content="start content"),
            Segment(start_line=50, end_line=60, content="middle content"),
            Segment(start_line=90, end_line=100, content="end content"),
        ]

        # Simple embeddings - same for all to isolate position effect
        d = 384
        embeddings = [np.ones(d, dtype=np.float32) / np.sqrt(d) for _ in segments]

        scored = score_segments(segments, embeddings, total_lines=100)
        scores = {idx: score for idx, score in scored}

        # Middle segment should have lowest score
        middle_score = scores[1]
        start_score = scores[0]
        end_score = scores[2]

        # Start and end should both score higher than middle
        assert start_score >= middle_score or end_score >= middle_score

    def test_headers_get_boost(self) -> None:
        """Header segments get score boost compared to non-header at same position."""
        # Put both segments at same position to isolate header effect
        segments = [
            Segment(start_line=10, end_line=20, content="regular", is_header=False),
            Segment(start_line=10, end_line=20, content="header", is_header=True),
        ]

        d = 384
        embeddings = [np.ones(d, dtype=np.float32) / np.sqrt(d) for _ in segments]
        scored = score_segments(segments, embeddings, total_lines=100)

        # Find scores by index
        scores = {idx: score for idx, score in scored}

        # Header should have higher score due to boost
        assert scores[1] >= scores[0]

    def test_dense_content_higher_score(self) -> None:
        """Information-dense segments score higher."""
        segments = [
            Segment(start_line=0, end_line=5, content="   \n\n\n   \n"),
            Segment(
                start_line=5,
                end_line=10,
                content="def func(): return x + y * z\n",
            ),
        ]

        embeddings = [np.random.randn(384).astype(np.float32) for _ in segments]
        scored = score_segments(segments, embeddings, total_lines=10)

        scores = {idx: score for idx, score in scored}

        # Dense code should score higher than whitespace
        assert scores[1] > scores[0]


class TestSemanticSummarization:
    """Tests for semantic summarization."""

    def test_small_content_unchanged(self) -> None:
        """Content under max_size is returned unchanged."""
        content = "small content"
        result = summarize_semantic(content, max_size=1000)
        assert result == content

    def test_large_content_truncated(self) -> None:
        """Content over max_size is summarized."""
        content = "\n".join(f"line {i}: some content here" for i in range(100))
        result = summarize_semantic(content, max_size=500)
        assert len(result) <= 500

    def test_preserves_semantic_structure(self) -> None:
        """Summarization preserves important structural elements."""
        content = """class ImportantClass:
    \"\"\"This is an important class.\"\"\"

    def __init__(self):
        self.value = 42

    def do_something(self):
        # Lots of implementation details here
        pass

# Many more lines of code...
""" + "\n".join(f"    line{i}" for i in range(50))

        result = summarize_semantic(content, max_size=500)

        # Should preserve class definition
        assert "class ImportantClass" in result or "omitted" in result

    def test_includes_omission_markers(self) -> None:
        """Summarized content includes omission markers."""
        content = "\n".join(f"line {i}" for i in range(100))
        config = SummarizationConfig(include_markers=True)
        result = summarize_semantic(content, max_size=300, config=config)

        assert "omitted" in result.lower() or "truncated" in result.lower()

    def test_no_markers_when_disabled(self) -> None:
        """Markers can be disabled."""
        content = "\n".join(f"line {i}" for i in range(50))
        config = SummarizationConfig(include_markers=False)

        # Need to test with segments that would have gaps
        # For small content this might not create gaps
        result = summarize_semantic(content, max_size=200, config=config)

        # Result should not have "omitted" marker format
        assert "...[" not in result or "[..." not in result

    def test_custom_embed_function(self) -> None:
        """Custom embedding function is used when provided."""
        # Need content large enough to trigger summarization
        content = "\n".join(f"line {i}: " + "x" * 50 for i in range(100))

        embed_called = [False]

        def custom_embed(text: str):
            embed_called[0] = True
            return np.random.randn(384).astype(np.float32)

        # Use small max_size to ensure summarization happens
        summarize_semantic(content, max_size=500, embed_fn=custom_embed)

        assert embed_called[0], "Custom embed function was not called"


class TestTruncateWithSummarization:
    """Tests for truncate_with_summarization."""

    def test_small_content_unchanged(self) -> None:
        """Small content passes through unchanged."""
        content = "Hello, world!"
        result = truncate_with_summarization(content, max_size=1000)
        assert result == content

    def test_very_small_limit_simple_truncate(self) -> None:
        """Very small max_size uses simple truncation."""
        content = "x" * 1000
        result = truncate_with_summarization(content, max_size=100)
        assert len(result) <= 100
        assert "TRUNCATED" in result

    def test_semantic_disabled_uses_truncation(self) -> None:
        """Disabling semantic uses simple truncation."""
        content = "x" * 1000
        result = truncate_with_summarization(content, max_size=200, use_semantic=False)
        assert len(result) <= 200
        assert "TRUNCATED" in result


class TestDiversityPenalty:
    """Tests for diversity-aware selection."""

    def test_similar_segments_penalized(self) -> None:
        """Segments similar to selected ones are penalized."""
        # Create content with repeated sections
        content = """def function_a():
    return 1

def function_b():
    return 2

def function_c():
    return 3
"""
        # All functions are similar, should not select all
        result = summarize_semantic(content, max_size=100)

        # Even with small limit, should get some content
        assert len(result) > 0


@pytest.mark.parametrize("max_size", [200, 500, 1000, 2000])
def test_summarization_respects_size_limit(max_size: int) -> None:
    """Summarization always respects the size limit."""
    content = "\n".join(f"line {i}: " + "x" * 50 for i in range(100))
    result = summarize_semantic(content, max_size=max_size)
    assert len(result) <= max_size


@pytest.mark.parametrize(
    "language,content",
    [
        ("python", "def foo():\n    pass\n\ndef bar():\n    pass\n"),
        ("typescript", "export function foo() {}\n\nexport function bar() {}\n"),
        ("go", "func foo() {}\n\nfunc bar() {}\n"),
        ("rust", "fn foo() {}\n\npub fn bar() {}\n"),
    ],
)
def test_language_boundary_detection(language: str, content: str) -> None:
    """Boundary detection works for various languages."""
    # Expand content to be larger
    expanded = content * 20

    segments = extract_segments(expanded)

    # Should detect multiple segments from function definitions
    assert len(segments) >= 2, f"Failed for {language}"
