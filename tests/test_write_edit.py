"""Tests for smart_write and smart_edit functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from semantic_cache_mcp.cache import (
    SemanticCache,
    _extract_line_range,
    _suppress_large_diff,
    smart_batch_edit,
    smart_edit,
    smart_read,
    smart_write,
)


class TestSmartWriteNewFile:
    """Tests for writing new files."""

    def test_write_new_file(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Writing a new file creates it and caches it."""
        file_path = temp_dir / "new_file.txt"
        content = "Hello, World!\n"

        result = smart_write(semantic_cache_no_embeddings, str(file_path), content)

        assert result.created is True
        assert file_path.exists()
        assert file_path.read_text() == content
        assert result.bytes_written == len(content.encode())
        assert result.diff_content is None

    def test_write_creates_parents(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Writing with create_parents=True creates missing directories."""
        file_path = temp_dir / "nested" / "dir" / "file.txt"
        content = "nested content\n"

        result = smart_write(semantic_cache_no_embeddings, str(file_path), content)

        assert result.created is True
        assert file_path.exists()
        assert file_path.read_text() == content

    def test_write_no_create_parents_fails(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Writing with create_parents=False fails if parent missing."""
        file_path = temp_dir / "missing" / "file.txt"

        with pytest.raises(FileNotFoundError, match="Parent directory"):
            smart_write(
                semantic_cache_no_embeddings,
                str(file_path),
                "content",
                create_parents=False,
            )


class TestSmartWriteOverwrite:
    """Tests for overwriting existing files."""

    def test_write_overwrites_existing(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Overwriting returns diff of changes."""
        file_path = temp_dir / "existing.txt"
        file_path.write_text("original content\n")

        result = smart_write(semantic_cache_no_embeddings, str(file_path), "new content\n")

        assert result.created is False
        assert result.diff_content is not None
        assert "-original" in result.diff_content
        assert "+new" in result.diff_content

    def test_write_updates_cache(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Write updates cache for subsequent reads."""
        file_path = temp_dir / "cached.txt"

        smart_write(semantic_cache_no_embeddings, str(file_path), "first\n")
        smart_write(semantic_cache_no_embeddings, str(file_path), "second\n")

        # Subsequent read should use cache
        result = smart_read(semantic_cache_no_embeddings, str(file_path))
        assert result.from_cache is True


class TestSmartWriteDryRun:
    """Tests for dry_run mode."""

    def test_write_dry_run_no_write(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """dry_run=True doesn't write file."""
        file_path = temp_dir / "dryrun.txt"

        result = smart_write(semantic_cache_no_embeddings, str(file_path), "content", dry_run=True)

        assert result.created is True
        assert not file_path.exists()

    def test_write_dry_run_shows_diff(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """dry_run=True still shows diff for overwrites."""
        file_path = temp_dir / "dryrun_diff.txt"
        file_path.write_text("old\n")

        result = smart_write(semantic_cache_no_embeddings, str(file_path), "new\n", dry_run=True)

        assert result.diff_content is not None
        assert file_path.read_text() == "old\n"  # Unchanged


class TestSmartWriteEdgeCases:
    """Tests for edge cases."""

    def test_write_binary_rejection(
        self, semantic_cache_no_embeddings: SemanticCache, binary_file: Path
    ) -> None:
        """Cannot overwrite binary file with text."""
        with pytest.raises(ValueError, match="Binary file"):
            smart_write(semantic_cache_no_embeddings, str(binary_file), "text")

    def test_write_returns_hash(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Write returns content hash for verification."""
        file_path = temp_dir / "hash_test.txt"

        result = smart_write(semantic_cache_no_embeddings, str(file_path), "content\n")

        assert result.content_hash
        assert len(result.content_hash) == 64  # BLAKE3 hex


class TestSmartEditBasic:
    """Tests for basic edit operations."""

    def test_edit_single_match(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Edit replaces single match."""
        file_path = temp_dir / "edit_single.txt"
        file_path.write_text("Hello World\n")

        result = smart_edit(semantic_cache_no_embeddings, str(file_path), "World", "Universe")

        assert result.matches_found == 1
        assert result.replacements_made == 1
        assert file_path.read_text() == "Hello Universe\n"

    def test_edit_returns_line_numbers(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Edit returns line numbers of matches."""
        file_path = temp_dir / "edit_lines.txt"
        file_path.write_text("line1\nfoo\nline3\n")

        result = smart_edit(semantic_cache_no_embeddings, str(file_path), "foo", "bar")

        assert result.line_numbers == [2]

    def test_edit_returns_diff(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Edit returns unified diff."""
        file_path = temp_dir / "edit_diff.txt"
        file_path.write_text("old value\n")

        result = smart_edit(semantic_cache_no_embeddings, str(file_path), "old", "new")

        assert "-old value" in result.diff_content
        assert "+new value" in result.diff_content


class TestSmartEditMultipleMatches:
    """Tests for multiple match handling."""

    def test_edit_multiple_matches_fails(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Multiple matches without replace_all raises error."""
        file_path = temp_dir / "edit_multi.txt"
        file_path.write_text("foo bar foo\n")

        with pytest.raises(ValueError, match="found 2 times"):
            smart_edit(semantic_cache_no_embeddings, str(file_path), "foo", "baz")

    def test_edit_replace_all(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """replace_all=True replaces all occurrences."""
        file_path = temp_dir / "edit_all.txt"
        file_path.write_text("foo bar foo\n")

        result = smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            "foo",
            "baz",
            replace_all=True,
        )

        assert result.matches_found == 2
        assert result.replacements_made == 2
        assert file_path.read_text() == "baz bar baz\n"


class TestSmartEditErrors:
    """Tests for error handling."""

    def test_edit_not_found_error(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Not found raises clear error."""
        file_path = temp_dir / "edit_notfound.txt"
        file_path.write_text("content\n")

        with pytest.raises(ValueError, match="not found"):
            smart_edit(semantic_cache_no_embeddings, str(file_path), "missing", "replacement")

    def test_edit_file_not_found(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """File not found raises FileNotFoundError."""
        file_path = temp_dir / "nonexistent.txt"

        with pytest.raises(FileNotFoundError):
            smart_edit(semantic_cache_no_embeddings, str(file_path), "a", "b")

    def test_edit_identical_strings(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """old_string == new_string raises error."""
        file_path = temp_dir / "edit_identical.txt"
        file_path.write_text("content\n")

        with pytest.raises(ValueError, match="identical"):
            smart_edit(semantic_cache_no_embeddings, str(file_path), "same", "same")

    def test_edit_binary_rejection(
        self, semantic_cache_no_embeddings: SemanticCache, binary_file: Path
    ) -> None:
        """Cannot edit binary files."""
        with pytest.raises(ValueError, match="Binary file"):
            smart_edit(semantic_cache_no_embeddings, str(binary_file), "a", "b")


class TestSmartEditCache:
    """Tests for cache integration."""

    def test_edit_uses_cache(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Edit uses cached content when available."""
        file_path = temp_dir / "edit_cache.txt"
        file_path.write_text("cached content\n")

        # Prime the cache
        smart_read(semantic_cache_no_embeddings, str(file_path))

        # Edit should use cache
        result = smart_edit(semantic_cache_no_embeddings, str(file_path), "cached", "modified")

        assert result.from_cache is True

    def test_edit_updates_cache(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Edit updates cache with new content."""
        file_path = temp_dir / "edit_update.txt"
        file_path.write_text("original\n")

        smart_edit(semantic_cache_no_embeddings, str(file_path), "original", "modified")

        # Subsequent read should be from cache and unchanged
        result = smart_read(semantic_cache_no_embeddings, str(file_path))
        assert result.from_cache is True


class TestSmartEditDryRun:
    """Tests for dry_run mode."""

    def test_edit_dry_run_no_write(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """dry_run=True doesn't modify file."""
        file_path = temp_dir / "edit_dryrun.txt"
        file_path.write_text("original\n")

        result = smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            "original",
            "modified",
            dry_run=True,
        )

        assert result.replacements_made == 1
        assert file_path.read_text() == "original\n"  # Unchanged


class TestTokenSavings:
    """Tests for token savings calculations."""

    def test_write_token_savings(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Write calculates token savings correctly."""
        file_path = temp_dir / "tokens.txt"
        # Create file with multiple lines for meaningful diff
        lines = [f"line {i}: some content here\n" for i in range(50)]
        file_path.write_text("".join(lines))

        # Small change at end
        new_lines = lines + ["new line added\n"]
        result = smart_write(semantic_cache_no_embeddings, str(file_path), "".join(new_lines))

        # Diff should be much smaller than full content
        assert result.tokens_saved > 0

    def test_edit_token_savings(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Edit tracks token savings from cached read."""
        file_path = temp_dir / "edit_tokens.txt"
        # Use unique marker that appears once
        file_path.write_text("start UNIQUE_MARKER end\n" * 10)

        # Prime cache
        smart_read(semantic_cache_no_embeddings, str(file_path))

        result = smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            "start UNIQUE_MARKER end",
            "start REPLACED_VALUE end",
            replace_all=True,
        )

        # Should save tokens from not returning full content
        assert result.tokens_saved > 0


class TestAutoFormat:
    """Tests for auto_format feature."""

    def test_write_auto_format_no_formatter(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Auto-format with no formatter available writes content unchanged."""
        file_path = temp_dir / "test.txt"  # .txt has no formatter
        content = "unformatted   content"

        result = smart_write(
            semantic_cache_no_embeddings,
            str(file_path),
            content,
            auto_format=True,
        )

        assert result.created is True
        assert file_path.read_text() == content

    def test_write_auto_format_false_default(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Auto-format defaults to False."""
        file_path = temp_dir / "test.py"
        # Intentionally badly formatted Python
        content = "x=1\ny=2\n"

        result = smart_write(
            semantic_cache_no_embeddings,
            str(file_path),
            content,
        )

        # Should be written as-is without formatting
        assert file_path.read_text() == content
        assert result.created is True

    def test_edit_auto_format_no_formatter(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Auto-format with no formatter available edits content unchanged."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("old content")

        result = smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            "old",
            "new",
            auto_format=True,
        )

        assert file_path.read_text() == "new content"
        assert result.replacements_made == 1


class TestSuppressLargeDiff:
    """Tests for _suppress_large_diff function."""

    def test_small_diff_passes_through(self) -> None:
        """Small diffs are returned unchanged."""
        diff = "@@ -1,3 +1,3 @@\n-old\n+new\n context\n"
        result = _suppress_large_diff(diff, full_tokens=500)
        assert result == diff

    def test_large_diff_returns_summary(self) -> None:
        """Large diffs return a summary string, not None."""
        # Generate a diff large enough to exceed MAX_RETURN_DIFF_TOKENS (8000)
        lines = []
        lines.append("--- old\n+++ new")
        lines.append("@@ -1,10000 +1,10000 @@")
        for i in range(5000):
            lines.append(f"-old line {i}")
            lines.append(f"+new line {i}")
        large_diff = "\n".join(lines)

        result = _suppress_large_diff(large_diff, full_tokens=50000)

        assert result is not None
        assert "[diff suppressed:" in result
        assert "+5000" in result
        assert "-5000" in result
        assert "1 hunks" in result

    def test_none_input_returns_none(self) -> None:
        """None input returns None."""
        assert _suppress_large_diff(None, full_tokens=100) is None

    def test_small_file_preserves_diff(self) -> None:
        """Files <=200 tokens always get full diff regardless of ratio."""
        # This diff is larger than the "file" (ratio > 0.9) but file is small
        diff = "@@ -1,3 +1,3 @@\n-old\n+new\n context line\n"
        result = _suppress_large_diff(diff, full_tokens=10)
        assert result == diff


class TestSmartWriteAppend:
    """Tests for append mode."""

    def test_append_creates_new_file(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Append to non-existent file creates it."""
        file_path = temp_dir / "append_new.txt"

        result = smart_write(semantic_cache_no_embeddings, str(file_path), "chunk1\n", append=True)

        assert result.created is True
        assert file_path.read_text() == "chunk1\n"

    def test_append_concatenates_content(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Append adds content to existing file."""
        file_path = temp_dir / "append_cat.txt"
        file_path.write_text("chunk1\n")

        smart_write(semantic_cache_no_embeddings, str(file_path), "chunk2\n", append=True)

        assert file_path.read_text() == "chunk1\nchunk2\n"

    def test_append_multiple_chunks(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Multiple appends build up file incrementally."""
        file_path = temp_dir / "append_multi.txt"

        smart_write(semantic_cache_no_embeddings, str(file_path), "line1\n")
        smart_write(semantic_cache_no_embeddings, str(file_path), "line2\n", append=True)
        smart_write(semantic_cache_no_embeddings, str(file_path), "line3\n", append=True)

        assert file_path.read_text() == "line1\nline2\nline3\n"

    def test_append_returns_diff(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Append returns diff showing added content."""
        file_path = temp_dir / "append_diff.txt"
        file_path.write_text("existing\n")

        result = smart_write(
            semantic_cache_no_embeddings, str(file_path), "appended\n", append=True
        )

        assert result.diff_content is not None
        assert "+appended" in result.diff_content


class TestExtractLineRange:
    """Tests for _extract_line_range helper."""

    def test_single_line(self) -> None:
        content = "aaa\nbbb\nccc\n"
        sub, cs, ce = _extract_line_range(content, 2, 2)
        assert sub == "bbb\n"
        assert content[:cs] + "XXX\n" + content[ce:] == "aaa\nXXX\nccc\n"

    def test_multi_line(self) -> None:
        content = "line1\nline2\nline3\nline4\n"
        sub, cs, ce = _extract_line_range(content, 2, 3)
        assert sub == "line2\nline3\n"
        assert content[:cs] + "replaced\n" + content[ce:] == "line1\nreplaced\nline4\n"

    def test_full_file(self) -> None:
        content = "a\nb\n"
        sub, cs, ce = _extract_line_range(content, 1, 2)
        assert sub == content
        assert cs == 0
        assert ce == len(content)

    def test_no_trailing_newline(self) -> None:
        content = "first\nsecond"
        sub, cs, ce = _extract_line_range(content, 2, 2)
        assert sub == "second"

    def test_empty_file_raises(self) -> None:
        with pytest.raises(ValueError, match="empty file"):
            _extract_line_range("", 1, 1)

    def test_start_line_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="start_line must be >= 1"):
            _extract_line_range("a\n", 0, 1)

    def test_end_before_start_raises(self) -> None:
        with pytest.raises(ValueError, match="end_line.*must be >= start_line"):
            _extract_line_range("a\nb\n", 2, 1)

    def test_start_exceeds_total_raises(self) -> None:
        with pytest.raises(ValueError, match="start_line.*exceeds total"):
            _extract_line_range("a\n", 3, 3)

    def test_end_exceeds_total_raises(self) -> None:
        with pytest.raises(ValueError, match="end_line.*exceeds total"):
            _extract_line_range("a\nb\n", 1, 5)


class TestSmartEditLineRange:
    """Tests for line-range editing (Modes B and C)."""

    # --- Mode B: scoped find/replace ---

    def test_mode_b_finds_match_in_range(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode B finds old_string within the specified line range."""
        file_path = temp_dir / "mode_b.txt"
        file_path.write_text("foo\nbar\nfoo\nbaz\n")

        # "foo" appears on lines 1 and 3; scope to line 3 only
        result = smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            "foo",
            "qux",
            start_line=3,
            end_line=3,
        )

        assert result.replacements_made == 1
        assert file_path.read_text() == "foo\nbar\nqux\nbaz\n"

    def test_mode_b_not_found_outside_range(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode B fails when old_string only exists outside the range."""
        file_path = temp_dir / "mode_b_miss.txt"
        file_path.write_text("foo\nbar\nbaz\n")

        with pytest.raises(ValueError, match="not found within lines 2-3"):
            smart_edit(
                semantic_cache_no_embeddings,
                str(file_path),
                "foo",
                "qux",
                start_line=2,
                end_line=3,
            )

    def test_mode_b_reports_absolute_line_numbers(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode B reports absolute line numbers, not relative to range."""
        file_path = temp_dir / "mode_b_abs.txt"
        file_path.write_text("aaa\nbbb\nccc\nddd\neee\n")

        result = smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            "ccc",
            "CCC",
            start_line=2,
            end_line=4,
        )

        assert result.line_numbers == [3]

    def test_mode_b_replace_all_within_range(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode B replace_all replaces all within the range only."""
        file_path = temp_dir / "mode_b_all.txt"
        file_path.write_text("x\nx\nx\nx\n")

        result = smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            "x",
            "y",
            replace_all=True,
            start_line=2,
            end_line=3,
        )

        assert result.replacements_made == 2
        assert file_path.read_text() == "x\ny\ny\nx\n"

    def test_mode_b_multiple_matches_without_replace_all_fails(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode B with multiple matches in range and no replace_all raises."""
        file_path = temp_dir / "mode_b_multi.txt"
        file_path.write_text("x\nx\nx\n")

        with pytest.raises(ValueError, match="found 2 times"):
            smart_edit(
                semantic_cache_no_embeddings,
                str(file_path),
                "x",
                "y",
                start_line=1,
                end_line=2,
            )

    # --- Mode C: line-range replacement ---

    def test_mode_c_replaces_line_range(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode C replaces exact line range with new_string."""
        file_path = temp_dir / "mode_c.txt"
        file_path.write_text("line1\nline2\nline3\nline4\n")

        result = smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            None,
            "replaced\n",
            start_line=2,
            end_line=3,
        )

        assert file_path.read_text() == "line1\nreplaced\nline4\n"
        assert result.replacements_made == 2  # 2 lines replaced

    def test_mode_c_single_line(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode C works on a single line."""
        file_path = temp_dir / "mode_c_single.txt"
        file_path.write_text("aaa\nbbb\nccc\n")

        result = smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            None,
            "BBB\n",
            start_line=2,
            end_line=2,
        )

        assert file_path.read_text() == "aaa\nBBB\nccc\n"
        assert result.replacements_made == 1

    def test_mode_c_multiline_new_string(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode C can insert more lines than it removes."""
        file_path = temp_dir / "mode_c_multi.txt"
        file_path.write_text("a\nb\nc\n")

        smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            None,
            "x\ny\nz\n",
            start_line=2,
            end_line=2,
        )

        assert file_path.read_text() == "a\nx\ny\nz\nc\n"

    def test_mode_c_dry_run(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode C dry_run previews without writing."""
        file_path = temp_dir / "mode_c_dry.txt"
        file_path.write_text("keep\ndelete\nkeep\n")

        result = smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            None,
            "",
            start_line=2,
            end_line=2,
            dry_run=True,
        )

        assert file_path.read_text() == "keep\ndelete\nkeep\n"  # Unchanged
        assert result.diff_content

    def test_mode_c_returns_diff(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode C returns a unified diff."""
        file_path = temp_dir / "mode_c_diff.txt"
        file_path.write_text("old1\nold2\nold3\n")

        result = smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            None,
            "new2\n",
            start_line=2,
            end_line=2,
        )

        assert "-old2" in result.diff_content
        assert "+new2" in result.diff_content

    # --- Validation errors ---

    def test_mode_c_reject_replace_all(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode C rejects replace_all."""
        file_path = temp_dir / "mode_c_ra.txt"
        file_path.write_text("a\nb\n")

        with pytest.raises(ValueError, match="replace_all is not supported"):
            smart_edit(
                semantic_cache_no_embeddings,
                str(file_path),
                None,
                "x\n",
                replace_all=True,
                start_line=1,
                end_line=1,
            )

    def test_old_string_none_without_line_range_raises(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """old_string=None without line range raises ValueError."""
        file_path = temp_dir / "no_range.txt"
        file_path.write_text("content\n")

        with pytest.raises(ValueError, match="old_string is required"):
            smart_edit(
                semantic_cache_no_embeddings,
                str(file_path),
                None,
                "new",
            )

    def test_only_start_line_raises(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Providing only start_line without end_line raises."""
        file_path = temp_dir / "half_range.txt"
        file_path.write_text("a\nb\n")

        with pytest.raises(ValueError, match="must both be provided"):
            smart_edit(
                semantic_cache_no_embeddings,
                str(file_path),
                "a",
                "b",
                start_line=1,
            )

    def test_out_of_bounds_raises(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Line range exceeding file length raises."""
        file_path = temp_dir / "oob.txt"
        file_path.write_text("one\ntwo\n")

        with pytest.raises(ValueError, match="exceeds total"):
            smart_edit(
                semantic_cache_no_embeddings,
                str(file_path),
                None,
                "x\n",
                start_line=1,
                end_line=5,
            )

    def test_end_before_start_raises(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """end_line < start_line raises."""
        file_path = temp_dir / "backwards.txt"
        file_path.write_text("a\nb\nc\n")

        with pytest.raises(ValueError, match="must be >= start_line"):
            smart_edit(
                semantic_cache_no_embeddings,
                str(file_path),
                None,
                "x\n",
                start_line=3,
                end_line=1,
            )

    def test_mode_c_no_trailing_newline(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode C auto-appends newline when new_string lacks one."""
        file_path = temp_dir / "mode_c_no_nl.txt"
        file_path.write_text("aaa\nbbb\nccc\nddd\n")

        result = smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            None,
            "BBB",  # no trailing \n
            start_line=2,
            end_line=2,
        )

        # Should NOT concatenate "BBB" with "ccc" on one line
        assert file_path.read_text() == "aaa\nBBB\nccc\nddd\n"
        assert result.replacements_made == 1

    def test_mode_c_no_trailing_newline_multiline(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode C auto-appends newline for multi-line replacement without trailing newline."""
        file_path = temp_dir / "mode_c_no_nl_multi.txt"
        file_path.write_text("aaa\nbbb\nccc\nddd\n")

        smart_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            None,
            "x\ny",  # no trailing \n
            start_line=2,
            end_line=3,
        )

        assert file_path.read_text() == "aaa\nx\ny\nddd\n"


class TestSmartBatchEditLineRange:
    """Tests for line-range editing in batch_edit."""

    def test_mode_c_in_batch(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode C works in batch edits."""
        file_path = temp_dir / "batch_c.txt"
        file_path.write_text("aaa\nbbb\nccc\nddd\n")

        result = smart_batch_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            [(None, "BBB\n", 2, 2)],
        )

        assert result.succeeded == 1
        assert result.failed == 0
        assert file_path.read_text() == "aaa\nBBB\nccc\nddd\n"

    def test_mode_b_in_batch(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode B works in batch edits."""
        file_path = temp_dir / "batch_b.txt"
        file_path.write_text("foo\nbar\nfoo\n")

        result = smart_batch_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            [("foo", "qux", 3, 3)],
        )

        assert result.succeeded == 1
        assert file_path.read_text() == "foo\nbar\nqux\n"

    def test_mixed_modes_in_batch(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mixed Mode A and Mode C edits in same batch."""
        file_path = temp_dir / "batch_mixed.txt"
        file_path.write_text("aaa\nbbb\nccc\nddd\n")

        result = smart_batch_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            [
                ("aaa", "AAA", None, None),  # Mode A: replace aaa
                (None, "DDD\n", 4, 4),  # Mode C: replace line 4
            ],
        )

        assert result.succeeded == 2
        assert result.failed == 0
        text = file_path.read_text()
        assert "AAA" in text
        assert "DDD" in text

    def test_invalid_range_becomes_failure(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Invalid line range in batch produces failure outcome, not exception."""
        file_path = temp_dir / "batch_fail.txt"
        file_path.write_text("a\nb\n")

        result = smart_batch_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            [(None, "x\n", 1, 99)],
        )

        assert result.failed == 1
        assert result.succeeded == 0
        assert result.outcomes[0].success is False
        assert "exceeds" in (result.outcomes[0].error or "")

    def test_batch_mode_c_no_trailing_newline(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mode C in batch auto-appends newline when new_string lacks one."""
        file_path = temp_dir / "batch_no_nl.txt"
        file_path.write_text("aaa\nbbb\nccc\nddd\n")

        result = smart_batch_edit(
            semantic_cache_no_embeddings,
            str(file_path),
            [(None, "XXX", 2, 2), (None, "YYY", 3, 3)],
        )

        assert result.succeeded == 2
        assert file_path.read_text() == "aaa\nXXX\nYYY\nddd\n"
