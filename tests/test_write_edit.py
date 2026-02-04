"""Tests for smart_write and smart_edit functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from semantic_cache_mcp.cache import SemanticCache, smart_edit, smart_read, smart_write


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
