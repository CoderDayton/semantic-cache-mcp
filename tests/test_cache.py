"""Tests for SemanticCache facade and smart_read function."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from semantic_cache_mcp.cache import SemanticCache, smart_read
from semantic_cache_mcp.types import EmbeddingVector


class TestSmartReadUnchangedFile:
    """Tests for smart_read with unchanged files."""

    def test_unchanged_file_returns_minimal_response(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """Unchanged file should return cached content."""
        file_path = sample_files["simple"]

        # First read - caches the file
        result1 = smart_read(semantic_cache_no_embeddings, str(file_path))
        assert result1.from_cache is False

        # Second read - file unchanged, should come from cache
        result2 = smart_read(semantic_cache_no_embeddings, str(file_path))
        assert result2.from_cache is True
        # For small files, full content is returned (more efficient than message)
        # For large files, "unchanged" message would be returned
        assert result2.content is not None

    def test_unchanged_file_tokens_saved(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """Unchanged file should save tokens."""
        file_path = sample_files["python"]

        # First read
        smart_read(semantic_cache_no_embeddings, str(file_path))

        # Second read
        result = smart_read(semantic_cache_no_embeddings, str(file_path))
        assert result.tokens_saved >= 0


class TestSmartReadChangedFile:
    """Tests for smart_read with changed files."""

    def test_changed_file_returns_diff(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Changed file should return unified diff when diff is significantly smaller."""
        # Create a larger file so diff is meaningful
        file_path = temp_dir / "large_original.txt"
        original_lines = [f"Line {i}: This is some content that adds tokens\n" for i in range(50)]
        file_path.write_text("".join(original_lines))

        # First read
        smart_read(semantic_cache_no_embeddings, str(file_path))

        # Small modification (diff should be much smaller than full content)
        file_path.write_text("".join(original_lines) + "New line added at the end\n")

        # Second read - should get diff since it's much smaller than full content
        result = smart_read(semantic_cache_no_embeddings, str(file_path))
        assert result.from_cache is True
        assert result.is_diff is True
        assert "Diff" in result.content or "+" in result.content

    def test_changed_file_updates_cache(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """Changed file should update cache entry."""
        file_path = sample_files["simple"]

        # First read
        smart_read(semantic_cache_no_embeddings, str(file_path))
        entry1 = semantic_cache_no_embeddings.get(str(file_path))

        # Modify and read again
        file_path.write_text("Modified content\n")
        smart_read(semantic_cache_no_embeddings, str(file_path))
        entry2 = semantic_cache_no_embeddings.get(str(file_path))

        assert entry1 is not None
        assert entry2 is not None
        assert entry1.content_hash != entry2.content_hash

    def test_changed_file_respects_max_size_when_full_is_cheapest(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Changed cached reads should still respect max_size when full beats diff."""
        file_path = temp_dir / "large_changed.txt"
        file_path.write_text("A" * 120_000)

        # Prime cache with original content.
        smart_read(semantic_cache_no_embeddings, str(file_path))

        # Rewrite completely so diff is expensive; bounded full content should win.
        file_path.write_text("B" * 120_000)
        result = smart_read(semantic_cache_no_embeddings, str(file_path), max_size=5_000)

        assert result.truncated is True
        assert len(result.content) <= 5_000


class TestSmartReadSemanticMatch:
    """Tests for smart_read with semantic similarity."""

    def test_semantic_match_with_similar_file(
        self, temp_dir: Path, mock_embeddings: EmbeddingVector
    ) -> None:
        """Should find and reference semantically similar cached file."""
        db_path = temp_dir / "semantic_test.db"

        # Mock embed to return consistent embeddings
        with patch("semantic_cache_mcp.cache.embed", return_value=mock_embeddings):
            cache = SemanticCache(db_path=db_path)

            # Create two similar files
            file1 = temp_dir / "similar1.py"
            file1.write_text("def hello():\n    return 'Hello'\n")

            file2 = temp_dir / "similar2.py"
            file2.write_text("def hello():\n    return 'Hi'\n")

            # Cache first file
            smart_read(cache, str(file1))

            # Read second file - might find semantic match
            result = smart_read(cache, str(file2))
            # The result should indicate it processed the file
            assert result.content is not None

    def test_semantic_diff_includes_base_file_context(
        self, temp_dir: Path, mock_embeddings: EmbeddingVector
    ) -> None:
        """Semantic diff responses should include the base cached path."""
        db_path = temp_dir / "semantic_context.db"

        with patch("semantic_cache_mcp.cache.embed", return_value=mock_embeddings):
            cache = SemanticCache(db_path=db_path)

            base = temp_dir / "base.py"
            variant = temp_dir / "variant.py"

            base_lines = [f"line {i}\n" for i in range(250)]
            variant_lines = base_lines.copy()
            variant_lines[123] = "line 123 changed\n"

            base.write_text("".join(base_lines))
            variant.write_text("".join(variant_lines))

            smart_read(cache, str(base))
            result = smart_read(cache, str(variant))

            assert result.is_diff is True
            assert result.semantic_match == str(base)
            assert f"// Similar to cached: {base}" in result.content


class TestSmartReadLargeFile:
    """Tests for smart_read with large files."""

    def test_large_file_truncation(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """Large files should be truncated."""
        file_path = sample_files["large"]

        result = smart_read(
            semantic_cache_no_embeddings,
            str(file_path),
            max_size=10000,  # Much smaller than 150KB file
        )
        assert result.truncated is True
        assert len(result.content) <= 10000

    def test_large_file_preserves_structure(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Truncation should preserve file structure (top + bottom)."""
        # Create large structured file
        lines = [f"Line {i}: Content here\n" for i in range(500)]
        large_file = temp_dir / "structured_large.txt"
        large_file.write_text("".join(lines))

        result = smart_read(
            semantic_cache_no_embeddings,
            str(large_file),
            max_size=5000,
        )
        assert "Line 0" in result.content
        # Semantic summarization uses "omitted" or "TRUNCATED" markers
        content_lower = result.content.lower()
        assert any(marker in content_lower for marker in ["truncated", "omitted", "lines omitted"])


class TestSmartReadDiffModeDisabled:
    """Tests for smart_read with diff_mode=False to get full content."""

    def test_diff_mode_false_bypasses_cache(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """diff_mode=False should return full content even if cached."""
        file_path = sample_files["python"]
        content = file_path.read_text()

        # First read to cache
        smart_read(semantic_cache_no_embeddings, str(file_path))

        # Read with diff_mode=False to get full content
        result = smart_read(
            semantic_cache_no_embeddings,
            str(file_path),
            diff_mode=False,
        )
        assert result.content == content
        assert result.is_diff is False


class TestCacheOperations:
    """Tests for SemanticCache operations."""

    def test_get_stats_returns_metrics(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """get_stats should return cache metrics."""
        # Cache some files
        smart_read(semantic_cache_no_embeddings, str(sample_files["simple"]))
        smart_read(semantic_cache_no_embeddings, str(sample_files["python"]))

        stats = semantic_cache_no_embeddings.get_stats()
        assert stats["files_cached"] == 2
        assert stats["total_tokens_cached"] > 0

    def test_clear_removes_all(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """clear should remove all cache entries."""
        # Cache files
        smart_read(semantic_cache_no_embeddings, str(sample_files["simple"]))
        smart_read(semantic_cache_no_embeddings, str(sample_files["python"]))

        count = semantic_cache_no_embeddings.clear()
        assert count == 2

        stats = semantic_cache_no_embeddings.get_stats()
        assert stats["files_cached"] == 0

    def test_get_embedding_without_embeddings_service(
        self, semantic_cache_no_embeddings: SemanticCache
    ) -> None:
        """get_embedding returns None when embed returns None."""
        result = semantic_cache_no_embeddings.get_embedding("test text")
        assert result is None

    def test_get_embedding_with_embeddings_service(
        self, temp_dir: Path, mock_embeddings: EmbeddingVector
    ) -> None:
        """get_embedding returns embedding when embed returns a value."""
        db_path = temp_dir / "emb_test.db"

        with patch("semantic_cache_mcp.cache.embed", return_value=mock_embeddings):
            cache = SemanticCache(db_path=db_path)
            result = cache.get_embedding("test text")

            assert result is not None
            assert len(result) == len(mock_embeddings)


class TestCacheReadResult:
    """Tests for ReadResult structure."""

    def test_read_result_fields(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """ReadResult should have all expected fields."""
        result = smart_read(semantic_cache_no_embeddings, str(sample_files["simple"]))

        assert hasattr(result, "content")
        assert hasattr(result, "from_cache")
        assert hasattr(result, "is_diff")
        assert hasattr(result, "tokens_original")
        assert hasattr(result, "tokens_returned")
        assert hasattr(result, "tokens_saved")
        assert hasattr(result, "truncated")
        assert hasattr(result, "compression_ratio")

    def test_compression_ratio_calculation(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """compression_ratio should be calculated correctly."""
        result = smart_read(semantic_cache_no_embeddings, str(sample_files["python"]))

        assert 0 < result.compression_ratio <= 1.0


class TestSmartReadDiffMode:
    """Tests for diff_mode parameter."""

    def test_diff_mode_disabled_returns_full(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """diff_mode=False should always return full content."""
        file_path = sample_files["simple"]

        # Cache the file
        smart_read(semantic_cache_no_embeddings, str(file_path))

        # Read again with diff_mode disabled
        result = smart_read(
            semantic_cache_no_embeddings,
            str(file_path),
            diff_mode=False,
        )
        # Should not be a diff response
        assert result.is_diff is False


class TestEmbeddingsIntegration:
    """Tests for FastEmbed integration."""

    def test_embed_function_called(self, temp_dir: Path, mock_embeddings: EmbeddingVector) -> None:
        """Verify embed function is called when generating embeddings."""
        db_path = temp_dir / "embed_test.db"

        with patch("semantic_cache_mcp.cache.embed", return_value=mock_embeddings) as mock_embed:
            cache = SemanticCache(db_path=db_path)

            # Create and read a file
            test_file = temp_dir / "test.txt"
            test_file.write_text("Test content for embedding")

            smart_read(cache, str(test_file))

            # Verify embed was called
            assert mock_embed.called

    def test_embedding_stored_in_cache(
        self, temp_dir: Path, mock_embeddings: EmbeddingVector
    ) -> None:
        """Verify embedding is stored with cache entry."""
        db_path = temp_dir / "store_test.db"

        with patch("semantic_cache_mcp.cache.embed", return_value=mock_embeddings):
            cache = SemanticCache(db_path=db_path)

            test_file = temp_dir / "test.txt"
            test_file.write_text("Test content")

            smart_read(cache, str(test_file))

            entry = cache.get(str(test_file))
            assert entry is not None
            assert entry.embedding is not None
            assert len(entry.embedding) == 768
