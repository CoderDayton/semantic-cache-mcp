"""End-to-end tests for semantic-cache-mcp.

Tests the complete flow from MCP tools through cache, storage, and back.
Verifies the full system works correctly in realistic scenarios.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semantic_cache_mcp.cache import SemanticCache, smart_read
from semantic_cache_mcp.types import EmbeddingVector


class TestFullCacheLifecycle:
    """Test complete cache lifecycle: read → cache → modify → diff → clear."""

    def test_first_read_caches_file(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """First read should return full content and cache it."""
        # Create test file
        test_file = temp_dir / "test.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        # First read - should cache
        result = smart_read(semantic_cache, str(test_file))

        assert result.from_cache is False
        assert result.is_diff is False
        assert "def hello():" in result.content
        assert result.tokens_original > 0
        assert result.tokens_saved == 0  # First read, nothing saved

    def test_second_read_returns_cached(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Second read of unchanged file should return cached response."""
        # Use a larger file so "unchanged" message is shorter than content
        test_file = temp_dir / "test.py"
        test_file.write_text("def hello():\n    return 'world'\n" * 20)

        # First read
        result1 = smart_read(semantic_cache, str(test_file))

        # Second read - should hit cache
        result2 = smart_read(semantic_cache, str(test_file))

        assert result2.from_cache is True
        assert result2.is_diff is False
        assert result2.tokens_saved > 0

    def test_modified_file_returns_diff(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Modified file should return diff instead of full content."""
        import os

        # Use large file with small change so diff saves >40% tokens
        test_file = temp_dir / "test.py"
        lines = [f"def func_{i}():\n    return {i}\n" for i in range(50)]
        test_file.write_text("".join(lines))

        # First read
        smart_read(semantic_cache, str(test_file))

        # Small modification - change just one line
        lines[25] = "def func_25():\n    return 'MODIFIED'\n"
        test_file.write_text("".join(lines))
        # Force mtime to be in the future
        future_time = time.time() + 10
        os.utime(test_file, (future_time, future_time))

        # Second read - should return diff (small change = big savings)
        result = smart_read(semantic_cache, str(test_file))

        assert result.from_cache is True
        assert result.is_diff is True
        assert result.tokens_saved > 0

    def test_force_full_bypasses_cache(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """force_full=True should return full content even if cached."""
        test_file = temp_dir / "test.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        # First read
        smart_read(semantic_cache, str(test_file))

        # Second read with force_full
        result = smart_read(semantic_cache, str(test_file), force_full=True)

        assert "def hello():" in result.content
        assert result.is_diff is False

    def test_diff_mode_disabled_returns_full(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """diff_mode=False should return full content for modified files."""
        test_file = temp_dir / "test.py"
        test_file.write_text("def hello():\n    return 'world'\n")

        # First read
        smart_read(semantic_cache, str(test_file))

        # Modify file
        time.sleep(0.01)
        test_file.write_text("def hello():\n    return 'universe'\n")

        # Read with diff_mode=False
        result = smart_read(semantic_cache, str(test_file), diff_mode=False)

        assert result.is_diff is False
        assert "def hello():" in result.content


class TestSemanticSimilarity:
    """Test semantic similarity detection between files."""

    def test_similar_file_detected(
        self, temp_dir: Path, mock_embeddings: EmbeddingVector
    ) -> None:
        """Similar files should be detected via embeddings."""
        db_path = temp_dir / "cache.db"

        # Create two similar files
        file1 = temp_dir / "utils.py"
        file1.write_text("def process_data(x):\n    return x * 2\n")

        file2 = temp_dir / "helpers.py"
        file2.write_text("def transform_data(x):\n    return x * 3\n")

        # Mock embeddings to return similar vectors
        call_count = [0]

        def mock_embed(text: str) -> EmbeddingVector:
            call_count[0] += 1
            # Return same embedding for both files (simulating similarity)
            return mock_embeddings

        with patch("semantic_cache_mcp.cache.embed", side_effect=mock_embed):
            cache = SemanticCache(db_path=db_path)

            # Cache first file
            smart_read(cache, str(file1))

            # Read second file - should detect similarity
            result = smart_read(cache, str(file2))

            # Embeddings should have been generated
            assert call_count[0] >= 2


class TestStatsAndClear:
    """Test cache statistics and clearing."""

    def test_stats_empty_cache(self, semantic_cache: SemanticCache) -> None:
        """Stats should work on empty cache."""
        stats = semantic_cache.get_stats()

        assert stats["files_cached"] == 0
        assert stats["total_tokens_cached"] == 0

    def test_stats_after_caching(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Stats should reflect cached files."""
        # Create and cache files
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"Content for file {i}\n" * 10)
            smart_read(semantic_cache, str(f))

        stats = semantic_cache.get_stats()

        assert stats["files_cached"] == 3
        assert stats["total_tokens_cached"] > 0
        assert stats["unique_chunks"] > 0

    def test_clear_removes_all(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Clear should remove all cache entries."""
        # Cache some files
        for i in range(3):
            f = temp_dir / f"file{i}.txt"
            f.write_text(f"Content for file {i}\n")
            smart_read(semantic_cache, str(f))

        # Verify files cached
        assert semantic_cache.get_stats()["files_cached"] == 3

        # Clear
        count = semantic_cache.clear()

        assert count == 3
        assert semantic_cache.get_stats()["files_cached"] == 0


class TestLargeFiles:
    """Test handling of large files."""

    def test_large_file_truncation(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Large files should be truncated to max_size."""
        large_file = temp_dir / "large.txt"
        large_file.write_text("x" * 200_000)  # 200KB

        result = smart_read(semantic_cache, str(large_file), max_size=50_000)

        assert result.truncated is True
        assert len(result.content) <= 50_000

    def test_large_file_preserves_structure(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Truncation should preserve code structure."""
        # Create large Python file
        large_py = temp_dir / "large.py"
        content = '"""Module docstring."""\n\n'
        for i in range(1000):
            content += f"def func_{i}():\n    pass\n\n"
        large_py.write_text(content)

        result = smart_read(semantic_cache, str(large_py), max_size=10_000)

        # Should preserve docstring at top
        assert '"""Module docstring."""' in result.content


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_nonexistent_file(self, semantic_cache: SemanticCache) -> None:
        """Reading nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            smart_read(semantic_cache, "/nonexistent/path/file.txt")

    def test_empty_file(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Empty files should be handled gracefully."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")

        result = smart_read(semantic_cache, str(empty_file))

        assert result.content == ""
        assert result.tokens_original == 0

    def test_binary_file_error(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Binary files should raise appropriate error."""
        binary_file = temp_dir / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\xff\xfe")

        with pytest.raises(ValueError, match="Binary file not supported"):
            smart_read(semantic_cache, str(binary_file))

    def test_unicode_content(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Unicode content should be handled correctly."""
        unicode_file = temp_dir / "unicode.txt"
        unicode_file.write_text("Hello \n Emoji \n Chinese ")

        result = smart_read(semantic_cache, str(unicode_file))

        assert "" in result.content
        assert "" in result.content
        assert "" in result.content

    def test_symlink_resolution(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Symlinks should be resolved to real path."""
        real_file = temp_dir / "real.txt"
        real_file.write_text("Real content")

        symlink = temp_dir / "link.txt"
        symlink.symlink_to(real_file)

        result = smart_read(semantic_cache, str(symlink))

        assert "Real content" in result.content


class TestConcurrentAccess:
    """Test concurrent cache access."""

    def test_multiple_reads_same_file(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Multiple reads of same file should be consistent."""
        test_file = temp_dir / "shared.txt"
        test_file.write_text("Shared content\n")

        results = []
        for _ in range(10):
            result = smart_read(semantic_cache, str(test_file))
            results.append(result)

        # First read is from disk, rest from cache
        assert results[0].from_cache is False
        for r in results[1:]:
            assert r.from_cache is True


class TestMCPToolIntegration:
    """Test MCP tool functions directly."""

    def test_read_tool_format(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Read tool should return properly formatted response."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        result = smart_read(semantic_cache, str(test_file))

        # Result should have all required fields
        assert hasattr(result, "content")
        assert hasattr(result, "from_cache")
        assert hasattr(result, "is_diff")
        assert hasattr(result, "tokens_saved")
        assert hasattr(result, "truncated")

    def test_stats_json_serializable(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Stats should be JSON serializable."""
        # Cache a file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")
        smart_read(semantic_cache, str(test_file))

        stats = semantic_cache.get_stats()

        # Should be JSON serializable
        json_str = json.dumps(stats)
        parsed = json.loads(json_str)

        assert parsed["files_cached"] == 1


class TestTokenSavings:
    """Test token savings calculations."""

    def test_unchanged_file_high_savings(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Unchanged file should show high token savings."""
        test_file = temp_dir / "code.py"
        test_file.write_text("def func():\n" * 100)  # ~1000 tokens

        # First read
        result1 = smart_read(semantic_cache, str(test_file))
        original_tokens = result1.tokens_original

        # Second read
        result2 = smart_read(semantic_cache, str(test_file))

        # Should save most tokens
        assert result2.tokens_saved > original_tokens * 0.9

    def test_diff_saves_tokens(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Diff should save tokens compared to full content."""
        test_file = temp_dir / "code.py"
        test_file.write_text("def func():\n    pass\n" * 50)

        # First read
        result1 = smart_read(semantic_cache, str(test_file))
        original_tokens = result1.tokens_original

        # Small modification
        time.sleep(0.01)
        test_file.write_text("def func():\n    return 1\n" + "def func():\n    pass\n" * 49)

        # Second read - diff
        result2 = smart_read(semantic_cache, str(test_file))

        # Diff should save significant tokens
        assert result2.is_diff is True
        assert result2.tokens_saved > original_tokens * 0.5
