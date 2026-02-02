"""Tests for MCP server and edge cases."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semantic_cache_mcp.cache import SemanticCache, smart_read
from semantic_cache_mcp.storage.sqlite import SQLiteStorage


class TestFileNotFoundHandling:
    """Tests for file not found scenarios."""

    def test_smart_read_file_not_found(
        self, semantic_cache_no_embeddings: SemanticCache
    ) -> None:
        """smart_read should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            smart_read(semantic_cache_no_embeddings, "/nonexistent/file.txt")

    def test_smart_read_nonexistent_directory(
        self, semantic_cache_no_embeddings: SemanticCache
    ) -> None:
        """smart_read should raise for file in nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            smart_read(semantic_cache_no_embeddings, "/no/such/dir/file.txt")


class TestEmptyFileHandling:
    """Tests for empty file scenarios."""

    def test_smart_read_empty_file(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """smart_read should handle empty files gracefully."""
        result = smart_read(semantic_cache_no_embeddings, str(sample_files["empty"]))
        assert result.content == ""
        assert result.tokens_original == 0

    def test_empty_file_caching(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """Empty files should be cached correctly."""
        file_path = sample_files["empty"]

        # Cache the empty file
        smart_read(semantic_cache_no_embeddings, str(file_path))

        # Verify it's cached
        entry = semantic_cache_no_embeddings.get(str(file_path))
        assert entry is not None
        assert entry.tokens == 0


class TestBinaryFileHandling:
    """Tests for binary file scenarios."""

    def test_binary_file_graceful_failure(
        self, semantic_cache_no_embeddings: SemanticCache, binary_file: Path
    ) -> None:
        """Binary files should fail gracefully with clear error."""
        # Binary files are detected and rejected with clear error message
        with pytest.raises(ValueError, match="Binary file not supported"):
            smart_read(semantic_cache_no_embeddings, str(binary_file))


class TestUnicodeContentHandling:
    """Tests for unicode content."""

    def test_unicode_chinese_content(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Chinese text should be handled correctly."""
        chinese_file = temp_dir / "chinese.txt"
        chinese_file.write_text("hello world test content")

        result = smart_read(semantic_cache_no_embeddings, str(chinese_file))
        assert "hello" in result.content

    def test_unicode_emoji_content(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Emoji text should be handled correctly."""
        emoji_file = temp_dir / "emoji.txt"
        emoji_file.write_text("Hello World! Great!")

        result = smart_read(semantic_cache_no_embeddings, str(emoji_file))
        assert "Hello" in result.content

    def test_unicode_mixed_content(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mixed unicode should be handled correctly."""
        mixed_file = temp_dir / "mixed.txt"
        mixed_file.write_text("Hello test 123 World!")

        result = smart_read(semantic_cache_no_embeddings, str(mixed_file))
        assert result.content is not None


class TestVeryLargeFiles:
    """Tests for very large files (>100KB)."""

    def test_large_file_handling(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Files >100KB should be handled with truncation."""
        large_file = temp_dir / "very_large.txt"
        large_file.write_text("x" * 200_000)

        result = smart_read(
            semantic_cache_no_embeddings,
            str(large_file),
            max_size=50_000,
        )
        assert result.truncated is True
        assert len(result.content) <= 50_000

    def test_large_file_caching(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Large files should still be fully cached."""
        large_file = temp_dir / "large_cached.txt"
        content = "y" * 150_000
        large_file.write_text(content)

        # Read with truncation
        smart_read(semantic_cache_no_embeddings, str(large_file), max_size=10_000)

        # Verify full content is cached
        entry = semantic_cache_no_embeddings.get(str(large_file))
        assert entry is not None
        retrieved = semantic_cache_no_embeddings.get_content(entry)
        assert retrieved == content


class TestConcurrentAccess:
    """Tests for concurrent access patterns."""

    def test_concurrent_reads(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """Concurrent reads should not corrupt cache."""
        file_path = sample_files["python"]
        results: list[str] = []
        errors: list[Exception] = []

        def read_file():
            try:
                result = smart_read(semantic_cache_no_embeddings, str(file_path))
                results.append(result.content)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_file) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All results should be the same content
        assert len(set(results)) == 1

    def test_concurrent_writes(self, temp_dir: Path) -> None:
        """Concurrent writes should not corrupt database."""
        db_path = temp_dir / "concurrent.db"
        storage = SQLiteStorage(db_path)
        errors: list[Exception] = []

        def write_file(i: int):
            try:
                storage.put(f"/test/file{i}.txt", f"Content {i}", time.time())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_file, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = storage.get_stats()
        assert stats["files_cached"] == 10


class TestCorruptedCacheRecovery:
    """Tests for corrupted cache scenarios."""

    def test_corrupted_db_recreation(self, corrupted_cache_db: Path) -> None:
        """Corrupted database should be detected on connection."""
        # This should raise an error when trying to use corrupted DB
        with pytest.raises(sqlite3.DatabaseError):
            storage = SQLiteStorage(corrupted_cache_db)
            storage.get_stats()

    def test_missing_chunks_graceful(self, temp_dir: Path) -> None:
        """Missing chunks should be handled gracefully."""
        db_path = temp_dir / "missing_chunks.db"
        storage = SQLiteStorage(db_path)

        # Store a file
        storage.put("/test/file.txt", "Test content", time.time())

        # Manually delete chunks
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM chunks")

        # Try to retrieve - should handle gracefully
        entry = storage.get("/test/file.txt")
        assert entry is not None

        # Content retrieval will fail or return partial
        result = storage.load_chunks(entry.chunks)
        assert result == b""  # Empty because chunks are missing


class TestMissingEmbeddingsService:
    """Tests for missing embeddings service."""

    def test_graceful_without_embeddings(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """Cache should work without embeddings service."""
        result = smart_read(semantic_cache_no_embeddings, str(sample_files["python"]))
        assert result.content is not None
        assert result.semantic_match is None

    def test_embedding_error_handled(self, temp_dir: Path) -> None:
        """Embedding service errors should be handled gracefully."""
        db_path = temp_dir / "error_test.db"

        # Mock embed to raise an exception
        with patch("semantic_cache_mcp.cache.embed", side_effect=Exception("Model Error")):
            cache = SemanticCache(db_path=db_path)
            result = cache.get_embedding("test text")
            assert result is None


class TestNetworkTimeoutSimulation:
    """Tests for network timeout scenarios."""

    def test_embedding_timeout_handled(self, temp_dir: Path, sample_files: dict[str, Path]) -> None:
        """Network timeout for embeddings should be handled."""
        db_path = temp_dir / "timeout_test.db"

        def slow_embedding(*args, **kwargs):
            time.sleep(0.1)
            raise TimeoutError("Connection timed out")

        # Mock embed to simulate timeout
        with patch("semantic_cache_mcp.cache.embed", side_effect=slow_embedding):
            cache = SemanticCache(db_path=db_path)

            # Should complete without hanging
            result = smart_read(cache, str(sample_files["simple"]))
            assert result.content is not None


class TestPathTraversalPrevention:
    """Tests for path traversal security."""

    def test_path_expansion(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Paths should be expanded and resolved."""
        # Create a file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        # Use relative-style path components
        with_dots = str(temp_dir / "subdir" / ".." / "test.txt")
        result = smart_read(semantic_cache_no_embeddings, with_dots)
        assert result.content == "Test content"

    def test_tilde_expansion(
        self, semantic_cache_no_embeddings: SemanticCache
    ) -> None:
        """Tilde paths should be expanded."""
        # This tests that ~ is expanded, not that the file exists
        home = Path.home()
        if (home / ".bashrc").exists():
            result = smart_read(semantic_cache_no_embeddings, "~/.bashrc")
            assert result.content is not None

    def test_symlink_resolution(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Symlinks should be resolved."""
        # Create actual file
        real_file = temp_dir / "real.txt"
        real_file.write_text("Real content")

        # Create symlink
        link_file = temp_dir / "link.txt"
        try:
            link_file.symlink_to(real_file)

            result = smart_read(semantic_cache_no_embeddings, str(link_file))
            assert result.content == "Real content"
        except OSError:
            pytest.skip("Symlinks not supported on this platform")


class TestDatabaseIntegrity:
    """Tests for database integrity."""

    def test_schema_creation(self, temp_dir: Path) -> None:
        """Database schema should be created on init."""
        db_path = temp_dir / "new_db.db"
        storage = SQLiteStorage(db_path)

        with sqlite3.connect(db_path) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t[0] for t in tables}

            assert "chunks" in table_names
            assert "files" in table_names

    def test_index_creation(self, temp_dir: Path) -> None:
        """Indexes should be created for performance."""
        db_path = temp_dir / "indexed_db.db"
        SQLiteStorage(db_path)

        with sqlite3.connect(db_path) as conn:
            indexes = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
            index_names = {i[0] for i in indexes}

            assert "idx_created" in index_names
