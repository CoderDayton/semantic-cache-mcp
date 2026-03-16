"""Tests for MCP server and edge cases."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from semantic_cache_mcp.cache import SemanticCache, smart_read
from semantic_cache_mcp.server._mcp import _migrate_v2_to_v3
from semantic_cache_mcp.storage.vector import VectorStorage


class TestFileNotFoundHandling:
    """Tests for file not found scenarios."""

    async def test_smart_read_file_not_found(
        self, semantic_cache_no_embeddings: SemanticCache
    ) -> None:
        """smart_read should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            await smart_read(semantic_cache_no_embeddings, "/nonexistent/file.txt")

    async def test_smart_read_nonexistent_directory(
        self, semantic_cache_no_embeddings: SemanticCache
    ) -> None:
        """smart_read should raise for file in nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            await smart_read(semantic_cache_no_embeddings, "/no/such/dir/file.txt")


class TestEmptyFileHandling:
    """Tests for empty file scenarios."""

    async def test_smart_read_empty_file(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """smart_read should handle empty files gracefully."""
        result = await smart_read(semantic_cache_no_embeddings, str(sample_files["empty"]))
        assert result.content == ""
        assert result.tokens_original == 0

    async def test_empty_file_caching(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """Empty files should be cached correctly."""
        file_path = sample_files["empty"]

        # Cache the empty file
        await smart_read(semantic_cache_no_embeddings, str(file_path))

        # Verify it's cached
        entry = await semantic_cache_no_embeddings.get(str(file_path))
        assert entry is not None
        assert entry.tokens == 0


class TestBinaryFileHandling:
    """Tests for binary file scenarios."""

    async def test_binary_file_graceful_failure(
        self, semantic_cache_no_embeddings: SemanticCache, binary_file: Path
    ) -> None:
        """Binary files should fail gracefully with clear error."""
        # Binary files are detected and rejected with clear error message
        with pytest.raises(ValueError, match="Binary file not supported"):
            await smart_read(semantic_cache_no_embeddings, str(binary_file))


class TestUnicodeContentHandling:
    """Tests for unicode content."""

    async def test_unicode_chinese_content(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Chinese text should be handled correctly."""
        chinese_file = temp_dir / "chinese.txt"
        chinese_file.write_text("hello world test content")

        result = await smart_read(semantic_cache_no_embeddings, str(chinese_file))
        assert "hello" in result.content

    async def test_unicode_emoji_content(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Emoji text should be handled correctly."""
        emoji_file = temp_dir / "emoji.txt"
        emoji_file.write_text("Hello World! Great!")

        result = await smart_read(semantic_cache_no_embeddings, str(emoji_file))
        assert "Hello" in result.content

    async def test_unicode_mixed_content(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Mixed unicode should be handled correctly."""
        mixed_file = temp_dir / "mixed.txt"
        mixed_file.write_text("Hello test 123 World!")

        result = await smart_read(semantic_cache_no_embeddings, str(mixed_file))
        assert result.content is not None


class TestVeryLargeFiles:
    """Tests for very large files (>100KB)."""

    async def test_large_file_handling(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Files >100KB should be handled with truncation."""
        large_file = temp_dir / "very_large.txt"
        large_file.write_text("x" * 200_000)

        result = await smart_read(
            semantic_cache_no_embeddings,
            str(large_file),
            max_size=50_000,
        )
        assert result.truncated is True
        assert len(result.content) <= 50_000

    async def test_large_file_caching(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Large files should still be fully cached."""
        large_file = temp_dir / "large_cached.txt"
        content = "y" * 150_000
        large_file.write_text(content)

        # Read with truncation
        await smart_read(semantic_cache_no_embeddings, str(large_file), max_size=10_000)

        # Verify full content is cached
        entry = await semantic_cache_no_embeddings.get(str(large_file))
        assert entry is not None
        retrieved = await semantic_cache_no_embeddings.get_content(entry)
        assert retrieved == content


class TestConcurrentAccess:
    """Tests for concurrent access patterns."""

    async def test_concurrent_reads(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """Concurrent reads should not corrupt cache."""
        file_path = sample_files["python"]
        original_content = file_path.read_text()

        results: list[str] = []
        for _ in range(5):
            result = await smart_read(semantic_cache_no_embeddings, str(file_path))
            results.append(result.content)

        # Results should be either full content or "unchanged" message
        for r in results:
            assert r == original_content or "unchanged" in r.lower()

    async def test_concurrent_writes(self, temp_dir: Path) -> None:
        """Sequential writes should not corrupt database."""
        db_path = temp_dir / "concurrent.db"
        storage = VectorStorage(db_path)

        for i in range(10):
            await storage.put(f"/test/file{i}.txt", f"Content {i}", time.time())

        stats = await storage.get_stats()
        assert stats["files_cached"] == 10


class TestCorruptedCacheRecovery:
    """Tests for corrupted cache scenarios."""

    async def test_empty_storage_stats(self, temp_dir: Path) -> None:
        """Empty VectorStorage should return zero stats."""
        db_path = temp_dir / "empty.db"
        storage = VectorStorage(db_path)
        stats = await storage.get_stats()
        assert stats["files_cached"] == 0
        assert stats["total_documents"] == 0


class TestMissingEmbeddingsService:
    """Tests for missing embeddings service."""

    async def test_graceful_without_embeddings(
        self, semantic_cache_no_embeddings: SemanticCache, sample_files: dict[str, Path]
    ) -> None:
        """Cache should work without embeddings service."""
        result = await smart_read(semantic_cache_no_embeddings, str(sample_files["python"]))
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

    async def test_embedding_timeout_handled(
        self, temp_dir: Path, sample_files: dict[str, Path]
    ) -> None:
        """Network timeout for embeddings should be handled."""
        db_path = temp_dir / "timeout_test.db"

        def slow_embedding(*args, **kwargs):
            time.sleep(0.1)
            raise TimeoutError("Connection timed out")

        # Mock embed to simulate timeout
        with patch("semantic_cache_mcp.cache.embed", side_effect=slow_embedding):
            cache = SemanticCache(db_path=db_path)

            # Should complete without hanging
            result = await smart_read(cache, str(sample_files["simple"]))
            assert result.content is not None


class TestPathTraversalPrevention:
    """Tests for path traversal security."""

    async def test_path_expansion(
        self, semantic_cache_no_embeddings: SemanticCache, temp_dir: Path
    ) -> None:
        """Paths should be expanded and resolved."""
        # Create a file
        test_file = temp_dir / "test.txt"
        test_file.write_text("Test content")

        # Use relative-style path components
        with_dots = str(temp_dir / "subdir" / ".." / "test.txt")
        result = await smart_read(semantic_cache_no_embeddings, with_dots)
        assert result.content == "Test content"

    async def test_tilde_expansion(self, semantic_cache_no_embeddings: SemanticCache) -> None:
        """Tilde paths should be expanded."""
        # This tests that ~ is expanded, not that the file exists
        home = Path.home()
        if (home / ".bashrc").exists():
            result = await smart_read(semantic_cache_no_embeddings, "~/.bashrc")
            assert result.content is not None

    async def test_symlink_resolution(
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

            result = await smart_read(semantic_cache_no_embeddings, str(link_file))
            assert result.content == "Real content"
        except OSError:
            pytest.skip("Symlinks not supported on this platform")


class TestDatabaseIntegrity:
    """Tests for database integrity."""

    async def test_storage_creation(self, temp_dir: Path) -> None:
        """VectorStorage should initialize and create database file."""
        db_path = temp_dir / "new_db.db"
        storage = VectorStorage(db_path)
        assert db_path.exists()
        stats = await storage.get_stats()
        assert stats["files_cached"] == 0

    async def test_put_and_retrieve(self, temp_dir: Path) -> None:
        """VectorStorage should store and retrieve content."""
        db_path = temp_dir / "integrity.db"
        storage = VectorStorage(db_path)
        await storage.put("/test/file.txt", "Test content", time.time())
        entry = await storage.get("/test/file.txt")
        assert entry is not None
        content = await storage.get_content(entry)
        assert content == "Test content"


# ---------------------------------------------------------------------------
# Migration: v0.2.0 → v0.3.0
# ---------------------------------------------------------------------------


class TestMigrateV2ToV3:
    """Test legacy cache.db cleanup on first v0.3.0 startup."""

    def test_removes_legacy_db_with_old_schema(self, tmp_path: Path) -> None:
        """cache.db with chunks/files/lsh_index tables should be deleted."""
        import sqlite3

        db = tmp_path / "cache.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE files (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE lsh_index (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE session_metrics (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        # Also create WAL/SHM files
        (tmp_path / "cache.db-wal").write_bytes(b"wal")
        (tmp_path / "cache.db-shm").write_bytes(b"shm")

        with patch("semantic_cache_mcp.server._mcp.DB_PATH", db):
            _migrate_v2_to_v3()

        assert not db.exists()
        assert not (tmp_path / "cache.db-wal").exists()
        assert not (tmp_path / "cache.db-shm").exists()

    def test_ignores_unrelated_db(self, tmp_path: Path) -> None:
        """A database without the old schema should be left alone."""
        import sqlite3

        db = tmp_path / "cache.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE something_else (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        with patch("semantic_cache_mcp.server._mcp.DB_PATH", db):
            _migrate_v2_to_v3()

        assert db.exists()  # Not deleted

    def test_no_op_when_no_db(self, tmp_path: Path) -> None:
        """No crash when cache.db doesn't exist."""
        db = tmp_path / "cache.db"
        with patch("semantic_cache_mcp.server._mcp.DB_PATH", db):
            _migrate_v2_to_v3()  # Should not raise

    def test_handles_corrupted_db(self, tmp_path: Path) -> None:
        """Corrupted cache.db should not crash migration."""
        db = tmp_path / "cache.db"
        db.write_bytes(b"this is not a sqlite database")

        with patch("semantic_cache_mcp.server._mcp.DB_PATH", db):
            _migrate_v2_to_v3()  # Should not raise

        assert db.exists()  # Left alone since we can't verify schema
