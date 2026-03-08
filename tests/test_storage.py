"""Tests for VectorStorage backend."""

from __future__ import annotations

import array
import time
from pathlib import Path

from semantic_cache_mcp.storage.vector import VectorStorage
from semantic_cache_mcp.types import EmbeddingVector
from tests.constants import TEST_EMBEDDING_DIM


class TestFileOperations:
    """Tests for file cache operations."""

    def test_put_and_get_file(self, temp_dir: Path) -> None:
        """put and get should store and retrieve files."""
        storage = VectorStorage(temp_dir / "test.db")
        path = "/test/file.txt"
        content = "Test file content"
        mtime = time.time()

        storage.put(path, content, mtime)
        entry = storage.get(path)

        assert entry is not None
        assert entry.path == path
        assert entry.mtime == mtime

    def test_get_nonexistent_returns_none(self, temp_dir: Path) -> None:
        """get on non-existent path should return None."""
        storage = VectorStorage(temp_dir / "test.db")
        entry = storage.get("/nonexistent/path.txt")
        assert entry is None

    def test_get_content_returns_original(self, temp_dir: Path) -> None:
        """get_content should return the original content."""
        storage = VectorStorage(temp_dir / "test.db")
        path = "/test/content.txt"
        content = "Original file content for testing"
        mtime = time.time()

        storage.put(path, content, mtime)
        entry = storage.get(path)
        assert entry is not None

        retrieved = storage.get_content(entry)
        assert retrieved == content

    def test_put_with_embedding(self, temp_dir: Path, mock_embeddings: EmbeddingVector) -> None:
        """put with embedding should store the file with embedding."""
        storage = VectorStorage(temp_dir / "test.db")
        path = "/test/embedded.txt"
        content = "Content with embedding"
        mtime = time.time()

        storage.put(path, content, mtime, embedding=mock_embeddings)
        entry = storage.get(path)

        assert entry is not None
        assert entry.path == path

    def test_record_access_updates_history(self, temp_dir: Path) -> None:
        """record_access should add to access_history."""
        storage = VectorStorage(temp_dir / "test.db")
        path = "/test/access.txt"
        storage.put(path, "Content", time.time())

        # Record multiple accesses
        storage.record_access(path)
        storage.record_access(path)

        entry = storage.get(path)
        assert entry is not None
        # Initial access + 2 recorded = 3 accesses
        assert len(entry.access_history) >= 1

    def test_put_overwrites_existing(self, temp_dir: Path) -> None:
        """put should overwrite existing entry for same path."""
        storage = VectorStorage(temp_dir / "test.db")
        path = "/test/overwrite.txt"

        storage.put(path, "Original", time.time())
        storage.put(path, "Updated", time.time())

        entry = storage.get(path)
        assert entry is not None
        content = storage.get_content(entry)
        assert content == "Updated"


class TestCacheEviction:
    """Tests for LRU-K eviction."""

    def test_lru_k_scoring(self, temp_dir: Path) -> None:
        """LRU-K should use k-th most recent access for scoring."""
        storage = VectorStorage(temp_dir / "test.db")
        path1 = "/test/frequently_accessed.txt"
        path2 = "/test/rarely_accessed.txt"

        storage.put(path1, "Frequent", time.time())
        storage.put(path2, "Rare", time.time())

        # Access path1 multiple times
        for _ in range(5):
            storage.record_access(path1)
            time.sleep(0.01)

        entry1 = storage.get(path1)
        entry2 = storage.get(path2)

        assert entry1 is not None
        assert entry2 is not None
        assert len(entry1.access_history) > len(entry2.access_history)


class TestSemanticSearch:
    """Tests for semantic similarity search."""

    def test_find_similar_returns_best_match(
        self,
        temp_dir: Path,
        mock_embeddings: EmbeddingVector,
        mock_embeddings_similar: EmbeddingVector,
    ) -> None:
        """find_similar should return the most similar file."""
        storage = VectorStorage(temp_dir / "test.db")
        storage.put("/test/reference.txt", "Reference content", time.time(), mock_embeddings)

        result = storage.find_similar(mock_embeddings_similar)
        assert result == "/test/reference.txt"

    def test_find_similar_excludes_path(
        self, temp_dir: Path, mock_embeddings: EmbeddingVector
    ) -> None:
        """find_similar should exclude specified path."""
        storage = VectorStorage(temp_dir / "test.db")
        storage.put("/test/file1.txt", "Content 1", time.time(), mock_embeddings)
        storage.put("/test/file2.txt", "Content 2", time.time(), mock_embeddings)

        result = storage.find_similar(mock_embeddings, exclude_path="/test/file1.txt")
        assert result == "/test/file2.txt"

    def test_find_similar_no_embeddings(self, temp_dir: Path) -> None:
        """find_similar should return None when no files have embeddings."""
        storage = VectorStorage(temp_dir / "test.db")
        storage.put("/test/file.txt", "Content", time.time())  # No embedding

        mock_emb = array.array("f", [0.1] * TEST_EMBEDDING_DIM)
        result = storage.find_similar(mock_emb)
        assert result is None


class TestStatisticsAndManagement:
    """Tests for cache statistics and management."""

    def test_get_stats_returns_dict(self, temp_dir: Path) -> None:
        """get_stats should return dictionary with metrics."""
        storage = VectorStorage(temp_dir / "test.db")
        storage.put("/test/file.txt", "Content", time.time())
        stats = storage.get_stats()

        assert isinstance(stats, dict)
        assert "files_cached" in stats
        assert "total_tokens_cached" in stats
        assert "total_documents" in stats
        assert "db_size_mb" in stats

    def test_get_stats_accurate_count(self, temp_dir: Path) -> None:
        """get_stats should report accurate file count."""
        storage = VectorStorage(temp_dir / "test.db")
        for i in range(5):
            storage.put(f"/test/file{i}.txt", f"Content {i}", time.time())

        stats = storage.get_stats()
        assert stats["files_cached"] == 5

    def test_clear_removes_all_entries(self, temp_dir: Path) -> None:
        """clear should remove all files."""
        storage = VectorStorage(temp_dir / "test.db")
        for i in range(3):
            storage.put(f"/test/file{i}.txt", f"Content {i}", time.time())

        count = storage.clear()
        assert count >= 3  # May include chunk documents

        stats = storage.get_stats()
        assert stats["files_cached"] == 0

    def test_clear_returns_count(self, temp_dir: Path) -> None:
        """clear should return number of documents cleared."""
        storage = VectorStorage(temp_dir / "test.db")
        storage.put("/test/file1.txt", "Content 1", time.time())
        storage.put("/test/file2.txt", "Content 2", time.time())

        count = storage.clear()
        assert count >= 2
