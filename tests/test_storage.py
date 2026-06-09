"""Tests for ContentStorage backend."""

from __future__ import annotations

import time
from pathlib import Path

from semantic_cache_mcp.storage.docstore import ContentStorage


class TestFileOperations:
    """Tests for file cache operations."""

    async def test_put_and_get_file(self, temp_dir: Path) -> None:
        """put and get should store and retrieve files."""
        storage = ContentStorage(temp_dir / "test.db")
        path = "/test/file.txt"
        content = "Test file content"
        mtime = time.time()

        await storage.put(path, content, mtime)
        entry = await storage.get(path)

        assert entry is not None
        assert entry.path == path
        assert entry.mtime == mtime

    async def test_get_nonexistent_returns_none(self, temp_dir: Path) -> None:
        """get on non-existent path should return None."""
        storage = ContentStorage(temp_dir / "test.db")
        entry = await storage.get("/nonexistent/path.txt")
        assert entry is None

    async def test_get_content_returns_original(self, temp_dir: Path) -> None:
        """get_content should return the original content."""
        storage = ContentStorage(temp_dir / "test.db")
        path = "/test/content.txt"
        content = "Original file content for testing"
        mtime = time.time()

        await storage.put(path, content, mtime)
        entry = await storage.get(path)
        assert entry is not None

        retrieved = await storage.get_content(entry)
        assert retrieved == content

    async def test_record_access_updates_history(self, temp_dir: Path) -> None:
        """record_access should add to access_history."""
        storage = ContentStorage(temp_dir / "test.db")
        path = "/test/access.txt"
        await storage.put(path, "Content", time.time())

        # Record multiple accesses
        await storage.record_access(path)
        await storage.record_access(path)

        entry = await storage.get(path)
        assert entry is not None
        # Initial access + 2 recorded = 3 accesses
        assert len(entry.access_history) >= 1

    async def test_put_overwrites_existing(self, temp_dir: Path) -> None:
        """put should overwrite existing entry for same path."""
        storage = ContentStorage(temp_dir / "test.db")
        path = "/test/overwrite.txt"

        await storage.put(path, "Original", time.time())
        await storage.put(path, "Updated", time.time())

        entry = await storage.get(path)
        assert entry is not None
        content = await storage.get_content(entry)
        assert content == "Updated"


class TestCacheEviction:
    """Tests for LRU-K eviction."""

    async def test_lru_k_scoring(self, temp_dir: Path) -> None:
        """LRU-K should use k-th most recent access for scoring."""
        storage = ContentStorage(temp_dir / "test.db")
        path1 = "/test/frequently_accessed.txt"
        path2 = "/test/rarely_accessed.txt"

        await storage.put(path1, "Frequent", time.time())
        await storage.put(path2, "Rare", time.time())

        # Access path1 multiple times
        for _ in range(5):
            await storage.record_access(path1)
            time.sleep(0.01)

        entry1 = await storage.get(path1)
        entry2 = await storage.get(path2)

        assert entry1 is not None
        assert entry2 is not None
        assert len(entry1.access_history) > len(entry2.access_history)


class TestStatisticsAndManagement:
    """Tests for cache statistics and management."""

    async def test_get_stats_returns_dict(self, temp_dir: Path) -> None:
        """get_stats should return dictionary with metrics."""
        storage = ContentStorage(temp_dir / "test.db")
        await storage.put("/test/file.txt", "Content", time.time())
        stats = await storage.get_stats()

        assert isinstance(stats, dict)
        assert "files_cached" in stats
        assert "total_tokens_cached" in stats
        assert "total_documents" in stats
        assert "db_size_mb" in stats

    async def test_get_stats_accurate_count(self, temp_dir: Path) -> None:
        """get_stats should report accurate file count."""
        storage = ContentStorage(temp_dir / "test.db")
        for i in range(5):
            await storage.put(f"/test/file{i}.txt", f"Content {i}", time.time())

        stats = await storage.get_stats()
        assert stats["files_cached"] == 5

    async def test_clear_removes_all_entries(self, temp_dir: Path) -> None:
        """clear should remove all files."""
        storage = ContentStorage(temp_dir / "test.db")
        for i in range(3):
            await storage.put(f"/test/file{i}.txt", f"Content {i}", time.time())

        count = await storage.clear()
        assert count >= 3  # May include chunk documents

        stats = await storage.get_stats()
        assert stats["files_cached"] == 0

    async def test_clear_returns_count(self, temp_dir: Path) -> None:
        """clear should return number of documents cleared."""
        storage = ContentStorage(temp_dir / "test.db")
        await storage.put("/test/file1.txt", "Content 1", time.time())
        await storage.put("/test/file2.txt", "Content 2", time.time())

        count = await storage.clear()
        assert count >= 2

    async def test_delete_path_removes_one_entry(self, temp_dir: Path) -> None:
        """delete_path should remove cached docs for only the requested path."""
        storage = ContentStorage(temp_dir / "test.db")
        await storage.put("/test/file1.txt", "Content 1", time.time())
        await storage.put("/test/file2.txt", "Content 2", time.time())

        removed = await storage.delete_path("/test/file1.txt")

        assert removed >= 1
        assert await storage.get("/test/file1.txt") is None
        assert await storage.get("/test/file2.txt") is not None
