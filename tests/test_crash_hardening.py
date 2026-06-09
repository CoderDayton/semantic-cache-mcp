"""Tests for crash-hardening in VectorStorage (embed/usearch/save paths)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from semantic_cache_mcp.storage.vector import VectorStorage


@pytest.fixture
def storage(tmp_path: Path) -> VectorStorage:
    """Create a real VectorStorage with a tmp_path db."""
    s = VectorStorage(tmp_path / "test.db")
    return s


class TestSaveSkipsWhenClosed:
    """Fix 3: save() silently returns when _closed is True."""

    def test_save_skips_when_closed(self, storage: VectorStorage) -> None:
        """After close(), save() must not raise or touch the index."""
        storage.close()
        # Should be a no-op, not an exception.
        storage.save()


class TestGetStatsHandlesDeletedDb:
    """Fix 6: get_stats() returns db_size_mb=0 when the file is gone."""

    @pytest.mark.asyncio
    async def test_get_stats_handles_deleted_db_file(self, storage: VectorStorage) -> None:
        """Deleting the db file must not crash get_stats()."""
        db_path = storage._db_path
        if db_path.exists():
            db_path.unlink()
        stats = await storage.get_stats()
        assert stats["db_size_mb"] == 0
        storage.close()


class TestRebindExecutorOwnershipTransfer:
    """rebind_executor must shut down the previously-owned executor and flip
    ownership so close() does not later shut down the caller's replacement.
    """

    def test_rebind_shuts_down_previously_owned_executor(self, tmp_path: Path) -> None:
        """When VectorStorage owned the executor, rebind shuts the old one down."""
        vs = VectorStorage(tmp_path / "owned.db")
        assert vs._owns_executor is True
        old = vs._io_executor

        replacement = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replace")
        try:
            vs.rebind_executor(replacement)
            # Old owned executor must be shut down (it rejects new submissions).
            with pytest.raises(RuntimeError):
                old.submit(lambda: None)
            # Ownership transfers OUT — caller now owns the replacement.
            assert vs._owns_executor is False
            assert vs._io_executor is replacement
            # Replacement must still be usable.
            assert replacement.submit(lambda: 42).result(timeout=1) == 42
        finally:
            vs.close()
            replacement.shutdown(wait=False)

    def test_close_after_rebind_does_not_shutdown_replacement(self, tmp_path: Path) -> None:
        """close() must NOT shut down a caller-owned replacement executor."""
        vs = VectorStorage(tmp_path / "owned2.db")
        replacement = ThreadPoolExecutor(max_workers=1, thread_name_prefix="keep")
        try:
            vs.rebind_executor(replacement)
            vs.close()
            # Replacement should still accept work after VectorStorage.close().
            assert replacement.submit(lambda: "alive").result(timeout=1) == "alive"
        finally:
            replacement.shutdown(wait=False)

    def test_rebind_with_injected_executor_does_not_shutdown(self, tmp_path: Path) -> None:
        """If VectorStorage never owned its executor, rebind must not touch
        the originally-injected one (the caller still owns it)."""
        injected = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inj")
        replacement = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rep")
        try:
            vs = VectorStorage(tmp_path / "inj.db", executor=injected)
            assert vs._owns_executor is False
            vs.rebind_executor(replacement)
            # Original injected executor must still be usable — we never owned it.
            assert injected.submit(lambda: "ok").result(timeout=1) == "ok"
            assert vs._owns_executor is False
            vs.close()
        finally:
            injected.shutdown(wait=False)
            replacement.shutdown(wait=False)
