"""Tests for crash-hardening in VectorStorage (embed/usearch/save paths)."""

from __future__ import annotations

import array
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest

from semantic_cache_mcp.storage.vector import VectorStorage
from tests.constants import TEST_EMBEDDING_DIM


@pytest.fixture
def storage(tmp_path: Path) -> VectorStorage:
    """Create a real VectorStorage with a tmp_path db."""
    s = VectorStorage(tmp_path / "test.db")
    return s


class TestResolveEmbeddingDimGuard:
    """Fix 1: _resolve_embedding rejects wrong-dimension vectors."""

    def test_put_rejects_wrong_embedding_dim(self, storage: VectorStorage) -> None:
        """Passing an embedding whose length != expected dim raises ValueError."""
        correct = array.array("f", [0.0] * TEST_EMBEDDING_DIM)
        storage._resolve_embedding(correct)

        wrong_dim = TEST_EMBEDDING_DIM + 128
        wrong = array.array("f", [0.0] * wrong_dim)
        with pytest.raises(ValueError, match="dimension mismatch"):
            storage._resolve_embedding(wrong)

    def test_resolve_embedding_none_uses_correct_dim(self, storage: VectorStorage) -> None:
        """_resolve_embedding(None) returns a zero-vector of the expected length."""
        with patch(
            "semantic_cache_mcp.core.embeddings.get_embedding_dim",
            return_value=TEST_EMBEDDING_DIM,
        ):
            result = storage._resolve_embedding(None)
        assert isinstance(result, list)
        assert len(result) == TEST_EMBEDDING_DIM
        assert all(v == 0.0 for v in result)

    def test_resolve_embedding_correct_dim_passes(self, storage: VectorStorage) -> None:
        """_resolve_embedding accepts a vector whose dim matches expected."""
        with patch(
            "semantic_cache_mcp.core.embeddings.get_embedding_dim",
            return_value=TEST_EMBEDDING_DIM,
        ):
            vec = array.array("f", [0.1] * TEST_EMBEDDING_DIM)
            result = storage._resolve_embedding(vec)
        assert isinstance(result, list)
        assert len(result) == TEST_EMBEDDING_DIM


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


class TestClearIfModelChangedDimMismatch:
    """Fix 5: clear_if_model_changed detects runtime dimension mismatch."""

    def test_clear_if_model_changed_detects_dim_mismatch(self, storage: VectorStorage) -> None:
        """When index dim != model dim, _reset_collection_sync must be called."""
        with (
            patch.object(
                type(storage._collection),
                "dim",
                new_callable=PropertyMock,
                return_value=TEST_EMBEDDING_DIM,
            ),
            patch.object(VectorStorage, "_reset_collection_sync", return_value=None) as mock_reset,
        ):
            storage.clear_if_model_changed("some-model", TEST_EMBEDDING_DIM * 2)
        mock_reset.assert_called()

    def test_clear_if_model_changed_no_clear_when_dim_matches(self, storage: VectorStorage) -> None:
        """When dims agree, _reset_collection_sync should not be called for dim reasons."""
        with (
            patch.object(
                type(storage._collection),
                "dim",
                new_callable=PropertyMock,
                return_value=TEST_EMBEDDING_DIM,
            ),
            patch.object(VectorStorage, "_reset_collection_sync", return_value=None) as mock_reset,
        ):
            storage.clear_if_model_changed("some-model", TEST_EMBEDDING_DIM)
        mock_reset.assert_not_called()
