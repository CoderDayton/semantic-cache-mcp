"""Tests for session metrics accumulator."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pytest

from semantic_cache_mcp.cache.metrics import SessionMetrics
from semantic_cache_mcp.storage.sqlite import SQLiteStorage
from semantic_cache_mcp.types import (
    BatchEditResult,
    BatchReadResult,
    DiffResult,
    EditResult,
    FileReadSummary,
    ReadResult,
    SingleEditOutcome,
    WriteResult,
)


@pytest.fixture
def storage(temp_dir: Path) -> SQLiteStorage:
    """Create a temporary SQLiteStorage for metrics tests."""
    return SQLiteStorage(temp_dir / "metrics_test.db")


@pytest.fixture
def metrics(storage: SQLiteStorage) -> SessionMetrics:
    """Create a SessionMetrics instance."""
    return SessionMetrics(storage._pool)


class TestInitialState:
    """Verify zero-state after construction."""

    def test_initial_counters_zero(self, metrics: SessionMetrics) -> None:
        assert metrics.tokens_saved == 0
        assert metrics.tokens_original == 0
        assert metrics.tokens_returned == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.files_read == 0
        assert metrics.files_written == 0
        assert metrics.files_edited == 0
        assert metrics.diffs_served == 0
        assert metrics.tool_calls == {}

    def test_valid_session_id(self, metrics: SessionMetrics) -> None:
        parsed = uuid.UUID(metrics.session_id)
        assert parsed.version == 4

    def test_started_at_set(self, metrics: SessionMetrics) -> None:
        assert metrics.started_at > 0


class TestRecordRead:
    """Test recording ReadResult metrics."""

    def test_cache_hit_diff(self, metrics: SessionMetrics) -> None:
        result = ReadResult(
            content="diff content",
            from_cache=True,
            is_diff=True,
            tokens_original=500,
            tokens_returned=50,
            tokens_saved=450,
            truncated=False,
            compression_ratio=0.1,
        )
        metrics.record("read", result)

        assert metrics.tokens_saved == 450
        assert metrics.tokens_original == 500
        assert metrics.tokens_returned == 50
        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 0
        assert metrics.diffs_served == 1
        assert metrics.files_read == 1
        assert metrics.tool_calls == {"read": 1}

    def test_cache_miss(self, metrics: SessionMetrics) -> None:
        result = ReadResult(
            content="full content",
            from_cache=False,
            is_diff=False,
            tokens_original=200,
            tokens_returned=200,
            tokens_saved=0,
            truncated=False,
            compression_ratio=1.0,
        )
        metrics.record("read", result)

        assert metrics.cache_misses == 1
        assert metrics.cache_hits == 0
        assert metrics.diffs_served == 0

    def test_multiple_reads_accumulate(self, metrics: SessionMetrics) -> None:
        for i in range(3):
            result = ReadResult(
                content=f"content {i}",
                from_cache=True,
                is_diff=False,
                tokens_original=100,
                tokens_returned=100,
                tokens_saved=50,
                truncated=False,
                compression_ratio=0.5,
            )
            metrics.record("read", result)

        assert metrics.tokens_saved == 150
        assert metrics.files_read == 3
        assert metrics.tool_calls["read"] == 3


class TestRecordBatchRead:
    """Test recording BatchReadResult metrics."""

    def test_batch_read_files_count(self, metrics: SessionMetrics) -> None:
        result = BatchReadResult(
            files=[
                FileReadSummary(path="/a.py", tokens=100, status="full", from_cache=True),
                FileReadSummary(path="/b.py", tokens=200, status="full", from_cache=False),
            ],
            contents={"/a.py": "a", "/b.py": "b"},
            total_tokens=300,
            tokens_saved=500,
            files_read=2,
            files_skipped=0,
        )
        metrics.record("batch_read", result)

        assert metrics.files_read == 2
        assert metrics.tokens_saved == 500
        assert metrics.tool_calls["batch_read"] == 1


class TestRecordWrite:
    """Test recording WriteResult metrics."""

    def test_write_increments_files_written(self, metrics: SessionMetrics) -> None:
        result = WriteResult(
            path="/test.py",
            bytes_written=100,
            tokens_written=25,
            created=True,
            diff_content=None,
            diff_stats=None,
            tokens_saved=0,
            content_hash="abc123",
            from_cache=False,
        )
        metrics.record("write", result)

        assert metrics.files_written == 1
        assert metrics.tool_calls["write"] == 1


class TestRecordEdit:
    """Test recording EditResult and BatchEditResult metrics."""

    def test_edit_increments_files_edited(self, metrics: SessionMetrics) -> None:
        result = EditResult(
            path="/test.py",
            matches_found=1,
            replacements_made=1,
            line_numbers=[10],
            diff_content="@@ -10 +10 @@",
            diff_stats={"insertions": 1, "deletions": 1},
            tokens_saved=100,
            content_hash="abc",
            from_cache=True,
        )
        metrics.record("edit", result)

        assert metrics.files_edited == 1
        assert metrics.cache_hits == 1
        assert metrics.tokens_saved == 100

    def test_batch_edit_increments_files_edited(self, metrics: SessionMetrics) -> None:
        result = BatchEditResult(
            path="/test.py",
            outcomes=[
                SingleEditOutcome(
                    old_string="old", new_string="new", success=True, line_number=5, error=None
                ),
            ],
            succeeded=1,
            failed=0,
            diff_content="diff",
            diff_stats={"insertions": 1},
            tokens_saved=200,
            content_hash="def",
            from_cache=True,
        )
        metrics.record("batch_edit", result)

        assert metrics.files_edited == 1
        assert metrics.tokens_saved == 200


class TestRecordDiff:
    """Test recording DiffResult metrics."""

    def test_diff_tuple_from_cache(self, metrics: SessionMetrics) -> None:
        result = DiffResult(
            path1="/a.py",
            path2="/b.py",
            diff_content="diff",
            diff_stats={"insertions": 1},
            tokens_saved=300,
            similarity=0.85,
            from_cache=(True, False),
        )
        metrics.record("diff", result)

        assert metrics.tokens_saved == 300
        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 1


class TestRecordNone:
    """Test that record(None) is safe for clear tool."""

    def test_record_none_safe(self, metrics: SessionMetrics) -> None:
        metrics.record("clear", None)
        assert metrics.tool_calls["clear"] == 1
        assert metrics.tokens_saved == 0

    def test_record_none_multiple(self, metrics: SessionMetrics) -> None:
        metrics.record("clear", None)
        metrics.record("clear", None)
        assert metrics.tool_calls["clear"] == 2


class TestSnapshot:
    """Test snapshot returns correct data."""

    def test_snapshot_structure(self, metrics: SessionMetrics) -> None:
        metrics.record(
            "read",
            ReadResult(
                content="x",
                from_cache=True,
                is_diff=False,
                tokens_original=100,
                tokens_returned=100,
                tokens_saved=50,
                truncated=False,
                compression_ratio=0.5,
            ),
        )
        snap = metrics.snapshot()

        assert snap["session_id"] == metrics.session_id
        assert isinstance(snap["uptime_s"], float)
        assert snap["uptime_s"] >= 0
        assert snap["tokens_saved"] == 50
        assert snap["tool_calls"] == {"read": 1}


class TestThreadSafety:
    """Test concurrent access doesn't corrupt counters."""

    def test_concurrent_records(self, metrics: SessionMetrics) -> None:
        num_threads = 10
        records_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def worker() -> None:
            barrier.wait()
            for _ in range(records_per_thread):
                result = ReadResult(
                    content="x",
                    from_cache=True,
                    is_diff=False,
                    tokens_original=10,
                    tokens_returned=10,
                    tokens_saved=5,
                    truncated=False,
                    compression_ratio=0.5,
                )
                metrics.record("read", result)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = num_threads * records_per_thread
        assert metrics.tool_calls["read"] == expected
        assert metrics.tokens_saved == expected * 5
        assert metrics.cache_hits == expected
        assert metrics.files_read == expected


class TestPersistAndLifetime:
    """Test persist writes to SQLite and lifetime aggregates."""

    def test_persist_and_lifetime(self, storage: SQLiteStorage) -> None:
        m = SessionMetrics(storage._pool)
        m.record(
            "read",
            ReadResult(
                content="x",
                from_cache=True,
                is_diff=True,
                tokens_original=100,
                tokens_returned=10,
                tokens_saved=90,
                truncated=False,
                compression_ratio=0.1,
            ),
        )
        m.persist()

        lifetime = storage.get_lifetime_stats()
        assert lifetime["total_sessions"] == 1
        assert lifetime["tokens_saved"] == 90
        assert lifetime["tokens_original"] == 100
        assert lifetime["tokens_returned"] == 10
        assert lifetime["cache_hits"] == 1
        assert lifetime["diffs_served"] == 1
        assert lifetime["files_read"] == 1

    def test_multiple_sessions(self, storage: SQLiteStorage) -> None:
        for i in range(2):
            m = SessionMetrics(storage._pool)
            m.record(
                "read",
                ReadResult(
                    content="x",
                    from_cache=True,
                    is_diff=False,
                    tokens_original=100,
                    tokens_returned=50,
                    tokens_saved=50,
                    truncated=False,
                    compression_ratio=0.5,
                ),
            )
            m.record(
                "write",
                WriteResult(
                    path=f"/f{i}.py",
                    bytes_written=10,
                    tokens_written=5,
                    created=True,
                    diff_content=None,
                    diff_stats=None,
                    tokens_saved=0,
                    content_hash="h",
                    from_cache=False,
                ),
            )
            m.persist()

        lifetime = storage.get_lifetime_stats()
        assert lifetime["total_sessions"] == 2
        assert lifetime["tokens_saved"] == 100  # 50 * 2
        assert lifetime["files_read"] == 2
        assert lifetime["files_written"] == 2
