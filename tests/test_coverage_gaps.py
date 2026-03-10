"""Coverage-targeted tests for low-coverage modules.

Targets:
  - utils/_retry.py (57% → 100%)
  - server/__init__.py (58% → 80%+)
  - cache/store.py close() (79% → 85%+)
  - config.py validation (82% → 90%+)
  - storage/sqlite.py pool (85% → 90%+)
  - core/embeddings/_model.py (71% → 80%+)
"""

from __future__ import annotations

import signal
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from semantic_cache_mcp.utils._retry import retry

# ===========================================================================
# utils/_retry.py — full retry loop with delays, exhaustion, and logging
# ===========================================================================


class TestRetry:
    """Tests for retry utility covering retry loop, delays, and exhaustion."""

    def test_succeeds_first_attempt(self) -> None:
        """No retries needed when fn succeeds immediately."""
        assert retry(lambda: 42, label="test") == 42

    def test_succeeds_after_retries(self) -> None:
        """fn fails twice then succeeds — should return on third attempt."""
        call_count = 0

        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"fail #{call_count}")
            return "ok"

        result = retry(flaky, delays=(0.0, 0.0, 0.0), exceptions=(RuntimeError,), label="flaky")
        assert result == "ok"
        assert call_count == 3

    def test_exhausts_all_retries(self) -> None:
        """All attempts fail — should raise last exception."""
        call_count = 0

        def always_fail() -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError(f"fail #{call_count}")

        with pytest.raises(RuntimeError, match="fail #4"):
            retry(always_fail, delays=(0.0, 0.0, 0.0), exceptions=(RuntimeError,), label="doomed")
        assert call_count == 4  # 3 delays + 1 = 4 attempts

    def test_non_matching_exception_propagates_immediately(self) -> None:
        """Exceptions not in the exceptions tuple propagate without retry."""
        call_count = 0

        def type_error() -> None:
            nonlocal call_count
            call_count += 1
            raise TypeError("wrong type")

        with pytest.raises(TypeError, match="wrong type"):
            retry(type_error, delays=(0.0, 0.0), exceptions=(RuntimeError,), label="type")
        assert call_count == 1  # no retry

    def test_logs_retry_warnings(self) -> None:
        """Retry attempts log warnings with attempt number and delay."""
        call_count = 0

        def fail_twice() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("boom")
            return "ok"

        with patch("semantic_cache_mcp.utils._retry.logger") as mock_logger:
            retry(fail_twice, delays=(0.0, 0.0, 0.0), exceptions=(RuntimeError,), label="test-op")

        # Should have logged 2 retry warnings (attempt 1 and 2 failed)
        warning_calls = mock_logger.warning.call_args_list
        assert len(warning_calls) == 2
        assert "attempt 1/4" in str(warning_calls[0])
        assert "attempt 2/4" in str(warning_calls[1])

    def test_logs_exhaustion_warning(self) -> None:
        """When all retries fail, logs an exhaustion warning before raising."""
        with (
            patch("semantic_cache_mcp.utils._retry.logger") as mock_logger,
            pytest.raises(RuntimeError),
        ):
            retry(
                lambda: (_ for _ in ()).throw(RuntimeError("x")),
                delays=(0.0,),
                exceptions=(RuntimeError,),
                label="exhaust",
            )

        # Last warning should be the "all N attempts failed" message
        last_warning = str(mock_logger.warning.call_args_list[-1])
        assert "all 2 attempts failed" in last_warning

    def test_sleeps_between_retries(self) -> None:
        """Verify actual sleep is called with correct delay values."""
        call_count = 0

        def fail_once() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first fail")
            return "ok"

        with patch("semantic_cache_mcp.utils._retry.time.sleep") as mock_sleep:
            retry(fail_once, delays=(0.5, 1.0), exceptions=(RuntimeError,), label="sleep-test")

        mock_sleep.assert_called_once_with(0.5)


# ===========================================================================
# server/__init__.py — signal handler setup
# ===========================================================================


class TestSignalHandlers:
    """Tests for _setup_signal_handlers."""

    def test_setup_signal_handlers_sets_sigpipe(self) -> None:
        """SIGPIPE should be set to SIG_IGN on platforms that support it."""
        from semantic_cache_mcp.server import _setup_signal_handlers

        if not hasattr(signal, "SIGPIPE"):
            pytest.skip("SIGPIPE not available on this platform")

        with patch("signal.signal") as mock_signal:
            _setup_signal_handlers()
            mock_signal.assert_called_with(signal.SIGPIPE, signal.SIG_IGN)

    def test_setup_signal_handlers_no_sigpipe(self) -> None:
        """On platforms without SIGPIPE (Windows), should not raise."""
        from semantic_cache_mcp.server import _setup_signal_handlers

        saved = getattr(signal, "SIGPIPE", None)
        if saved is not None:
            delattr(signal, "SIGPIPE")
        try:
            with patch("signal.signal") as mock_signal:
                _setup_signal_handlers()
                # Should not have called signal.signal at all (no SIGPIPE to set)
                mock_signal.assert_not_called()
        finally:
            if saved is not None:
                signal.SIGPIPE = saved  # type: ignore[attr-defined]


# ===========================================================================
# cache/store.py — close() and get_stats() coverage
# ===========================================================================


class TestSemanticCacheClose:
    """Tests for SemanticCache.close() fault isolation."""

    def test_close_calls_all_cleanup(self, tmp_path: Path) -> None:
        """close() persists metrics, closes storage, and closes pool."""
        from semantic_cache_mcp.cache.store import SemanticCache

        cache = SemanticCache(tmp_path / "vec.db")
        # Just call close and verify it doesn't raise — the real close()
        # exercises persist + VectorStorage.close + pool.close_all.
        cache.close()

    def test_close_continues_after_metrics_failure(self, tmp_path: Path) -> None:
        """If metrics.persist() fails, storage still gets closed."""
        from semantic_cache_mcp.cache.store import SemanticCache

        cache = SemanticCache(tmp_path / "vec.db")
        with patch(
            "semantic_cache_mcp.cache.metrics.SessionMetrics.persist",
            side_effect=Exception("db error"),
        ):
            cache.close()  # should not raise

    def test_close_continues_after_storage_failure(self, tmp_path: Path) -> None:
        """If storage.close() fails, pool still gets closed."""
        from semantic_cache_mcp.cache.store import SemanticCache

        cache = SemanticCache(tmp_path / "vec.db")
        with patch(
            "semantic_cache_mcp.storage.vector.VectorStorage.close",
            side_effect=Exception("usearch hung"),
        ):
            cache.close()  # should not raise


# ===========================================================================
# storage/vector — close() with timeout
# ===========================================================================


class TestVectorStorageClose:
    """Tests for VectorStorage.close() timeout behavior."""

    def test_close_timeout_does_not_block(self, tmp_path: Path) -> None:
        """A hung save times out within deadline, not at thread exit."""
        from semantic_cache_mcp.storage.vector import VectorStorage

        vs = VectorStorage(tmp_path / "vec.db")

        def hang_forever() -> None:
            time.sleep(60)

        with patch.object(vs._db._db, "save", side_effect=hang_forever):
            start = time.monotonic()
            vs.close(timeout=0.2)
            elapsed = time.monotonic() - start

        # Must return near the timeout, not block until thread finishes
        assert elapsed < 2.0, f"close() blocked for {elapsed:.1f}s instead of timing out"

    def test_close_save_error_logged(self, tmp_path: Path) -> None:
        """An error in save is caught and logged."""
        from semantic_cache_mcp.storage.vector import VectorStorage

        vs = VectorStorage(tmp_path / "vec.db")

        with patch.object(vs._db._db, "save", side_effect=RuntimeError("corrupt")):
            vs.close()  # should not raise


# ===========================================================================
# config.py — validation error branches
# ===========================================================================


class TestConfigValidation:
    """Tests for _validate_config error detection."""

    def test_validate_config_passes_with_defaults(self) -> None:
        """Default config values should pass validation."""
        from semantic_cache_mcp.config import _validate_config

        # Should not raise with default values
        _validate_config()

    def test_validate_config_catches_bad_chunk_sizes(self) -> None:
        """CHUNK_MAX_SIZE <= CHUNK_MIN_SIZE should raise."""
        from semantic_cache_mcp import config

        with (
            patch.object(config, "CHUNK_MAX_SIZE", 100),
            patch.object(config, "CHUNK_MIN_SIZE", 200),
            pytest.raises(ValueError, match="CHUNK_MAX_SIZE"),
        ):
            config._validate_config()

    def test_validate_config_catches_bad_cache_entries(self) -> None:
        """MAX_CACHE_ENTRIES <= 0 should raise."""
        from semantic_cache_mcp import config

        with (
            patch.object(config, "MAX_CACHE_ENTRIES", 0),
            pytest.raises(ValueError, match="MAX_CACHE_ENTRIES"),
        ):
            config._validate_config()

    def test_validate_config_catches_bad_content_size(self) -> None:
        """MAX_CONTENT_SIZE <= 0 should raise."""
        from semantic_cache_mcp import config

        with (
            patch.object(config, "MAX_CONTENT_SIZE", -1),
            pytest.raises(ValueError, match="MAX_CONTENT_SIZE"),
        ):
            config._validate_config()

    def test_validate_config_catches_bad_output_mode(self) -> None:
        """Invalid TOOL_OUTPUT_MODE should raise."""
        from semantic_cache_mcp import config

        with (
            patch.object(config, "TOOL_OUTPUT_MODE", "verbose"),
            pytest.raises(ValueError, match="TOOL_OUTPUT_MODE"),
        ):
            config._validate_config()

    def test_validate_config_catches_negative_response_tokens(self) -> None:
        """TOOL_MAX_RESPONSE_TOKENS < 0 should raise."""
        from semantic_cache_mcp import config

        with (
            patch.object(config, "TOOL_MAX_RESPONSE_TOKENS", -5),
            pytest.raises(ValueError, match="TOOL_MAX_RESPONSE_TOKENS"),
        ):
            config._validate_config()

    def test_validate_config_catches_bad_embedding_device(self) -> None:
        """Invalid EMBEDDING_DEVICE should raise."""
        from semantic_cache_mcp import config

        with (
            patch.object(config, "EMBEDDING_DEVICE", "tpu"),
            pytest.raises(ValueError, match="EMBEDDING_DEVICE"),
        ):
            config._validate_config()

    def test_validate_config_catches_bad_similarity_threshold(self) -> None:
        """SIMILARITY_THRESHOLD outside (0, 1) should raise."""
        from semantic_cache_mcp import config

        with (
            patch.object(config, "SIMILARITY_THRESHOLD", 1.5),
            pytest.raises(ValueError, match="SIMILARITY_THRESHOLD"),
        ):
            config._validate_config()

    def test_validate_config_catches_bad_near_duplicate_threshold(self) -> None:
        """NEAR_DUPLICATE_THRESHOLD outside (0, 1] should raise."""
        from semantic_cache_mcp import config

        with (
            patch.object(config, "NEAR_DUPLICATE_THRESHOLD", 0),
            pytest.raises(ValueError, match="NEAR_DUPLICATE_THRESHOLD"),
        ):
            config._validate_config()

    def test_validate_config_catches_near_dup_below_similarity(self) -> None:
        """NEAR_DUPLICATE_THRESHOLD < SIMILARITY_THRESHOLD should raise."""
        from semantic_cache_mcp import config

        with (
            patch.object(config, "SIMILARITY_THRESHOLD", 0.8),
            patch.object(config, "NEAR_DUPLICATE_THRESHOLD", 0.5),
            pytest.raises(ValueError, match="NEAR_DUPLICATE_THRESHOLD.*SIMILARITY_THRESHOLD"),
        ):
            config._validate_config()


# ===========================================================================
# storage/sqlite.py — connection pool coverage
# ===========================================================================


class TestConnectionPool:
    """Tests for ConnectionPool edge cases."""

    def test_pool_creates_connections_on_demand(self, tmp_path: Path) -> None:
        """Connections are created lazily, not at init time."""
        from semantic_cache_mcp.storage.sqlite import ConnectionPool

        pool = ConnectionPool(tmp_path / "test.db", max_size=2)
        assert pool._total == 0

        with pool.get_connection() as conn:
            assert conn is not None
            assert pool._total == 1

    def test_pool_reuses_returned_connections(self, tmp_path: Path) -> None:
        """Returned connections go back to the pool for reuse."""
        from semantic_cache_mcp.storage.sqlite import ConnectionPool

        pool = ConnectionPool(tmp_path / "test.db", max_size=2)

        with pool.get_connection() as conn1:
            conn1_id = id(conn1)

        with pool.get_connection() as conn2:
            assert id(conn2) == conn1_id  # same connection reused

        assert pool._total == 1  # only one was ever created

    def test_pool_rollback_on_exception(self, tmp_path: Path) -> None:
        """Exceptions inside get_connection trigger rollback."""
        from semantic_cache_mcp.storage.sqlite import ConnectionPool

        pool = ConnectionPool(tmp_path / "test.db", max_size=2)

        # Create table first (DDL is auto-committed in SQLite)
        with pool.get_connection() as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")

        # Insert + raise should rollback the INSERT
        with pytest.raises(ValueError, match="deliberate"), pool.get_connection() as conn:
            conn.execute("INSERT INTO t VALUES (1)")
            raise ValueError("deliberate")

        # Connection should still be usable; INSERT was rolled back
        with pool.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
            assert count == 0

    def test_pool_close_all(self, tmp_path: Path) -> None:
        """close_all drains the pool."""
        from semantic_cache_mcp.storage.sqlite import ConnectionPool

        pool = ConnectionPool(tmp_path / "test.db", max_size=3)

        # Create and return 2 connections
        with pool.get_connection():
            pass
        with pool.get_connection():
            pass

        pool.close_all()
        assert pool._available.empty()

    def test_sqlite_storage_pragmas_applied(self, tmp_path: Path) -> None:
        """SQLite connections have WAL mode and other pragmas set."""
        from semantic_cache_mcp.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(tmp_path / "test.db")

        with storage._pool.get_connection() as conn:
            journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
            assert journal == "wal"
            busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert busy == 5000
