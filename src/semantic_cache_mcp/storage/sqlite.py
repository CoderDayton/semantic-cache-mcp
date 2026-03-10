"""SQLite storage backend for session metrics persistence."""

from __future__ import annotations

import contextlib
import logging
import queue
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import DB_PATH

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------


class ConnectionPool:
    """Thread-safe SQLite connection pool with WAL mode.

    Attributes:
        db_path: Path to SQLite database
        max_size: Maximum connections in pool
    """

    __slots__ = ("_all_conns", "_available", "_closed", "_lock", "_total", "db_path", "max_size")

    def __init__(self, db_path: Path, max_size: int = 5) -> None:
        self.db_path = db_path
        self.max_size = max_size
        self._available: queue.Queue[sqlite3.Connection] = queue.Queue()
        self._all_conns: set[sqlite3.Connection] = set()
        self._total = 0
        self._lock = threading.Lock()
        self._closed = False

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimized settings."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30,
            check_same_thread=False,
        )

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-8000")  # 8MB
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA wal_autocheckpoint=1000")
        self._all_conns.add(conn)
        return conn

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a connection from the pool."""
        conn: sqlite3.Connection | None = None
        try:
            conn = self._available.get_nowait()
        except queue.Empty:
            with self._lock:
                if self._total < self.max_size:
                    conn = self._create_connection()
                    self._total += 1

            if conn is None:
                try:
                    conn = self._available.get(timeout=10)
                except queue.Empty as e:
                    msg = "Connection pool exhausted"
                    raise RuntimeError(msg) from e

        try:
            yield conn
            conn.commit()
        except Exception:
            with contextlib.suppress(Exception):
                conn.rollback()
            raise
        finally:
            try:
                self._available.put_nowait(conn)
            except queue.Full:
                with contextlib.suppress(Exception):
                    conn.close()

    def close_all(self) -> None:
        """Close all connections — both pooled and in-use. Idempotent."""
        if self._closed:
            return
        self._closed = True

        # Drain the available queue first
        while True:
            try:
                self._available.get_nowait()
            except queue.Empty:
                break

        # Close every connection we ever created (includes in-use ones)
        for conn in self._all_conns:
            with contextlib.suppress(Exception):
                conn.close()
        self._all_conns.clear()


# ---------------------------------------------------------------------------
# Metrics-only SQLite storage
# ---------------------------------------------------------------------------


class SQLiteStorage:
    """SQLite storage for session metrics persistence.

    Only handles session_metrics table — file content storage is handled
    by VectorStorage via simplevecdb.
    """

    __slots__ = ("_pool", "db_path")

    def __init__(self, db_path: Path = DB_PATH, pool_size: int = 5) -> None:
        """Initialize storage with connection pool.

        Args:
            db_path: Path to SQLite database file
            pool_size: Maximum connections in pool (default: 5)
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._pool = ConnectionPool(db_path, max_size=pool_size)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize session metrics table."""
        with self._pool.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS session_metrics (
                    session_id       TEXT PRIMARY KEY,
                    started_at       REAL NOT NULL,
                    ended_at         REAL,
                    tokens_saved     INTEGER NOT NULL DEFAULT 0,
                    tokens_original  INTEGER NOT NULL DEFAULT 0,
                    tokens_returned  INTEGER NOT NULL DEFAULT 0,
                    cache_hits       INTEGER NOT NULL DEFAULT 0,
                    cache_misses     INTEGER NOT NULL DEFAULT 0,
                    files_read       INTEGER NOT NULL DEFAULT 0,
                    files_written    INTEGER NOT NULL DEFAULT 0,
                    files_edited     INTEGER NOT NULL DEFAULT 0,
                    diffs_served     INTEGER NOT NULL DEFAULT 0,
                    tool_calls_json  TEXT NOT NULL DEFAULT '{}'
                ) WITHOUT ROWID;
            """)

    def save_session(self, data: dict[str, object]) -> None:
        """Persist a session metrics snapshot.

        Args:
            data: Dict with session_id, started_at, ended_at, counters, tool_calls_json
        """
        with self._pool.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO session_metrics
                (session_id, started_at, ended_at,
                 tokens_saved, tokens_original, tokens_returned,
                 cache_hits, cache_misses,
                 files_read, files_written, files_edited,
                 diffs_served, tool_calls_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["session_id"],
                    data["started_at"],
                    data["ended_at"],
                    data["tokens_saved"],
                    data["tokens_original"],
                    data["tokens_returned"],
                    data["cache_hits"],
                    data["cache_misses"],
                    data["files_read"],
                    data["files_written"],
                    data["files_edited"],
                    data["diffs_served"],
                    data["tool_calls_json"],
                ),
            )

    def get_lifetime_stats(self) -> dict[str, int]:
        """Aggregate metrics across all completed sessions.

        Returns:
            Dict with total_sessions, sum of each counter
        """
        with self._pool.get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*),
                    COALESCE(SUM(tokens_saved), 0),
                    COALESCE(SUM(tokens_original), 0),
                    COALESCE(SUM(tokens_returned), 0),
                    COALESCE(SUM(cache_hits), 0),
                    COALESCE(SUM(cache_misses), 0),
                    COALESCE(SUM(files_read), 0),
                    COALESCE(SUM(files_written), 0),
                    COALESCE(SUM(files_edited), 0),
                    COALESCE(SUM(diffs_served), 0)
                FROM session_metrics
                WHERE ended_at IS NOT NULL
                """
            ).fetchone()

        return {
            "total_sessions": row[0],
            "tokens_saved": row[1],
            "tokens_original": row[2],
            "tokens_returned": row[3],
            "cache_hits": row[4],
            "cache_misses": row[5],
            "files_read": row[6],
            "files_written": row[7],
            "files_edited": row[8],
            "diffs_served": row[9],
        }
