"""Session-level metrics accumulator.

Thread-safe in-memory counters fed by each tool call, persisted to SQLite on
shutdown. Exposed via the ``stats`` tool as ``session`` (current) and
``lifetime`` (aggregated historical) dicts.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import TYPE_CHECKING

from ..types import (
    BatchEditResult,
    BatchReadResult,
    DiffResult,
    EditResult,
    ReadResult,
    WriteResult,
)

if TYPE_CHECKING:
    from ..storage.sqlite import ConnectionPool

logger = logging.getLogger(__name__)


class SessionMetrics:
    """Thread-safe in-memory counters.

    ``record()`` feeds from tool call results; ``persist()`` flushes to SQLite on shutdown.
    """

    __slots__ = (
        "_pool",
        "_lock",
        "session_id",
        "started_at",
        "tokens_saved",
        "tokens_original",
        "tokens_returned",
        "cache_hits",
        "cache_misses",
        "files_read",
        "files_written",
        "files_edited",
        "diffs_served",
        "tool_calls",
    )

    def __init__(self, pool: ConnectionPool) -> None:
        self._pool = pool
        self._lock = threading.Lock()
        self.session_id = str(uuid.uuid4())
        self.started_at = time.time()
        self.tokens_saved = 0
        self.tokens_original = 0
        self.tokens_returned = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.files_read = 0
        self.files_written = 0
        self.files_edited = 0
        self.diffs_served = 0
        self.tool_calls: dict[str, int] = {}

    def record(self, tool_name: str, result: object) -> None:
        """Record metrics from a tool result. Thread-safe.

        Metrics must stay internally coherent: any saved-token contribution must
        carry a matching original/returned denominator. This means token-savings
        rate metrics are derived only from read flows, where the result models
        expose both original and returned token counts directly or derivably.
        """
        with self._lock:
            self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1

            if result is None:
                return

            if isinstance(result, ReadResult):
                self.tokens_saved += result.tokens_saved
                self.tokens_original += result.tokens_original
                self.tokens_returned += result.tokens_returned
                if result.from_cache:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1
                if result.is_diff:
                    self.diffs_served += 1
                if tool_name == "read":
                    self.files_read += 1
                return

            if isinstance(result, BatchReadResult):
                self.tokens_saved += result.tokens_saved
                self.tokens_original += result.total_tokens + result.tokens_saved
                self.tokens_returned += result.total_tokens
                self.files_read += result.files_read
                for file_result in result.files:
                    if file_result.status == "skipped":
                        continue
                    if file_result.from_cache:
                        self.cache_hits += 1
                    else:
                        self.cache_misses += 1
                    if file_result.status == "diff":
                        self.diffs_served += 1
                return

            if isinstance(result, DiffResult):
                self.cache_hits += sum(1 for v in result.from_cache if v)
                self.cache_misses += sum(1 for v in result.from_cache if not v)
                return

            if isinstance(result, WriteResult):
                if result.from_cache:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1
                if tool_name == "write" and not result.dry_run:
                    self.files_written += 1
                return

            if isinstance(result, EditResult):
                if result.from_cache:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1
                if tool_name == "edit" and not result.dry_run:
                    self.files_edited += 1
                return

            if isinstance(result, BatchEditResult):
                if result.from_cache:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1
                if tool_name == "batch_edit" and not result.dry_run and result.succeeded > 0:
                    self.files_edited += 1

    def snapshot(self) -> dict[str, object]:
        """Thread-safe snapshot of current metrics with computed uptime."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "uptime_s": round(time.time() - self.started_at, 1),
                "tokens_saved": self.tokens_saved,
                "tokens_original": self.tokens_original,
                "tokens_returned": self.tokens_returned,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "files_read": self.files_read,
                "files_written": self.files_written,
                "files_edited": self.files_edited,
                "diffs_served": self.diffs_served,
                "tool_calls": dict(self.tool_calls),
            }

    def persist(self) -> None:
        """Flush session metrics to SQLite."""
        now = time.time()
        with self._lock:
            data: dict[str, object] = {
                "session_id": self.session_id,
                "started_at": self.started_at,
                "ended_at": now,
                "tokens_saved": self.tokens_saved,
                "tokens_original": self.tokens_original,
                "tokens_returned": self.tokens_returned,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "files_read": self.files_read,
                "files_written": self.files_written,
                "files_edited": self.files_edited,
                "diffs_served": self.diffs_served,
                "tool_calls_json": json.dumps(self.tool_calls),
            }
            total_calls = sum(self.tool_calls.values())

        # Use pool directly to avoid importing SQLiteStorage instance
        try:
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
            logger.info(
                f"Session {self.session_id[:8]} persisted: "
                f"{self.tokens_saved} tokens saved, {total_calls} tool calls"
            )
        except Exception as e:
            logger.error(f"Failed to persist session metrics: {e}")
