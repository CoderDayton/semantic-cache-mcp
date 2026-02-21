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

if TYPE_CHECKING:
    from ..storage.sqlite import ConnectionPool

logger = logging.getLogger(__name__)


class SessionMetrics:
    """Thread-safe session metrics accumulator.

    Each tool call feeds ``record(tool_name, result)`` after success.
    ``persist()`` writes the final snapshot to SQLite for lifetime aggregation.
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

        Dispatches via ``hasattr`` on the result object — no per-type methods,
        no decorators. ``result=None`` is safe (e.g. clear tool).
        """
        with self._lock:
            self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1

            if result is None:
                return

            # tokens_saved — present on ReadResult, WriteResult, EditResult,
            # BatchReadResult, BatchEditResult, DiffResult
            if hasattr(result, "tokens_saved"):
                self.tokens_saved += result.tokens_saved

            # tokens_original / tokens_returned — ReadResult only
            if hasattr(result, "tokens_original"):
                self.tokens_original += result.tokens_original
            if hasattr(result, "tokens_returned"):
                self.tokens_returned += result.tokens_returned

            # Cache hit/miss — ReadResult, WriteResult, EditResult, BatchEditResult
            if hasattr(result, "from_cache"):
                val = result.from_cache
                if isinstance(val, bool):
                    if val:
                        self.cache_hits += 1
                    else:
                        self.cache_misses += 1
                elif isinstance(val, tuple):
                    # DiffResult.from_cache is tuple[bool, bool]
                    self.cache_hits += sum(1 for v in val if v)
                    self.cache_misses += sum(1 for v in val if not v)

            # Diff tracking — ReadResult
            if hasattr(result, "is_diff") and result.is_diff:
                self.diffs_served += 1

            # File counts by tool
            if tool_name == "read":
                self.files_read += 1
            elif tool_name == "batch_read" and hasattr(result, "files_read"):
                self.files_read += result.files_read
            elif tool_name == "write":
                self.files_written += 1
            elif tool_name in ("edit", "batch_edit"):
                self.files_edited += 1

    def snapshot(self) -> dict[str, object]:
        """Return a thread-safe copy of current metrics with computed uptime."""
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
        """Write session metrics to SQLite. Called on server shutdown."""
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
