"""Per-session tracker: has this client already seen this file's content?

Without this, `unchanged: true` only signals cache freshness — not whether
the calling agent has actually been sent the bytes in this session. The
audit found models always immediately re-read after `unchanged: true`,
defeating the token saving the marker was designed for.

This tracker bridges that gap with a small LRU keyed by
`(session_id, abs_path)`. The read tool consults it to decide whether
to send full content or just a metadata marker; mutations (`edit`,
`write`, `delete`) invalidate the entry so the next read after a change
returns fresh content.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock

# Cap on tracked (session, path) pairs across all sessions. With 256 entries
# the in-memory footprint is negligible (~50KB at typical path lengths), and
# active sessions overwhelmingly re-read the same hot files.
_MAX_ENTRIES = 256

# Fallback session id when FastMCP doesn't expose one. Treats the whole
# process as one "session", which is correct for stdio transport (one
# connection per process) and degrades gracefully otherwise.
_PROC_SENTINEL = "_proc_"


class _ReadSessionTracker:
    """Thread-safe LRU set of (session_id, abs_path) pairs seen by clients."""

    __slots__ = ("_entries", "_lock")

    def __init__(self) -> None:
        self._entries: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def _normalize_session(session_id: str | None) -> str:
        return session_id if session_id else _PROC_SENTINEL

    def seen(self, session_id: str | None, abs_path: str) -> bool:
        """Return True if this client has already been sent `abs_path`.

        Refreshes LRU position on hit.
        """
        key = (self._normalize_session(session_id), abs_path)
        with self._lock:
            if key not in self._entries:
                return False
            self._entries.move_to_end(key)
            return True

    def mark(self, session_id: str | None, abs_path: str) -> None:
        """Record that we've sent `abs_path` to this client."""
        key = (self._normalize_session(session_id), abs_path)
        with self._lock:
            self._entries[key] = None
            self._entries.move_to_end(key)
            while len(self._entries) > _MAX_ENTRIES:
                self._entries.popitem(last=False)

    def invalidate(self, abs_path: str) -> None:
        """Drop every session's entry for `abs_path` after a mutation."""
        with self._lock:
            stale = [k for k in self._entries if k[1] == abs_path]
            for key in stale:
                del self._entries[key]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# Process-wide singleton. The lifespan dict could expose it via context,
# but a module-level singleton is fine: the tracker has no per-cache state
# and resets cleanly on process restart.
_TRACKER = _ReadSessionTracker()


def get_tracker() -> _ReadSessionTracker:
    return _TRACKER
