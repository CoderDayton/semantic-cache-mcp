"""SemanticCache — orchestration facade over VectorStorage and SQLite metrics."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections import OrderedDict
from concurrent.futures import Executor
from pathlib import Path
from typing import Any

from ..config import CACHE_DIR, TOOL_TIMEOUT
from ..core import count_tokens
from ..logger import log_marker
from ..storage import SQLiteStorage, VectorStorage
from ..storage.vector import VECDB_PATH
from ..types import CacheEntry
from ..utils import DetachedExecutor
from .metrics import SessionMetrics

logger = logging.getLogger(__name__)


def _get_rss_mb() -> float | None:
    """Cross-platform resident set size in MB. Returns None on failure."""
    try:
        if sys.platform == "linux":
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return round(int(line.split()[1]) / 1024, 1)
            return None
        if sys.platform == "darwin":
            import resource  # noqa: PLC0415

            # macOS ru_maxrss is in bytes
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1)
        if sys.platform == "win32":
            import ctypes  # noqa: PLC0415
            import ctypes.wintypes  # noqa: PLC0415

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = ProcessMemoryCounters()
            pmc.cb = ctypes.sizeof(pmc)
            handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[union-attr]
            if ctypes.windll.kernel32.K32GetProcessMemoryInfo(  # type: ignore[union-attr]
                handle, ctypes.byref(pmc), pmc.cb
            ):
                return round(pmc.WorkingSetSize / (1024 * 1024), 1)
            return None
    except Exception:
        return None
    return None


class SemanticCache:
    """Facade over VectorStorage (simplevecdb) and SQLite metrics."""

    __slots__ = (
        "_storage",
        "_metrics_storage",
        "_metrics",
        "_closed",
        "_shutting_down",
        "_inflight",
        "_drained",
        "_io_executor",
        "_stale_paths",
        "_search_cache",
    )

    # Grace period for in-flight operations to finish during shutdown.
    _DRAIN_TIMEOUT: float = 8.0

    def __init__(self, db_path: Path = VECDB_PATH) -> None:
        # Single-thread executor shared by ALL blocking operations:
        # file I/O and vecdb index saves.
        # MUST be single-threaded — usearch is not safe under concurrent
        # access from different threads (heap corruption / segfaults).
        # Passed to VectorStorage so simplevecdb's own operations
        # (add_texts, keyword_search, etc.) also serialize on this thread
        # via the AsyncVectorCollection wrapper VectorStorage builds.
        self._io_executor: Executor = DetachedExecutor(thread_name_prefix="semantic-cache-io")
        self._storage = VectorStorage(db_path, executor=self._io_executor)
        metrics_db = CACHE_DIR / "metrics.db"
        self._metrics_storage = SQLiteStorage(metrics_db)
        self._metrics = SessionMetrics(self._metrics_storage._pool)
        self._closed = False
        self._shutting_down = False
        self._inflight = 0
        self._drained: asyncio.Event = asyncio.Event()
        self._drained.set()  # starts drained (no inflight ops)
        self._stale_paths: set[str] = set()
        # In-session search-result cache. Bounded LRU keyed on
        # (query, k, directory). Bumped on every write so callers see fresh
        # results once any file changes.
        self._search_cache: OrderedDict[tuple[str, int, str | None], Any] = OrderedDict()

    def reset_executor(self) -> None:
        """Replace the IO executor with a fresh one after a timeout/hang.

        Abandons the stuck thread (it will be GC'd when its task completes or
        the process exits) and creates a new single-threaded executor. All
        references on VectorStorage and simplevecdb are updated atomically.
        """
        logger.warning("Resetting IO executor — previous thread may be stuck")
        old = self._io_executor
        # Don't wait for the old executor — the current call may be wedged in
        # a blocking C extension or a kernel I/O wait and cannot be cancelled.
        old.shutdown(wait=False, cancel_futures=True)

        new_executor: Executor = DetachedExecutor(thread_name_prefix="semantic-cache-io")
        self._io_executor = new_executor
        self._storage.rebind_executor(new_executor)
        logger.debug("IO executor replaced with fresh instance")

    @property
    def metrics(self) -> SessionMetrics:
        """Current session metrics accumulator."""
        return self._metrics

    def request_shutdown(self) -> None:
        """Signal that shutdown has been requested. New operations will be rejected."""
        self._shutting_down = True

    def begin_operation(self) -> bool:
        """Mark the start of an in-flight operation.

        Returns False if shutdown is in progress (caller should bail out).
        No lock needed: asyncio is cooperative and there is no await between
        the guard check and the counter increment.
        """
        if self._shutting_down:
            return False
        self._inflight += 1
        self._drained.clear()
        return True

    def end_operation(self) -> None:
        """Mark the end of an in-flight operation."""
        self._inflight = max(0, self._inflight - 1)
        if self._inflight == 0:
            self._drained.set()

    async def async_close(self) -> None:
        """Graceful shutdown: drain in-flight ops, persist metrics, close backends.

        Called from the lifespan finally block. Waits up to _DRAIN_TIMEOUT
        seconds for in-flight operations to finish before forcing close.
        Idempotent — safe to call multiple times.
        """
        if self._closed:
            return
        self._shutting_down = True

        # Wait for in-flight operations to drain.
        # Catch CancelledError so cleanup proceeds even if our task is cancelled
        # during asyncio.run()'s shutdown (after loop.stop from signal handler).
        if self._inflight > 0:
            logger.debug(f"Waiting for {self._inflight} in-flight operation(s) to finish...")
            try:
                await asyncio.wait_for(self._drained.wait(), timeout=self._DRAIN_TIMEOUT)
                logger.debug("All in-flight operations drained")
            except TimeoutError:
                logger.warning(
                    f"Drain timeout ({self._DRAIN_TIMEOUT}s) expired with "
                    f"{self._inflight} operation(s) still running — forcing close"
                )
            except asyncio.CancelledError:
                logger.warning(
                    f"Drain interrupted by cancellation with "
                    f"{self._inflight} operation(s) still running — forcing close"
                )

        self._closed = True

        try:
            self._metrics.persist()
        except Exception as e:
            logger.warning(f"Failed to persist metrics on close: {e}")

        try:
            self._storage.close()
        except Exception as e:
            logger.warning(f"Failed to close VectorStorage: {e}")

        try:
            self._metrics_storage._pool.close_all()
        except Exception as e:
            logger.warning(f"Failed to close metrics pool: {e}")

        self._io_executor.shutdown(wait=False)

        # Remove crash sentinel — signals clean shutdown.
        VectorStorage._remove_sentinel()

    def close(self) -> None:
        """Synchronous close fallback (no drain wait).

        Prefer async_close() in async contexts. This exists for atexit
        and signal handler safety.
        """
        if self._closed:
            return
        self._shutting_down = True
        self._closed = True

        try:
            self._metrics.persist()
        except Exception as e:
            logger.warning(f"Failed to persist metrics on close: {e}")

        try:
            self._storage.close()
        except Exception as e:
            logger.warning(f"Failed to close VectorStorage: {e}")

        try:
            self._metrics_storage._pool.close_all()
        except Exception as e:
            logger.warning(f"Failed to close metrics pool: {e}")

        self._io_executor.shutdown(wait=False)

        # Remove crash sentinel — signals clean shutdown.
        VectorStorage._remove_sentinel()

    # -------------------------------------------------------------------------
    # Delegated operations
    # -------------------------------------------------------------------------

    async def get(self, path: str) -> CacheEntry | None:
        if path in self._stale_paths:
            logger.debug(f"Treating stale cache entry as miss: {path}")
            return None
        entry = await self._storage.get(path)
        if entry:
            logger.debug(f"Cache hit: {path}")
        return entry

    def mark_stale(self, path: str) -> None:
        self._stale_paths.add(path)

    def clear_stale(self, path: str) -> None:
        self._stale_paths.discard(path)

    def is_stale(self, path: str) -> bool:
        return path in self._stale_paths

    def _compute_refresh_timeout(self) -> float:
        """Timeout for refresh_path(): a single chunk + store write, no embedding."""
        return min(max(1.0, TOOL_TIMEOUT * 0.1), 2.0)

    async def put(
        self,
        path: str,
        content: str,
        mtime: float,
    ) -> None:
        tokens = count_tokens(content)
        started = time.perf_counter()
        log_marker(
            logger,
            "cache.put.begin",
            path=path,
            tokens=tokens,
        )
        await self._storage.put(path, content, mtime)
        self._bump_search_cache()
        log_marker(
            logger,
            "cache.put.end",
            path=path,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        logger.debug(f"Cached file: {path} ({tokens} tokens)")

    def _bump_search_cache(self) -> None:
        """Invalidate the in-session search-result cache.

        Called on every cache mutation so semantic_search never returns
        results that predate a write.
        """
        self._search_cache.clear()

    async def refresh_path(
        self,
        path: str,
        content: str,
        mtime: float,
        *,
        timeout: float | None = None,
    ) -> bool:
        refresh_timeout = self._compute_refresh_timeout() if timeout is None else timeout
        started = time.perf_counter()
        log_marker(
            logger,
            "cache.refresh.begin",
            path=path,
            timeout_s=refresh_timeout,
        )

        async def _refresh() -> None:
            await self.put(path, content, mtime)

        try:
            await asyncio.wait_for(_refresh(), timeout=refresh_timeout)
        except TimeoutError:
            self.mark_stale(path)
            log_marker(
                logger,
                "cache.refresh.timeout",
                path=path,
                timeout_s=refresh_timeout,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            logger.warning(f"Cache refresh timed out for {path}; marking stale and resetting IO")
            self.reset_executor()
            return False
        except Exception as e:
            self.mark_stale(path)
            log_marker(
                logger,
                "cache.refresh.fail",
                path=path,
                error=type(e).__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            logger.warning(f"Cache refresh failed for {path}: {e}")
            return False

        self.clear_stale(path)
        log_marker(
            logger,
            "cache.refresh.end",
            path=path,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return True

    async def get_content(self, entry: CacheEntry, *, max_bytes: int | None = None) -> str:
        """Reassemble cached content for ``entry``.

        ``max_bytes`` (UTF-8 bytes) caps how much is returned; truncation
        happens on a code-point boundary, so the result may be slightly
        shorter than the cap. ``None`` returns the full content. See
        :meth:`VectorStorage.get_content` for the full contract.
        """
        return await self._storage.get_content(entry, max_bytes=max_bytes)

    async def record_access(self, path: str) -> None:
        await self._storage.record_access(path)

    async def update_mtime(self, path: str, new_mtime: float) -> None:
        """Update cached mtime without re-storing content or re-embedding."""
        await self._storage.update_mtime(path, new_mtime)
        # Bump the search cache: the mtime change reflects an external write
        # whose content matched the cached hash. Even though the indexed
        # content didn't change, the invariant "every mtime write clears the
        # search cache" keeps result freshness simple to reason about.
        self._bump_search_cache()

    async def get_stats(self) -> dict[str, Any]:
        """Cache statistics: occupancy, process memory, session, and lifetime metrics."""
        stats: dict[str, Any] = {**await self._storage.get_stats()}

        # Add process memory stats
        rss = _get_rss_mb()
        if rss is not None:
            stats["process_rss_mb"] = rss

        # Add merge cache stats
        from ..core.tokenizer import _tokenizer  # noqa: PLC0415

        if _tokenizer is not None:
            stats["merge_cache_entries"] = len(_tokenizer._merge_cache)
            stats["merge_cache_maxsize"] = _tokenizer._merge_cache_maxsize

        # Session metrics
        stats["session"] = self._metrics.snapshot()

        # Lifetime metrics (aggregated from all completed sessions)
        try:
            stats["lifetime"] = self._metrics_storage.get_lifetime_stats()
        except Exception as e:
            logger.warning(f"Failed to load lifetime stats: {e}")

        return stats

    async def clear(self) -> int:
        result = await self._storage.clear()
        self._bump_search_cache()
        return result

    async def delete_path(self, path: str) -> int:
        """Delete one cached path and clear any stale marker for it."""
        removed = await self._storage.delete_path(path)
        self.clear_stale(path)
        if removed:
            self._bump_search_cache()
        return removed
