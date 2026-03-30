"""Single-worker executor that can be abandoned safely after a timeout.

The standard ``ThreadPoolExecutor`` is a poor fit for this server's timeout
path: ``shutdown(wait=False)`` leaves non-daemon workers alive, and Python's
``concurrent.futures`` exit hook force-joins those threads during interpreter
shutdown. A single stuck ``run_in_executor()`` call can therefore outlive the
tool timeout and pin the MCP process.

This executor keeps the same ``Executor`` interface used by
``asyncio.run_in_executor()`` but runs work on one daemon thread that is not
registered with ``concurrent.futures.thread._threads_queues``. When the
executor is reset after a timeout, the old worker can be abandoned safely and
the process can still exit.
"""

from __future__ import annotations

import logging
import queue
import threading
from concurrent.futures import Executor, Future
from typing import Any

logger = logging.getLogger(__name__)

_WorkItem = tuple[Future[Any], Any, tuple[Any, ...], dict[str, Any]]


class DetachedExecutor(Executor):
    """Single-thread executor with daemon worker semantics.

    ``shutdown(wait=False)`` abandons the current worker instead of blocking on
    a potentially stuck call. This is intentional for the timeout recovery path.
    """

    def __init__(self, *, thread_name_prefix: str = "semantic-cache-io") -> None:
        self._thread_name_prefix = thread_name_prefix
        self._work_queue: queue.Queue[_WorkItem | None] = queue.Queue()
        self._shutdown = False
        self._shutdown_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker,
            name=f"{thread_name_prefix}_0",
            daemon=True,
        )
        self._thread.start()

    def submit(self, fn, /, *args, **kwargs):
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            future: Future[Any] = Future()
            self._work_queue.put((future, fn, args, kwargs))
            return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        with self._shutdown_lock:
            if self._shutdown:
                return
            self._shutdown = True

        if cancel_futures:
            self._cancel_pending()

        self._work_queue.put(None)
        if wait and threading.current_thread() is not self._thread:
            self._thread.join()

    def _cancel_pending(self) -> None:
        while True:
            try:
                item = self._work_queue.get_nowait()
            except queue.Empty:
                return

            if item is None:
                self._work_queue.put(None)
                return

            future, _fn, _args, _kwargs = item
            future.cancel()

    def _worker(self) -> None:
        while True:
            item = self._work_queue.get()
            if item is None:
                return

            future, fn, args, kwargs = item
            if not future.set_running_or_notify_cancel():
                continue

            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                future.set_exception(exc)
                logger.debug("DetachedExecutor worker raised", exc_info=exc)
            else:
                future.set_result(result)
