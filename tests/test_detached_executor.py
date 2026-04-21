"""Regression tests for the timeout recovery executor."""

from __future__ import annotations

import subprocess
import sys

from semantic_cache_mcp.utils import DetachedExecutor


def test_detached_executor_runs_submitted_work() -> None:
    """Submitted work should run and return through a normal Future."""
    executor = DetachedExecutor(thread_name_prefix="test-detached")
    try:
        future = executor.submit(lambda: 42)
        assert future.result(timeout=1.0) == 42
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_detached_executor_does_not_pin_process_exit() -> None:
    """A stuck worker abandoned with shutdown(wait=False) must not hang exit."""
    code = """
from __future__ import annotations

import threading
import time

from semantic_cache_mcp.utils import DetachedExecutor

started = threading.Event()
executor = DetachedExecutor(thread_name_prefix="probe")

def hang() -> None:
    started.set()
    time.sleep(60)

executor.submit(hang)
if not started.wait(1.0):
    raise SystemExit("worker never started")

executor.shutdown(wait=False, cancel_futures=True)
print("subprocess-exit-ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "subprocess-exit-ok" in result.stdout


def test_utils_import_does_not_eagerly_import_server_stack() -> None:
    """Importing semantic_cache_mcp.utils should stay isolated and cheap."""
    code = """
from __future__ import annotations

import sys

import semantic_cache_mcp.utils  # noqa: F401

assert "semantic_cache_mcp.server" not in sys.modules
assert "semantic_cache_mcp.cache" not in sys.modules
print("utils-import-ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "utils-import-ok" in result.stdout
