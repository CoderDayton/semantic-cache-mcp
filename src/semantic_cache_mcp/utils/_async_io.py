"""Async wrappers for blocking file I/O operations.

Prevents synchronous file operations from blocking the asyncio event loop
when multiple concurrent MCP tool calls are in flight.

All operations are routed through a single-threaded executor to prevent
segfaults from ONNX Runtime / usearch allocator conflicts when blocking
operations run on different threads concurrently.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
from concurrent.futures import Executor
from pathlib import Path


async def aread_bytes(path: Path, executor: Executor | None = None) -> bytes:
    """Read file bytes without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, path.read_bytes)


async def aread_text(
    path: Path,
    encoding: str = "utf-8",
    errors: str = "strict",
    executor: Executor | None = None,
) -> str:
    """Read file text without blocking the event loop."""
    loop = asyncio.get_running_loop()

    def _read() -> str:
        return path.read_text(encoding=encoding, errors=errors)

    return await loop.run_in_executor(executor, _read)


async def astat(path: Path, executor: Executor | None = None) -> os.stat_result:
    """Stat a file without blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, path.stat)


def _atomic_write_sync(path: Path, content: str) -> None:
    """Atomic write via temp-file + rename. Preserves original permissions."""
    import stat as stat_mod  # noqa: PLC0415

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(content)
        if path.exists():
            try:
                original_mode = path.stat().st_mode
                os.chmod(tmp_path, stat_mod.S_IMODE(original_mode))
            except OSError:
                pass
        Path(tmp_path).replace(path)
    except BaseException:
        with contextlib.suppress(OSError):
            Path(tmp_path).unlink(missing_ok=True)
        raise


async def awrite_atomic(path: Path, content: str, executor: Executor | None = None) -> None:
    """Atomic write without blocking the event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, _atomic_write_sync, path, content)
