"""Async wrappers for blocking file I/O operations.

Prevents synchronous file operations from blocking the asyncio event loop
when multiple concurrent MCP tool calls are in flight.

All operations are routed through a single-threaded executor so blocking
file I/O stays serialized with the rest of the storage layer instead of
racing across threads.
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


async def aread_head(path: Path, size: int, executor: Executor | None = None) -> bytes:
    """Read at most *size* leading bytes without blocking the event loop.

    For callers that only need a prefix — sniffing whether a file is binary,
    say. Reading the whole file to look at its first 8 KB allocates the entire
    file to answer a question about a fixed-size window, which on a multi-
    gigabyte file or a network mount is the difference between cheap and not.
    """
    loop = asyncio.get_running_loop()

    def _read() -> bytes:
        with path.open("rb") as handle:
            return handle.read(size)

    return await loop.run_in_executor(executor, _read)


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


def _fsync_directory(directory: Path) -> None:
    """Flush a completed rename into the directory entry itself.

    Best-effort by design: POSIX needs this for the new name to survive a
    crash, while Windows exposes no directory descriptor and needs nothing.
    A failure here never means data loss — the bytes are already durable from
    the file fsync; only the rename could be lost.
    """
    try:
        dir_fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _atomic_write_sync(path: Path, content: str) -> None:
    """Atomic write via temp-file + rename. Preserves original permissions.

    The temp file is fsync'd before the rename and the directory after it.
    Rename alone is atomic only against a concurrent *reader*; against a crash
    it is not, because the rename can reach disk while the data behind it has
    not, leaving the file empty or half-written. These writes are the user's
    source files, so one flush per write is the right price for the guarantee
    the name "atomic write" already implies.
    """
    import stat as stat_mod  # noqa: PLC0415

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
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
    _fsync_directory(path.parent)


async def awrite_atomic(path: Path, content: str, executor: Executor | None = None) -> None:
    """Atomic write without blocking the event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, _atomic_write_sync, path, content)


async def aunlink(
    path: Path, *, missing_ok: bool = False, executor: Executor | None = None
) -> None:
    """Unlink a file or symlink without blocking the event loop."""
    loop = asyncio.get_running_loop()

    def _unlink() -> None:
        path.unlink(missing_ok=missing_ok)

    await loop.run_in_executor(executor, _unlink)
