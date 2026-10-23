"""The shared daemon behind :mod:`semantic_cache_mcp.launcher`.

One FastMCP lifespan — one cache, one tool worker — serves every connection on
the endpoint. Each connection opens with one JSON preamble line from the
launcher (its working directory, plus the token on ``tcp``), then carries MCP's
stdio framing, which the SDK's own ``stdio_server`` adapter turns into a
session. The lifespan is reference-counted, so each session's ``Server.run``
reuses it.

On ``tcp`` the daemon listens on a random loopback port and publishes the port
and a fresh 256-bit token to its ``.port`` file (mode 0600). A connection whose
preamble lacks that token is closed before any MCP message is read.

The endpoint lock is held for as long as the daemon accepts connections, so a
second daemon for the same endpoint exits at once. After ``SCMCP_IDLE_TIMEOUT``
seconds (default 600) with no connection, the daemon stops accepting, removes
its socket or port file and releases the lock, then finishes any connection
still open.
"""

from __future__ import annotations

import contextlib
import hmac
import json
import logging
import os
import secrets
import signal
import socket
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import anyio
from anyio.abc import ByteStream, SocketListener
from anyio.streams.buffered import BufferedByteReceiveStream
from fastmcp.server.context import set_transport
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.stdio import stdio_server

from ..launcher import LOOPBACK, READY, Endpoint, endpoint, try_lock
from ._mcp import mcp
from .tools import CONNECTION_CWD

logger = logging.getLogger(__name__)

IDLE_TIMEOUT = 600.0
LOCK_WAIT = 0.5
_PREAMBLE_MAX = 64 * 1024
_PREAMBLE_TIMEOUT = 10.0
_MESSAGE_MAX = 256 * 1024 * 1024
_STREAM_ENDS = (
    anyio.EndOfStream,
    anyio.IncompleteRead,
    anyio.DelimiterNotFound,
    anyio.BrokenResourceError,
    anyio.ClosedResourceError,
)


def serve(environ: Mapping[str, str]) -> None:
    """Run the daemon for this configuration, or return at once when another
    daemon already holds its endpoint. The short wait rides out a launcher
    probing the lock (``launcher.daemon_alive``) at the same instant."""
    target = endpoint(environ)
    lock_fd = try_lock(target.lock, wait=LOCK_WAIT)
    if lock_fd is None:
        return
    lock = _Lock(lock_fd)
    try:
        anyio.run(_serve, target, lock, _idle_timeout(environ))
    finally:
        lock.release()


def _idle_timeout(environ: Mapping[str, str]) -> float:
    try:
        return max(0.0, float(environ.get("SCMCP_IDLE_TIMEOUT", IDLE_TIMEOUT)))
    except ValueError:
        return IDLE_TIMEOUT


class _Lock:
    __slots__ = ("_fd",)

    def __init__(self, fd: int) -> None:
        self._fd: int | None = fd

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class _Connections:
    """Open-connection count, with a wake-up whenever it changes."""

    def __init__(self) -> None:
        self.active = 0
        self.last_closed = time.monotonic()
        self._changed = anyio.Event()

    def opened(self) -> None:
        self.active += 1
        self._notify()

    def closed(self) -> None:
        self.active -= 1
        self.last_closed = time.monotonic()
        self._notify()

    def _notify(self) -> None:
        self._changed.set()
        self._changed = anyio.Event()

    async def wait_idle(self, idle: float) -> None:
        while True:
            changed = self._changed
            if self.active:
                await changed.wait()
                continue
            remaining = idle - (time.monotonic() - self.last_closed)
            if remaining <= 0:
                return
            with anyio.move_on_after(remaining):
                await changed.wait()

    async def wait_drained(self) -> None:
        while self.active:
            await self._changed.wait()


async def _serve(target: Endpoint, lock: _Lock, idle: float) -> None:
    set_transport("stdio")
    async with mcp._lifespan_manager():
        with contextlib.suppress(FileNotFoundError):
            target.address.unlink()
        listener, token = await _listen(target)
        connections = _Connections()
        stop = anyio.Event()
        logger.info(f"shared server listening on {target.address}")
        async with anyio.create_task_group() as tg:
            tg.start_soon(_accept, listener, token, connections, tg)
            tg.start_soon(_stop_when_idle, connections, idle, stop)
            if sys.platform != "win32":
                tg.start_soon(_stop_on_signal, stop)
            await stop.wait()
            await listener.aclose()
            with contextlib.suppress(FileNotFoundError):
                target.address.unlink()
            lock.release()
            await connections.wait_drained()
            tg.cancel_scope.cancel()
        logger.info("shared server stopped")


async def _listen(target: Endpoint) -> tuple[SocketListener, str | None]:
    """The listener for ``target`` and the token its connections must send
    (None on ``unix``, where the directory's permissions admit only this user)."""
    if target.kind == "unix":
        return await anyio.create_unix_listener(target.address, mode=0o600), None
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((LOOPBACK, 0))
        sock.listen()
        sock.setblocking(False)
        listener = await SocketListener.from_socket(sock)
    except BaseException:
        sock.close()
        raise
    token = secrets.token_hex(32)
    _publish_port(target.address, sock.getsockname()[1], token)
    return listener, token


def _publish_port(path: Path, port: int, token: str) -> None:
    """Write the port file atomically, readable only by this user."""
    staging = path.with_suffix(".tmp")
    with contextlib.suppress(FileNotFoundError):
        staging.unlink()
    fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump({"port": port, "token": token}, handle)
    os.replace(staging, path)


async def _accept(
    listener: SocketListener,
    token: str | None,
    connections: _Connections,
    tg: anyio.abc.TaskGroup,
) -> None:
    while True:
        try:
            stream = await listener.accept()
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            return
        connections.opened()
        tg.start_soon(_session, stream, token, connections)


async def _stop_when_idle(connections: _Connections, idle: float, stop: anyio.Event) -> None:
    await connections.wait_idle(idle)
    stop.set()


async def _stop_on_signal(stop: anyio.Event) -> None:
    with anyio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signals:
        async for _ in signals:
            stop.set()
            return


def _admitted(hello: object, token: str | None) -> bool:
    if token is None:
        return True
    given = hello.get("token") if isinstance(hello, dict) else None
    return isinstance(given, str) and hmac.compare_digest(
        given.encode("utf-8"), token.encode("utf-8")
    )


async def _session(stream: ByteStream, token: str | None, connections: _Connections) -> None:
    try:
        async with stream:
            buffered = BufferedByteReceiveStream(stream)
            line = b""
            with anyio.move_on_after(_PREAMBLE_TIMEOUT):
                line = await buffered.receive_until(b"\n", _PREAMBLE_MAX)
            if not line:
                logger.warning("shared server dropped a connection that sent no preamble")
                return
            hello = json.loads(line)
            if not _admitted(hello, token):
                logger.warning("shared server dropped a connection without its token")
                return
            cwd = hello.get("cwd") if isinstance(hello, dict) else None
            CONNECTION_CWD.set(
                Path(cwd).resolve() if isinstance(cwd, str) and os.path.isabs(cwd) else None
            )
            await stream.send(READY)
            reader = cast(Any, _Lines(buffered))
            writer = cast(Any, _Writer(stream))
            async with stdio_server(reader, writer) as (read_stream, write_stream):
                await mcp._mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp._mcp_server.create_initialization_options(
                        notification_options=NotificationOptions(tools_changed=True)
                    ),
                )
    except (*_STREAM_ENDS, json.JSONDecodeError):
        logger.debug("shared server connection closed", exc_info=True)
    except Exception:
        logger.exception("shared server session failed")
    finally:
        connections.closed()


class _Lines:
    """The connection's newline-delimited messages, as ``stdio_server`` reads stdin."""

    def __init__(self, buffered: BufferedByteReceiveStream) -> None:
        self._buffered = buffered

    def __aiter__(self) -> _Lines:
        return self

    async def __anext__(self) -> str:
        try:
            line = await self._buffered.receive_until(b"\n", _MESSAGE_MAX)
        except _STREAM_ENDS as exc:
            raise StopAsyncIteration from exc
        return line.decode("utf-8", errors="replace")


class _Writer:
    """The connection's send side, as ``stdio_server`` writes stdout."""

    def __init__(self, stream: ByteStream) -> None:
        self._stream = stream

    async def write(self, text: str) -> None:
        await self._stream.send(text.encode("utf-8"))

    async def flush(self) -> None:
        return None
