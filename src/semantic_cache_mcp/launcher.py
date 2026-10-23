"""The command a client launches: a thin relay to one shared server.

The command connects to a daemon and relays its stdio to it byte for byte,
starting the daemon when none is listening. The daemon
(:mod:`semantic_cache_mcp.server._daemon`) runs one cache and one tool worker,
and one MCP session per connection, so each client keeps its own session and
roots.

Two transports reach the daemon:

- ``unix`` (the POSIX default): a Unix socket in a directory only this user
  can enter.
- ``tcp`` (the Windows default, or ``SCMCP_SHARED_TRANSPORT=tcp``): a loopback
  port. The daemon writes its port and a random token to a ``.port`` file
  readable only by this user, and drops any connection whose preamble does not
  carry that token, since any local user can connect to a loopback port.

A daemon is keyed by version, install location and every environment variable
the configuration reads, so clients with different settings never share one.
Anything that stops the relay from starting — shared mode off, an unsafe
runtime directory, a daemon that never answers — falls back to serving
in-process.

This module stays import-light: the relay never loads the server.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import stat
import subprocess  # nosec B404
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from blake3 import blake3

from . import __version__

CONFIG_ENV = frozenset(
    {
        "HOME",
        "LOCALAPPDATA",
        "LOG_DIR",
        "LOG_LEVEL",
        "MAX_CACHE_ENTRIES",
        "MAX_CONTENT_SIZE",
        "PATH",
        "SEMANTIC_CACHE_DIR",
        "TOOL_MAX_RESPONSE_TOKENS",
        "TOOL_OUTPUT_MODE",
        "TOOL_TIMEOUT",
        "USERPROFILE",
        "XDG_CACHE_HOME",
    }
)
CONFIG_PREFIX = "SCMCP_"
LOOPBACK = "127.0.0.1"
READY = b'{"scmcp": "ready"}\n'

SPAWN_TIMEOUT = 60.0
READY_TIMEOUT = 10.0
ATTEMPTS = 2
_CHUNK = 65536
_FALSE_WORDS = frozenset({"0", "false", "no", "off"})
_INSTALL = Path(__file__).resolve().parent

Transport = Literal["unix", "tcp"]


@dataclass(frozen=True, slots=True)
class Endpoint:
    """Where a daemon listens: its socket (``unix``) or port file (``tcp``),
    and the lock file its owner holds."""

    kind: Transport
    address: Path
    lock: Path


def shared_enabled(environ: Mapping[str, str]) -> bool:
    """Shared mode is on unless ``SCMCP_SHARED_SERVER`` turns it off."""
    return environ.get("SCMCP_SHARED_SERVER", "1").strip().lower() not in _FALSE_WORDS


def transport_kind(environ: Mapping[str, str]) -> Transport:
    """``tcp`` on Windows, on a system without Unix sockets, or when
    ``SCMCP_SHARED_TRANSPORT=tcp``; ``unix`` otherwise."""
    if os.name != "posix" or not hasattr(socket, "AF_UNIX"):
        return "tcp"
    chosen = environ.get("SCMCP_SHARED_TRANSPORT", "unix").strip().lower()
    return "tcp" if chosen == "tcp" else "unix"


def daemon_key(environ: Mapping[str, str]) -> str:
    """16 hex characters naming the daemon for this build and configuration.

    ``PATH`` is part of it because formatters run as subprocesses of the
    daemon, ``HOME``/``USERPROFILE`` because ``~`` and the default cache
    directory resolve against them, and every ``SCMCP_`` variable because
    tools read them directly.
    """
    settings = sorted(
        (name, value)
        for name, value in environ.items()
        if name in CONFIG_ENV or name.startswith(CONFIG_PREFIX)
    )
    payload = json.dumps([__version__, str(_INSTALL), settings])
    return blake3(payload.encode("utf-8")).hexdigest()[:16]


def runtime_dir(environ: Mapping[str, str]) -> Path:
    """The per-user directory holding daemon endpoints.

    On POSIX it is ``$XDG_RUNTIME_DIR/semantic-cache-mcp-<uid>`` (or under the
    temp directory), created mode 0700. Raises PermissionError when it exists
    but is not a directory owned by this user and closed to everyone else,
    since anyone who can enter it can talk to the daemon. On Windows it is
    under ``%LOCALAPPDATA%``, whose ACL already admits only this user.
    """
    if sys.platform == "win32":
        base = Path(environ.get("LOCALAPPDATA") or tempfile.gettempdir())
        directory = base / "semantic-cache-mcp" / "run"
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    base = Path(environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir())
    directory = base / f"semantic-cache-mcp-{os.getuid()}"
    with contextlib.suppress(FileExistsError):
        directory.mkdir(mode=0o700)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise PermissionError(f"{directory} must be a directory owned by this user, mode 0700")
    return directory


def endpoint(environ: Mapping[str, str]) -> Endpoint:
    directory = runtime_dir(environ)
    key = daemon_key(environ)
    kind = transport_kind(environ)
    suffix = ".sock" if kind == "unix" else ".port"
    return Endpoint(kind=kind, address=directory / f"{key}{suffix}", lock=directory / f"{key}.lock")


def try_lock(path: Path, *, wait: float = 0.0) -> int | None:
    """An fd holding the exclusive lock on ``path``, or None when another
    process still holds it after ``wait`` seconds. The OS drops the lock when
    the holder exits."""
    deadline = time.monotonic() + wait
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    while True:
        try:
            _lock_fd(fd)
        except OSError:
            if time.monotonic() >= deadline:
                os.close(fd)
                return None
            time.sleep(0.02)
            continue
        return fd


def _lock_fd(fd: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def daemon_alive(target: Endpoint) -> bool:
    """Whether a daemon holds ``target``'s lock. Probing takes the lock for an
    instant; a daemon starting at that moment waits for it (``try_lock``'s
    ``wait``)."""
    fd = try_lock(target.lock)
    if fd is None:
        return True
    os.close(fd)
    return False


def read_port_file(path: Path) -> tuple[int, str] | None:
    """The ``(port, token)`` a TCP daemon published, or None when the file is
    missing or malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        port, token = data["port"], data["token"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(port, int) or not isinstance(token, str):
        return None
    return port, token


def main() -> None:
    """Console entry point: relay to the shared daemon, else serve in-process.

    ``semantic-cache-mcp serve`` runs the daemon itself.
    """
    if sys.argv[1:2] == ["serve"]:
        from .server._daemon import serve

        serve(os.environ)
        return
    if not relay_to_daemon(os.environ):
        from .server import main as serve_in_process

        serve_in_process()


def relay_to_daemon(environ: Mapping[str, str]) -> bool:
    """Relay this process's stdio to the shared daemon until either side
    closes. False, before any byte is relayed, when that is not possible."""
    if not shared_enabled(environ):
        return False
    for _ in range(ATTEMPTS):
        try:
            target = endpoint(environ)
            link = _connect(target) or _spawn_and_connect(target, environ)
        except OSError as exc:
            _warn(f"shared server unavailable ({exc}); serving in-process")
            return False
        if link is None:
            _warn("shared server did not start; serving in-process")
            return False
        conn, token = link
        try:
            early = _handshake(conn, token)
        except OSError as exc:
            conn.close()
            _warn(f"shared server dropped the connection ({exc})")
            continue
        _relay(conn, early)
        return True
    _warn("shared server never answered; serving in-process")
    return False


def _connect(target: Endpoint) -> tuple[socket.socket, str | None] | None:
    """A connection to the daemon and the token its preamble must carry, or
    None when no daemon is listening there."""
    token: str | None = None
    if target.kind == "unix":
        if sys.platform == "win32":
            return None
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        address: str | tuple[str, int] = str(target.address)
    else:
        published = read_port_file(target.address)
        if published is None or not daemon_alive(target):
            return None
        port, token = published
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        address = (LOOPBACK, port)
    try:
        conn.connect(address)
    except (FileNotFoundError, ConnectionRefusedError):
        conn.close()
        return None
    except OSError:
        conn.close()
        raise
    return conn, token


def _spawn_and_connect(
    target: Endpoint, environ: Mapping[str, str]
) -> tuple[socket.socket, str | None] | None:
    """Start a daemon and wait for its endpoint. A second launcher racing this
    one starts a daemon too; the endpoint lock lets only one of them serve."""
    daemon = _spawn_daemon(environ)
    deadline = time.monotonic() + SPAWN_TIMEOUT
    while time.monotonic() < deadline:
        link = _connect(target)
        if link is not None:
            return link
        if daemon.poll():
            return None
        time.sleep(0.05)
    return None


def _spawn_daemon(environ: Mapping[str, str]) -> subprocess.Popen[bytes]:
    argv = [sys.executable, "-m", "semantic_cache_mcp.server", "serve"]
    null = subprocess.DEVNULL
    if sys.platform == "win32":
        return subprocess.Popen(  # noqa: S603  # nosec B603
            argv,
            env=dict(environ),
            cwd=tempfile.gettempdir(),
            stdin=null,
            stdout=null,
            stderr=null,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    return subprocess.Popen(  # noqa: S603  # nosec B603
        argv,
        env=dict(environ),
        cwd="/",
        stdin=null,
        stdout=null,
        stderr=null,
        start_new_session=True,
        close_fds=True,
    )


def _handshake(conn: socket.socket, token: str | None) -> bytes:
    """Send the preamble line (this process's working directory and the
    token) and wait for the daemon's ``READY`` line, which it sends only to a
    connection it will serve. Returns any bytes received after it.

    Raises OSError when the daemon closes the connection, answers anything
    else, or stays silent for ``READY_TIMEOUT`` seconds — as a daemon that is
    shutting down does to a connection still in its listen backlog.
    """
    try:
        cwd: str | None = os.getcwd()
    except OSError:
        cwd = None
    hello: dict[str, str | None] = {"cwd": cwd}
    if token is not None:
        hello["token"] = token
    conn.settimeout(READY_TIMEOUT)
    conn.sendall((json.dumps(hello) + "\n").encode("utf-8"))
    received = b""
    while b"\n" not in received:
        chunk = conn.recv(_CHUNK)
        if not chunk:
            raise ConnectionResetError("closed before it was ready")
        received += chunk
    line, _, early = received.partition(b"\n")
    if line + b"\n" != READY:
        raise ConnectionError("unexpected handshake reply")
    conn.settimeout(None)
    return early


def _relay(conn: socket.socket, early: bytes = b"") -> None:
    """Write ``early`` to stdout, then copy stdin to the socket and the socket
    to stdout until the daemon closes the connection."""

    def pump_stdin() -> None:
        try:
            while chunk := os.read(0, _CHUNK):
                conn.sendall(chunk)
        except OSError:
            pass
        finally:
            with contextlib.suppress(OSError):
                conn.shutdown(socket.SHUT_WR)

    def write_out(data: bytes) -> None:
        view = memoryview(data)
        while view:
            view = view[os.write(1, view) :]

    threading.Thread(target=pump_stdin, name="stdin-relay", daemon=True).start()
    try:
        write_out(early)
        while chunk := conn.recv(_CHUNK):
            write_out(chunk)
    except OSError:
        pass
    finally:
        conn.close()


def _warn(message: str) -> None:
    with contextlib.suppress(OSError):
        sys.stderr.write(f"semantic-cache-mcp: {message}\n")
        sys.stderr.flush()
