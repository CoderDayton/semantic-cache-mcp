"""One shared server per user and configuration, reached through thin launchers."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import anyio
import pytest
from anyio.abc import ByteStream
from fastmcp.client import Client
from fastmcp.client.transports.stdio import StdioTransport

from semantic_cache_mcp import launcher as _daemon
from semantic_cache_mcp.config import CACHE_DIR

REPO = Path(__file__).resolve().parents[1]

posix_only = pytest.mark.skipif(os.name != "posix", reason="Unix sockets and flock")


class TestDaemonKey:
    def test_unrelated_environment_is_ignored(self) -> None:
        assert _daemon.daemon_key({"TERM": "a"}) == _daemon.daemon_key({"TERM": "b"})

    def test_the_home_directory_changes_the_key(self) -> None:
        assert _daemon.daemon_key({"HOME": "/a"}) != _daemon.daemon_key({"HOME": "/b"})
        assert _daemon.daemon_key({"USERPROFILE": "/a"}) != _daemon.daemon_key({})

    def test_configuration_changes_the_key(self) -> None:
        base = _daemon.daemon_key({})
        assert _daemon.daemon_key({"TOOL_TIMEOUT": "1"}) != base
        assert _daemon.daemon_key({"SEMANTIC_CACHE_DIR": "/x"}) != base
        assert _daemon.daemon_key({"SCMCP_FORMAT_TIMEOUT_S": "3"}) != base
        assert _daemon.daemon_key({"PATH": "/bin"}) != base


@posix_only
class TestRuntimeDir:
    def test_the_directory_is_private(self, tmp_path: Path) -> None:
        directory = _daemon.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)})
        assert directory.parent == tmp_path
        assert directory.stat().st_mode & 0o777 == 0o700

    def test_a_directory_others_can_write_is_refused(self, tmp_path: Path) -> None:
        shared = tmp_path / f"semantic-cache-mcp-{os.getuid()}"
        shared.mkdir()
        shared.chmod(0o777)
        with pytest.raises(PermissionError):
            _daemon.runtime_dir({"XDG_RUNTIME_DIR": str(tmp_path)})


class TestSharedEnabled:
    def test_it_can_be_turned_off(self) -> None:
        assert _daemon.shared_enabled({"SCMCP_SHARED_SERVER": "0"}) is False

    def test_it_is_on_by_default(self) -> None:
        assert _daemon.shared_enabled({}) is True


class TestTransportKind:
    def test_the_default_follows_the_platform(self, monkeypatch) -> None:
        monkeypatch.setattr(_daemon.os, "name", "posix")
        assert _daemon.transport_kind({}) == "unix"
        monkeypatch.setattr(_daemon.os, "name", "nt")
        assert _daemon.transport_kind({}) == "tcp"

    def test_tcp_can_be_chosen(self) -> None:
        assert _daemon.transport_kind({"SCMCP_SHARED_TRANSPORT": "tcp"}) == "tcp"

    def test_windows_cannot_choose_unix_sockets(self, monkeypatch) -> None:
        monkeypatch.setattr(_daemon.os, "name", "nt")
        assert _daemon.transport_kind({"SCMCP_SHARED_TRANSPORT": "unix"}) == "tcp"


@pytest.fixture
def runtime() -> Iterator[Path]:
    """A short runtime dir: AF_UNIX paths are capped near 108 bytes."""
    directory = Path(tempfile.mkdtemp(prefix="scmcp-", dir="/tmp"))
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


def _env(tmp_path: Path, runtime: Path, **extra: str) -> dict[str, str]:
    tokenizer = CACHE_DIR / "tokenizer" / "o200k_base.tiktoken"
    if not tokenizer.exists():
        pytest.skip("integration test requires cached tokenizer assets")
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    if not (cache / "tokenizer").exists():
        (cache / "tokenizer").symlink_to(CACHE_DIR / "tokenizer", target_is_directory=True)
    env = os.environ.copy()
    env.update(
        SEMANTIC_CACHE_DIR=str(cache),
        XDG_RUNTIME_DIR=str(runtime),
        SCMCP_IDLE_TIMEOUT="1",
        LOG_LEVEL="WARNING",
        **extra,
    )
    return env


def _client(env: dict[str, str], cwd: Path, log: Path) -> Client:
    transport = StdioTransport(
        command="uv",
        args=["run", "--project", str(REPO), "semantic-cache-mcp"],
        env=env,
        cwd=str(cwd),
        keep_alive=False,
        log_file=log,
    )
    return Client(transport, timeout=30, init_timeout=60)


def _text(result: object) -> str:
    return "\n".join(getattr(item, "text", "") for item in result.content)  # type: ignore[attr-defined]


def _sockets(runtime: Path) -> list[Path]:
    return list(runtime.glob("*/*.sock")) + list(runtime.glob("*/*.port"))


async def _wait_for_no_sockets(runtime: Path, seconds: float) -> bool:
    for _ in range(int(seconds * 10)):
        if not _sockets(runtime):
            return True
        await asyncio.sleep(0.1)
    return False


@posix_only
class TestSharedServer:
    @pytest.mark.parametrize("transport", ["unix", "tcp"])
    async def test_two_clients_share_one_daemon_and_keep_their_own_directory(
        self, tmp_path: Path, runtime: Path, transport: str
    ) -> None:
        env = _env(tmp_path, runtime, SCMCP_SHARED_TRANSPORT=transport)
        dir_a, dir_b = tmp_path / "a", tmp_path / "b"
        for directory, word in ((dir_a, "alpha"), (dir_b, "beta")):
            directory.mkdir()
            (directory / "x.txt").write_text(f"{word}\n")

        async with (
            _client(env, dir_a, tmp_path / "a.log") as a,
            _client(env, dir_b, tmp_path / "b.log") as b,
        ):
            read_a = _text(await a.call_tool("read", {"path": "x.txt"}))
            read_b = _text(await b.call_tool("read", {"path": "x.txt"}))
            assert "alpha" in read_a and "beta" not in read_a
            assert "beta" in read_b and "alpha" not in read_b
            endpoints = _sockets(runtime)
            assert [p.suffix for p in endpoints] == [".port" if transport == "tcp" else ".sock"]
            assert endpoints[0].stat().st_mode & 0o777 == 0o600
            stats = _text(await a.call_tool("stats", {}))
            assert "**2** files" in stats, stats

        assert await _wait_for_no_sockets(runtime, 15), "idle daemon did not exit"

    async def test_a_tcp_connection_without_the_token_is_dropped(
        self, tmp_path: Path, runtime: Path
    ) -> None:
        env = _env(tmp_path, runtime, SCMCP_SHARED_TRANSPORT="tcp")
        async with _client(env, tmp_path, tmp_path / "t.log") as c:
            await c.call_tool("stats", {})
            (port_file,) = _sockets(runtime)
            port = json.loads(port_file.read_text())["port"]
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            initialize = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "intruder", "version": "0"},
                },
            }
            hello = {"cwd": str(tmp_path), "token": "0" * 64}
            writer.write((json.dumps(hello) + "\n" + json.dumps(initialize) + "\n").encode())
            await writer.drain()
            assert await asyncio.wait_for(reader.read(), timeout=10) == b""
            writer.close()

    async def test_turned_off_it_serves_in_process(self, tmp_path: Path, runtime: Path) -> None:
        env = _env(tmp_path, runtime, SCMCP_SHARED_SERVER="0")
        (tmp_path / "x.txt").write_text("gamma\n")
        async with _client(env, tmp_path, tmp_path / "c.log") as c:
            assert "gamma" in _text(await c.call_tool("read", {"path": "x.txt"}))
            assert _sockets(runtime) == []

    async def test_an_unsafe_runtime_dir_falls_back_to_in_process(
        self, tmp_path: Path, runtime: Path
    ) -> None:
        shared = runtime / f"semantic-cache-mcp-{os.getuid()}"
        shared.mkdir()
        shared.chmod(0o777)
        env = _env(tmp_path, runtime)
        (tmp_path / "x.txt").write_text("delta\n")
        async with _client(env, tmp_path, tmp_path / "d.log") as c:
            assert "delta" in _text(await c.call_tool("read", {"path": "x.txt"}))
            assert _sockets(runtime) == []


class _SilentStream(ByteStream):
    """A peer that connects and never sends a byte."""

    async def receive(self, max_bytes: int = 65536) -> bytes:
        await anyio.sleep_forever()
        raise AssertionError

    async def send(self, item: bytes) -> None:
        return None

    async def send_eof(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


@posix_only
class TestRelay:
    def test_a_daemon_gone_before_the_preamble_falls_back(self, monkeypatch, runtime) -> None:
        import socket

        ours, theirs = socket.socketpair()
        theirs.close()
        monkeypatch.setattr(_daemon, "_connect", lambda target: (ours, None))
        assert _daemon.relay_to_daemon({"XDG_RUNTIME_DIR": str(runtime)}) is False

    def test_a_daemon_that_never_says_ready_is_retried_then_skipped(
        self, monkeypatch, runtime
    ) -> None:
        import socket
        import threading

        attempts = []

        def connect(target: object) -> tuple[socket.socket, None]:
            ours, theirs = socket.socketpair()

            def hang_up() -> None:
                theirs.recv(65536)
                theirs.close()

            threading.Thread(target=hang_up, daemon=True).start()
            attempts.append(ours)
            return ours, None

        monkeypatch.setattr(_daemon, "_connect", connect)
        assert _daemon.relay_to_daemon({"XDG_RUNTIME_DIR": str(runtime)}) is False
        assert len(attempts) == 2

    def test_a_port_file_without_a_live_daemon_is_not_trusted(self, runtime) -> None:
        import socket

        env = {"XDG_RUNTIME_DIR": str(runtime), "SCMCP_SHARED_TRANSPORT": "tcp"}
        target = _daemon.endpoint(env)
        stranger = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        stranger.bind(("127.0.0.1", 0))
        stranger.listen()
        stranger.setblocking(False)
        port = stranger.getsockname()[1]
        target.address.write_text(json.dumps({"port": port, "token": "t" * 64}))
        try:
            assert _daemon._connect(target) is None
            with pytest.raises(BlockingIOError):
                stranger.accept()
        finally:
            stranger.close()

    def test_a_lock_freed_shortly_after_is_still_taken(self, runtime) -> None:
        import threading

        path = runtime / "x.lock"
        holder = _daemon.try_lock(path)
        assert holder is not None
        threading.Timer(0.2, os.close, args=(holder,)).start()
        assert _daemon.try_lock(path) is None
        taken = _daemon.try_lock(path, wait=2.0)
        assert taken is not None
        os.close(taken)

    def test_a_daemon_that_crashes_on_start_is_not_waited_for(self, monkeypatch, runtime) -> None:
        import subprocess
        import sys
        import time

        def crash(environ: object) -> subprocess.Popen[bytes]:
            return subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])  # noqa: S603

        monkeypatch.setattr(_daemon, "_spawn_daemon", crash)
        monkeypatch.setattr(_daemon, "SPAWN_TIMEOUT", 10.0)
        target = _daemon.endpoint({"XDG_RUNTIME_DIR": str(runtime)})
        started = time.monotonic()
        assert _daemon._spawn_and_connect(target, {}) is None
        assert time.monotonic() - started < 5


class TestPreamble:
    async def test_a_silent_connection_is_dropped(self, monkeypatch) -> None:
        from semantic_cache_mcp.server import _daemon as daemon

        monkeypatch.setattr(daemon, "_PREAMBLE_TIMEOUT", 0.05)
        connections = daemon._Connections()
        connections.opened()
        with anyio.fail_after(5):
            await daemon._session(_SilentStream(), None, connections)
        assert connections.active == 0
