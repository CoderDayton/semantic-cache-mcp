"""Client-root resolution over the MCP session.

`Context.list_roots()` was removed in fastmcp 4.0 — the sessionless 2026-07-28
era has no back-channel for a server-initiated request. The capability still
exists on handshake-era connections through the raw session, so that is where
the root now comes from, and a connection without a back-channel falls back to
the server's working directory rather than failing the call.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import ListRootsResult, Root

from semantic_cache_mcp.server import tools


@pytest.fixture(autouse=True)
def _reset_client_root() -> Any:
    """`_resolve_client_root` caches per session; isolate each test from it."""
    tools._session_roots.clear()
    yield
    tools._session_roots.clear()


class _Session:
    def __init__(self, list_roots: Any) -> None:
        self.list_roots = list_roots


def _ctx_with_roots(*uris: str) -> Any:
    async def list_roots() -> ListRootsResult:
        return ListRootsResult(roots=[Root(uri=uri) for uri in uris])

    return SimpleNamespace(session=_Session(list_roots))


def _ctx_without_back_channel() -> Any:
    async def list_roots() -> ListRootsResult:
        raise RuntimeError("connection has no back-channel for server-initiated requests")

    return SimpleNamespace(session=_Session(list_roots))


class TestResolveClientRoot:
    async def test_first_file_root_becomes_the_client_root(self) -> None:
        ctx = _ctx_with_roots("file:///home/user/project")
        # Compared against the resolved form, not the literal URI path: cached
        # document paths are stored resolved, so the root has to be too or the
        # prefix test misses every one of them. On macOS `/home` is a firmlink
        # to `/System/Volumes/Data/home`, which makes the difference visible.
        assert await tools._resolve_client_root(ctx) == Path("/home/user/project").resolve()

    async def test_non_file_roots_are_ignored(self) -> None:
        ctx = _ctx_with_roots("https://example.com/repo")
        assert await tools._resolve_client_root(ctx) is None

    async def test_no_roots_returns_none(self) -> None:
        ctx = _ctx_with_roots()
        assert await tools._resolve_client_root(ctx) is None

    async def test_missing_back_channel_returns_none_without_raising(self) -> None:
        assert await tools._resolve_client_root(_ctx_without_back_channel()) is None

    async def test_context_without_a_session_returns_none(self) -> None:
        """Sessionless era and unit-test contexts alike: absent, not an error."""
        assert await tools._resolve_client_root(SimpleNamespace()) is None

    async def test_a_session_property_that_raises_returns_none(self) -> None:
        """fastmcp's `Context.session` raises RuntimeError when none is established."""

        class _NoSession:
            @property
            def session(self) -> Any:
                raise RuntimeError("session is not available")

        assert await tools._resolve_client_root(_NoSession()) is None

    async def test_result_is_resolved_once_and_cached(self) -> None:
        calls = 0

        async def list_roots() -> ListRootsResult:
            nonlocal calls
            calls += 1
            return ListRootsResult(roots=[Root(uri="file:///srv/app")])

        ctx = SimpleNamespace(session=_Session(list_roots))
        assert await tools._resolve_client_root(ctx) == Path("/srv/app")
        assert await tools._resolve_client_root(ctx) == Path("/srv/app")
        assert calls == 1

    async def test_each_session_keeps_its_own_root(self) -> None:
        first = _ctx_with_roots("file:///srv/one")
        second = _ctx_with_roots("file:///srv/two")
        assert await tools._resolve_client_root(first) == Path("/srv/one")
        assert await tools._resolve_client_root(second) == Path("/srv/two")
        assert await tools._resolve_client_root(first) == Path("/srv/one")

    async def test_no_roots_falls_back_to_the_connection_cwd(self) -> None:
        token = tools.CONNECTION_CWD.set(Path("/srv/launched-here"))
        try:
            assert await tools._resolve_client_root(_ctx_with_roots()) == Path("/srv/launched-here")
            assert await tools._resolve_client_root(SimpleNamespace()) == Path("/srv/launched-here")
        finally:
            tools.CONNECTION_CWD.reset(token)


class TestResolvePath:
    def test_relative_path_joins_the_client_root(self) -> None:
        assert tools._resolve_path("src/main.py", Path("/home/user/project")) == (
            "/home/user/project/src/main.py"
        )

    def test_absolute_path_ignores_the_client_root(self) -> None:
        assert tools._resolve_path("/etc/hosts", Path("/home/user/project")) == "/etc/hosts"

    def test_relative_path_without_a_root_resolves_against_cwd(self) -> None:
        assert tools._resolve_path("src/main.py", None) == str(Path("src/main.py").resolve())
