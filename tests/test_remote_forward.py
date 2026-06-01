"""Guard: every remote-forwarding tool forwards its full parameter set.

In remote/supervisor mode a tool hands its call to the worker process via
``_maybe_call_remote_tool``. Historically each forward payload was a
hand-listed dict, so adding a parameter without updating that dict silently
dropped it in remote mode (no error, wrong behavior only under the
supervisor). ``_forward_kwargs()`` now derives the forwarded set from the
tool's own signature; this test locks that invariant in so the silent drop
cannot come back.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastmcp import Context

from semantic_cache_mcp.server import tools as tools_mod

# Tools that intentionally do NOT forward to the worker: read_image streams
# image bytes via direct disk I/O, bypassing the cache and worker entirely, so
# it never calls _maybe_call_remote_tool. Every other tool reads or mutates
# through the cache, which under the supervisor lives in the worker.
_NON_FORWARDING = {"read_image"}

_FORWARDING_TOOLS = [
    "read",
    "stats",
    "clear",
    "delete",
    "write",
    "edit",
    "edit_preview",
    "batch_edit",
    "search",
    "batch_read",
    "glob",
    "grep",
]


class _FakeRemoteCache:
    """Stand-in for the supervisor-backed tool runtime.

    ``_is_remote_runtime`` checks the ``_is_tool_process_supervisor`` flag, so
    setting it routes every tool down the remote-forward path. ``call_tool``
    records the forwarded kwargs and returns a sentinel payload.
    """

    _is_tool_process_supervisor = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(
        self,
        tool: str,
        kwargs: dict[str, Any],
        *,
        output_mode: str,
        max_response_tokens: int | None,
        timeout: float,
    ) -> dict[str, Any]:
        self.calls.append((tool, kwargs))
        return {"ok": True, "tool": tool, "forwarded": True}


def _make_ctx(cache: _FakeRemoteCache) -> MagicMock:
    ctx = MagicMock(spec=Context)
    ctx.lifespan_context = {"cache": cache}
    return ctx


def _required_args(fn: Any) -> dict[str, Any]:
    """Supply a value for each required (no-default) param; defaults fill the rest.

    Every required tool parameter is a string (path/content/pattern/query/...),
    so a single placeholder satisfies the signature up to the forward point.
    """
    args: dict[str, Any] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if name == "ctx":
            continue
        if param.default is inspect.Parameter.empty:
            args[name] = "x"
    return args


def _param_names(fn: Any) -> set[str]:
    return {name for name in inspect.signature(fn).parameters if name != "ctx"}


@pytest.mark.parametrize("tool_name", _FORWARDING_TOOLS)
async def test_remote_forward_covers_all_params(tool_name: str) -> None:
    fn = getattr(tools_mod, tool_name)
    cache = _FakeRemoteCache()
    ctx = _make_ctx(cache)

    result = await fn(ctx, **_required_args(fn))

    assert result == {"ok": True, "tool": tool_name, "forwarded": True}
    assert cache.calls, f"{tool_name} did not forward to the remote runtime"

    forwarded_tool, forwarded_kwargs = cache.calls[0]
    assert forwarded_tool == tool_name
    assert set(forwarded_kwargs) == _param_names(fn), (
        f"{tool_name} forwarded {set(forwarded_kwargs)} but its signature "
        f"declares {_param_names(fn)} — a parameter would be silently dropped "
        f"in remote mode"
    )


def test_non_forwarding_tools_unchanged() -> None:
    """Lock the set of tools that deliberately skip remote forwarding."""
    for name in _NON_FORWARDING:
        assert hasattr(tools_mod, name), f"{name} missing from tools module"
    assert not (_NON_FORWARDING & set(_FORWARDING_TOOLS)), (
        "a tool is listed as both forwarding and non-forwarding"
    )


def test_forward_kwargs_includes_keyword_only_params() -> None:
    """A keyword-only param must be forwarded, not dropped.

    Slicing ``co_varnames`` at ``co_argcount`` alone excludes keyword-only
    args; a future tool with ``def tool(ctx, x, *, flag)`` would silently drop
    ``flag`` in remote mode. This pins the full-signature behavior directly,
    independent of whether any real tool currently uses keyword-only params.
    """

    def fake_tool(ctx: object, path: str, *, flag: bool = True) -> dict[str, Any]:
        return tools_mod._forward_kwargs()

    assert fake_tool("ctx", "/p", flag=False) == {"path": "/p", "flag": False}


def test_forward_kwargs_rejects_varargs_tool() -> None:
    """A ``*args``/``**kwargs`` tool cannot be forwarded by name — fail loud."""

    def vararg_tool(ctx: object, path: str, **extra: Any) -> dict[str, Any]:
        return tools_mod._forward_kwargs()

    with pytest.raises(TypeError, match="no stable forwarded name"):
        vararg_tool("ctx", "/p", foo=1)


def test_forward_kwargs_rejects_unknown_override() -> None:
    """A typo'd override key must raise, not silently forward the raw value."""

    def fake_tool(ctx: object, paths: str) -> dict[str, Any]:
        # ``path_list`` is not a parameter — without the guard the raw,
        # unencoded ``paths`` would be forwarded instead of the override.
        return tools_mod._forward_kwargs(overrides={"path_list": "[]"})

    with pytest.raises(KeyError, match="path_list"):
        fake_tool("ctx", "a,b")
