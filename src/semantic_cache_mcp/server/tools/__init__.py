"""MCP tool handlers."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import os
import stat as stat_module
import warnings
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Final, TypeVar, cast

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.shared.exceptions import MCPDeprecationWarning
from mcp.types import ImageContent, TextContent

from ...cache import (
    SemanticCache,
    batch_smart_read,
    find_edit_anchors,
    glob_with_cache_status,
    semantic_search,
    smart_batch_edit,
    smart_edit,
    smart_read,
    smart_write,
)
from ...cache._helpers import _diff_context_lines, _PhaseTimer
from ...cache.read import _DIFF_MAX_RATIO, _DIFF_MIN_TOKENS, _sniff_image_mime
from ...config import MAX_CONTENT_SIZE, TOOL_TIMEOUT
from ...core import (
    count_tokens,
    extract_outline,
    generate_diff,
    rebase_diff_hunks,
    render_outline,
)
from ...core.hashing import hash_matches, short_hash
from ...types import ReadResult
from ...utils import aread_bytes, astat
from ...utils._async_io import aunlink
from .._coverage import (
    EMPTY_SPANS,
    LineSpans,
    decode_coverage_token,
    encode_coverage_token,
)
from .._mcp import mcp
from .._paths import relativize, shared_root
from .._tool_models import (
    BatchEditResponse,
    BatchReadResponse,
    ClearResponse,
    DeleteResponse,
    EditPreviewResponse,
    EditResponse,
    GlobResponse,
    GrepResponse,
    ReadImageResponse,
    ReadResponse,
    SearchResponse,
    StatsResponse,
    WarmResponse,
    WriteResponse,
    register_response_model,
)
from ..response import (
    _MODE_DEBUG,
    _MODE_NORMAL,
    _diff_state,
    _finalize_payload,
    _raise_tool_error,
    _response_mode,
    _response_token_cap,
)

logger = logging.getLogger(__name__)


# Tool timeout from config (env TOOL_TIMEOUT, default 30s).
_TOOL_TIMEOUT: float = TOOL_TIMEOUT

# Chars per token to assume when sizing a payload of serialized JSON matches.
# Prose runs ~4; JSON punctuation, quoted keys and line numbers run denser, so
# budgeting at the prose rate overshoots the response cap.
_JSON_CHARS_PER_TOKEN: Final = 2

# Per-line JSON envelope in a grep response: the quotes, comma, line number and
# separator wrapping one rendered `"<n>:<text>"` entry. It replaced a 32-char
# `{"line_number":N,"line":"..."}` object per match.
_LINE_ENVELOPE_CHARS: Final = 10

# Separators closing a grep line's number prefix: `:` for a line that matched,
# `-` for a context line, as ripgrep renders them.
_MATCH_SEPARATOR: Final = ":"
_CONTEXT_SEPARATOR: Final = "-"

# What a grep response may carry. `paths` and `count` exist because "which
# files mention X" is a large share of real greps and used to pay for every
# matching line.
_GREP_OUTPUT_MODES: Final = ("matches", "paths", "count")

# Ceilings for `read(outline=true)`. An outline exists to be cheap; a generated
# or vendored file with tens of thousands of definitions would otherwise cost
# more than the summary it replaces. Whatever is dropped is counted and stated
# in the rendered output, never silently omitted.
_OUTLINE_MAX_ENTRIES: Final = 2000
_OUTLINE_MAX_TOKENS: Final = 4000

# `warm` bounds. It exists to be called on a whole tree, so every limit is a
# refusal that gets *reported* rather than a silent stop: a file left out of
# the index is a file `grep` will never mention again.
_WARM_MAX_FILES: Final = 2000
_WARM_MAX_FILE_BYTES: Final = 4 * 1024 * 1024
_WARM_MAX_TOTAL_BYTES: Final = 64 * 1024 * 1024
_WARM_MAX_FAILURES_REPORTED: Final = 20


def _ranged_metrics(tokens_original: int, tokens_returned: int, *, from_cache: bool) -> ReadResult:
    """Build a ReadResult so a ranged read records accurate token accounting.

    A ranged read materializes the whole file to address lines but returns only
    the requested slice, so the file's full token count is the original, the
    slice is what was returned, and the difference is saved versus a naive full
    read. Previously ranged reads recorded original == returned and zero saved,
    billing a few lines as if the entire file had been sent.
    """
    return ReadResult(
        content="",
        from_cache=from_cache,
        is_diff=False,
        tokens_original=tokens_original,
        tokens_returned=tokens_returned,
        tokens_saved=max(0, tokens_original - tokens_returned),
        truncated=False,
        compression_ratio=(tokens_returned / tokens_original) if tokens_original else 1.0,
    )


# Global tool mutex: only one tool call executes at a time.
# Prevents concurrent coroutines from interleaving executor tasks,
# catalog reads, and blocking I/O, the root cause of hangs when
# multiple subagents fire tool calls simultaneously.
#
# We bind the lock to the running event loop so that test runners which
# create a fresh loop per test (pytest-asyncio function scope) get a fresh
# lock too — a stale Lock from a closed loop would deadlock or raise on
# acquire. Production runs see a single loop, so the rebind path is dead
# code in normal operation.
_tool_lock: asyncio.Lock | None = None
_tool_lock_loop: asyncio.AbstractEventLoop | None = None
_RemoteToolReturnT = TypeVar("_RemoteToolReturnT")


# Cached client root — resolved once per session via the session's roots/list.
_client_root: Path | None = None
_client_root_resolved: bool = False


async def _resolve_client_root(ctx: Context) -> Path | None:
    """Fetch and cache the MCP client's project root (first roots/list entry).

    fastmcp 4.0 removed ``Context.list_roots``: the sessionless 2026-07-28 era has
    no back-channel for a server-initiated request. Handshake-era connections still
    carry one, reachable through the raw session, so ask there. Every way of not
    getting an answer — no session, no back-channel, no roots — means the same
    thing here, and leaves relative paths resolving against the working directory.
    """
    global _client_root, _client_root_resolved
    if not _client_root_resolved:
        try:
            with warnings.catch_warnings():
                # roots/list is deprecated as of 2026-07-28 (SEP-2577). The
                # handshake-era back-channel is still the only way to ask, and
                # the era that dropped it also drops the question. Filter only
                # that one warning — `catch_warnings` mutates process-global
                # state, and this block spans an await, so a blanket ignore
                # would swallow warnings from concurrent tasks.
                warnings.filterwarnings("ignore", category=MCPDeprecationWarning)
                result = await ctx.session.list_roots()
            roots = result.roots
            if roots:
                uri = str(roots[0].uri)
                if uri.startswith("file://"):
                    _client_root = Path(uri[7:])
                    logger.debug(f"Client root: {_client_root}")
        except Exception:
            logger.debug("Could not resolve client roots", exc_info=True)
        _client_root_resolved = True
    return _client_root


def _resolve_path(path: str, root: Path | None) -> str:
    """Resolve *path* — absolute passes through, relative joins to *root*."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    if root is not None:
        return str(root / p)
    return str(p.resolve())


@dataclass(frozen=True, slots=True)
class _ToolCallState:
    cache: Any
    mode: str
    max_response_tokens: int | None
    client_root: Path | None

    def resolve(self, path: str) -> str:
        """Resolve a path against the client's project root."""
        return _resolve_path(path, self.client_root)


def _parse_path_list(raw: str) -> list[str]:
    """Parse comma-separated or JSON-array path inputs."""
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [p.strip() for p in text.split(",") if p.strip()]


def _resolve_path_list(raw: str, state: _ToolCallState) -> list[str]:
    """Parse and resolve each path against the client root."""
    return [state.resolve(path) for path in _parse_path_list(raw)]


def _parse_known_hashes(raw: str, state: _ToolCallState) -> dict[str, str]:
    """Parse a caller's ``path -> content_hash`` possession claims.

    Keys are resolved exactly as ``paths`` is, so a claim lands on the file it
    names however the caller spelled it.
    """
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"known_hashes must be a JSON object of path -> content_hash: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError("known_hashes must be a JSON object of path -> content_hash")
    return {state.resolve(str(p)): str(h) for p, h in parsed.items() if h}


def _get_tool_lock() -> asyncio.Lock:
    """Return the per-event-loop tool lock, creating it lazily on first use.

    Re-creates the lock if the running event loop has changed (which only
    happens in test scenarios that spin up a fresh loop per test).
    """
    global _tool_lock, _tool_lock_loop
    loop = asyncio.get_running_loop()
    if _tool_lock is None or _tool_lock_loop is not loop:
        _tool_lock = asyncio.Lock()
        _tool_lock_loop = loop
    return _tool_lock


def _delete_cache_candidates(path: Path) -> list[str]:
    """Return cache-key candidates for a filesystem delete path.

    Real files are cached by resolved path. Symlinks are deleted as links, so we
    avoid resolving them to prevent evicting the target file's cache entry.
    """
    if path.is_symlink():
        return [str(path)]
    return list(dict.fromkeys((str(path.resolve(strict=False)), str(path))))


def _is_remote_runtime(value: Any) -> bool:
    """True when *value* is the supervisor-backed tool runtime."""
    return getattr(value, "_is_tool_process_supervisor", False) is True


async def _tool_call_state(ctx: Context) -> _ToolCallState:
    return _ToolCallState(
        cache=ctx.lifespan_context["cache"],
        mode=_response_mode(),
        max_response_tokens=_response_token_cap(),
        client_root=await _resolve_client_root(ctx),
    )


def _show_diff_requested(mode: str, show_diff: bool) -> bool:
    """Debug mode and explicit show_diff both count as a verbose diff request."""
    return show_diff or mode == _MODE_DEBUG


def _apply_mutation_diff(
    payload: dict[str, Any],
    *,
    diff_content: str | None,
    mode: str,
    show_diff: bool,
    partial: bool = False,
) -> None:
    """Attach diff fields only when they materially help the next decision."""
    actual_state = _diff_state(diff_content) or "unchanged"
    include_diff = partial or _show_diff_requested(mode, show_diff)

    if actual_state == "unchanged":
        payload["diff_state"] = "unchanged"
        if include_diff and diff_content:
            payload["diff"] = diff_content
        return

    if include_diff and diff_content:
        payload["diff"] = diff_content
        payload["diff_state"] = actual_state
        return

    payload["diff_state"] = "omitted"
    payload["diff_omitted"] = True


async def _maybe_call_remote_tool(
    state: _ToolCallState,
    tool: str,
    kwargs: dict[str, Any],
    *,
    timeout: float,
) -> _RemoteToolReturnT | None:
    if not _is_remote_runtime(state.cache):
        return None

    try:
        return cast(
            _RemoteToolReturnT,
            await state.cache.call_tool(
                tool,
                kwargs,
                output_mode=state.mode,
                max_response_tokens=state.max_response_tokens,
                timeout=timeout,
            ),
        )
    except TimeoutError:
        _raise_tool_error(tool, f"timed out after {timeout}s", state.max_response_tokens)


def _forward_kwargs(*, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build remote-forward kwargs from the *calling tool's* own signature.

    Reads the caller frame's parameters (every tool parameter except ``ctx``)
    and pulls their current values from the caller's locals. Because the
    forwarded set is derived from the signature rather than hand-listed, a
    newly added tool parameter is forwarded automatically — it can never be
    silently dropped in remote/supervisor mode. ``overrides`` supplies values
    for parameters that are transformed before forwarding (e.g. ``batch_read``
    encodes its resolved path list as JSON).
    """
    frame = inspect.currentframe()
    caller = frame.f_back if frame is not None else None
    if caller is None:  # pragma: no cover - CPython always provides a frame
        raise RuntimeError("_forward_kwargs requires a caller frame")
    code = caller.f_code
    # ``co_varnames`` lays out positional-or-keyword args first
    # (``co_argcount``), then keyword-only args (``co_kwonlyargcount``). Both
    # are real parameters that must be forwarded; slicing at ``co_argcount``
    # alone would silently drop a keyword-only param — the exact regression
    # this helper exists to prevent. A ``*args``/``**kwargs`` tool has no
    # stable name to forward under, so fail loudly rather than drop it.
    if code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS):
        raise TypeError(
            f"_forward_kwargs cannot forward {code.co_name}: "
            "*args/**kwargs parameters have no stable forwarded name"
        )
    names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
    local_vars = caller.f_locals
    over = overrides or {}
    unknown = set(over) - set(names)
    if unknown:
        raise KeyError(
            f"_forward_kwargs override(s) {sorted(unknown)} are not parameters "
            f"of {code.co_name} — a typo'd override would forward the raw value"
        )
    return {
        name: (over[name] if name in over else local_vars[name]) for name in names if name != "ctx"
    }


def _serialized(fn):
    """Decorator: acquire the global tool lock before running the handler.

    Ensures only one tool call executes at a time, preventing concurrent
    coroutines from interleaving executor tasks and causing hangs.

    Lock acquisition has NO timeout — tools always join the queue.
    The tool *holding* the lock will release it within TOOL_TIMEOUT
    (via asyncio.wait_for for reads, asyncio.timeout for writes).
    """
    import functools  # noqa: PLC0415

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        async with _get_tool_lock():
            return await fn(*args, **kwargs)

    return wrapper


def _handle_timeout(cache: SemanticCache, tool: str, detail: str = "") -> None:
    """Reset the executor after a timeout so subsequent calls don't hang."""
    msg = f"{tool} timed out after {_TOOL_TIMEOUT}s"
    if detail:
        msg += f": {detail}"
    logger.warning(msg)
    cache.reset_executor()


async def _shielded_write(cache: SemanticCache, coro: Any, *, timeout: float | None = None) -> Any:
    """Run a write coroutine protected from cancellation during shutdown.

    Uses asyncio.shield so the inner task runs to completion even when
    the tool handler's task is cancelled (e.g. SIGTERM). end_operation()
    fires only after the write actually finishes, keeping the drain
    counter accurate for async_close().

    Timeout is enforced INSIDE the shield via asyncio.timeout, NOT by
    wrapping this function in asyncio.wait_for.  wait_for + shield is
    broken: wait_for cancels the wrapper → shield catches CancelledError
    and re-awaits the inner task → wait_for blocks forever waiting for
    the wrapper to finish.  asyncio.timeout works because it cancels the
    *shield future* (which is immediately "done"), not the inner task.
    """
    if timeout is None:
        timeout = _TOOL_TIMEOUT
    if not cache.begin_operation():
        coro.close()  # prevent 'coroutine was never awaited' warning
        raise RuntimeError("Server is shutting down")
    task = asyncio.ensure_future(coro)
    task.set_name("shielded-write")
    # Pair begin_operation() with end_operation() that fires exactly once
    # when the underlying write actually finishes — success, error, or
    # cancellation. Wiring it as a done_callback (instead of a finally
    # branch) is what makes the drain counter accurate when the awaiter
    # gives up on a timeout while the shielded task keeps running.
    task.add_done_callback(lambda _t: cache.end_operation())
    try:
        async with asyncio.timeout(timeout):
            return await asyncio.shield(task)
    except asyncio.CancelledError:
        # Not our timeout — genuine cancellation (SIGTERM / graceful shutdown).
        # Give the write a brief grace period to finish disk I/O before
        # the process exits.
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            # Inner task is intentionally NOT cancelled — it keeps running on
            # the IO executor until completion, at which point the
            # done_callback fires end_operation() and unblocks async_close().
            # async_close() waits up to _DRAIN_TIMEOUT (8s) on _drained, which
            # is longer than the 2s grace above, so the drain counter is
            # reliably balanced before the loop shuts down.
            raise asyncio.CancelledError() from None


def _binary_read_payload(path: str, result: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "tool": "read",
        "path": path,
        "is_binary": True,
        "size": result.size,
        "mime": result.mime,
    }


# A partial or summarized read reports the file's hash under `file_hash`, and
# never as a bare content hash: a caller echoing it back as `known_hash` would
# be claiming possession of a file it saw only part of. Namespacing the value
# makes that claim impossible to make by accident instead of merely discouraged
# in prose, while still letting the caller compare it across reads to notice the
# file moved.
_PARTIAL_HASH_PREFIX = "partial:"


def _stamp_result_hash(payload: dict[str, Any], new_hash: str, *, caller_holds: bool) -> None:
    """Attach a mutation's resulting hash under the key its evidence supports.

    A write or edit leaves the caller holding the new content only when the
    caller could derive that content itself: it supplied the whole file, or it
    supplied a transformation and had proved it held what the file started
    from. Otherwise the hash goes out as `file_hash` — a caller that edited a
    file it never read must not be handed a token that would later buy it an
    `unchanged` reply for content it has never seen.
    """
    if caller_holds:
        payload["content_hash"] = short_hash(new_hash)
    else:
        payload["file_hash"] = _PARTIAL_HASH_PREFIX + short_hash(new_hash)


def _stamp_ranged_possession(
    payload: dict[str, Any],
    content_hash: str,
    covered: LineSpans,
    total_lines: int,
) -> None:
    """Record what a ranged read leaves the caller able to prove it holds.

    Coverage spanning every line means the whole file has now been delivered
    for this version, so the caller earns a claimable `content_hash`. Anything
    narrower keeps reporting the file's identity under `file_hash` — unchanged
    from before, so a caller comparing it across reads still learns only
    whether the file moved — and adds a signed `coverage_token` naming the
    windows it does hold, which its next ranged read can redeem for
    `unchanged` instead of a re-send.
    """
    if covered.covers_all(total_lines):
        payload["content_hash"] = short_hash(content_hash)
        return
    payload["file_hash"] = _PARTIAL_HASH_PREFIX + short_hash(content_hash)
    # The token carries the wire form too: it is compared with `hash_matches`,
    # which accepts either length, and the short one keeps the token readable.
    payload["coverage_token"] = encode_coverage_token(short_hash(content_hash), covered)


async def _outline_read(
    *,
    cache: Any,
    path: str,
    max_size: int,
    mode: str,
    max_response_tokens: int | None,
) -> dict[str, Any]:
    """Answer `read(outline=true)`: the file's definitions, not its text.

    Serves from the cache when disk has not moved past it, the same freshness
    rule the rest of `read` uses; otherwise it reads and seeds the cache, so an
    outline also makes the file visible to `grep` and `search`.

    The result is never possession: an outline is a map of the file and a
    caller holding it cannot reconstruct a single line, so the hash goes out
    under `file_hash` where it structurally cannot be redeemed.
    """
    abs_path = str(Path(path).expanduser().resolve())
    entry = await cache.get(abs_path)
    st = None
    if entry is not None:
        try:
            st = await astat(Path(abs_path), cache._io_executor)
        except OSError:
            st = None

    if entry is not None and st is not None and entry.mtime >= st.st_mtime:
        full_text: str = await cache.get_content(entry)
        full_tokens: int = entry.tokens
        from_cache = True
        content_hash: str | None = entry.content_hash
    else:
        result = await asyncio.wait_for(
            smart_read(
                cache=cache,
                path=path,
                max_size=max_size,
                diff_mode=False,
                force_full=True,
                refresh_cache=False,  # smart_read still refreshes when stale
                # An outline maps the real file. Summarizing first would map a
                # digest and hand back line numbers belonging to nothing.
                summarize=False,
            ),
            timeout=_TOOL_TIMEOUT,
        )
        if result.is_binary:
            cache.metrics.record("read", result)
            return _finalize_payload(_binary_read_payload(path, result), max_response_tokens)
        full_text = result.content
        full_tokens = result.tokens_original
        from_cache = result.from_cache
        content_hash = result.content_hash

    structure = extract_outline(
        full_text,
        filename=Path(path).name,
        max_entries=_OUTLINE_MAX_ENTRIES,
        max_tokens=_OUTLINE_MAX_TOKENS,
        count_fn=count_tokens,
    )
    rendered = render_outline(structure)
    tokens_returned = count_tokens(rendered)
    cache.metrics.record(
        "read", _ranged_metrics(full_tokens, tokens_returned, from_cache=from_cache)
    )

    payload: dict[str, Any] = {
        "ok": True,
        "tool": "read",
        "path": path,
        "outline": True,
        "symbols": len(structure.entries),
        "total_lines": structure.total_lines,
    }
    if content_hash is not None:
        payload["file_hash"] = _PARTIAL_HASH_PREFIX + short_hash(content_hash)
    if structure.entries:
        payload["content"] = rendered
    else:
        # An empty body with no explanation reads as a broken tool. Say which
        # of the two possible answers this is, and what to call instead.
        payload["reason"] = "no_definitions_found"
        payload["hint"] = (
            "no class/function definitions were recognized in this file; "
            "read it with offset/limit, or without outline"
        )
    if structure.truncated:
        payload["truncated"] = True
    if mode == _MODE_DEBUG:
        payload["from_cache"] = from_cache
        payload["tokens_saved"] = max(0, full_tokens - tokens_returned)

    return _finalize_payload(payload, max_response_tokens)


@mcp.tool(
    output_schema=register_response_model("read", ReadResponse),
    meta={"version": _pkg_version("semantic-cache-mcp")},
)
@_serialized
async def read(
    ctx: Context,
    path: str,
    max_size: int = MAX_CONTENT_SIZE,
    offset: int | None = None,
    limit: int | None = None,
    known_hash: str | None = None,
    outline: bool = False,
    line_numbers: bool = False,
) -> dict[str, Any]:
    """Read a file, returning as few tokens as possible. For 2+ files, use `batch_read`.

    The first read returns the file's full content plus a `content_hash`
    (lines come back as-is; a ranged read numbers them only with
    `line_numbers=true`).
    Echo that hash back as `known_hash` and a later read of an unchanged file
    answers `"unchanged": true` with no body; a changed file returns a unified
    diff. Omit it and the file is always sent in full. Reading also caches the
    file so `grep`, `search`, and `batch_read` can see it.

    Whenever you re-read a file you have read before, pass back `known_hash`
    (the `content_hash` from your last read of it). It is the server's only
    proof that you still hold the content, so use it every time you can; the
    server then skips re-sending unchanged bytes. Use `offset`/`limit` to read
    or recover an exact line range, for example after a large file was
    summarized. A read that returns only part of a file — a line range, or a
    summary of a large one — reports its hash as `file_hash` (prefixed
    `partial:`) rather than `content_hash`: it identifies the file across reads
    but is not proof you hold it, and cannot be redeemed as `known_hash`.

    A ranged read also returns a `coverage_token` recording the lines it just
    sent you. Pass that back as `known_hash` on your next ranged read of the
    file: a window you already hold answers `unchanged` instead of being
    re-sent, and a new window widens the token's coverage. Once the windows add
    up to the whole file you are handed a claimable `content_hash` for it.

    For a large or unfamiliar file, `outline=true` is the cheap first read: one
    line per class/function as `<line>: <signature>`, typically a small fraction
    of the file, and every number is an `offset` you can read next. An outline
    is a map, not the file, so it comes back as `file_hash`.

    What the body holds is always labelled: `is_diff` marks a unified diff and
    `truncated` marks a summary, so you never have to infer it from the bytes.
    A binary file returns metadata instead of content; for images use
    `read_image`.

    Args:
        path: File path (absolute, or relative to the project root). Use an
            absolute path for files outside the project root.
        max_size: Byte threshold above which the file is semantically
            summarized; recover exact lines afterward with `offset`/`limit`.
        offset: 1-based first line for a ranged read; omit or pass 0 to start
            from the first line.
        limit: Number of lines to return starting at `offset`.
        known_hash: The `content_hash` from your last read of this file — or the
            `coverage_token` from your last ranged read of it — passed back to
            get `"unchanged"` instead of the content re-sent. Omit only on a
            first read, or when you no longer hold what it vouches for.
        outline: Return the file's definitions and their line numbers instead
            of its text. Cannot be combined with `offset`/`limit`.
        line_numbers: Prefix each line of a ranged read with its number. Costs
            about 17% more tokens; the range is in `lines` either way. Requires
            `offset` or `limit`.
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens

    # Validate bounds locally before any remote forwarding. `offset=0` is
    # accepted and treated as from-start.
    if offset is not None and offset < 0:
        _raise_tool_error(
            "read", "offset must be >= 0 (1-based; 0 is from start)", max_response_tokens
        )
    if limit is not None and limit < 1:
        _raise_tool_error("read", "limit must be >= 1", max_response_tokens)
    # Two incompatible requests must not resolve by precedence: a caller that
    # asked for both an outline and a window has to learn which one it is not
    # getting, or it reads the wrong thing believing it read the right one.
    ranged = offset is not None or limit is not None
    if outline and ranged:
        _raise_tool_error(
            "read",
            "outline cannot be combined with offset/limit — call it alone to "
            "locate the file, then read the line ranges it names",
            max_response_tokens,
        )
    if line_numbers and not ranged:
        _raise_tool_error(
            "read",
            "line_numbers applies to a ranged read; pass offset and/or limit",
            max_response_tokens,
        )

    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state, "read", _forward_kwargs(), timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result
    max_size = max(1, min(max_size, MAX_CONTENT_SIZE * 10))

    try:
        if outline:
            return await _outline_read(
                cache=cache,
                path=path,
                max_size=max_size,
                mode=mode,
                max_response_tokens=max_response_tokens,
            )

        # If offset/limit specified, read specific lines (still caches full file)
        if ranged:
            ranged_abs = str(Path(path).expanduser().resolve())
            ranged_entry = await cache.get(ranged_abs)
            ranged_st = None
            if ranged_entry is not None:
                try:
                    ranged_st = await astat(Path(ranged_abs), cache._io_executor)
                except OSError:
                    ranged_st = None
            # Trust the cache when disk has not moved past it (mtime), the same
            # freshness rule the rest of the read tool uses. A fresh entry lets us
            # slice from cached bytes instead of re-reading the whole file off disk.
            ranged_fresh = (
                ranged_entry is not None
                and ranged_st is not None
                and ranged_entry.mtime >= ranged_st.st_mtime
            )

            # A coverage token names the version its windows came from, so it is
            # decoded up front and matched below against the bytes actually
            # served. Anything that is not a token this process signed — a bare
            # `content_hash`, a `partial:` file hash, an invented string —
            # decodes to nothing and therefore claims nothing.
            claimed = decode_coverage_token(known_hash)

            # Materialize the file to address specific lines. Serve from the cache
            # when it is fresh (no disk read); only touch disk when the cache is
            # missing or stale, in which case smart_read also refreshes it.
            superseded_text: str | None = None
            if ranged_fresh:
                full_text = await cache.get_content(ranged_entry)
                full_tokens = ranged_entry.tokens
                ranged_from_cache = True
                ranged_hash: str | None = ranged_entry.content_hash
            else:
                # A token can name the version the cache is about to lose:
                # smart_read refreshes a stale entry, and those retired bytes
                # are the only base a window diff could be built against. Fetch
                # them first, and only when a token actually claims that
                # version, so the ordinary path pays nothing for this.
                if (
                    claimed is not None
                    and ranged_entry is not None
                    and hash_matches(claimed[0], ranged_entry.content_hash)
                ):
                    superseded_text = await cache.get_content(ranged_entry)
                result = await asyncio.wait_for(
                    smart_read(
                        cache=cache,
                        path=path,
                        max_size=max_size,
                        diff_mode=False,  # Line ranges bypass diff mode
                        force_full=True,
                        refresh_cache=False,  # smart_read still refreshes when stale
                        summarize=False,  # Line ranges need literal lines, not a summary
                    ),
                    timeout=_TOOL_TIMEOUT,
                )
                if result.is_binary:
                    cache.metrics.record("read", result)
                    return _finalize_payload(
                        _binary_read_payload(path, result), max_response_tokens
                    )
                full_text = result.content
                full_tokens = result.tokens_original
                ranged_from_cache = result.from_cache
                ranged_hash = result.content_hash

            lines = full_text.splitlines(keepends=True)
            start = max(0, (offset or 0) - 1)  # Convert to 0-based; offset 0/None both start at 0
            end = start + (limit or len(lines) - start)
            selected = lines[start:end]
            # A ranged read only ever shows the caller the window it asked for.
            # A window spanning the whole file mints or redeems a `content_hash`
            # outright. Anything narrower is credited to `covered`: what the
            # caller can prove it holds once this window lands, being whatever
            # coverage it carried in widened by what is served now.
            covers_whole_file = start == 0 and end >= len(lines)
            window_end = min(end, len(lines))
            claimed_spans = (
                claimed[1]
                if claimed is not None
                and ranged_hash is not None
                and hash_matches(claimed[0], ranged_hash)
                else EMPTY_SPANS
            )
            covered = claimed_spans.merge(start, window_end)
            line_info = {
                # Empty window (offset past EOF / empty file): report
                # start==end==total instead of a start that exceeds end.
                "start": start + 1 if selected else len(lines),
                "end": window_end,
                "total": len(lines),
            }

            # Hash-gated short-circuit: the caller still holds this exact file
            # and asked for all of it, so there is nothing left to re-send.
            # Recorded as a cache hit that saved the whole file.
            if (
                covers_whole_file
                and known_hash
                and ranged_fresh
                and ranged_entry is not None
                and hash_matches(known_hash, ranged_entry.content_hash)
            ):
                cache.metrics.record(
                    "read", _ranged_metrics(ranged_entry.tokens, 0, from_cache=True)
                )
                return _finalize_payload(
                    {
                        "ok": True,
                        "tool": "read",
                        "path": path,
                        "unchanged": True,
                        "content_hash": short_hash(ranged_entry.content_hash),
                        "lines": {"total": len(lines)},
                    },
                    max_response_tokens,
                )

            # Coverage short-circuit: the caller carried in a token proving it
            # was already sent these exact lines of this exact version, so the
            # window would be re-sent for nothing. A whole-file `content_hash`
            # is deliberately NOT accepted here — holding the file says nothing
            # about still holding a window the caller is now asking for, and
            # answering that request with no body strands it.
            if selected and ranged_hash is not None and claimed_spans.covers(start, window_end):
                cache.metrics.record("read", _ranged_metrics(full_tokens, 0, from_cache=True))
                held_payload: dict[str, Any] = {
                    "ok": True,
                    "tool": "read",
                    "path": path,
                    "unchanged": True,
                    "lines": line_info,
                }
                _stamp_ranged_possession(held_payload, ranged_hash, covered, len(lines))
                return _finalize_payload(held_payload, max_response_tokens)

            # Window diff: the caller holds this window of the version just
            # superseded, so send what changed inside it instead of the whole
            # window again. Applying this diff to the lines it holds yields the
            # lines being served, which is what a re-read would have delivered.
            # Coverage does not carry over — outside this window the caller's
            # copy is now the old file — so `covered` is this window alone.
            if (
                selected
                and superseded_text is not None
                and ranged_hash is not None
                and claimed is not None
                and claimed[1].covers(start, window_end)
            ):
                base_window = "".join(superseded_text.splitlines(keepends=True)[start:end])
                # Diffing a slice numbers its hunks from the slice's first
                # line. Rebase onto the file so `@@` means what it means in
                # every other diff this server sends — a caller that reads a
                # window-relative number as a file line goes and re-reads the
                # wrong place, which costs far more than the header saved.
                window_diff = rebase_diff_hunks(
                    generate_diff(
                        base_window,
                        "".join(selected),
                        context_lines=_diff_context_lines(base_window),
                    ),
                    start,
                )
                window_tokens = count_tokens("".join(selected))
                diff_tokens = count_tokens(window_diff)
                # Same gate the whole-file diff uses: below a floor the @@-header
                # overhead outweighs the saving, and a diff that is not much
                # smaller than the window is not worth the caller reassembling.
                if (
                    window_tokens >= _DIFF_MIN_TOKENS
                    and diff_tokens < window_tokens * _DIFF_MAX_RATIO
                ):
                    cache.metrics.record(
                        "read", _ranged_metrics(full_tokens, diff_tokens, from_cache=True)
                    )
                    diff_payload: dict[str, Any] = {
                        "ok": True,
                        "tool": "read",
                        "path": path,
                        "content": window_diff,
                        "is_diff": True,
                        "lines": line_info,
                    }
                    _stamp_ranged_possession(diff_payload, ranged_hash, covered, len(lines))
                    return _finalize_payload(diff_payload, max_response_tokens)

            # Only the line terminator is stripped, never trailing whitespace
            # within the line. A window covering the whole file mints a
            # claimable `content_hash`, and that claim is only true if the
            # caller can reconstruct the bytes it is vouching for — rstrip()
            # silently dropped trailing spaces and tabs, certifying possession
            # of content that was never sent.
            #
            # The number gutter is opt-in. It measured ~17% of a window on real
            # source, and `lines` already carries the range, so the caller can
            # map any offset into the window without paying per line. Generator
            # expressions avoid materializing the intermediate list; `selected`
            # may be thousands of lines on partial reads of large files.
            if line_numbers:
                content = "\n".join(
                    f"{i:6d}\t{line.removesuffix(chr(10)).removesuffix(chr(13))}"
                    for i, line in enumerate(selected, start=start + 1)
                )
            else:
                content = "\n".join(
                    line.removesuffix(chr(10)).removesuffix(chr(13)) for line in selected
                )
            # Bill only the slice actually returned; the rest of the file is saved
            # versus a naive full read.
            ranged_returned = count_tokens(content)
            cache.metrics.record(
                "read",
                _ranged_metrics(full_tokens, ranged_returned, from_cache=ranged_from_cache),
            )
            payload: dict[str, Any] = {
                "ok": True,
                "tool": "read",
                "path": path,
                "content": content,
                "lines": line_info,
            }
            if ranged_hash is not None:
                _stamp_ranged_possession(payload, ranged_hash, covered, len(lines))
            if mode in _MODE_NORMAL:
                payload["truncated"] = False
            if mode == _MODE_DEBUG:
                payload["from_cache"] = ranged_from_cache
                payload["tokens_saved"] = max(0, full_tokens - ranged_returned)

            return _finalize_payload(payload, max_response_tokens)

        # `unchanged` is driven entirely by the caller asserting freshness: it
        # passes back the content_hash it already holds. If that matches the
        # cached hash and the file is cache-fresh (disk == cache), the caller
        # provably has the current bytes, so we skip re-sending. cache_fresh
        # still gates this, so a stale known_hash can never mask a real change.
        # With no known_hash the server makes no assumption and sends content.
        #
        # Possession also decides how the file is read, so it is settled before
        # reading. A proven caller can take the cache-hit shortcuts: the
        # `unchanged` marker, or a diff against the bytes it holds. An unproven
        # one needs literal bytes, and those have to come back through the size
        # budget — serving a cached entry directly returns the whole file
        # however large, which is how `max_size` came to bound the first read of
        # a file and nothing after it.
        abs_path = str(Path(path).expanduser().resolve())
        prior_entry = await cache.get(abs_path)
        caller_has_current = prior_entry is not None and hash_matches(
            known_hash, prior_entry.content_hash
        )
        result = await asyncio.wait_for(
            smart_read(
                cache=cache,
                path=path,
                max_size=max_size,
                diff_mode=caller_has_current,
                force_full=not caller_has_current,
                refresh_cache=caller_has_current,
            ),
            timeout=_TOOL_TIMEOUT,
        )
        cache.metrics.record("read", result)

        # Binary fallback: structured metadata instead of an error so callers
        # can branch on is_binary without parsing the error string.
        if result.is_binary:
            return _finalize_payload(_binary_read_payload(path, result), max_response_tokens)

        cache_fresh = result.from_cache and not result.is_diff
        unchanged = cache_fresh and caller_has_current
        # Re-read the entry now that the read has run: it seeds a file the cache
        # had never seen and refreshes one that moved on disk, so the pre-read
        # entry is the wrong thing to stamp a hash from.
        entry = prior_entry if unchanged else await cache.get(abs_path)

        payload = {
            "ok": True,
            "tool": "read",
            "path": path,
        }
        if unchanged:
            # Skip sending content; give the model enough metadata to decide
            # locally whether a ranged re-read is worth it.
            payload["unchanged"] = True
            if entry is not None:
                payload["content_hash"] = short_hash(entry.content_hash)
                # smart_read returns either the full content (small files) or
                # an "// File unchanged" marker (large files). Reuse the bytes
                # if we already have them; only re-fetch from cache otherwise.
                if result.content and result.tokens_returned >= result.tokens_original > 0:
                    cached_text: str | None = result.content
                else:
                    cached_text = await cache.get_content(entry)
                if cached_text:
                    payload["total_lines"] = cached_text.count("\n") + (
                        0 if cached_text.endswith("\n") else 1
                    )
        else:
            # Always literal bytes: an unproven caller is read with force_full,
            # so smart_read returns real content bounded by max_size rather than
            # the "// File unchanged" marker. Nothing is re-fetched from the
            # cache here — that was the path that handed back whole files
            # regardless of the budget the caller asked for.
            payload["content"] = result.content
        # Always surface the current hash so the caller can echo it back as
        # known_hash on its next read and skip a re-send. A summarized read
        # delivered a digest rather than the file, so its hash goes out under
        # `file_hash` and cannot be redeemed as a possession claim.
        if entry is not None and "content_hash" not in payload:
            if result.truncated:
                payload["file_hash"] = _PARTIAL_HASH_PREFIX + short_hash(entry.content_hash)
            else:
                payload["content_hash"] = short_hash(entry.content_hash)
        # A diff and a summary are not the file. Whatever the response mode,
        # the payload has to say which one it carries — otherwise the caller
        # has to sniff the bytes for `@@` headers to find out what it holds.
        if result.is_diff:
            payload["is_diff"] = True
        if result.truncated:
            payload["truncated"] = True
        if mode in _MODE_NORMAL and result.truncated:
            # Truncated reads use semantic summarization — the returned
            # content is non-contiguous, so line numbers don't map to the
            # original file. Don't hint a specific offset; instead tell
            # the caller to use offset/limit to read specific line ranges.
            total_tokens = entry.tokens if entry else result.tokens_original
            payload["total_tokens"] = total_tokens
            payload["hint"] = (
                f"File was semantically summarized ({total_tokens} tokens total). "
                f"Use read with offset=<line> and limit=<n> to read specific "
                f"sections of the original file."
            )
        if mode == _MODE_DEBUG:
            payload["from_cache"] = result.from_cache
            payload["tokens_saved"] = result.tokens_saved
            payload["tokens_original"] = result.tokens_original
            payload["tokens_returned"] = result.tokens_returned
            payload["params"] = {
                "max_size": max_size,
                "offset": offset,
                "limit": limit,
            }

        return _finalize_payload(payload, max_response_tokens)

    except FileNotFoundError as e:
        _raise_tool_error("read", str(e), max_response_tokens)
    except TimeoutError:
        _handle_timeout(cache, "read", path)
        _raise_tool_error("read", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except ToolError:
        raise
    except Exception as e:
        _raise_tool_error("read", f"reading failed: {e}", max_response_tokens)


# Image pass-through: maximum bytes inlined as a single MCP image block.
# Anthropic's vision API rejects images over ~5MB; cap defends both response
# budget and upstream contract. Override via SCMCP_MAX_IMAGE_BYTES.
_DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024

# Wire-side cap on the base64-encoded payload Anthropic will accept (~5 MB).
# Raw bytes expand 4/3 on encoding, so 5 MiB raw → ~6.99 MB on the wire and
# upstream rejects it. Guard the encoded size explicitly so the failure is a
# tool-level error with a clear message rather than an opaque upstream 400.
_DEFAULT_MAX_ENCODED_IMAGE_BYTES = 5_000_000


def _parse_max_image_bytes() -> int:
    raw = os.environ.get("SCMCP_MAX_IMAGE_BYTES", str(_DEFAULT_MAX_IMAGE_BYTES))
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid SCMCP_MAX_IMAGE_BYTES=%r; using default %d",
            raw,
            _DEFAULT_MAX_IMAGE_BYTES,
        )
        return _DEFAULT_MAX_IMAGE_BYTES
    return max(1024, value)


def _parse_max_encoded_image_bytes() -> int:
    raw = os.environ.get("SCMCP_MAX_ENCODED_IMAGE_BYTES", str(_DEFAULT_MAX_ENCODED_IMAGE_BYTES))
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Invalid SCMCP_MAX_ENCODED_IMAGE_BYTES=%r; using default %d",
            raw,
            _DEFAULT_MAX_ENCODED_IMAGE_BYTES,
        )
        return _DEFAULT_MAX_ENCODED_IMAGE_BYTES
    return max(1024, value)


def _predicted_base64_len(n: int) -> int:
    return 4 * ((n + 2) // 3)


_MAX_IMAGE_BYTES: int = _parse_max_image_bytes()
_MAX_ENCODED_IMAGE_BYTES: int = _parse_max_encoded_image_bytes()


# read_image deliberately omits @_serialized: it never touches the cache or
# the serialized SQLite executor (it reads bytes via the default loop executor),
# so it has nothing to serialize against and need not queue behind other tools.
@mcp.tool(
    output_schema=register_response_model("read_image", ReadImageResponse),
    meta={"version": _pkg_version("semantic-cache-mcp")},
)
async def read_image(
    ctx: Context,
    path: str,
) -> ToolResult:
    """Read an image file so the model can see it.

    Returns an MCP image block (base64 data + mime type) plus a small JSON
    metadata sidecar (`size`, `mime`). Use this only when the model needs to
    view the image; for text or any other file type use `read`.

    The format is detected from the file's magic bytes, not its extension, so
    a mis-named image still works and a non-image (e.g. text saved as `.png`)
    is rejected. Supports PNG, JPEG, GIF, TIFF, BMP, and WebP. Images are
    never cached — every call re-reads from disk. Oversized images are
    rejected before encoding; the cap is `SCMCP_MAX_IMAGE_BYTES` (default
    5 MiB), bounded by Anthropic's ~5 MB upload limit.

    Args:
        path: Image file path (absolute, or relative to the project root).
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    max_response_tokens = state.max_response_tokens

    # Image reads bypass the cache (and the worker process). They use the
    # default loop executor — aread_bytes/astat accept `None` and fall back
    # to asyncio's default ThreadPoolExecutor, which is the right thing in
    # the server process (no contention with the worker's SQLite IO thread).
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        _raise_tool_error("read_image", f"File not found: {path}", max_response_tokens)

    try:
        st = await astat(file_path, None)
    except OSError as e:
        _raise_tool_error("read_image", f"Cannot stat file: {e}", max_response_tokens)

    if not stat_module.S_ISREG(st.st_mode):
        _raise_tool_error("read_image", f"Not a regular file: {path}", max_response_tokens)

    if st.st_size > _MAX_IMAGE_BYTES:
        _raise_tool_error(
            "read_image",
            (
                f"image too large: {st.st_size} bytes exceeds limit {_MAX_IMAGE_BYTES} "
                f"(raise via SCMCP_MAX_IMAGE_BYTES)"
            ),
            max_response_tokens,
        )

    try:
        raw = await asyncio.wait_for(aread_bytes(file_path, None), timeout=_TOOL_TIMEOUT)
    except TimeoutError:
        _raise_tool_error("read_image", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except OSError as e:
        _raise_tool_error("read_image", f"I/O error: {e}", max_response_tokens)

    # Re-check size against the bytes actually read: the pre-read st_size
    # check races a file that grows — or a swapped symlink target — between
    # the stat and the read. Reject here so an oversized image never reaches
    # the base64 step or the response budget.
    if len(raw) > _MAX_IMAGE_BYTES:
        _raise_tool_error(
            "read_image",
            (
                f"image too large: {len(raw)} bytes exceeds limit {_MAX_IMAGE_BYTES} "
                f"(raise via SCMCP_MAX_IMAGE_BYTES)"
            ),
            max_response_tokens,
        )

    # Base64 expands 4/3 — guard the on-the-wire size against Anthropic's
    # ~5 MB upload cap before paying for the encode. Cheap arithmetic check;
    # no allocation.
    predicted_encoded = _predicted_base64_len(len(raw))
    if predicted_encoded > _MAX_ENCODED_IMAGE_BYTES:
        _raise_tool_error(
            "read_image",
            (
                f"image too large after base64: {predicted_encoded} encoded bytes "
                f"exceeds limit {_MAX_ENCODED_IMAGE_BYTES} "
                f"(raise via SCMCP_MAX_ENCODED_IMAGE_BYTES)"
            ),
            max_response_tokens,
        )

    # Verify by magic bytes, not by extension — a file named `x.png` that
    # holds text must be refused, and a real image with a wrong/missing
    # extension must still be accepted. Supports PNG, JPEG, GIF, TIFF, BMP,
    # and WebP.
    mime = _sniff_image_mime(raw)
    if mime is None:
        _raise_tool_error(
            "read_image",
            (
                f"not a recognized image: {path} — content is not PNG/JPEG/GIF/"
                "TIFF/BMP/WebP; use `read` for non-image files"
            ),
            max_response_tokens,
        )

    # Run base64 off the event loop and under the tool timeout. For a 5 MiB
    # image the encode is ~tens of ms of pure CPU; doing it inline blocks
    # every other coroutine and is unbounded if the buffer is unexpectedly
    # large. (The MCP response write itself is framework-controlled and not
    # covered by this timeout.)
    try:
        encoded = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, base64.b64encode, raw),
            timeout=_TOOL_TIMEOUT,
        )
    except TimeoutError:
        _raise_tool_error(
            "read_image",
            f"base64 encoding timed out after {_TOOL_TIMEOUT}s",
            max_response_tokens,
        )

    metadata: dict[str, Any] = {
        "ok": True,
        "tool": "read_image",
        "path": path,
        "size": st.st_size,
        "mime": mime,
    }
    image_block = ImageContent(
        type="image",
        data=encoded.decode("ascii"),
        mime_type=mime,
    )
    text_block = TextContent(type="text", text=json.dumps(metadata))
    return ToolResult(content=[text_block, image_block], structured_content=metadata)


@mcp.tool(output_schema=register_response_model("stats", StatsResponse))
@_serialized
async def stats(
    ctx: Context,
) -> ToolResult:
    """Report cache health, token savings, and runtime diagnostics.

    Returns storage occupancy (files, tokens, documents, DB size), session and
    lifetime token savings and cache hit rates, per-tool call counts, and
    process memory. Use it to measure or debug — not as a routine step in
    read/edit loops. Takes no arguments.
    """
    state = await _tool_call_state(ctx)
    cache = state.cache
    mode = state.mode
    remote_result: ToolResult | None = await _maybe_call_remote_tool(
        state, "stats", _forward_kwargs(), timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result

    cache_stats = await cache.get_stats()

    session = cache_stats.get("session", {})
    lifetime = cache_stats.get("lifetime", {})

    # Session savings
    s_saved = session.get("tokens_saved", 0)
    s_original = session.get("tokens_original", 0)
    s_pct = round(s_saved / s_original * 100, 1) if s_original > 0 else 0.0
    s_hits = session.get("cache_hits", 0)
    s_misses = session.get("cache_misses", 0)
    s_total = s_hits + s_misses
    s_hit_pct = round(s_hits / s_total * 100) if s_total > 0 else 0

    # Lifetime savings
    lt_saved = lifetime.get("tokens_saved", 0)
    lt_original = lifetime.get("tokens_original", 0)
    lt_pct = round(lt_saved / lt_original * 100, 1) if lt_original > 0 else 0.0
    lt_hits = lifetime.get("cache_hits", 0)
    lt_misses = lifetime.get("cache_misses", 0)
    lt_total = lt_hits + lt_misses
    lt_hit_pct = round(lt_hits / lt_total * 100) if lt_total > 0 else 0
    lt_sessions = lifetime.get("total_sessions", 0)

    # Helpers
    def _n(v: int) -> str:
        return f"{v:,}"

    def _mb(v: float) -> str:
        return f"{v:.2f} MB"

    def _uptime(s: float) -> str:
        s = int(s)
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m {s % 60}s"
        return f"{s // 3600}h {(s % 3600) // 60}m"

    structured_payload: dict[str, Any] = {
        "mode": mode,
        "storage": {
            "files_cached": cache_stats.get("files_cached", 0),
            "total_tokens_cached": cache_stats.get("total_tokens_cached", 0),
            "total_documents": cache_stats.get("total_documents", 0),
            "db_size_mb": cache_stats.get("db_size_mb", 0.0),
        },
        "session": {
            "uptime_s": session.get("uptime_s", 0),
            "tokens_saved": s_saved,
            "tokens_original": s_original,
            "tokens_returned": session.get("tokens_returned", 0),
            "cache_hits": s_hits,
            "cache_misses": s_misses,
            "hit_rate_pct": s_hit_pct,
            "files_read": session.get("files_read", 0),
            "files_written": session.get("files_written", 0),
            "files_edited": session.get("files_edited", 0),
            "diffs_served": session.get("diffs_served", 0),
            "tool_calls": dict(session.get("tool_calls", {})),
        },
        "lifetime": {
            "total_sessions": lt_sessions,
            "tokens_saved": lt_saved,
            "tokens_original": lt_original,
            "tokens_returned": lifetime.get("tokens_returned", 0),
            "cache_hits": lt_hits,
            "cache_misses": lt_misses,
            "hit_rate_pct": lt_hit_pct,
            "files_read": lifetime.get("files_read", 0),
            "files_written": lifetime.get("files_written", 0),
            "files_edited": lifetime.get("files_edited", 0),
        },
        "process_rss_mb": cache_stats.get("process_rss_mb"),
    }

    if mode == "compact":
        lines = [
            "## Semantic Cache",
            "",
            f"Storage: **{_n(cache_stats.get('files_cached', 0))}** files · "
            f"**{_n(cache_stats.get('total_tokens_cached', 0))}** tokens · "
            f"**{_mb(cache_stats.get('db_size_mb', 0.0))}**",
            "",
            f"Session: {_n(s_saved)} saved ({s_pct}%) · {s_hit_pct}% hit",
            f"Lifetime: {_n(lt_saved)} saved ({lt_pct}%) · {lt_hit_pct}% hit",
            "",
            (f"*{lt_sessions} completed session{'s' if lt_sessions != 1 else ''}*"),
        ]
        return ToolResult(content="\n".join(lines), structured_content=structured_payload)

    if mode == "normal":
        uptime = _uptime(session.get("uptime_s", 0))
        files_read = session.get("files_read", 0)
        files_written = session.get("files_written", 0)
        files_edited = session.get("files_edited", 0)
        diffs = session.get("diffs_served", 0)
        tool_calls: dict[str, int] = session.get("tool_calls", {})
        top_tools = sorted(tool_calls.items(), key=lambda x: x[1], reverse=True)[:5]

        lt_files_read = lifetime.get("files_read", 0)
        lt_files_written = lifetime.get("files_written", 0)
        lt_files_edited = lifetime.get("files_edited", 0)

        rss = cache_stats.get("process_rss_mb")
        mem_str = f"{rss:.0f} MB RSS" if rss is not None else "—"

        lines = [
            "# Semantic Cache Stats",
            "",
            "## Storage",
            (
                f"{_n(cache_stats.get('files_cached', 0))} files · "
                f"{_n(cache_stats.get('total_tokens_cached', 0))} tokens · "
                f"{_n(cache_stats.get('total_documents', 0))} documents · "
                f"{_mb(cache_stats.get('db_size_mb', 0.0))}"
            ),
            "",
            f"## Session  ·  uptime {uptime}",
            (
                f"Saved {_n(s_saved)} tokens ({s_pct}%) · returned "
                f"{_n(session.get('tokens_returned', 0))} · hit rate {s_hit_pct}%"
            ),
            (
                f"Activity: read {files_read} · written {files_written} · "
                f"edited {files_edited} · diffs served {diffs}"
            ),
        ]

        if top_tools:
            lines += [
                "",
                "**Tool calls:** " + " · ".join(f"`{t}` ×{c}" for t, c in top_tools),
            ]

        lines += [
            "",
            f"## Lifetime  ·  {lt_sessions} session{'s' if lt_sessions != 1 else ''}",
            (
                f"Saved {_n(lt_saved)} tokens ({lt_pct}%) · returned "
                f"{_n(lifetime.get('tokens_returned', 0))} · hit rate {lt_hit_pct}%"
            ),
            (
                f"Activity: read {_n(lt_files_read)} · written {_n(lt_files_written)} · "
                f"edited {_n(lt_files_edited)}"
            ),
            "",
            "## System",
            f"{mem_str}",
        ]
        return ToolResult(content="\n".join(lines), structured_content=structured_payload)

    # debug — full raw dump. Serialized compactly: `indent=2` is 15-25% pure
    # whitespace tokens on a nested dict, and this dump exists to be read by
    # whatever is measuring, not laid out.
    return ToolResult(
        content=f"```json\n{json.dumps(cache_stats, separators=(',', ':'))}\n```",
        structured_content=structured_payload,
    )


@mcp.tool(output_schema=register_response_model("clear", ClearResponse))
@_serialized
async def clear(
    ctx: Context,
) -> dict[str, Any]:
    """Empty the cache. Does not touch any project file.

    Removes every cached file entry and returns how many were dropped; the
    next `read`/`batch_read` re-seeds from disk. Use rarely — only to recover
    from stale cache state or force a cold re-seed. Normal reads already
    refresh changed files on their own, so this is seldom needed. Takes no
    arguments.
    """
    state = await _tool_call_state(ctx)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state, "clear", _forward_kwargs(), timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result

    count = await cache.clear()
    cache.metrics.record("clear", None)
    payload: dict[str, Any] = {"ok": True, "tool": "clear", "status": "cleared", "count": count}
    if mode == _MODE_DEBUG:
        payload["output_mode"] = mode
    return _finalize_payload(payload, max_response_tokens)


@mcp.tool(output_schema=register_response_model("delete", DeleteResponse))
@_serialized
async def delete(
    ctx: Context,
    path: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delete one file or symlink and evict its cache entries.

    Use this for explicit single-path removal instead of shelling out. A
    missing path is reported as status `not_found`, not an error.

    Statuses: `deleted` (removed), `would_delete` (dry-run preview only, and
    `dry_run: true` comes back with it), `not_found` (nothing was there).
    Constraints: one path only — no globs, no recursion, no real-directory
    deletes. A symlink path deletes the link itself, never its target.

    Args:
        path: File or symlink path (absolute, or relative to the project root).
        dry_run: Preview the outcome without deleting or evicting the cache.
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state, "delete", _forward_kwargs(), timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result

    target = Path(path).expanduser()
    is_symlink = target.is_symlink()
    exists = target.exists() or is_symlink
    if target.is_dir() and not is_symlink:
        _raise_tool_error(
            "delete",
            "directory deletion is not supported; delete only removes one file or symlink path",
            max_response_tokens,
        )

    try:
        if dry_run:
            payload: dict[str, Any] = {
                "ok": True,
                "tool": "delete",
                "status": "would_delete" if exists else "not_found",
                "path": path,
                "deleted": False,
                "dry_run": True,
                "cache_removed": False,
            }
            if mode == _MODE_DEBUG:
                payload["symlink"] = is_symlink
            return _finalize_payload(payload, max_response_tokens)

        deleted = False
        if exists:
            await aunlink(target, executor=cache._io_executor)
            deleted = True

        cache_removed_count = 0
        for candidate in _delete_cache_candidates(target):
            cache_removed_count += await cache.delete_path(candidate)

        cache.metrics.record("delete", None)
        payload = {
            "ok": True,
            "tool": "delete",
            "status": "deleted" if deleted else "not_found",
            "path": path,
            "deleted": deleted,
            "dry_run": False,
            "cache_removed": cache_removed_count > 0,
        }
        if mode == _MODE_DEBUG:
            payload["symlink"] = is_symlink
        return _finalize_payload(payload, max_response_tokens)

    except FileNotFoundError:
        payload = {
            "ok": True,
            "tool": "delete",
            "status": "not_found",
            "path": path,
            "deleted": False,
            "dry_run": False,
            "cache_removed": False,
        }
        if mode == _MODE_DEBUG:
            payload["symlink"] = is_symlink
        return _finalize_payload(payload, max_response_tokens)
    except PermissionError as e:
        _raise_tool_error("delete", f"permission denied - {e}", max_response_tokens)
    except OSError as e:
        _raise_tool_error("delete", f"I/O operation failed - {e}", max_response_tokens)
    except ToolError:
        raise
    except Exception as e:
        logger.exception("Unexpected error in delete")
        _raise_tool_error("delete", str(e), max_response_tokens)


@mcp.tool(output_schema=register_response_model("write", WriteResponse))
@_serialized
async def write(
    ctx: Context,
    path: str,
    content: str,
    create_parents: bool = True,
    dry_run: bool = False,
    auto_format: bool = False,
    show_diff: bool = False,
    append: bool = False,
    known_hash: str | None = None,
) -> dict[str, Any]:
    """Create a file or replace its entire contents.

    Use this for new files or full rewrites; for localized changes prefer
    `edit` or `batch_edit`. Status is `created` for a new path or `updated`
    for an existing one; an update reports `diff_state`, and includes the diff
    against the previous content only when you ask with `show_diff`. A
    `dry_run` writes nothing and says so: the status is
    `would_create`/`would_update` and `dry_run: true` comes back with it.
    Writing refreshes the cache so later reads, `grep`, and `search` see the
    new text. The response carries the new `content_hash`;
    pass it back as `read`'s `known_hash`, or as a `batch_read` `known_hashes`
    entry, to get `unchanged` instead of re-reading the file you just wrote.
    Missing parent directories are created unless `create_parents=false`.

    A full write supplies the whole file, so its hash is yours to keep. An
    append only adds a tail, so pass `known_hash` to show you held the rest —
    without it you get `file_hash` instead, since you cannot vouch for a file
    you have only seen the end of. `auto_format` reports `file_hash` too: the
    formatter's output is not what you sent.

    Args:
        path: File path to create or replace (absolute, or relative to root).
        content: Full file content, or the text to append when `append=true`.
        create_parents: Create any missing parent directories.
        dry_run: Preview the result without writing.
        auto_format: Run the formatter after writing.
        show_diff: Return the unified diff even on a deterministic write.
        append: Append `content` to the end of the file instead of overwriting.
        known_hash: The `content_hash` you hold for this file. Only needed for
            `append`, to prove you hold the part you are not resending.
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state, "write", _forward_kwargs(), timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result

    try:
        result = await _shielded_write(
            cache,
            smart_write(
                cache=cache,
                path=path,
                content=content,
                create_parents=create_parents,
                dry_run=dry_run,
                auto_format=auto_format,
                append=append,
            ),
        )
        cache.metrics.record("write", result)

        # A preview is not a mutation, so it does not report one. `would_*`
        # mirrors the vocabulary `delete` already uses for its dry run, and the
        # explicit `dry_run` flag lets a caller branch on either.
        if result.dry_run:
            status = "would_create" if result.created else "would_update"
        else:
            status = "created" if result.created else "updated"
        payload: dict[str, Any] = {
            "ok": True,
            "tool": "write",
            "status": status,
            "path": result.path,
        }
        if result.dry_run:
            payload["dry_run"] = True
        # Surface the resulting hash so the caller can echo it as known_hash on a
        # later read and get `unchanged` instead of a re-send. Only when the write
        # actually landed; a dry_run leaves disk and cache untouched. A full write
        # supplies the whole file, so it needs no prior possession; an append only
        # extends a base the caller must already have held, and a formatter pass
        # rewrites the file on its own terms, so neither is derivable without it.
        if not result.dry_run:
            base_held = (
                not append
                or result.previous_hash is None
                or hash_matches(known_hash, result.previous_hash)
            )
            _stamp_result_hash(
                payload, result.content_hash, caller_holds=base_held and not auto_format
            )
        if result.created:
            payload["diff_state"] = "none"
        else:
            _apply_mutation_diff(
                payload,
                diff_content=result.diff_content,
                mode=mode,
                show_diff=show_diff,
            )

        if mode in _MODE_NORMAL:
            payload["created"] = result.created
            payload["tokens_saved"] = result.tokens_saved
        if mode == _MODE_DEBUG:
            payload["bytes_written"] = result.bytes_written
            payload["tokens_written"] = result.tokens_written
            payload["diff_stats"] = result.diff_stats
            payload["from_cache"] = result.from_cache

        return _finalize_payload(payload, max_response_tokens)

    except RuntimeError as e:
        if "shutting down" in str(e):
            _raise_tool_error("write", "server is shutting down", max_response_tokens)
        _raise_tool_error("write", str(e), max_response_tokens)
    except FileNotFoundError as e:
        _raise_tool_error("write", str(e), max_response_tokens)
    except PermissionError as e:
        _raise_tool_error("write", f"permission denied - {e}", max_response_tokens)
    except ValueError as e:
        _raise_tool_error("write", str(e), max_response_tokens)
    except TimeoutError:
        _handle_timeout(cache, "write", path)
        _raise_tool_error("write", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except OSError as e:
        logger.warning(f"I/O error in write: {e}")
        _raise_tool_error("write", f"I/O operation failed - {e}", max_response_tokens)
    except ToolError:
        raise
    except Exception:
        logger.exception("Unexpected error in write")
        _raise_tool_error(
            "write",
            "Internal error occurred while writing file",
            max_response_tokens,
        )


@mcp.tool(output_schema=register_response_model("edit", EditResponse))
@_serialized
async def edit(
    ctx: Context,
    path: str,
    old_string: str | None = None,
    new_string: str = "",
    replace_all: bool = False,
    dry_run: bool = False,
    auto_format: bool = False,
    show_diff: bool = False,
    start_line: int | None = None,
    end_line: int | None = None,
    known_hash: str | None = None,
) -> dict[str, Any]:
    """Edit one file by exact text replacement.

    Three modes:
    - find/replace: `old_string` + `new_string` (the default).
    - scoped: add `start_line`/`end_line` to confine the search to a range.
    - line-range: omit `old_string` and give both lines to replace them wholesale.

    `old_string` must match exactly — whitespace and indentation included —
    and, unless `replace_all=true`, must be unique, or the edit fails. Use
    `edit_preview` first if you're unsure an anchor is unique. Returns the
    replacement count and the affected line numbers, and refreshes the cache.
    The diff itself is omitted unless you ask for it with `show_diff`;
    `diff_state` always tells you which you got. A `dry_run` writes nothing
    and says so: the status is `would_edit` and `dry_run: true` comes back
    with it. For several edits to
    one file use `batch_edit`; for a full rewrite use `write`.

    Pass `known_hash` — the hash you hold for this file — and the response
    carries the new `content_hash`, so you never need a read after an edit just
    to learn what the file now contains: you held the old text and you know what
    you replaced. Without it (or with `auto_format`, whose output is not what
    you asked for) the result comes back as `file_hash`, which cannot be
    redeemed as `known_hash` — editing a file is not the same as having read it.

    Args:
        path: File path to modify (absolute, or relative to root).
        old_string: Exact text to find. Omit only for a line-range replacement.
        new_string: Replacement text (an empty string deletes the match).
        replace_all: Replace every occurrence instead of requiring a unique match.
        dry_run: Preview without writing.
        auto_format: Run the formatter after editing.
        show_diff: Return the diff even on a deterministic edit.
        start_line: 1-based inclusive start line for a scoped or line-range edit.
        end_line: 1-based inclusive end line for a scoped or line-range edit.
        known_hash: The `content_hash` from your last read of this file. Proves
            you hold the text being edited, so the result can be handed back as
            a claimable `content_hash`.
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state, "edit", _forward_kwargs(), timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result

    timer = _PhaseTimer()
    try:
        result = await _shielded_write(
            cache,
            smart_edit(
                cache=cache,
                path=path,
                old_string=old_string,
                new_string=new_string,
                replace_all=replace_all,
                dry_run=dry_run,
                auto_format=auto_format,
                start_line=start_line,
                end_line=end_line,
                timer=timer,
            ),
        )
        cache.metrics.record("edit", result)

        payload: dict[str, Any] = {
            "ok": True,
            "tool": "edit",
            # A preview is not a mutation, so it does not report one.
            "status": "would_edit" if dry_run else "edited",
            "path": result.path,
            # matches_found always equals replacements_made — collapse to one field
            "replaced": result.replacements_made,
            "line_numbers": result.line_numbers,
        }
        if dry_run:
            payload["dry_run"] = True
        # Surface the resulting hash so the caller can echo it as known_hash on a
        # later read and skip re-reading the file it just edited. Skip on dry_run,
        # which leaves disk and cache untouched. An edit is only derivable by a
        # caller that held the pre-edit content — it knows what it replaced — and
        # a formatter pass rewrites the result beyond what it asked for.
        if not dry_run:
            base_held = hash_matches(known_hash, result.previous_hash)
            _stamp_result_hash(
                payload, result.content_hash, caller_holds=base_held and not auto_format
            )
        _apply_mutation_diff(
            payload,
            diff_content=result.diff_content,
            mode=mode,
            show_diff=show_diff,
        )
        if mode in _MODE_NORMAL:
            payload["tokens_saved"] = result.tokens_saved
        if mode == _MODE_DEBUG:
            payload["diff_stats"] = result.diff_stats
            payload["from_cache"] = result.from_cache
            payload["params"] = {
                "replace_all": replace_all,
                "dry_run": dry_run,
                "auto_format": auto_format,
                "show_diff": show_diff,
            }

        return _finalize_payload(payload, max_response_tokens)

    except RuntimeError as e:
        if "shutting down" in str(e):
            _raise_tool_error("edit", "server is shutting down", max_response_tokens)
        _raise_tool_error("edit", str(e), max_response_tokens)
    except FileNotFoundError as e:
        _raise_tool_error("edit", str(e), max_response_tokens)
    except PermissionError as e:
        _raise_tool_error("edit", f"permission denied - {e}", max_response_tokens)
    except ValueError as e:
        _raise_tool_error("edit", str(e), max_response_tokens)
    except TimeoutError:
        _handle_timeout(cache, "edit", path)
        _raise_tool_error(
            "edit",
            f"timed out in phase '{timer.current_phase}' after "
            f"{timer.elapsed():.1f}s (budget {_TOOL_TIMEOUT}s)",
            max_response_tokens,
        )
    except OSError as e:
        logger.warning(f"I/O error in edit: {e}")
        _raise_tool_error("edit", f"I/O operation failed - {e}", max_response_tokens)
    except ToolError:
        raise
    except Exception:
        logger.exception("Unexpected error in edit")
        _raise_tool_error("edit", "Internal error occurred while editing file", max_response_tokens)


@mcp.tool(output_schema=register_response_model("edit_preview", EditPreviewResponse))
@_serialized
async def edit_preview(
    ctx: Context,
    path: str,
    old_string: str,
) -> dict[str, Any]:
    """Show where `old_string` would match in a file, without editing it.

    Returns the match count, 1-based line numbers, and short snippets so you
    can confirm an anchor is unique before calling `edit`. Read-only and cheap
    (kept under ~200 tokens), so use it freely as a probe. Raises an error on
    a binary file or an empty `old_string`.

    Args:
        path: File path to search (absolute, or relative to root).
        old_string: Anchor text to locate. Must match exactly, including
            whitespace and indentation. Cannot be empty.
    """
    state = await _tool_call_state(ctx)
    # Resolve against the client root *before* forwarding, like read/edit/write:
    # _forward_kwargs() snapshots this frame's locals, and the worker process
    # has no client root, so a raw relative path would resolve against the
    # worker cwd and preview the wrong file.
    path = state.resolve(path)
    # edit_preview reads through the cache via smart_read, so under the
    # supervisor runtime the real cache lives in the worker — forward there
    # like read/grep/search rather than calling smart_read on the proxy.
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state, "edit_preview", _forward_kwargs(), timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result

    cache = state.cache
    max_response_tokens = state.max_response_tokens

    if not old_string:
        _raise_tool_error("edit_preview", "old_string cannot be empty", max_response_tokens)

    try:
        result = await asyncio.wait_for(
            smart_read(
                cache=cache,
                path=path,
                max_size=MAX_CONTENT_SIZE * 10,
                diff_mode=False,
                force_full=True,
                refresh_cache=False,
                # Anchors and their line numbers only mean anything against the
                # literal file. Summarizing first made this tool answer
                # "found: false" for text that is in the file, and quote line
                # numbers belonging to the summary — a confident wrong answer to
                # the one question it exists to settle.
                summarize=False,
            ),
            timeout=_TOOL_TIMEOUT,
        )
    except FileNotFoundError as e:
        _raise_tool_error("edit_preview", str(e), max_response_tokens)
    except TimeoutError:
        _raise_tool_error("edit_preview", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except ValueError as e:
        # smart_read raises ValueError for a non-regular file — a directory,
        # FIFO, or device. Surface its message as a clean ToolError.
        _raise_tool_error("edit_preview", str(e), max_response_tokens)
    except OSError as e:
        # An unreadable regular file reaches smart_read as an OSError
        # (PermissionError, ...). Surface a clean ToolError instead of
        # leaking an internal -32603, matching read/read_image.
        _raise_tool_error("edit_preview", f"cannot read file: {e}", max_response_tokens)

    if result.is_binary:
        _raise_tool_error("edit_preview", f"binary file not supported: {path}", max_response_tokens)

    content = result.content
    match_count, line_numbers = find_edit_anchors(content, old_string, max_results=50)

    # Build small context snippets — cap at 5 entries and 120 chars/line so
    # the response fits well under 200 tokens even on dense files.
    lines = content.splitlines()
    context: list[dict[str, Any]] = []
    for ln in line_numbers[:5]:
        if 1 <= ln <= len(lines):
            snippet = lines[ln - 1].rstrip()[:120]
            context.append({"line": ln, "snippet": snippet})

    payload: dict[str, Any] = {
        "ok": True,
        "tool": "edit_preview",
        "path": path,
        "found": match_count > 0,
        "match_count": match_count,
        "line_numbers": line_numbers,
        "context": context,
    }
    if match_count > len(line_numbers):
        payload["truncated"] = True

    return _finalize_payload(payload, max_response_tokens)


@mcp.tool(output_schema=register_response_model("batch_edit", BatchEditResponse))
@_serialized
async def batch_edit(
    ctx: Context,
    path: str,
    edits: str,
    dry_run: bool = False,
    auto_format: bool = False,
    show_diff: bool = False,
    known_hash: str | None = None,
) -> dict[str, Any]:
    """Apply many exact edits to one file in a single atomic call.

    Preferred over repeated `edit` calls on the same file: one response,
    applied atomically, faster on large files. Partial success is allowed —
    any failed edits are returned with their reason so you can retry just the
    misses (status is `edited` when all apply, `partial` when some fail,
    `no_changes` when none do). A `dry_run` writes nothing and says so: the
    status becomes `would_edit`/`would_partial` and `dry_run: true` comes back
    with it. For edits across different files, call the tool once per file.

    `edits` is a JSON array; each entry is one of:
    - `[old, new]` — exact find/replace.
    - `[old, new, start_line, end_line]` — find/replace confined to a range.
    - `[null, new, start_line, end_line]` — replace that line range wholesale.
    - `{"old": ..., "new": ..., "start_line": ..., "end_line": ...}` — object form.

    Prefer line-range entries when you already have line numbers from `read`.
    Pass `known_hash` — the hash you hold for this file — and the response
    carries the new `content_hash`, so you need no read afterwards to learn what
    the file now contains. Without it (or with `auto_format`) the result comes
    back as `file_hash`, which cannot be redeemed as `known_hash`.

    Args:
        path: File path to modify (absolute, or relative to root).
        edits: JSON array of edit entries, in any of the forms above.
        dry_run: Preview without writing.
        auto_format: Run the formatter after all edits.
        show_diff: Return the full diff even on a deterministic all-success batch.
        known_hash: The `content_hash` from your last read of this file. Proves
            you hold the text being edited, so the result can be handed back as
            a claimable `content_hash`.
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state, "batch_edit", _forward_kwargs(), timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result

    try:
        # Parse edits JSON
        edits_str = edits.strip()
        if not edits_str.startswith("["):
            _raise_tool_error(
                "batch_edit",
                "edits must be a JSON array of [old, new] pairs",
                max_response_tokens,
            )

        edit_list = json.loads(edits_str)

        # Convert to list of 4-tuples: (old | None, new, start_line | None, end_line | None)
        edit_tuples: list[tuple[str | None, str, int | None, int | None]] = []
        for item in edit_list:
            if isinstance(item, list) and len(item) == 2:
                old = str(item[0]) if item[0] is not None else None
                edit_tuples.append((old, str(item[1]), None, None))
            elif isinstance(item, list) and len(item) == 4:
                old = str(item[0]) if item[0] is not None else None
                sl = int(item[2]) if item[2] is not None else None
                el = int(item[3]) if item[3] is not None else None
                edit_tuples.append((old, str(item[1]), sl, el))
            elif isinstance(item, dict) and "new" in item:
                old = str(item["old"]) if item.get("old") is not None else None
                sl = int(item["start_line"]) if item.get("start_line") is not None else None
                el = int(item["end_line"]) if item.get("end_line") is not None else None
                edit_tuples.append((old, str(item["new"]), sl, el))
            else:
                _raise_tool_error(
                    "batch_edit",
                    "Each edit must be [old, new], [old, new, start, end], "
                    "or {old, new, start_line?, end_line?}",
                    max_response_tokens,
                )

        result = await _shielded_write(
            cache,
            smart_batch_edit(
                cache=cache,
                path=path,
                edits=edit_tuples,
                dry_run=dry_run,
                auto_format=auto_format,
            ),
        )
        cache.metrics.record("batch_edit", result)

        status = (
            "edited"
            if result.failed == 0
            else ("partial" if result.succeeded > 0 else "no_changes")
        )
        # A partial batch always ships its diff, so the caller can see which
        # edits landed. Captured before the dry-run rename below, which would
        # otherwise make `would_partial` fail the check and omit the diff on
        # exactly the preview that needs it most.
        is_partial = status == "partial"
        # A preview is not a mutation, so it does not report one. The counts
        # still describe what would happen; only the verb changes. `no_changes`
        # needs no prefix — nothing would change either way.
        if dry_run:
            status = {"edited": "would_edit", "partial": "would_partial"}.get(status, status)
        payload: dict[str, Any] = {
            "ok": True,
            "tool": "batch_edit",
            "status": status,
            "path": result.path,
            "succeeded": result.succeeded,
        }
        if dry_run:
            payload["dry_run"] = True
        # Omit failed when 0 — saves tokens in the common all-succeed case
        if result.failed:
            payload["failed"] = result.failed
            # Surface failure details so LLM can retry without a separate debug call
            payload["failures"] = [
                {
                    "old": (o.old_string[:60] + "..." if len(o.old_string) > 60 else o.old_string),
                    "error": o.error,
                }
                for o in result.outcomes
                if not o.success
            ]
        # Surface the resulting hash so the caller can echo it as known_hash on a
        # later read and skip re-reading the file it just edited. Only when an edit
        # actually applied (cache refreshed); dry_run and all-failed leave disk and
        # cache untouched. Derivable only for a caller that held the pre-edit
        # content and did not hand the result to a formatter.
        if not dry_run and result.succeeded > 0:
            base_held = hash_matches(known_hash, result.previous_hash)
            _stamp_result_hash(
                payload, result.content_hash, caller_holds=base_held and not auto_format
            )
        _apply_mutation_diff(
            payload,
            diff_content=result.diff_content,
            mode=mode,
            show_diff=show_diff,
            partial=is_partial,
        )
        if mode in _MODE_NORMAL:
            payload["tokens_saved"] = result.tokens_saved
        if mode == _MODE_DEBUG:
            payload["outcomes"] = [
                {
                    "old": o.old_string,
                    "new": o.new_string,
                    "success": o.success,
                    "line_number": o.line_number,
                    "error": o.error,
                }
                for o in result.outcomes
            ]
            payload["diff_stats"] = result.diff_stats
            payload["from_cache"] = result.from_cache
            payload["params"] = {
                "dry_run": dry_run,
                "auto_format": auto_format,
                "show_diff": show_diff,
            }

        return _finalize_payload(payload, max_response_tokens)

    except RuntimeError as e:
        if "shutting down" in str(e):
            _raise_tool_error("batch_edit", "server is shutting down", max_response_tokens)
        _raise_tool_error("batch_edit", str(e), max_response_tokens)
    except json.JSONDecodeError as e:
        _raise_tool_error("batch_edit", f"Invalid JSON in edits - {e}", max_response_tokens)
    except FileNotFoundError as e:
        _raise_tool_error("batch_edit", str(e), max_response_tokens)
    except PermissionError as e:
        _raise_tool_error("batch_edit", f"permission denied - {e}", max_response_tokens)
    except ValueError as e:
        _raise_tool_error("batch_edit", str(e), max_response_tokens)
    except TimeoutError:
        _handle_timeout(cache, "batch_edit", path)
        _raise_tool_error("batch_edit", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except ToolError:
        raise
    except Exception:
        logger.exception("Unexpected error in batch_edit")
        _raise_tool_error(
            "batch_edit",
            "Internal error occurred while editing file",
            max_response_tokens,
        )


@mcp.tool(output_schema=register_response_model("search", SearchResponse))
@_serialized
async def search(
    ctx: Context,
    query: str,
    k: int = 10,
    directory: str | None = None,
    show_preview: bool = False,
) -> dict[str, Any]:
    """Find cached files by keyword relevance (BM25 ranking).

    Searches only files already in the cache — index them first with `warm`,
    which returns counts rather than content (thin results usually mean too
    few files are cached).
    Ranks by BM25 term relevance, so multi-word and keyword queries work
    well; matching is lexical, not embedding-based, so synonyms won't match a
    word that isn't present. Terms are OR'd — a word the corpus happens not to
    contain costs you ranking, not the whole result set. For an exact string or
    regex use `grep`; to pull more of the repo into the cache use `batch_read`.
    Returns matches with a normalized 0–1 relevance score (best match = 1.0)
    and a short preview.

    Args:
        query: Keywords to rank by. Natural-language phrasing is fine, but
            ranking is on the individual words.
        k: Maximum number of matches to return.
        directory: Restrict matches to files under this directory.
        show_preview: Include a short preview line for each match.
    """
    state = await _tool_call_state(ctx)
    directory = state.resolve(directory) if directory else None
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state, "search", _forward_kwargs(), timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result

    try:
        result = await asyncio.wait_for(
            semantic_search(cache, query, k=k, directory=directory),
            timeout=_TOOL_TIMEOUT,
        )
        cache.metrics.record("search", result)

        match_payload: list[dict[str, Any]] = []
        root = shared_root(
            [m.path for m in result.matches],
            str(state.client_root) if state.client_root else None,
        )
        for m in result.matches:
            # Two decimals of a normalized rank score: the extra digits change
            # no decision and cost ~2 tokens each.
            item: dict[str, Any] = {
                "path": relativize(m.path, root),
                "similarity": round(m.similarity, 2),
            }
            if mode in _MODE_NORMAL:
                item["tokens"] = m.tokens
            if show_preview or mode == _MODE_DEBUG:
                item["preview"] = m.preview
            match_payload.append(item)

        payload: dict[str, Any] = {
            "ok": True,
            "tool": "search",
            "matches": match_payload,
        }
        if root is not None:
            payload["root"] = root
        # `count` is not emitted: it is `len(matches)`, and the array is right
        # there. Echoing `query` back is wasteful for the same reason — the
        # caller just sent it — so it is kept for debug traceability only.
        if mode in _MODE_NORMAL:
            payload["cached_files"] = result.cached_files
        if mode == _MODE_DEBUG:
            payload["query"] = query
            payload["k"] = k
            payload["directory"] = directory
            payload["show_preview"] = show_preview

        return _finalize_payload(payload, max_response_tokens)

    except TimeoutError:
        _handle_timeout(cache, "search", query[:50])
        _raise_tool_error("search", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except ToolError:
        raise
    except Exception as e:
        logger.exception("Error in search")
        _raise_tool_error("search", str(e), max_response_tokens)


@mcp.tool(output_schema=register_response_model("batch_read", BatchReadResponse))
@_serialized
async def batch_read(
    ctx: Context,
    paths: str,
    max_total_tokens: int = 50000,
    priority: str = "",
    known_hashes: str = "",
) -> dict[str, Any]:
    """Read several files at once under a shared token budget.

    Cheaper than many single `read` calls. To make files searchable without
    reading them at all, use `warm` instead — this tool returns their text.
    Every file is returned in full unless you
    prove you still hold it: pass `known_hashes` and each proven file collapses
    to an `unchanged` count, or to a diff when it moved on disk. Each file sent
    in full comes back with its `content_hash` — keep those and pass them next
    time. A file large enough to come back summarized carries none: a summary is
    not the file, so it cannot be vouched for. Smallest files are read first, and
    a file too big for the remaining budget is listed under `skipped` while
    smaller ones keep being read — so one large file cannot starve the rest of
    the batch. Recover anything skipped with `read` using `offset`/`limit`.

    Args:
        paths: The files to read — a comma-separated list, a JSON array, or
            glob patterns (expanded for you).
        max_total_tokens: Total token budget shared across the whole batch.
        priority: Optional paths to read first, ahead of the remaining files.
            Ordering only — a priority file still has to fit the budget, and is
            skipped like any other when it does not.
        known_hashes: JSON object mapping a path to the `content_hash` you
            still hold for it, e.g. `{"src/a.py": "8f3c..."}`. Any file you
            cannot vouch for this way is sent in full.
    """
    state = await _tool_call_state(ctx)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens

    try:
        path_list = _resolve_path_list(paths, state)
        priority_list = _resolve_path_list(priority, state) if priority.strip() else None
        try:
            hash_claims = _parse_known_hashes(known_hashes, state)
        except ValueError as e:
            _raise_tool_error("batch_read", str(e), max_response_tokens)
        remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
            state,
            "batch_read",
            _forward_kwargs(
                overrides={
                    "paths": json.dumps(path_list),
                    "priority": json.dumps(priority_list) if priority_list else "",
                    "known_hashes": json.dumps(hash_claims) if hash_claims else "",
                }
            ),
            timeout=_TOOL_TIMEOUT * 2,
        )
        if remote_result is not None:
            return remote_result

        # Expand glob patterns off the event loop: a single blocking glob()
        # step can't be interrupted by _expand_globs' internal deadline, so
        # run it in the IO executor and bound the wait here.
        try:
            path_list = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    cache._io_executor, _expand_globs, path_list
                ),
                timeout=_EXPAND_GLOBS_TIMEOUT + 1,
            )
        except TimeoutError:
            logger.warning("Glob expansion exceeded hard timeout; using raw paths")

        result = await asyncio.wait_for(
            batch_smart_read(
                cache,
                path_list,
                max_total_tokens=max_total_tokens,
                priority=priority_list,
                known_hashes=hash_claims,
            ),
            timeout=_TOOL_TIMEOUT * 2,
        )  # batch gets double timeout
        cache.metrics.record("batch_read", result)

        # Build restructured response — separate unchanged, skipped, and content files
        summary: dict[str, Any] = {
            "files_read": result.files_read,
            "files_skipped": result.files_skipped,
        }
        if mode in _MODE_NORMAL:
            summary["total_tokens"] = result.total_tokens
            summary["tokens_saved"] = result.tokens_saved
        if result.unchanged_paths:
            summary["unchanged_count"] = len(result.unchanged_paths)
            if mode == _MODE_DEBUG:
                summary["unchanged"] = result.unchanged_paths

        skipped_items: list[dict[str, Any]] = []
        file_items: list[dict[str, Any]] = []
        for f in result.files:
            if f.status == "skipped":
                skipped_item: dict[str, Any] = {"path": f.path}
                if f.est_tokens is not None:
                    skipped_item["est_tokens"] = f.est_tokens
                if mode == _MODE_DEBUG:
                    skipped_item["hint"] = "use read with offset/limit"
                skipped_items.append(skipped_item)
            elif f.status == "unchanged":
                # Already captured in summary.unchanged — no per-file entry needed
                continue
            else:
                # full, diff, truncated — entries with actual content
                item: dict[str, Any] = {"path": f.path, "status": f.status}
                if f.path in result.contents:
                    item["content"] = result.contents[f.path]
                if f.status == "truncated":
                    content = result.contents.get(f.path, "")
                    returned_lines = content.count("\n") + 1 if content else 0
                    item["hint"] = (
                        f"Truncated. Use read with offset={returned_lines + 1} "
                        f"to continue. Do NOT re-read from the beginning."
                    )
                # Only exact deliveries carry a hash — pass it back as a
                # `known_hashes` entry next time to skip re-sending this file.
                if f.content_hash is not None:
                    item["content_hash"] = short_hash(f.content_hash)
                if mode == _MODE_DEBUG:
                    item["tokens"] = f.tokens
                    item["from_cache"] = f.from_cache
                file_items.append(item)

        payload: dict[str, Any] = {
            "ok": True,
            "tool": "batch_read",
            "summary": summary,
        }
        if skipped_items:
            summary["hint"] = "Use read with offset/limit for skipped files."
            payload["skipped"] = skipped_items
        payload["files"] = file_items

        return _finalize_payload(payload, max_response_tokens)

    except json.JSONDecodeError:
        _raise_tool_error(
            "batch_read",
            "Invalid paths format. Use comma-separated or JSON array.",
            max_response_tokens,
        )
    except TimeoutError:
        _handle_timeout(cache, "batch_read")
        _raise_tool_error(
            "batch_read",
            f"timed out after {_TOOL_TIMEOUT * 2}s",
            max_response_tokens,
        )
    except ToolError:
        raise
    except Exception as e:
        logger.exception("Error in batch_read")
        _raise_tool_error("batch_read", str(e), max_response_tokens)


@mcp.tool(output_schema=register_response_model("warm", WarmResponse))
@_serialized
async def warm(
    ctx: Context,
    paths: str,
    max_files: int = 200,
) -> dict[str, Any]:
    """Index files into the cache so `grep` and `search` can see them, returning counts only.

    `grep` and `search` read only cached files, and every other way to cache a
    file returns its text. Warming a tree with `batch_read` therefore costs the
    whole tree in tokens before the first search runs. This costs a few dozen
    tokens however many files it indexes: no content, no previews, no paths for
    the files that succeeded — just how many were indexed, how many were
    already current, and how many were not.

    Anything not indexed is counted in `skipped`, and the first few come back
    under `failures` with a reason (`not_found`, `not_a_file`, `binary`,
    `too_large`, `unreadable`, `timeout`). If a cap stopped the walk early you
    get `truncated` or `incomplete` rather than a short count that looks
    complete.

    Use it before searching an unfamiliar tree, then `grep` for the exact
    string or `search` for the concept, and `read` only what those name.

    Args:
        paths: Files to index — a comma-separated list, a JSON array, or glob
            patterns (expanded for you, e.g. `src/**/*.py`).
        max_files: Cap on files indexed in this call. Matches beyond it are
            left out and flagged with `truncated`.
    """
    state = await _tool_call_state(ctx)
    cache = state.cache
    max_response_tokens = state.max_response_tokens

    if not 1 <= max_files <= _WARM_MAX_FILES:
        _raise_tool_error(
            "warm",
            f"max_files must be between 1 and {_WARM_MAX_FILES}, got {max_files}",
            max_response_tokens,
        )

    try:
        path_list = _resolve_path_list(paths, state)
    except json.JSONDecodeError as e:
        _raise_tool_error(
            "warm",
            f"paths must be a comma-separated list or a JSON array: {e}",
            max_response_tokens,
        )
    if not path_list:
        _raise_tool_error(
            "warm", "paths is empty — name at least one file or glob", max_response_tokens
        )

    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state,
        "warm",
        _forward_kwargs(overrides={"paths": json.dumps(path_list)}),
        timeout=_TOOL_TIMEOUT * 2,
    )
    if remote_result is not None:
        return remote_result

    import time  # noqa: PLC0415  # module-local, matching _expand_globs_detailed

    # Ask for one more than the cap so a full page can be told apart from an
    # overflowing one; without it a caller whose tree is exactly `max_files`
    # long would be warned of a truncation that never happened.
    try:
        expanded, glob_truncated = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(
                cache._io_executor, _expand_globs_detailed, path_list, max_files + 1
            ),
            timeout=_EXPAND_GLOBS_TIMEOUT + 1,
        )
    except TimeoutError:
        logger.warning("warm: glob expansion exceeded hard timeout; using raw paths")
        expanded, glob_truncated = path_list, True

    truncated = glob_truncated or len(expanded) > max_files
    targets = expanded[:max_files]

    warmed = 0
    already_current = 0
    skipped = 0
    tokens_indexed = 0
    total_bytes = 0
    incomplete = False
    failures: list[dict[str, str]] = []

    def _skip(path: str, reason: str) -> None:
        """Count a file that will not be indexed, and name it while there is room."""
        nonlocal skipped
        skipped += 1
        if len(failures) < _WARM_MAX_FAILURES_REPORTED:
            failures.append({"path": path, "reason": reason})

    deadline = time.monotonic() + _TOOL_TIMEOUT * 2

    for raw_path in targets:
        if time.monotonic() > deadline:
            incomplete = True
            break

        candidate = Path(raw_path).expanduser()
        try:
            st = await astat(candidate, cache._io_executor)
        except FileNotFoundError:
            _skip(raw_path, "not_found")
            continue
        except OSError:
            _skip(raw_path, "unreadable")
            continue

        if not stat_module.S_ISREG(st.st_mode):
            _skip(raw_path, "not_a_file")
            continue
        if st.st_size > _WARM_MAX_FILE_BYTES:
            _skip(raw_path, "too_large")
            continue
        if total_bytes + st.st_size > _WARM_MAX_TOTAL_BYTES:
            # Stop rather than skip: the budget is spent, and every remaining
            # file would report the same reason. `incomplete` says so once.
            incomplete = True
            break

        abs_path = str(candidate.resolve())
        entry = await cache.get(abs_path)
        if entry is not None and entry.mtime >= st.st_mtime:
            already_current += 1
            tokens_indexed += entry.tokens
            continue

        try:
            result = await asyncio.wait_for(
                smart_read(
                    cache=cache,
                    path=str(candidate),
                    diff_mode=False,
                    force_full=True,
                    refresh_cache=False,  # smart_read still refreshes when stale
                    # Indexing needs the real bytes; a summary would make grep
                    # answer for text the file does not contain.
                    summarize=False,
                ),
                timeout=_TOOL_TIMEOUT,
            )
        except TimeoutError:
            _skip(raw_path, "timeout")
            continue
        except (OSError, ValueError):
            _skip(raw_path, "unreadable")
            continue

        if result.is_binary:
            _skip(raw_path, "binary")
            continue

        warmed += 1
        tokens_indexed += result.tokens_original
        total_bytes += st.st_size

    payload: dict[str, Any] = {
        "ok": True,
        "tool": "warm",
        "warmed": warmed,
        "already_current": already_current,
        "skipped": skipped,
        "tokens_indexed": tokens_indexed,
    }
    if failures:
        payload["failures"] = failures
    # Both caps can fire in one call, and each names a different remedy, so the
    # hints are joined rather than one silently overwriting the other.
    hints: list[str] = []
    if truncated:
        payload["truncated"] = True
        hints.append(
            f"more files matched than max_files={max_files}; raise it or narrow the pattern"
        )
    if incomplete:
        payload["incomplete"] = True
        hints.append(
            "stopped on the time or byte budget before reaching every file; "
            "warm the rest in smaller batches"
        )
    if hints:
        payload["hint"] = "; ".join(hints)

    return _finalize_payload(payload, max_response_tokens)


@mcp.tool(output_schema=register_response_model("glob", GlobResponse))
@_serialized
async def glob(
    ctx: Context,
    pattern: str,
    directory: str = ".",
    cached_only: bool = False,
) -> dict[str, Any]:
    """List files matching a glob and show which are already cached.

    Use it to discover files and see what `search`/`grep` can already access
    before you spend reads. Each match carries a `cached` flag; set
    `cached_only=true` to list only files already in the cache. Pair it with
    `batch_read` to pull in whatever isn't cached yet.

    Args:
        pattern: Glob pattern to match (e.g. `src/**/*.py`).
        directory: Base directory the pattern is evaluated from.
        cached_only: Return only files that are already cached.
    """
    state = await _tool_call_state(ctx)
    directory = state.resolve(directory)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state, "glob", _forward_kwargs(), timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result

    try:
        result = await glob_with_cache_status(
            cache,
            pattern,
            directory=directory,
            cached_only=cached_only,
        )
        cache.metrics.record("glob", result)
        matches_payload: list[dict[str, Any]] = []
        # Up to 1000 matches come back, each carrying the same absolute prefix.
        # Name the directory once and report the rest relative to it.
        root = shared_root(
            [m.path for m in result.matches],
            str(state.client_root) if state.client_root else None,
        )
        # When all matches are uncached and we're not in debug mode, drop the
        # redundant `cached: false` field — saves ~13 chars per match.
        all_uncached = result.cached_count == 0 and mode != _MODE_DEBUG
        for m in result.matches:
            item: dict[str, Any] = {"path": relativize(m.path, root)}
            if not all_uncached:
                item["cached"] = m.cached
            if mode == _MODE_DEBUG:
                item["tokens"] = m.tokens
                item["mtime"] = m.mtime
            matches_payload.append(item)

        payload: dict[str, Any] = {
            "ok": True,
            "tool": "glob",
            "matches": matches_payload,
            "total_matches": result.total_matches,
            "cached_count": result.cached_count,
        }
        if root is not None:
            payload["root"] = root
        # Echoing pattern/directory back is wasteful in compact mode; the
        # caller already knows what they sent.
        if mode in _MODE_NORMAL:
            payload["pattern"] = pattern
            payload["directory"] = result.directory
        if mode == _MODE_DEBUG:
            payload["total_cached_tokens"] = result.total_cached_tokens

        return _finalize_payload(payload, max_response_tokens)

    except ToolError:
        raise
    except Exception as e:
        logger.exception("Error in glob")
        _raise_tool_error("glob", str(e), max_response_tokens)


def _render_grep_lines(matches: list[dict[str, Any]], context_lines: int) -> list[str]:
    """Render one file's matches as `"<n>:<text>"` / `"<n>-<text>"` strings.

    Two savings live here. The per-match JSON object becomes a string, which
    drops ~16 tokens of braces and repeated key names per match. And context
    windows are merged instead of sliced fresh per match: with
    `context_lines=3`, two matches two lines apart used to re-send the same
    source line up to four times in one response.

    A line that matched always wins over the same line delivered as another
    match's context, so the `:` separator never understates a hit.
    """
    # line number -> (text, matched). Insertion order is irrelevant; the result
    # is sorted by line number, which is also what makes the merge work.
    rendered: dict[int, tuple[str, bool]] = {}

    for match in matches:
        line_number = int(match["line_number"])
        if context_lines > 0:
            before = match.get("before") or ()
            # `before` is the contiguous run ending just above the match, so
            # its first line is that many lines back.
            for offset, text in enumerate(before, start=line_number - len(before)):
                rendered.setdefault(offset, (text, False))
            after = match.get("after") or ()
            for offset, text in enumerate(after, start=line_number + 1):
                rendered.setdefault(offset, (text, False))
        rendered[line_number] = (str(match["line"]), True)

    return [
        f"{number}{_MATCH_SEPARATOR if matched else _CONTEXT_SEPARATOR}{text}"
        for number, (text, matched) in sorted(rendered.items())
    ]


@mcp.tool(output_schema=register_response_model("grep", GrepResponse))
@_serialized
async def grep(
    ctx: Context,
    pattern: str,
    path: str | None = None,
    fixed_string: bool = False,
    case_sensitive: bool = True,
    context_lines: int = 0,
    max_matches: int = 100,
    max_files: int = 50,
    output: str = "matches",
) -> dict[str, Any]:
    """Search cached file contents for an exact string or regex.

    Fast, exact, line-numbered matching over files already in the cache — it
    does NOT touch disk, so index files first with `warm`, which costs a few
    dozen tokens however many files it covers (empty results usually mean the
    files aren't cached). A pattern that is not a
    valid regex is an error, never an empty result, so zero matches always
    means zero matches. For concept-level questions where you don't know the
    exact term, use `search` instead.

    Counts are complete unless the response says otherwise: if a cap stops the
    scan, `complete` comes back false with `limit_reached` naming which one, so
    `total_matches` is never mistaken for the total that exists.

    Each file's hits come back as `"<line>:<text>"` strings under `lines`, with
    context lines using `-` instead of `:`. Paths are relative to the `root`
    the response names, when there is one worth naming.

    A repeated group wrapping an unbounded quantifier (`(a+)+`) is rejected
    rather than run — it can take exponential time and cannot be interrupted
    once started. Drop the redundant repeat, or pass `fixed_string=true`.

    Args:
        pattern: A regular expression, or a literal string when
            `fixed_string=true`.
        path: Optional filter — an exact path, a path suffix, a directory
            (matching every cached file beneath it), or a glob.
        fixed_string: Match `pattern` literally instead of as a regex.
        case_sensitive: Match case-sensitively.
        context_lines: Lines of surrounding context to include around each
            match. Overlapping windows are merged, so no line is sent twice.
        max_matches: Cap on total matches returned across all files.
        max_files: Cap on the number of files returned.
        output: How much to return — `matches` (default), `paths` for the
            matching files without their lines, or `count` for the totals
            alone.
    """
    state = await _tool_call_state(ctx)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    # Validated before forwarding, so a typo fails the same way in both
    # runtimes and never silently falls back to a mode nobody asked for.
    if output not in _GREP_OUTPUT_MODES:
        _raise_tool_error(
            "grep",
            f"output must be one of {', '.join(_GREP_OUTPUT_MODES)}, got {output!r}",
            max_response_tokens,
        )
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state, "grep", _forward_kwargs(), timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result

    try:
        # `context_lines` is an explicit request for more output, so it is
        # honoured in every response mode — the rule `show_diff` already
        # follows. Compact mode suppresses context nobody asked for, which is
        # what the default of 0 already does on its own.
        # Bounded like every other tool. The pattern comes from the caller and
        # `re` has no interrupt, so the scan runs off the event loop (see
        # `_grep.grep`); that is what lets this timeout actually fire.
        results = await asyncio.wait_for(
            cache._storage.grep(
                pattern,
                path=path,
                fixed_string=fixed_string,
                case_sensitive=case_sensitive,
                context_lines=context_lines,
                max_matches=max_matches,
                max_files=max_files,
            ),
            timeout=_TOOL_TIMEOUT,
        )
        cache.metrics.record("grep", None)

        total_matches = sum(len(r["matches"]) for r in results)

        # Apply a soft char budget so a wide regex on a large repo doesn't
        # spend the entire response token cap on match lines. The hard
        # token cap in _finalize_payload still applies as a backstop.
        char_budget: int | None = None
        if max_response_tokens is not None and max_response_tokens > 0:
            # Leave ~512 tokens for the response envelope/metadata, and budget
            # the rest at _JSON_CHARS_PER_TOKEN rather than the ~4 chars/token
            # of prose. Serialized matches are not prose: quoted keys, braces,
            # commas and line numbers all tokenize far denser, so a prose-rate
            # estimate assembles a payload over the hard cap and the whole
            # thing is cut down to the minimal form — losing every match line
            # the budget existed to preserve.
            char_budget = max(1024, (max_response_tokens - 512) * _JSON_CHARS_PER_TOKEN)

        # Every path the store returns is absolute and fully resolved. Naming
        # the directory they share once, and reporting the rest relative to it,
        # roughly halves the bytes a path costs.
        root = shared_root(
            [file_result["path"] for file_result in results],
            str(state.client_root) if state.client_root else None,
        )

        files_payload: list[dict[str, Any]] = []
        truncated_matches = 0
        truncated_files = 0
        running_chars = 0
        budget_exceeded = False

        for file_result in results:
            if output == "count":
                # Nothing per-file goes out, so neither the rendering nor the
                # budget walk has anything to protect. Truncation counters stay
                # at zero — the answer is the totals, and they are complete.
                continue
            entry: dict[str, Any] = {"path": relativize(file_result["path"], root)}
            if output == "paths":
                files_payload.append(entry)
                continue
            if budget_exceeded:
                truncated_files += 1
                truncated_matches += len(file_result["matches"])
                continue

            matches = file_result["matches"]
            kept = matches
            if char_budget is not None:
                # Charge each match its rendered line plus the string envelope,
                # and stop at the first one that would cross the budget. Cutting
                # here leaves the earlier matches in place; the payload-wide cap
                # in `_finalize_payload` is the backstop, not the first line of
                # defence.
                for index, m in enumerate(matches):
                    running_chars += len(m["line"]) + _LINE_ENVELOPE_CHARS
                    if context_lines > 0:
                        for ctx_line in (*m.get("before", ()), *m.get("after", ())):
                            running_chars += len(ctx_line) + _LINE_ENVELOPE_CHARS
                    if running_chars > char_budget:
                        budget_exceeded = True
                        # Keep at least one match: a file listed with no lines
                        # says nothing the `paths` mode would not have said.
                        kept = matches[: max(1, index)]
                        truncated_matches += len(matches) - len(kept)
                        break

            entry["lines"] = _render_grep_lines(kept, context_lines)
            files_payload.append(entry)

        payload: dict[str, Any] = {
            "ok": True,
            "tool": "grep",
            "total_matches": total_matches,
            "files_matched": len(results),
        }
        if root is not None:
            payload["root"] = root
        if output != "count":
            payload["files"] = files_payload
        # Echoing the pattern and path back is wasteful — the caller just sent
        # them — so they are kept for traceability only, matching `search` and
        # `glob`. `_minimal_payload` still preserves `pattern` when present.
        if mode in _MODE_NORMAL:
            payload["pattern"] = pattern
            if path is not None:
                payload["path"] = path
        if truncated_matches > 0 or truncated_files > 0:
            payload["truncated_matches"] = truncated_matches
            payload["files_in_response"] = len(files_payload)
            if truncated_files > 0:
                payload["truncated_files"] = truncated_files
        # The scan itself stops at max_matches/max_files, so the corpus past
        # that point was never examined. Say so explicitly: `total_matches`
        # otherwise reads as the complete count of what exists, and a caller
        # that believes it moves on without ever looking at the rest.
        if not results.complete:
            payload["complete"] = False
            payload["limit_reached"] = "max_matches" if results.truncated_matches else "max_files"
            if results.files_not_searched:
                payload["files_not_searched"] = results.files_not_searched
            # No prose here: `complete: false` and `limit_reached` already say
            # the count is a floor and name the cap that stopped the scan.
        # Distinguish "no files cached under that path" from "no matches".
        # The audit found 22/29 empty greps fit the cache-miss shape, so the
        # caller should know whether to seed via batch_read/glob.
        if total_matches == 0 and path is not None:
            has_cached = await cache._storage.has_cached_paths_under(path)
            if not has_cached:
                payload["reason"] = "no_files_cached_under_path"
                payload["hint"] = "use warm to index this path, then grep again"
        if mode == _MODE_DEBUG:
            payload["fixed_string"] = fixed_string
            payload["case_sensitive"] = case_sensitive
            payload["context_lines"] = context_lines

        return _finalize_payload(payload, max_response_tokens)

    except ToolError:
        raise
    except TimeoutError:
        # A pattern that backtracks catastrophically leaves the executor thread
        # wedged; reset it so the next call is not stuck behind it.
        _handle_timeout(cache, "grep", pattern[:50])
        _raise_tool_error(
            "grep",
            f"timed out after {_TOOL_TIMEOUT}s — the pattern is too expensive to "
            "run over the cached files; simplify it or narrow `path`",
            max_response_tokens,
        )
    except ValueError as e:
        # Bad input (an uncompilable pattern), not a server fault — report it
        # without a traceback, and never as an empty result set.
        _raise_tool_error("grep", str(e), max_response_tokens)
    except Exception as e:
        logger.exception("Error in grep")
        _raise_tool_error("grep", str(e), max_response_tokens)


_EXPAND_GLOBS_TIMEOUT = 5  # seconds — matches GLOB_TIMEOUT_SECONDS


def _expand_globs(raw_paths: list[str], max_files: int = 50) -> list[str]:
    """Expand glob patterns in path list. Non-glob paths pass through unchanged."""
    return _expand_globs_detailed(raw_paths, max_files)[0]


def _expand_globs_detailed(raw_paths: list[str], max_files: int = 50) -> tuple[list[str], bool]:
    """Expand globs, and say whether a limit cut the expansion short.

    Uses a deadline to prevent recursive ``**`` patterns from blocking
    the caller for an unbounded amount of time. The second element is True
    when the deadline or ``max_files`` stopped the walk, so a caller can tell
    "these are the matches" from "these are the first N matches".
    """
    import time  # noqa: PLC0415

    truncated = False

    deadline = time.monotonic() + _EXPAND_GLOBS_TIMEOUT
    expanded: list[str] = []
    glob_chars = frozenset("*?[")
    for p in raw_paths:
        if time.monotonic() > deadline:
            logger.warning(f"Glob expansion timed out after {_EXPAND_GLOBS_TIMEOUT}s")
            truncated = True
            break
        if any(c in p for c in glob_chars):
            try:
                # Split into directory + pattern for Path.glob
                pp = Path(p)
                if pp.is_absolute():
                    # Find the first component with glob chars
                    parts = pp.parts
                    base_parts: list[str] = []
                    pattern_parts: list[str] = []
                    found_glob = False
                    for part in parts:
                        if not found_glob and not any(c in part for c in glob_chars):
                            base_parts.append(part)
                        else:
                            found_glob = True
                            pattern_parts.append(part)
                    base = Path(*base_parts) if base_parts else Path("/")
                    pattern = str(Path(*pattern_parts)) if pattern_parts else "*"
                else:
                    base = Path(".")
                    pattern = p
                if not base.is_dir():
                    expanded.append(p)  # Base doesn't exist — treat as literal
                else:
                    # Iterate lazily with deadline to avoid materializing huge trees
                    remaining = max_files - len(expanded)
                    matches: list[str] = []
                    for m in base.glob(pattern):
                        if time.monotonic() > deadline:
                            logger.warning(f"Glob pattern timed out: {pattern}")
                            truncated = True
                            break
                        if m.is_file():
                            matches.append(str(m))
                            if len(matches) >= remaining:
                                truncated = True
                                break
                    matches.sort()
                    expanded.extend(matches)
            except (OSError, ValueError):
                expanded.append(p)  # Treat invalid pattern as literal
        else:
            expanded.append(p)
        if len(expanded) >= max_files:
            logger.debug(f"Glob expansion truncated at {max_files} files")
            truncated = True
            break
    return expanded[:max_files], truncated or len(expanded) > max_files
