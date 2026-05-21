"""MCP tool handlers."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import stat as stat_module
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, TypeVar, cast

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from mcp.types import ImageContent, TextContent

from ...cache import (
    SemanticCache,
    batch_smart_read,
    compare_files,
    find_edit_anchors,
    find_similar_files,
    glob_with_cache_status,
    semantic_search,
    smart_batch_edit,
    smart_edit,
    smart_read,
    smart_write,
)
from ...cache._helpers import _PhaseTimer
from ...cache.read import _sniff_image_mime
from ...config import MAX_CONTENT_SIZE, TOOL_TIMEOUT
from ...core.embeddings import get_model_info
from ...utils import aread_bytes, astat
from ...utils._async_io import aunlink
from .._mcp import mcp
from .._read_session import get_tracker as _get_read_session_tracker
from .._tool_models import (
    BatchEditResponse,
    BatchReadResponse,
    ClearResponse,
    DeleteResponse,
    DiffResponse,
    EditPreviewResponse,
    EditResponse,
    GlobResponse,
    GrepResponse,
    ReadImageResponse,
    ReadResponse,
    SearchResponse,
    SimilarResponse,
    StatsResponse,
    WriteResponse,
    output_schema,
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

_read_session_tracker = _get_read_session_tracker()

logger = logging.getLogger(__name__)


# Tool timeout from config (env TOOL_TIMEOUT, default 30s).
_TOOL_TIMEOUT: float = TOOL_TIMEOUT

# Global tool mutex: only one tool call executes at a time.
# Prevents concurrent coroutines from interleaving executor tasks,
# catalog reads, and ONNX calls — the root cause of hangs when
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


# Cached client root — resolved once per session via ctx.list_roots().
_client_root: Path | None = None
_client_root_resolved: bool = False


async def _resolve_client_root(ctx: Context) -> Path | None:
    """Fetch and cache the MCP client's project root (first list_roots entry)."""
    global _client_root, _client_root_resolved
    if not _client_root_resolved:
        try:
            roots = await ctx.list_roots()
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


@mcp.tool(
    output_schema=output_schema(ReadResponse),
    meta={
        "version": _pkg_version("semantic-cache-mcp"),
        "author": "Dayton Dunbar",
        "github": "https://github.com/CoderDayton/semantic-cache-mcp",
    },
)
@_serialized
async def read(
    ctx: Context,
    path: str,
    max_size: int = MAX_CONTENT_SIZE,
    offset: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Read a file with token-efficient caching. For 2+ files, use `batch_read`.

    Returns full content on first read, `"unchanged": true` on re-read of an
    unchanged file (content already in your context — do NOT re-read), or a
    unified diff when modified. Use `offset`/`limit` to recover line ranges
    after truncation.

    Args:
        path: File path (absolute or relative to project root). Use absolute
            paths for files outside the current project root.
        max_size: Maximum content size to return before summarization.
        offset: 1-based starting line number for targeted reads. `0` is
            treated as "from the start" (equivalent to omitting).
        limit: Number of lines to return from `offset`.
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state,
        "read",
        {
            "path": path,
            "max_size": max_size,
            "offset": offset,
            "limit": limit,
        },
        timeout=_TOOL_TIMEOUT,
    )
    if remote_result is not None:
        return remote_result

    # Validate bounds. `offset=0` is accepted and treated as from-start.
    if offset is not None and offset < 0:
        _raise_tool_error(
            "read", "offset must be >= 0 (1-based; 0 is from start)", max_response_tokens
        )
    if limit is not None and limit < 1:
        _raise_tool_error("read", "limit must be >= 1", max_response_tokens)
    max_size = max(1, min(max_size, MAX_CONTENT_SIZE * 10))

    try:
        # If offset/limit specified, read specific lines (still caches full file)
        if offset is not None or limit is not None:
            result = await asyncio.wait_for(
                smart_read(
                    cache=cache,
                    path=path,
                    max_size=max_size,
                    diff_mode=False,  # Line ranges bypass diff mode
                    force_full=True,
                    refresh_cache=False,
                ),
                timeout=_TOOL_TIMEOUT,
            )
            cache.metrics.record("read", result)
            if result.is_binary:
                return _finalize_payload(_binary_read_payload(path, result), max_response_tokens)
            lines = result.content.splitlines(keepends=True)
            start = max(0, (offset or 0) - 1)  # Convert to 0-based; offset 0/None both start at 0
            end = start + (limit or len(lines) - start)
            selected = lines[start:end]

            # Format with line numbers like built-in Read tool. Generator
            # expression avoids materializing the intermediate list — `selected`
            # may be thousands of lines on partial reads of large files.
            content = "\n".join(
                f"{i:6d}\t{line.rstrip()}" for i, line in enumerate(selected, start=start + 1)
            )
            line_info = {
                "start": start + 1,
                "end": min(end, len(lines)),
                "total": len(lines),
            }
            payload: dict[str, Any] = {
                "ok": True,
                "tool": "read",
                "path": path,
                "content": content,
                "lines": line_info,
            }
            if mode in _MODE_NORMAL:
                payload["truncated"] = result.truncated
            if mode == _MODE_DEBUG:
                payload["from_cache"] = result.from_cache
                payload["tokens_saved"] = result.tokens_saved

            return _finalize_payload(payload, max_response_tokens)

        result = await asyncio.wait_for(
            smart_read(
                cache=cache,
                path=path,
                max_size=max_size,
            ),
            timeout=_TOOL_TIMEOUT,
        )
        cache.metrics.record("read", result)

        # Binary fallback: structured metadata instead of an error so callers
        # can branch on is_binary without parsing the error string.
        if result.is_binary:
            return _finalize_payload(_binary_read_payload(path, result), max_response_tokens)

        # Detect unchanged files: from_cache=True + is_diff=False means the
        # cached file matches the on-disk file. This is a cache fact, not proof
        # that the current client already has the file text in context — we
        # also gate on the per-session tracker so the first read of a session
        # always returns content even when the cache already has it.
        cache_fresh = result.from_cache and not result.is_diff
        abs_path = str(Path(path).expanduser().resolve())
        session_id = getattr(ctx, "session_id", None) or getattr(ctx, "client_id", None)
        already_seen = _read_session_tracker.seen(session_id, abs_path)
        unchanged = cache_fresh and already_seen

        payload = {
            "ok": True,
            "tool": "read",
            "path": path,
        }
        if unchanged:
            # Skip sending content; give the model enough metadata to decide
            # locally whether a ranged re-read is worth it.
            payload["unchanged"] = True
            entry = await cache.get(abs_path)
            if entry is not None:
                payload["content_hash"] = entry.content_hash
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
            payload["content"] = result.content
            # Only mark as "seen" when the model actually received the full
            # bytes. Diff payloads carry only the delta — marking would later
            # cause an unchanged:true response for a file the model has never
            # seen in full.
            if not result.is_diff:
                _read_session_tracker.mark(session_id, abs_path)
        if mode in _MODE_NORMAL:
            if result.is_diff:
                payload["is_diff"] = True
            if result.truncated:
                payload["truncated"] = True
            if result.semantic_match:
                payload["semantic_match"] = result.semantic_match
            if result.truncated:
                # Truncated reads use semantic summarization — the returned
                # content is non-contiguous, so line numbers don't map to the
                # original file. Don't hint a specific offset; instead tell
                # the caller to use offset/limit to read specific line ranges.
                entry = await cache.get(str(Path(path).expanduser().resolve()))
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


_MAX_IMAGE_BYTES: int = _parse_max_image_bytes()


# read_image deliberately omits @_serialized: it never touches the cache or
# the ONNX worker executor (it reads bytes via the default loop executor),
# so it has nothing to serialize against and need not queue behind other tools.
@mcp.tool(
    output_schema=output_schema(ReadImageResponse),
    meta={
        "version": _pkg_version("semantic-cache-mcp"),
        "author": "Dayton Dunbar",
        "github": "https://github.com/CoderDayton/semantic-cache-mcp",
    },
)
async def read_image(
    ctx: Context,
    path: str,
) -> ToolResult:
    """Read an image file and pass the bytes through to the model.

    Returns an MCP image content block (base64-encoded with mime type) plus a
    JSON metadata sidecar. Use this when the model needs to actually see the
    image; for any other file type use `read`.

    Images are NOT cached — every call re-reads from disk. Cap is
    `SCMCP_MAX_IMAGE_BYTES` (default 5 MiB) to protect both the response
    budget and Anthropic's ~5 MB upload limit.

    Args:
        path: Image file path (absolute or relative to project root).
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    max_response_tokens = state.max_response_tokens

    # Image reads bypass the cache (and the worker process). They use the
    # default loop executor — aread_bytes/astat accept `None` and fall back
    # to asyncio's default ThreadPoolExecutor, which is the right thing in
    # the server process (no GIL contention with the worker's ONNX thread).
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

    metadata: dict[str, Any] = {
        "ok": True,
        "tool": "read_image",
        "path": path,
        "size": st.st_size,
        "mime": mime,
    }
    image_block = ImageContent(
        type="image",
        data=base64.b64encode(raw).decode("ascii"),
        mimeType=mime,
    )
    text_block = TextContent(type="text", text=json.dumps(metadata))
    return ToolResult(content=[text_block, image_block], structured_content=metadata)


@mcp.tool(output_schema=output_schema(StatsResponse))
@_serialized
async def stats(
    ctx: Context,
) -> ToolResult:
    """Inspect cache health, token savings, and runtime diagnostics.

    Use this for debugging or measurement, not as a normal step in routine
    read/edit loops.

    Returns cache occupancy, hit rates, token savings, tool-call counts,
    embedding model info, and process memory usage.
    """
    state = await _tool_call_state(ctx)
    cache = state.cache
    mode = state.mode
    remote_result: ToolResult | None = await _maybe_call_remote_tool(
        state, "stats", {}, timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result

    cache_stats = await cache.get_stats()
    model_info = get_model_info()

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

    model_name = str(model_info.get("model", "unknown"))
    provider = str(cache_stats.get("embedding_provider", "CPU"))
    ready = model_info.get("ready", False)
    provider_str = f"{provider} ✓" if ready else f"{provider} ✗"

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
        "embedding": {
            "model": model_name,
            "provider": provider,
            "ready": ready,
            "process_rss_mb": cache_stats.get("process_rss_mb"),
        },
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
            (
                f"*{lt_sessions} completed session"
                f"{'s' if lt_sessions != 1 else ''} · {model_name} · {provider_str}*"
            ),
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
            f"`{model_name}` · {provider_str} · {mem_str}",
        ]
        return ToolResult(content="\n".join(lines), structured_content=structured_payload)

    # debug — full raw dump
    return ToolResult(
        content=f"```json\n{json.dumps(cache_stats | {'embedding': model_info}, indent=2)}\n```",
        structured_content=structured_payload,
    )


@mcp.tool(output_schema=output_schema(ClearResponse))
@_serialized
async def clear(
    ctx: Context,
) -> dict[str, Any]:
    """Clear the semantic cache only; does not modify project files.

    Use this rarely, mainly to recover from stale cache state or to force cold
    re-seeding. Prefer normal `read`/`batch_read` refresh behavior when
    possible.

    Returns the number of cached entries removed.
    """
    state = await _tool_call_state(ctx)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state, "clear", {}, timeout=_TOOL_TIMEOUT
    )
    if remote_result is not None:
        return remote_result

    count = await cache.clear()
    cache.metrics.record("clear", None)
    _read_session_tracker.clear()
    payload: dict[str, Any] = {"ok": True, "tool": "clear", "status": "cleared", "count": count}
    if mode == _MODE_DEBUG:
        payload["output_mode"] = mode
    return _finalize_payload(payload, max_response_tokens)


@mcp.tool(output_schema=output_schema(DeleteResponse))
@_serialized
async def delete(
    ctx: Context,
    path: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delete one file or one symlink path and evict cache entries for that path.

    Use this for explicit single-path removal instead of shelling out.

    Normal statuses:
    - `deleted`: file or symlink path was removed
    - `would_delete`: dry-run preview only
    - `not_found`: path did not exist; this is not an error

    Constraints:
    - No globs
    - No recursive delete
    - No real-directory delete
    - If `path` is a symlink, deletes the link itself, not the target

    Args:
        path: File or symlink path (absolute or relative to project root).
        dry_run: Preview without deleting or evicting cache.
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state, "delete", {"path": path, "dry_run": dry_run}, timeout=_TOOL_TIMEOUT
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
        _read_session_tracker.invalidate(str(Path(path).expanduser().resolve()))
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


@mcp.tool(output_schema=output_schema(WriteResponse))
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
) -> dict[str, Any]:
    """Create or replace a file. Prefer `edit`/`batch_edit` for localized changes.

    Args:
        path: File path to create or replace.
        content: Full content to write, or appended content when
            `append=true`.
        create_parents: Create missing parent directories when needed.
        dry_run: Preview without writing.
        auto_format: Run formatter after write.
        show_diff: Return the diff explicitly even for deterministic writes.
        append: Append instead of overwrite.
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state,
        "write",
        {
            "path": path,
            "content": content,
            "create_parents": create_parents,
            "dry_run": dry_run,
            "auto_format": auto_format,
            "show_diff": show_diff,
            "append": append,
        },
        timeout=_TOOL_TIMEOUT,
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
        _read_session_tracker.invalidate(str(Path(result.path).expanduser().resolve()))

        payload: dict[str, Any] = {
            "ok": True,
            "tool": "write",
            "status": "created" if result.created else "updated",
            "path": result.path,
        }
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
            payload["dry_run"] = dry_run
            payload["tokens_saved"] = result.tokens_saved
        if mode == _MODE_DEBUG:
            payload["bytes_written"] = result.bytes_written
            payload["tokens_written"] = result.tokens_written
            payload["diff_stats"] = result.diff_stats
            payload["content_hash"] = result.content_hash
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


@mcp.tool(output_schema=output_schema(EditResponse))
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
) -> dict[str, Any]:
    """Edit one file via exact replacement.

    For multiple edits to the same file, use `batch_edit` (single response,
    atomic, faster). For full rewrites, use `write`.

    Modes: find/replace (`old_string`+`new_string`), scoped (add `start_line`/`end_line`),
    or line-range (omit `old_string`, provide both lines). Keep `old_string` short
    and unique; add line bounds when ambiguous.

    Args:
        path: File path to modify.
        old_string: Exact text to find. Omit only for line-range replacement.
        new_string: Replacement text.
        replace_all: Replace all matches instead of requiring uniqueness.
        dry_run: Preview without writing.
        auto_format: Run formatter after editing.
        show_diff: Return the diff explicitly for successful deterministic edits.
        start_line: 1-based inclusive start line for scoped or line-range edit.
        end_line: 1-based inclusive end line for scoped or line-range edit.
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state,
        "edit",
        {
            "path": path,
            "old_string": old_string,
            "new_string": new_string,
            "replace_all": replace_all,
            "dry_run": dry_run,
            "auto_format": auto_format,
            "show_diff": show_diff,
            "start_line": start_line,
            "end_line": end_line,
        },
        timeout=_TOOL_TIMEOUT,
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
        _read_session_tracker.invalidate(str(Path(result.path).expanduser().resolve()))

        payload: dict[str, Any] = {
            "ok": True,
            "tool": "edit",
            "status": "edited",
            "path": result.path,
            # matches_found always equals replacements_made — collapse to one field
            "replaced": result.replacements_made,
            "line_numbers": result.line_numbers,
        }
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
            payload["content_hash"] = result.content_hash
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


@mcp.tool(output_schema=output_schema(EditPreviewResponse))
@_serialized
async def edit_preview(
    ctx: Context,
    path: str,
    old_string: str,
) -> dict[str, Any]:
    """Preview where `old_string` matches in a file without modifying it.

    Returns match count, 1-based line numbers, and small snippets so the
    caller can confirm an anchor is unique before committing to `edit`.
    Read-only and intentionally cheap — under ~200 tokens — so it can be
    called freely as a probe.

    Args:
        path: File path to search.
        old_string: Anchor text. Must match exactly (whitespace, indentation).
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
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


@mcp.tool(output_schema=output_schema(BatchEditResponse))
@_serialized
async def batch_edit(
    ctx: Context,
    path: str,
    edits: str,
    dry_run: bool = False,
    auto_format: bool = False,
    show_diff: bool = False,
) -> dict[str, Any]:
    """Apply multiple exact edits to one file in a single call.

    Preferred over repeated `edit` calls on the same file: single response,
    atomic across all edits, and faster on large files. For cross-file work,
    call the relevant tools per file instead of trying to batch across files.

    Supported entry forms:
    - `[old, new]` for full-file exact replacement
    - `[old, new, start_line, end_line]` for scoped replacement
    - `[null, new, start_line, end_line]` for line-range replacement
    - `{"old": ..., "new": ..., "start_line": ..., "end_line": ...}`

    Behavior:
    - Partial success is allowed.
    - Failed edits are returned so you can retry only the misses.
    - Prefer line-range entries when you already have line numbers from `read`.
    - Deterministic all-success batches omit full diffs unless `show_diff=true`
      or debug mode is enabled.

    Args:
        path: File path to modify.
        edits: JSON array of edit entries for that file.
        dry_run: Preview without writing.
        auto_format: Run formatter after edits.
        show_diff: Return the diff explicitly for successful deterministic batches.
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state,
        "batch_edit",
        {
            "path": path,
            "edits": edits,
            "dry_run": dry_run,
            "auto_format": auto_format,
            "show_diff": show_diff,
        },
        timeout=_TOOL_TIMEOUT,
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
        _read_session_tracker.invalidate(str(Path(result.path).expanduser().resolve()))

        status = (
            "edited"
            if result.failed == 0
            else ("partial" if result.succeeded > 0 else "no_changes")
        )
        payload: dict[str, Any] = {
            "ok": True,
            "tool": "batch_edit",
            "status": status,
            "path": result.path,
            "succeeded": result.succeeded,
        }
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
        _apply_mutation_diff(
            payload,
            diff_content=result.diff_content,
            mode=mode,
            show_diff=show_diff,
            partial=status == "partial",
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
            payload["content_hash"] = result.content_hash
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


@mcp.tool(output_schema=output_schema(SearchResponse))
@_serialized
async def search(
    ctx: Context,
    query: str,
    k: int = 10,
    directory: str | None = None,
    show_preview: bool = False,
) -> dict[str, Any]:
    """Cache-only semantic search by meaning.

    Use `grep` for exact strings, `glob` to discover files. Empty results
    usually mean files weren't seeded via `read`/`batch_read`.

    Args:
        query: Natural-language query, keywords, or a mixture of both.
        k: Maximum number of matches to return.
        directory: Optional directory filter applied after retrieval.
        show_preview: Include match previews explicitly.
    """
    state = await _tool_call_state(ctx)
    directory = state.resolve(directory) if directory else None
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state,
        "search",
        {"query": query, "k": k, "directory": directory, "show_preview": show_preview},
        timeout=_TOOL_TIMEOUT,
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
        for m in result.matches:
            item: dict[str, Any] = {"path": m.path, "similarity": round(m.similarity, 4)}
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
        # Echoing `query` back is wasteful — the caller just sent it.
        # Only include it in debug mode for traceability.
        if mode in _MODE_NORMAL:
            payload["count"] = len(match_payload)
            payload["cached_files"] = result.cached_files
        if mode == _MODE_DEBUG:
            payload["query"] = query
            payload["files_searched"] = result.files_searched
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


@mcp.tool(output_schema=output_schema(DiffResponse))
@_serialized
async def diff(
    ctx: Context,
    path1: str,
    path2: str,
    context_lines: int = 3,
) -> dict[str, Any]:
    """Compare two files side by side and return a unified diff.

    Use this for explicit file-to-file comparison. For "what changed since I
    last read this file?", use `read` instead of `diff`.

    Behavior:
    - Returns unified diff plus semantic similarity score.
    - Reuses cached content when possible.
    - Large diffs may be suppressed to stay within token budget.

    Args:
        path1: First file path.
        path2: Second file path.
        context_lines: Number of context lines to include around changes.
    """
    state = await _tool_call_state(ctx)
    path1, path2 = state.resolve(path1), state.resolve(path2)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state,
        "diff",
        {"path1": path1, "path2": path2, "context_lines": context_lines},
        timeout=_TOOL_TIMEOUT,
    )
    if remote_result is not None:
        return remote_result

    try:
        result = await asyncio.wait_for(
            compare_files(cache, path1, path2, context_lines=context_lines),
            timeout=_TOOL_TIMEOUT,
        )
        cache.metrics.record("diff", result)

        payload: dict[str, Any] = {
            "ok": True,
            "tool": "diff",
            "path1": result.path1,
            "path2": result.path2,
            "diff": result.diff_content,
            "diff_state": _diff_state(result.diff_content),
        }
        if mode in _MODE_NORMAL:
            payload["similarity"] = round(result.similarity, 4)
            payload["diff_stats"] = result.diff_stats
        if mode == _MODE_DEBUG:
            payload["tokens_saved"] = result.tokens_saved
            payload["from_cache"] = result.from_cache
            payload["context_lines"] = context_lines

        return _finalize_payload(payload, max_response_tokens)

    except FileNotFoundError as e:
        _raise_tool_error("diff", str(e), max_response_tokens)
    except TimeoutError:
        _handle_timeout(cache, "diff")
        _raise_tool_error("diff", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except ToolError:
        raise
    except Exception as e:
        logger.exception("Error in diff")
        _raise_tool_error("diff", str(e), max_response_tokens)


@mcp.tool(output_schema=output_schema(BatchReadResponse))
@_serialized
async def batch_read(
    ctx: Context,
    paths: str,
    max_total_tokens: int = 50000,
    priority: str = "",
) -> dict[str, Any]:
    """Read multiple files under a token budget.

    Use to seed cache before `search`/`grep`; prefer over repeated `read`
    calls. Returns diffs for modified files, full content for new ones;
    large files skipped when budget exhausted (use `read` with
    `offset`/`limit` to recover).

    Args:
        paths: Comma-separated paths, JSON array, or glob patterns.
        max_total_tokens: Token budget across the batch.
        priority: Optional paths to read first before the remaining files.
    """
    state = await _tool_call_state(ctx)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens

    try:
        path_list = _resolve_path_list(paths, state)
        priority_list = _resolve_path_list(priority, state) if priority.strip() else None
        remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
            state,
            "batch_read",
            {
                "paths": json.dumps(path_list),
                "max_total_tokens": max_total_tokens,
                "priority": json.dumps(priority_list) if priority_list else "",
            },
            timeout=_TOOL_TIMEOUT * 2,
        )
        if remote_result is not None:
            return remote_result

        # Expand glob patterns
        path_list = _expand_globs(path_list)

        result = await asyncio.wait_for(
            batch_smart_read(
                cache,
                path_list,
                max_total_tokens=max_total_tokens,
                priority=priority_list,
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


@mcp.tool(output_schema=output_schema(SimilarResponse))
@_serialized
async def similar(
    ctx: Context,
    path: str,
    k: int = 5,
) -> dict[str, Any]:
    """Find cached files semantically similar to one source file.

    Neighbors must already be cached (seed with `batch_read` first).

    Args:
        path: Source file path.
        k: Maximum number of similar files to return.
    """
    state = await _tool_call_state(ctx)
    path = state.resolve(path)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state,
        "similar",
        {"path": path, "k": k},
        timeout=_TOOL_TIMEOUT,
    )
    if remote_result is not None:
        return remote_result

    try:
        result = await asyncio.wait_for(find_similar_files(cache, path, k=k), timeout=_TOOL_TIMEOUT)
        cache.metrics.record("similar", result)

        similar_payload = [
            {"path": f.path, "similarity": round(f.similarity, 4)}
            if mode == "compact"
            else {"path": f.path, "similarity": round(f.similarity, 4), "tokens": f.tokens}
            for f in result.similar_files
        ]
        payload: dict[str, Any] = {
            "ok": True,
            "tool": "similar",
            "source_path": result.source_path,
            "similar_files": similar_payload,
        }
        if mode in _MODE_NORMAL:
            payload["source_tokens"] = result.source_tokens
            payload["files_searched"] = result.files_searched
        if mode == _MODE_DEBUG:
            payload["k"] = k

        return _finalize_payload(payload, max_response_tokens)

    except FileNotFoundError as e:
        _raise_tool_error("similar", str(e), max_response_tokens)
    except TimeoutError:
        _handle_timeout(cache, "similar", path)
        _raise_tool_error("similar", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except ToolError:
        raise
    except Exception as e:
        logger.exception("Error in similar")
        _raise_tool_error("similar", str(e), max_response_tokens)


@mcp.tool(output_schema=output_schema(GlobResponse))
@_serialized
async def glob(
    ctx: Context,
    pattern: str,
    directory: str = ".",
    cached_only: bool = False,
) -> dict[str, Any]:
    """Discover files by glob and show which are already cached.

    Use before `batch_read`/`search`/`grep`. `cached_only=true` shows what
    search/grep can see without more reads.

    Args:
        pattern: Glob pattern to expand.
        directory: Base directory for the glob.
        cached_only: Restrict results to already cached files.
    """
    state = await _tool_call_state(ctx)
    directory = state.resolve(directory)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state,
        "glob",
        {
            "pattern": pattern,
            "directory": directory,
            "cached_only": cached_only,
        },
        timeout=_TOOL_TIMEOUT,
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
        # When all matches are uncached and we're not in debug mode, drop the
        # redundant `cached: false` field — saves ~13 chars per match.
        all_uncached = result.cached_count == 0 and mode != _MODE_DEBUG
        for m in result.matches:
            item: dict[str, Any] = {"path": m.path}
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


@mcp.tool(output_schema=output_schema(GrepResponse))
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
) -> dict[str, Any]:
    """Cache-only ripgrep: exact string/regex with line numbers.

    Use `search` for semantic intent. Seed files with `batch_read` first;
    empty results often mean files not cached.

    Args:
        pattern: Regex pattern, or a literal if `fixed_string=true`.
        path: Optional exact path, suffix, or glob filter.
        fixed_string: Treat `pattern` as a literal instead of regex.
        case_sensitive: Whether matching is case-sensitive.
        context_lines: Number of context lines to include around matches.
        max_matches: Maximum total matches across all files.
        max_files: Maximum number of files to return.
    """
    state = await _tool_call_state(ctx)
    cache = state.cache
    mode = state.mode
    max_response_tokens = state.max_response_tokens
    remote_result: dict[str, Any] | None = await _maybe_call_remote_tool(
        state,
        "grep",
        {
            "pattern": pattern,
            "path": path,
            "fixed_string": fixed_string,
            "case_sensitive": case_sensitive,
            "context_lines": context_lines,
            "max_matches": max_matches,
            "max_files": max_files,
        },
        timeout=_TOOL_TIMEOUT,
    )
    if remote_result is not None:
        return remote_result

    try:
        # In compact mode, context lines are dropped in the response anyway —
        # skip fetching them to avoid wasted storage work.
        effective_context = context_lines if mode in _MODE_NORMAL else 0
        results = await cache._storage.grep(
            pattern,
            path=path,
            fixed_string=fixed_string,
            case_sensitive=case_sensitive,
            context_lines=effective_context,
            max_matches=max_matches,
            max_files=max_files,
        )
        cache.metrics.record("grep", None)

        total_matches = sum(len(r["matches"]) for r in results)

        # Apply a soft char budget so a wide regex on a large repo doesn't
        # spend the entire response token cap on match lines. The hard
        # token cap in _finalize_payload still applies as a backstop.
        char_budget: int | None = None
        if max_response_tokens is not None and max_response_tokens > 0:
            # Leave ~512 tokens for the response envelope/metadata.
            char_budget = max(1024, (max_response_tokens - 512) * 4)

        # Build response
        files_payload: list[dict[str, Any]] = []
        truncated_matches = 0
        truncated_files = 0
        running_chars = 0
        budget_exceeded = False
        for file_result in results:
            if budget_exceeded:
                truncated_files += 1
                truncated_matches += len(file_result["matches"])
                continue
            match_items: list[dict[str, Any]] = []
            for m in file_result["matches"]:
                item: dict[str, Any] = {
                    "line_number": m["line_number"],
                    "line": m["line"],
                }
                if effective_context > 0:
                    if "before" in m:
                        item["before"] = m["before"]
                    if "after" in m:
                        item["after"] = m["after"]
                match_items.append(item)
                if char_budget is not None:
                    # ~32 chars JSON envelope per match (line_number, line keys
                    # + braces + commas + quotes). Add context-line bytes when
                    # present so the soft budget accounts for them.
                    running_chars += len(m["line"]) + 32
                    if effective_context > 0:
                        for ctx_line in m.get("before", ()):
                            running_chars += len(ctx_line) + 4
                        for ctx_line in m.get("after", ()):
                            running_chars += len(ctx_line) + 4
                    if running_chars > char_budget:
                        budget_exceeded = True
                        # Count remaining matches in this file as truncated
                        remaining = file_result["matches"][len(match_items) :]
                        truncated_matches += len(remaining)
                        break
            files_payload.append(
                {
                    "path": file_result["path"],
                    "count": len(match_items),
                    "matches": match_items,
                }
            )

        payload: dict[str, Any] = {
            "ok": True,
            "tool": "grep",
            "pattern": pattern,
            "path": path,
            "total_matches": total_matches,
            "files_matched": len(results),
            "files": files_payload,
        }
        if truncated_matches > 0 or truncated_files > 0:
            payload["truncated_matches"] = truncated_matches
            payload["files_in_response"] = len(files_payload)
            if truncated_files > 0:
                payload["truncated_files"] = truncated_files
        # Distinguish "no files cached under that path" from "no matches".
        # The audit found 22/29 empty greps fit the cache-miss shape, so the
        # caller should know whether to seed via batch_read/glob.
        if total_matches == 0 and path is not None:
            has_cached = await cache._storage.has_cached_paths_under(path)
            if not has_cached:
                payload["reason"] = "no_files_cached_under_path"
                payload["hint"] = "use batch_read or glob to seed the cache"
        if mode == _MODE_DEBUG:
            payload["fixed_string"] = fixed_string
            payload["case_sensitive"] = case_sensitive
            payload["context_lines"] = context_lines

        return _finalize_payload(payload, max_response_tokens)

    except ToolError:
        raise
    except Exception as e:
        logger.exception("Error in grep")
        _raise_tool_error("grep", str(e), max_response_tokens)


_EXPAND_GLOBS_TIMEOUT = 5  # seconds — matches GLOB_TIMEOUT_SECONDS


def _expand_globs(raw_paths: list[str], max_files: int = 50) -> list[str]:
    """Expand glob patterns in path list. Non-glob paths pass through unchanged.

    Uses a deadline to prevent recursive ``**`` patterns from blocking
    the caller for an unbounded amount of time.
    """
    import time  # noqa: PLC0415

    deadline = time.monotonic() + _EXPAND_GLOBS_TIMEOUT
    expanded: list[str] = []
    glob_chars = frozenset("*?[")
    for p in raw_paths:
        if time.monotonic() > deadline:
            logger.warning(f"Glob expansion timed out after {_EXPAND_GLOBS_TIMEOUT}s")
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
                            break
                        if m.is_file():
                            matches.append(str(m))
                            if len(matches) >= remaining:
                                break
                    matches.sort()
                    expanded.extend(matches)
            except (OSError, ValueError):
                expanded.append(p)  # Treat invalid pattern as literal
        else:
            expanded.append(p)
        if len(expanded) >= max_files:
            logger.debug(f"Glob expansion truncated at {max_files} files")
            break
    return expanded[:max_files]
