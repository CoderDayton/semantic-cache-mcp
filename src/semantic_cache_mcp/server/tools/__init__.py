"""MCP tool handlers."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, TypeVar, cast

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult

from ...cache import (
    SemanticCache,
    batch_smart_read,
    compare_files,
    find_similar_files,
    glob_with_cache_status,
    semantic_search,
    smart_batch_edit,
    smart_edit,
    smart_read,
    smart_write,
)
from ...config import MAX_CONTENT_SIZE, TOOL_TIMEOUT
from ...core.embeddings import get_model_info
from ...utils._async_io import aunlink
from .._mcp import mcp
from .._tool_models import (
    BatchEditResponse,
    BatchReadResponse,
    ClearResponse,
    DeleteResponse,
    DiffResponse,
    EditResponse,
    GlobResponse,
    GrepResponse,
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

logger = logging.getLogger(__name__)


# Tool timeout from config (env TOOL_TIMEOUT, default 30s).
_TOOL_TIMEOUT: float = TOOL_TIMEOUT

# Global tool mutex: only one tool call executes at a time.
# Prevents concurrent coroutines from interleaving executor tasks,
# catalog reads, and ONNX calls — the root cause of hangs when
# multiple subagents fire tool calls simultaneously.
_tool_lock: asyncio.Lock | None = None
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
    """Lazy-init the lock (must be created inside a running event loop)."""
    global _tool_lock
    if _tool_lock is None:
        _tool_lock = asyncio.Lock()
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
    timed_out = False
    try:
        async with asyncio.timeout(timeout):
            return await asyncio.shield(task)
    except TimeoutError:
        timed_out = True
        raise
    except asyncio.CancelledError:
        # Not our timeout — genuine cancellation (SIGTERM / graceful shutdown).
        # Give the write a brief grace period to finish disk I/O before
        # the process exits.
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=2.0)
        except (TimeoutError, asyncio.CancelledError):
            raise asyncio.CancelledError() from None
    finally:
        if not timed_out:
            cache.end_operation()


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
    """Read a file. Automatically returns the most token-efficient response.

    Use this for a single file. For 2+ files, prefer `batch_read`.

    Behavior (automatic — no configuration needed):
    - First read: returns full content and seeds the cache.
    - Unchanged re-read: returns `"unchanged": true` (content already in context).
    - Modified re-read: returns a unified diff of changes.
    - External changes: detected automatically via mtime + content hash.

    If response contains `"unchanged": true`, do NOT re-read — you already
    have the full content from a prior read. Use `offset`/`limit` to recover
    specific line ranges after truncation or context loss.

    Args:
        path: File path (absolute or relative to project root). Use absolute
            paths for files outside the current project root.
        max_size: Maximum content size to return before summarization.
        offset: 1-based starting line number for targeted reads.
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

    # Validate bounds
    if offset is not None and offset < 1:
        _raise_tool_error("read", "offset must be >= 1 (1-based)", max_response_tokens)
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
            lines = result.content.splitlines(keepends=True)
            start = (offset or 1) - 1  # Convert to 0-based
            end = start + (limit or len(lines) - start)
            selected = lines[start:end]

            # Format with line numbers like built-in Read tool
            numbered = []
            for i, line in enumerate(selected, start=start + 1):
                numbered.append(f"{i:6d}\t{line.rstrip()}")
            content = "\n".join(numbered)
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
        # Detect unchanged files: from_cache=True + is_diff=False means
        # the LLM already has this file's content from a prior read.
        unchanged = result.from_cache and not result.is_diff

        payload = {
            "ok": True,
            "tool": "read",
            "path": path,
        }
        if unchanged:
            payload["unchanged"] = True
        if not unchanged or mode == _MODE_DEBUG:
            payload["content"] = result.content
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
    """Create or replace a file, with cache refresh and optional overwrite diffs.

    Use this when you already know the full new file content. For targeted
    changes inside an existing file, prefer `edit` or `batch_edit`.

    Routing rules:
    - New file or full replacement: use `write`.
    - Small localized change: use `edit`.
    - Multiple localized changes in one file: use `batch_edit`.

    Behavior:
    - Deterministic successful overwrites omit full diffs by default.
    - Set `show_diff=true` or use debug mode to include the diff explicitly.
    - New files return creation status.
    - `append=true` supports chunked construction for large files.
    - `dry_run=true` previews without writing.
    - `auto_format=true` is best used near the end of an edit cycle.

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
    """Edit one file using cache-aware exact replacement.

    Prefer this over `write` when you want to preserve the rest of the file.
    Use line numbers from `read` whenever possible to keep the edit precise.

    Modes:
    - Find/replace: `old_string` + `new_string`
    - Scoped replace: add `start_line` + `end_line`
    - Line-range replace: omit `old_string`, provide `start_line` + `end_line`

    Routing rules:
    - One localized change: use `edit`
    - Multiple independent changes in the same file: use `batch_edit`
    - Full-file rewrite: use `write`

    Precision rules:
    - Keep `old_string` exact and as short as possible, ideally one line.
    - If a match is ambiguous, add more context or line bounds.
    - Use `replace_all=true` only when every match should change.
    - Deterministic successful edits omit full diffs unless `show_diff=true`
      or debug mode is enabled.

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
            ),
        )
        cache.metrics.record("edit", result)

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
        _raise_tool_error("edit", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except OSError as e:
        logger.warning(f"I/O error in edit: {e}")
        _raise_tool_error("edit", f"I/O operation failed - {e}", max_response_tokens)
    except ToolError:
        raise
    except Exception:
        logger.exception("Unexpected error in edit")
        _raise_tool_error("edit", "Internal error occurred while editing file", max_response_tokens)


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

    Use this when several independent edits belong in the same file. For one
    change, prefer `edit`. For cross-file work, call the relevant tools per
    file instead of trying to batch across files.

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
    """Search cached files by meaning or mixed keyword intent.

    This is a cache-only semantic search. If results are empty, the likely
    cause is that the relevant files were never seeded with `read` or
    `batch_read`.

    Routing rules:
    - Use `search` for meaning-based queries such as concepts, behavior, or
      intent.
    - Use `grep` for exact symbols, strings, or regex patterns.
    - Use `glob` to discover candidate files before seeding the cache.

    Usage guidance:
    - Seed likely files with `batch_read` first.
    - Start with small `k` such as 3–5.
    - Use `directory` to keep large codebases focused.
    - Set `show_preview=true` only when snippet text changes the next decision.

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
            "query": query,
            "matches": match_payload,
        }
        if mode in _MODE_NORMAL:
            payload["count"] = len(match_payload)
            payload["cached_files"] = result.cached_files
        if mode == _MODE_DEBUG:
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
    """Read multiple files under a token budget. Automatically cache-aware.

    Use this to seed the cache, gather several files at once, or expand globs
    before `search`, `similar`, or `grep`. Prefer over repeated `read` calls.

    Behavior (automatic — no configuration needed):
    - Unchanged files counted in `summary.unchanged_count` (path list in debug mode).
    - Modified files return diffs.
    - New files return full content.
    - Large files skipped once token budget is exhausted.

    If a file is skipped for budget, use `read` with `offset`/`limit` or
    raise the budget.

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

    Use this to discover related implementations, tests, or configs after the
    surrounding code has already been seeded into the cache.

    Important constraint:
    - The source file is handled automatically.
    - Candidate neighbor files must already be cached, typically via
      `batch_read`, or they will not appear.

    Usage guidance:
    - Seed a directory with `batch_read` first.
    - Start with `k=3` to `k=5`.
    - Empty results usually mean either only the source file is cached or the
      relevant neighbors were never seeded.

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
    """Discover files by glob and show whether each one is already cached.

    Use this before `batch_read`, `search`, `similar`, or `grep` when you need
    candidate files or want to inspect cache coverage.

    Routing rules:
    - Use `glob` to discover paths.
    - Use `batch_read` to seed or read them.
    - Use `cached_only=true` when you want to know what search/grep can already
      see without more reads.

    Usage guidance:
    - Keep patterns specific.
    - Avoid broad patterns like `**/*` unless truly necessary.
    - Matches can be fed directly into `batch_read`.

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
        matches_payload = []
        for m in result.matches:
            item: dict[str, Any] = {"path": m.path, "cached": m.cached}
            if mode == _MODE_DEBUG:
                item["tokens"] = m.tokens
                item["mtime"] = m.mtime
            matches_payload.append(item)

        payload: dict[str, Any] = {
            "ok": True,
            "tool": "glob",
            "pattern": pattern,
            "directory": result.directory,
            "matches": matches_payload,
            "total_matches": result.total_matches,
            "cached_count": result.cached_count,
        }
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
    """Search cached files for an exact string or regex, with line numbers.

    This is the cache-only exact-search tool. It is intentionally closer to
    "ripgrep on cached content" than to live filesystem search.

    Routing rules:
    - Use `grep` for exact symbols, literals, imports, error strings, or regex.
    - Use `search` for semantic or fuzzy intent.
    - Seed candidate files with `batch_read` first; empty results may simply
      mean the relevant files are not cached yet.

    Usage guidance:
    - Set `fixed_string=true` for literals containing regex metacharacters.
    - Add `path` to limit scope to one file, a suffix, or a glob.
    - Add `context_lines=2` or `3` when surrounding code matters.

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

        # Build response
        files_payload: list[dict[str, Any]] = []
        for file_result in results:
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
            "files_matched": len(files_payload),
            "files": files_payload,
        }
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
