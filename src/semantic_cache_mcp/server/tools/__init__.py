"""MCP tool handlers."""

from __future__ import annotations

import asyncio
import json
import logging
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from fastmcp import Context

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
from .._mcp import mcp
from ..response import (
    _MODE_DEBUG,
    _MODE_NORMAL,
    _render_error,
    _render_response,
    _response_mode,
    _response_token_cap,
)

logger = logging.getLogger(__name__)


# Tool timeout from config (env TOOL_TIMEOUT, default 20s).
_TOOL_TIMEOUT: float = TOOL_TIMEOUT

# Global tool mutex: only one tool call executes at a time.
# Prevents concurrent coroutines from interleaving executor tasks,
# catalog reads, and ONNX calls — the root cause of hangs when
# multiple subagents fire tool calls simultaneously.
_tool_lock: asyncio.Lock | None = None


def _get_tool_lock() -> asyncio.Lock:
    """Lazy-init the lock (must be created inside a running event loop)."""
    global _tool_lock
    if _tool_lock is None:
        _tool_lock = asyncio.Lock()
    return _tool_lock


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
    meta={
        "version": _pkg_version("semantic-cache-mcp"),
        "author": "Dayton Dunbar",
        "github": "https://github.com/CoderDayton/semantic-cache-mcp",
    }
)
@_serialized
async def read(
    ctx: Context,
    path: str,
    max_size: int = MAX_CONTENT_SIZE,
    diff_mode: bool = True,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    """Read a file with token-efficient caching and diffs.

    Behavior with diff_mode=true (default):
    - First read: returns full content and caches it.
    - Subsequent read, file unchanged: returns a short "unchanged" marker.
      Set diff_mode=false after context compression to get full content again.
    - Subsequent read, file modified: returns a unified diff of changes.

    When response contains "unchanged":true, the file has NOT changed since
    your last read — the full content is already in your conversation context.
    Do NOT re-read it.

    For 2+ files, prefer batch_read. Use offset/limit to read specific line
    ranges without re-reading the whole file.

    Truncated files: use read(path, offset=N, limit=M) to continue — do NOT
    re-read from the beginning.

    Args:
        path: File path (absolute or relative)
        max_size: Maximum content size to return (default: 100000)
        diff_mode: Return diff if previously read (default: true)
        offset: Line number to start reading from (1-based)
        limit: Number of lines to read from offset
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

    # Validate bounds
    if offset is not None and offset < 1:
        return _render_error("read", "offset must be >= 1 (1-based)", max_response_tokens)
    if limit is not None and limit < 1:
        return _render_error("read", "limit must be >= 1", max_response_tokens)
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

            return _render_response(payload, max_response_tokens)

        result = await asyncio.wait_for(
            smart_read(
                cache=cache,
                path=path,
                max_size=max_size,
                diff_mode=diff_mode,
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
            "content": result.content,
        }
        if unchanged:
            payload["unchanged"] = True
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
                "diff_mode": diff_mode,
                "offset": offset,
                "limit": limit,
            }

        return _render_response(payload, max_response_tokens)

    except FileNotFoundError as e:
        return _render_error("read", str(e), max_response_tokens)
    except TimeoutError:
        _handle_timeout(cache, "read", path)
        return _render_error("read", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except Exception as e:
        return _render_error("read", f"reading failed: {e}", max_response_tokens)


@mcp.tool()
@_serialized
async def stats(
    ctx: Context,
) -> str:
    """Get cache statistics, session activity, and lifetime metrics.

    Returns cache occupancy, token savings, tool call counts, embedding model
    info, and process memory usage. Useful for monitoring cache effectiveness
    and diagnosing performance issues.
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
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

    if mode == "compact":
        lines = [
            "## Semantic Cache",
            "",
            f"**{_n(cache_stats.get('files_cached', 0))}** files · "
            f"**{_n(cache_stats.get('total_tokens_cached', 0))}** tokens stored · "
            f"**{_mb(cache_stats.get('db_size_mb', 0.0))}**",
            "",
            "| | Saved | Rate | Hit Rate |",
            "|---|---:|---:|---:|",
            f"| **Session** | {_n(s_saved)} | **{s_pct}%** | {s_hit_pct}% ({s_hits}/{s_total}) |",
            (
                f"| **Lifetime** | {_n(lt_saved)} | **{lt_pct}%** "
                f"| {lt_hit_pct}% ({lt_hits}/{lt_total}) |"
            ),
            "",
            (
                f"*{lt_sessions} completed session"
                f"{'s' if lt_sessions != 1 else ''} · {model_name} · {provider_str}*"
            ),
        ]
        return "\n".join(lines)

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
            "---",
            "",
            "## Storage",
            "",
            "| Files Cached | Tokens Stored | Documents | DB Size |",
            "|---:|---:|---:|---:|",
            f"| **{_n(cache_stats.get('files_cached', 0))}** "
            f"| **{_n(cache_stats.get('total_tokens_cached', 0))}** "
            f"| {_n(cache_stats.get('total_documents', 0))} "
            f"| {_mb(cache_stats.get('db_size_mb', 0.0))} |",
            "",
            "---",
            "",
            f"## This Session  ·  uptime {uptime}",
            "",
            "| Metric | Tokens | Rate |",
            "|---|---:|---:|",
            f"| Tokens saved | **{_n(s_saved)}** | **{s_pct}%** |",
            f"| Tokens returned | {_n(session.get('tokens_returned', 0))} | — |",
            "",
            "| Cache hits | Cache misses | Hit rate |",
            "|---:|---:|---:|",
            f"| **{_n(s_hits)}** | {_n(s_misses)} | **{s_hit_pct}%** |",
            "",
            f"Files read: **{files_read}** · "
            f"written: **{files_written}** · "
            f"edited: **{files_edited}** · "
            f"diffs served: **{diffs}**",
        ]

        if top_tools:
            lines += [
                "",
                "**Tool calls:** " + " · ".join(f"`{t}` ×{c}" for t, c in top_tools),
            ]

        lines += [
            "",
            "---",
            "",
            f"## Lifetime  ·  {lt_sessions} session{'s' if lt_sessions != 1 else ''}",
            "",
            "| Metric | Tokens | Rate |",
            "|---|---:|---:|",
            f"| Tokens saved | **{_n(lt_saved)}** | **{lt_pct}%** |",
            f"| Tokens returned | {_n(lifetime.get('tokens_returned', 0))} | — |",
            "",
            "| Cache hits | Cache misses | Hit rate |",
            "|---:|---:|---:|",
            f"| **{_n(lt_hits)}** | {_n(lt_misses)} | **{lt_hit_pct}%** |",
            "",
            f"Files read: **{_n(lt_files_read)}** · "
            f"written: **{_n(lt_files_written)}** · "
            f"edited: **{_n(lt_files_edited)}**",
            "",
            "---",
            "",
            "## System",
            "",
            "| Model | Provider | Memory |",
            "|---|---|---:|",
            f"| `{model_name}` | {provider_str} | {mem_str} |",
        ]
        return "\n".join(lines)

    # debug — full raw dump
    return f"```json\n{json.dumps(cache_stats | {'embedding': model_info}, indent=2)}\n```"


@mcp.tool()
@_serialized
async def clear(
    ctx: Context,
) -> str:
    """Clear all cache entries (content, embeddings, indexes). Returns count removed."""
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()
    count = await cache.clear()
    cache.metrics.record("clear", None)
    payload: dict[str, Any] = {"ok": True, "tool": "clear", "status": "cleared", "count": count}
    if mode == _MODE_DEBUG:
        payload["output_mode"] = mode
    return _render_response(payload, max_response_tokens)


@mcp.tool()
@_serialized
async def write(
    ctx: Context,
    path: str,
    content: str,
    create_parents: bool = True,
    dry_run: bool = False,
    auto_format: bool = False,
    append: bool = False,
) -> str:
    """Write file content with cache integration.

    Timing:
    - Use when creating files or replacing most/all content.
    - For targeted substitutions, prefer edit or batch_edit.
    - Use auto_format=true at stabilization points, not during rapid iteration.

    Large files: use append=true to write in chunks:
      write(path, chunk1)                               # creates file
      write(path, chunk2, append=true)                   # appends
      write(path, chunk3, append=true, auto_format=true) # final chunk + format

    Response: diff of changes for overwrites, status-only for new files.

    Args:
        path: File path (absolute or relative)
        content: Content to write (or content to append when append=true)
        create_parents: Create parent directories (default: true)
        dry_run: Preview changes without writing (default: false)
        auto_format: Run formatter after write (default: false)
        append: Append content to existing file instead of overwriting (default: false)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

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
        if result.diff_content:
            payload["diff"] = result.diff_content
        elif not result.created:
            payload["diff_omitted"] = True

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

        return _render_response(payload, max_response_tokens)

    except RuntimeError as e:
        if "shutting down" in str(e):
            return _render_error("write", "server is shutting down", max_response_tokens)
        return _render_error("write", str(e), max_response_tokens)
    except FileNotFoundError as e:
        return _render_error("write", str(e), max_response_tokens)
    except PermissionError as e:
        return _render_error("write", f"permission denied - {e}", max_response_tokens)
    except ValueError as e:
        return _render_error("write", str(e), max_response_tokens)
    except TimeoutError:
        _handle_timeout(cache, "write", path)
        return _render_error("write", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except OSError as e:
        logger.warning(f"I/O error in write: {e}")
        return _render_error("write", f"I/O operation failed - {e}", max_response_tokens)
    except Exception:
        logger.exception("Unexpected error in write")
        return _render_error(
            "write", "Internal error occurred while writing file", max_response_tokens
        )


@mcp.tool()
@_serialized
async def edit(
    ctx: Context,
    path: str,
    old_string: str | None = None,
    new_string: str = "",
    replace_all: bool = False,
    dry_run: bool = False,
    auto_format: bool = False,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Edit file using find/replace with cached reads.

    Three modes — use line numbers from `read` to save tokens:
    - **find/replace**: old_string + new_string. Searches entire file.
    - **scoped**: old_string + new_string + start_line/end_line. Shorter context.
    - **line replace**: new_string + start_line/end_line only. Maximum savings.

    IMPORTANT: keep old_string minimal (one line). Prefer line replace when
    you have line numbers. For 2+ changes use batch_edit.

    Multiple matches: fails with hint. Add context or set replace_all=true.

    Args:
        path: File path (absolute or relative)
        old_string: Exact string to find. Keep to one line. Omit for line-replace.
        new_string: Replacement string
        replace_all: Replace all occurrences (default: false)
        dry_run: Preview without writing (default: false)
        auto_format: Run formatter after edit (default: false)
        start_line: Start of line range (1-based). Must pair with end_line.
        end_line: End of line range (1-based). Must pair with start_line.
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

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
        if result.diff_content:
            payload["diff"] = result.diff_content
        else:
            payload["diff_omitted"] = True
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
            }

        return _render_response(payload, max_response_tokens)

    except RuntimeError as e:
        if "shutting down" in str(e):
            return _render_error("edit", "server is shutting down", max_response_tokens)
        return _render_error("edit", str(e), max_response_tokens)
    except FileNotFoundError as e:
        return _render_error("edit", str(e), max_response_tokens)
    except PermissionError as e:
        return _render_error("edit", f"permission denied - {e}", max_response_tokens)
    except ValueError as e:
        return _render_error("edit", str(e), max_response_tokens)
    except TimeoutError:
        _handle_timeout(cache, "edit", path)
        return _render_error("edit", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except OSError as e:
        logger.warning(f"I/O error in edit: {e}")
        return _render_error("edit", f"I/O operation failed - {e}", max_response_tokens)
    except Exception:
        logger.exception("Unexpected error in edit")
        return _render_error(
            "edit", "Internal error occurred while editing file", max_response_tokens
        )


@mcp.tool()
@_serialized
async def batch_edit(
    ctx: Context,
    path: str,
    edits: str,
    dry_run: bool = False,
    auto_format: bool = False,
) -> str:
    """Apply multiple edits to a file in one call. Max 50 edits.

    Edit modes per entry:
    - [old, new] — find/replace
    - [old, new, start_line, end_line] — scoped find/replace
    - [null, new, start_line, end_line] — line replace (preferred)
    Also accepts {"old": ..., "new": ..., "start_line": ..., "end_line": ...}.

    IMPORTANT: prefer line replace [null, new, start, end] when you have
    line numbers. Keep old to one line when using find/replace.

    Partial success: response includes 'failed' count and 'failures' array.

    Args:
        path: Absolute path to file
        edits: JSON array of edit entries
        dry_run: Preview without writing (default: false)
        auto_format: Run formatter after edits (default: false)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

    try:
        # Parse edits JSON
        edits_str = edits.strip()
        if not edits_str.startswith("["):
            return _render_error(
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
                return _render_error(
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
        if result.diff_content:
            payload["diff"] = result.diff_content
        else:
            payload["diff_omitted"] = True
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
            payload["params"] = {"dry_run": dry_run, "auto_format": auto_format}

        return _render_response(payload, max_response_tokens)

    except RuntimeError as e:
        if "shutting down" in str(e):
            return _render_error("batch_edit", "server is shutting down", max_response_tokens)
        return _render_error("batch_edit", str(e), max_response_tokens)
    except json.JSONDecodeError as e:
        return _render_error("batch_edit", f"Invalid JSON in edits - {e}", max_response_tokens)
    except FileNotFoundError as e:
        return _render_error("batch_edit", str(e), max_response_tokens)
    except PermissionError as e:
        return _render_error("batch_edit", f"permission denied - {e}", max_response_tokens)
    except ValueError as e:
        return _render_error("batch_edit", str(e), max_response_tokens)
    except TimeoutError:
        _handle_timeout(cache, "batch_edit", path)
        return _render_error("batch_edit", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except Exception:
        logger.exception("Unexpected error in batch_edit")
        return _render_error(
            "batch_edit",
            "Internal error occurred while editing file",
            max_response_tokens,
        )


@mcp.tool()
@_serialized
async def search(
    ctx: Context,
    query: str,
    k: int = 10,
    directory: str | None = None,
) -> str:
    """Search cached files by meaning and keywords (hybrid BM25 + vector RRF).

    Only searches files already in the cache. Seed with batch_read first.
    Use `glob cached_only=true` to check what's cached.

    Timing:
    - batch_read target files first, then search across them.
    - Start with k=3–5; increase only if recall is insufficient.
    - Use directory to limit scope on large codebases.

    Args:
        query: Natural language or keyword search (both work — results are fused)
        k: Max results (default: 10, max: 100)
        directory: Optional directory path to limit search scope
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

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

        return _render_response(payload, max_response_tokens)

    except TimeoutError:
        _handle_timeout(cache, "search", query[:50])
        return _render_error("search", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except Exception as e:
        logger.exception("Error in search")
        return _render_error("search", str(e), max_response_tokens)


@mcp.tool()
@_serialized
async def diff(
    ctx: Context,
    path1: str,
    path2: str,
    context_lines: int = 3,
) -> str:
    """Compare two files. Returns unified diff and semantic similarity score.

    Use for explicit side-by-side comparison. For checking changes to a single
    file over time, use read (which returns diffs automatically).

    Large diffs are truncated to stay within token budget.

    Args:
        path1: First file path
        path2: Second file path
        context_lines: Lines of context in diff (default: 3)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

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
        }
        if mode in _MODE_NORMAL:
            payload["similarity"] = round(result.similarity, 4)
            payload["diff_stats"] = result.diff_stats
        if mode == _MODE_DEBUG:
            payload["tokens_saved"] = result.tokens_saved
            payload["from_cache"] = result.from_cache
            payload["context_lines"] = context_lines

        return _render_response(payload, max_response_tokens)

    except FileNotFoundError as e:
        return _render_error("diff", str(e), max_response_tokens)
    except TimeoutError:
        _handle_timeout(cache, "diff")
        return _render_error("diff", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except Exception as e:
        logger.exception("Error in diff")
        return _render_error("diff", str(e), max_response_tokens)


@mcp.tool()
@_serialized
async def batch_read(
    ctx: Context,
    paths: str,
    max_total_tokens: int = 50000,
    priority: str = "",
    diff_mode: bool = True,
) -> str:
    """Read multiple files under a token budget. Supports glob patterns in paths.

    Prefer over repeated read calls for 2+ files. Use early to seed the cache
    before search/similar/grep. Set diff_mode=false after context compression.

    Per-file status: full, diff, or skipped (with est_tokens).
    Unchanged files listed in summary.unchanged (already in your context).

    Args:
        paths: Comma-separated paths, JSON array, or glob patterns (e.g. "src/**/*.py")
        max_total_tokens: Token budget (default: 50000, max: 200000)
        priority: Comma-separated or JSON array of paths to read first
        diff_mode: When false, always return full content
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

    try:
        # Parse paths (comma-separated or JSON array)
        paths_str = paths.strip()
        if paths_str.startswith("["):
            path_list: list[str] = json.loads(paths_str)
        else:
            path_list = [p.strip() for p in paths_str.split(",") if p.strip()]

        # Expand glob patterns
        path_list = _expand_globs(path_list)

        # Parse priority
        priority_list: list[str] | None = None
        priority_str = priority.strip()
        if priority_str:
            if priority_str.startswith("["):
                priority_list = json.loads(priority_str)
            else:
                priority_list = [p.strip() for p in priority_str.split(",") if p.strip()]

        result = await asyncio.wait_for(
            batch_smart_read(
                cache,
                path_list,
                max_total_tokens=max_total_tokens,
                priority=priority_list,
                diff_mode=diff_mode,
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
            summary["unchanged"] = result.unchanged_paths
            summary["unchanged_count"] = len(result.unchanged_paths)

        skipped_items: list[dict[str, Any]] = []
        file_items: list[dict[str, Any]] = []
        for f in result.files:
            if f.status == "skipped":
                skipped_item: dict[str, Any] = {"path": f.path}
                if f.est_tokens is not None:
                    skipped_item["est_tokens"] = f.est_tokens
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
            payload["skipped"] = skipped_items
        payload["files"] = file_items

        return _render_response(payload, max_response_tokens)

    except json.JSONDecodeError:
        return _render_error(
            "batch_read",
            "Invalid paths format. Use comma-separated or JSON array.",
            max_response_tokens,
        )
    except TimeoutError:
        _handle_timeout(cache, "batch_read")
        return _render_error(
            "batch_read", f"timed out after {_TOOL_TIMEOUT * 2}s", max_response_tokens
        )
    except Exception as e:
        logger.exception("Error in batch_read")
        return _render_error("batch_read", str(e), max_response_tokens)


@mcp.tool()
@_serialized
async def similar(
    ctx: Context,
    path: str,
    k: int = 5,
) -> str:
    """Find files semantically similar to a given file using cached embeddings.

    Compares against files already in the cache. The source file is cached
    automatically, but neighbors must be cached first (via batch_read) to appear.

    Timing:
    - Use to discover related implementations, tests, or config files.
    - batch_read a directory first, then use similar to find connections.
    - Start with k=3–5.

    Args:
        path: Source file path (absolute or relative)
        k: Max results (default: 5, max: 50)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

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

        return _render_response(payload, max_response_tokens)

    except FileNotFoundError as e:
        return _render_error("similar", str(e), max_response_tokens)
    except TimeoutError:
        _handle_timeout(cache, "similar", path)
        return _render_error("similar", f"timed out after {_TOOL_TIMEOUT}s", max_response_tokens)
    except Exception as e:
        logger.exception("Error in similar")
        return _render_error("similar", str(e), max_response_tokens)


@mcp.tool()
@_serialized
async def glob(
    ctx: Context,
    pattern: str,
    directory: str = ".",
    cached_only: bool = False,
) -> str:
    """Find files by glob pattern with cache status. Max 1000 matches, 5s timeout.

    Timing:
    - Use first to discover files, then batch_read the results.
    - Keep patterns specific (e.g. "src/**/*.py" not "**/*").
    - Use cached_only=true to see what's already cached before search/grep.

    Each match shows: path, cached (bool), tokens, mtime. Pipe results
    directly into batch_read paths.

    Args:
        pattern: Glob pattern (e.g., "**/*.py")
        directory: Base directory (default: current)
        cached_only: Only return files already in cache (default: false)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

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
            if mode in _MODE_NORMAL:
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

        return _render_response(payload, max_response_tokens)

    except Exception as e:
        logger.exception("Error in glob")
        return _render_error("glob", str(e), max_response_tokens)


@mcp.tool()
@_serialized
async def grep(
    ctx: Context,
    pattern: str,
    fixed_string: bool = False,
    case_sensitive: bool = True,
    context_lines: int = 0,
    max_matches: int = 100,
    max_files: int = 50,
) -> str:
    """Exact pattern search across cached files with line numbers and context.

    Like ripgrep on the cache. For semantic/fuzzy queries, use `search` instead.
    Only searches cached files — seed with batch_read first.

    Timing:
    - Use for exact code patterns: function names, imports, error strings.
    - Use `search` for meaning-based queries like "error handling logic".
    - Set fixed_string=true for literals with regex chars (e.g. "foo.bar()").
    - Add context_lines=2–3 to see surrounding code.

    Args:
        pattern: Regex pattern (or literal string if fixed_string=true)
        fixed_string: Treat pattern as literal, not regex (default: false)
        case_sensitive: Case-sensitive matching (default: true)
        context_lines: Lines of context before/after each match (default: 0)
        max_matches: Total match limit across all files (default: 100)
        max_files: Maximum files to return (default: 50)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

    try:
        # In compact mode, context lines are dropped in the response anyway —
        # skip fetching them to avoid wasted storage work.
        effective_context = context_lines if mode in _MODE_NORMAL else 0
        results = await cache._storage.grep(
            pattern,
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
            "total_matches": total_matches,
            "files_matched": len(files_payload),
            "files": files_payload,
        }
        if mode == _MODE_DEBUG:
            payload["fixed_string"] = fixed_string
            payload["case_sensitive"] = case_sensitive
            payload["context_lines"] = context_lines

        return _render_response(payload, max_response_tokens)

    except Exception as e:
        logger.exception("Error in grep")
        return _render_error("grep", str(e), max_response_tokens)


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
