"""FastMCP server for semantic file caching.

Provides smart_read tool that achieves 80%+ token reduction through:
- Content-addressable storage with deduplication
- Semantic similarity for related file detection (local FastEmbed)
- Diff-based updates for changed files
- LRU-K eviction for optimal cache utilization
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan

from .cache import (
    SemanticCache,
    batch_smart_read,
    compare_files,
    find_similar_files,
    glob_with_cache_status,
    semantic_search,
    smart_edit,
    smart_multi_edit,
    smart_read,
    smart_write,
)
from .config import MAX_CONTENT_SIZE, TOOL_MAX_RESPONSE_TOKENS, TOOL_OUTPUT_MODE
from .core import count_tokens
from .core.embeddings import get_model_info, warmup
from .core.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

_MODE_NORMAL = {"normal", "debug"}
_MODE_DEBUG = "debug"


def _response_mode() -> str:
    """Global response mode from environment-backed config."""
    return TOOL_OUTPUT_MODE


def _response_token_cap() -> int | None:
    """Global response token cap from environment-backed config."""
    return TOOL_MAX_RESPONSE_TOKENS if TOOL_MAX_RESPONSE_TOKENS > 0 else None


def _minimal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build minimal JSON payload when response exceeds token budget."""
    keep_order = (
        "ok",
        "tool",
        "status",
        "path",
        "path1",
        "path2",
        "summary",
        "skipped",
        "files_read",
        "files_skipped",
        "succeeded",
        "failed",
        "message",
        "error",
    )
    minimal: dict[str, Any] = {}
    for key in keep_order:
        if key in payload:
            minimal[key] = payload[key]
    minimal["truncated"] = True
    if "message" not in minimal:
        minimal["message"] = "Response truncated by max_response_tokens"
    return minimal


def _render_response(payload: dict[str, Any], max_response_tokens: int | None) -> str:
    """Render tool response as compact JSON with optional token cap."""
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    if max_response_tokens is not None and max_response_tokens > 0:
        if count_tokens(body) > max_response_tokens:
            body = json.dumps(_minimal_payload(payload), separators=(",", ":"), ensure_ascii=False)
        if count_tokens(body) > max_response_tokens:
            body = json.dumps({"ok": False, "truncated": True}, separators=(",", ":"))
    return body


def _render_error(tool: str, message: str, max_response_tokens: int | None) -> str:
    """Render consistent error responses."""
    payload = {"ok": False, "tool": tool, "error": message}
    return _render_response(payload, max_response_tokens)


@lifespan
async def app_lifespan(server: FastMCP):
    """Initialize cache and embedding model on startup."""
    logger.info("Semantic cache MCP server starting...")

    # Warmup tokenizer (loads 200K vocab from disk, ~600ms one-time cost)
    logger.info("Initializing tokenizer...")
    get_tokenizer()

    # Warmup embedding model (downloads if needed, loads into memory)
    logger.info("Initializing embedding model...")
    warmup()

    model_info = get_model_info()
    if not model_info.get("ready", False):
        logger.error(
            "Embedding model failed to initialize. "
            "Semantic similarity features will be disabled. "
            "Check network connectivity and disk space."
        )
    else:
        logger.info(f"Embedding model ready: {model_info['model']}")

    # Initialize cache
    cache = SemanticCache()
    logger.info("Semantic cache MCP server started")

    try:
        yield {"cache": cache}
    finally:
        logger.info("Semantic cache MCP server stopped")


mcp = FastMCP("semantic-cache-mcp", lifespan=app_lifespan)


@mcp.tool(
    meta={
        "version": "1.0.0",
        "author": "Dayton Dunbar",
        "github": "https://github.com/CoderDayton/semantic-cache-mcp",
    }
)
def read(
    ctx: Context,
    path: str,
    max_size: int = MAX_CONTENT_SIZE,
    diff_mode: bool = True,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    """Read files with token-efficient caching and diffs.

    Timing guidance:
    - Use for single-file inspection and verification.
    - For 2+ files, prefer batch_read first.
    - Keep diff_mode=true during iteration; set false only for full uncached content.
    - Use offset/limit to read specific line ranges without loading full content into context.

    Args:
        path: Path to the file to read
        max_size: Maximum content size to return (default: 100000)
        diff_mode: Return diff if previously read (default: true). Set false for full content.
        offset: Line number to start reading from (1-based). Only provide if file is too large.
        limit: Number of lines to read. Only provide if file is too large.
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

    try:
        # If offset/limit specified, read specific lines (still caches full file)
        if offset is not None or limit is not None:
            result = smart_read(
                cache=cache,
                path=path,
                max_size=max_size,
                diff_mode=False,  # Line ranges bypass diff mode
                force_full=True,
            )
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

        result = smart_read(
            cache=cache,
            path=path,
            max_size=max_size,
            diff_mode=diff_mode,
        )
        payload = {
            "ok": True,
            "tool": "read",
            "path": path,
            "content": result.content,
        }
        if mode in _MODE_NORMAL:
            payload["is_diff"] = result.is_diff
            payload["truncated"] = result.truncated
            payload["semantic_match"] = result.semantic_match
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
    except Exception as e:
        return _render_error("read", f"reading failed: {e}", max_response_tokens)


@mcp.tool()
def stats(
    ctx: Context,
) -> str:
    """Get cache statistics: files tracked, tokens, compression ratios, embedding status."""
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()
    cache_stats = cache.get_stats()

    # Add embedding model info
    model_info = get_model_info()
    result: dict[str, int | float | str | bool] = {
        **cache_stats,
        "embedding_model": model_info["model"],
        "embedding_ready": model_info["ready"],
    }
    payload: dict[str, Any] = {"ok": True, "tool": "stats"}
    if mode == "compact":
        payload.update(
            {
                "files_cached": result["files_cached"],
                "total_tokens_cached": result["total_tokens_cached"],
                "embedding_ready": result["embedding_ready"],
            }
        )
    elif mode == "normal":
        payload.update(result)
    else:
        payload.update(result)
        payload["output_mode"] = mode

    return _render_response(payload, max_response_tokens)


@mcp.tool()
def clear(
    ctx: Context,
) -> str:
    """Clear all cache entries (content, embeddings, indexes). Returns count removed."""
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()
    count = cache.clear()
    payload: dict[str, Any] = {"ok": True, "tool": "clear", "status": "cleared", "count": count}
    if mode == _MODE_DEBUG:
        payload["output_mode"] = mode
    return _render_response(payload, max_response_tokens)


@mcp.tool()
def write(
    ctx: Context,
    path: str,
    content: str,
    create_parents: bool = True,
    dry_run: bool = False,
    auto_format: bool = False,
) -> str:
    """Write full file content with cache integration.

    Timing guidance:
    - Use when creating files or replacing most/all content.
    - For focused substitutions, prefer edit or multi_edit.
    - Keep auto_format=false during rapid iterations; enable at stabilization points.

    For files too large to write in a single output, write in two steps:
      1. write(path, first_portion)
      2. edit(path, last_line, last_line + remainder)  # last_line must be unique

    Args:
        path: Absolute path to file
        content: Content to write
        create_parents: Create parent directories (default: true)
        dry_run: Preview changes without writing (default: false)
        auto_format: Run formatter after write (default: false)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

    try:
        result = smart_write(
            cache=cache,
            path=path,
            content=content,
            create_parents=create_parents,
            dry_run=dry_run,
            auto_format=auto_format,
        )

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

    except FileNotFoundError as e:
        return _render_error("write", str(e), max_response_tokens)
    except PermissionError as e:
        return _render_error("write", f"permission denied - {e}", max_response_tokens)
    except ValueError as e:
        return _render_error("write", str(e), max_response_tokens)
    except OSError as e:
        logger.warning(f"I/O error in write: {e}")
        return _render_error("write", f"I/O operation failed - {e}", max_response_tokens)
    except Exception:
        logger.exception("Unexpected error in write")
        return _render_error(
            "write", "Internal error occurred while writing file", max_response_tokens
        )


@mcp.tool()
def edit(
    ctx: Context,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    dry_run: bool = False,
    auto_format: bool = False,
) -> str:
    """Edit file using find/replace with cached reads.

    Timing guidance:
    - Use for a single targeted replacement.
    - For 2+ independent replacements in one file, use multi_edit.
    - Use dry_run when validating match uniqueness before committing edits.

    Args:
        path: Absolute path to file
        old_string: Exact string to find (whitespace-sensitive)
        new_string: Replacement string
        replace_all: Replace all occurrences (default: false)
        dry_run: Preview without writing (default: false)
        auto_format: Run formatter after edit (default: false)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

    try:
        result = smart_edit(
            cache=cache,
            path=path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            dry_run=dry_run,
            auto_format=auto_format,
        )

        payload: dict[str, Any] = {
            "ok": True,
            "tool": "edit",
            "status": "edited",
            "path": result.path,
            "matches_found": result.matches_found,
            "replacements_made": result.replacements_made,
        }
        if result.diff_content:
            payload["diff"] = result.diff_content
        else:
            payload["diff_omitted"] = True
        if mode in _MODE_NORMAL:
            payload["line_numbers"] = result.line_numbers
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

    except FileNotFoundError as e:
        return _render_error("edit", str(e), max_response_tokens)
    except PermissionError as e:
        return _render_error("edit", f"permission denied - {e}", max_response_tokens)
    except ValueError as e:
        return _render_error("edit", str(e), max_response_tokens)
    except OSError as e:
        logger.warning(f"I/O error in edit: {e}")
        return _render_error("edit", f"I/O operation failed - {e}", max_response_tokens)
    except Exception:
        logger.exception("Unexpected error in edit")
        return _render_error(
            "edit", "Internal error occurred while editing file", max_response_tokens
        )


@mcp.tool()
def multi_edit(
    ctx: Context,
    path: str,
    edits: str,
    dry_run: bool = False,
    auto_format: bool = False,
) -> str:
    """Apply multiple independent edits to a file.

    Timing guidance:
    - Use when applying 2+ edits to the same file in one step.
    - More token-efficient than repeated edit calls on the same file.
    - Use dry_run when testing risky edit batches.

    Args:
        path: Absolute path to file
        edits: JSON array of [old, new] pairs or {"old": ..., "new": ...} objects
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
                "multi_edit",
                "edits must be a JSON array of [old, new] pairs",
                max_response_tokens,
            )

        edit_list = json.loads(edits_str)
        if not isinstance(edit_list, list):
            return _render_error("multi_edit", "edits must be a JSON array", max_response_tokens)

        # Convert to list of tuples
        edit_tuples: list[tuple[str, str]] = []
        for item in edit_list:
            if isinstance(item, list) and len(item) == 2:
                edit_tuples.append((str(item[0]), str(item[1])))
            elif isinstance(item, dict) and "old" in item and "new" in item:
                edit_tuples.append((str(item["old"]), str(item["new"])))
            else:
                return _render_error(
                    "multi_edit",
                    "Each edit must be [old, new] or {old, new}",
                    max_response_tokens,
                )

        result = smart_multi_edit(
            cache=cache,
            path=path,
            edits=edit_tuples,
            dry_run=dry_run,
            auto_format=auto_format,
        )

        payload: dict[str, Any] = {
            "ok": True,
            "tool": "multi_edit",
            "status": "edited",
            "path": result.path,
            "succeeded": result.succeeded,
            "failed": result.failed,
        }
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

    except json.JSONDecodeError as e:
        return _render_error("multi_edit", f"Invalid JSON in edits - {e}", max_response_tokens)
    except FileNotFoundError as e:
        return _render_error("multi_edit", str(e), max_response_tokens)
    except PermissionError as e:
        return _render_error("multi_edit", f"permission denied - {e}", max_response_tokens)
    except ValueError as e:
        return _render_error("multi_edit", str(e), max_response_tokens)
    except Exception:
        logger.exception("Unexpected error in multi_edit")
        return _render_error(
            "multi_edit",
            "Internal error occurred while editing file",
            max_response_tokens,
        )


@mcp.tool()
def search(
    ctx: Context,
    query: str,
    k: int = 10,
    directory: str | None = None,
) -> str:
    """Search cached files by semantic meaning.

    Timing guidance:
    - Seed cache first via read or batch_read.
    - Start with k=3 to k=5 and increase only if recall is insufficient.
    - Use directory filter early to limit response size.

    Args:
        query: Search query (what you're looking for)
        k: Max results (default: 10, max: 100)
        directory: Optional directory to limit search
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

    try:
        result = semantic_search(cache, query, k=k, directory=directory)

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

    except Exception as e:
        logger.exception("Error in search")
        return _render_error("search", str(e), max_response_tokens)


@mcp.tool()
def diff(
    ctx: Context,
    path1: str,
    path2: str,
    context_lines: int = 3,
) -> str:
    """Compare two files using cache.

    Timing guidance:
    - Use only for explicit two-file comparisons.
    - Prefer read for normal iterative file updates.
    - Lower context_lines for tighter outputs when reviewing many diffs.

    Args:
        path1: First file path
        path2: Second file path
        context_lines: Lines of context in diff (default: 3)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

    try:
        result = compare_files(cache, path1, path2, context_lines=context_lines)

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
    except Exception as e:
        logger.exception("Error in diff")
        return _render_error("diff", str(e), max_response_tokens)


def _expand_globs(raw_paths: list[str], max_files: int = 50) -> list[str]:
    """Expand glob patterns in path list. Non-glob paths pass through unchanged."""
    expanded: list[str] = []
    glob_chars = frozenset("*?[")
    for p in raw_paths:
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
                    matches = sorted(str(m) for m in base.glob(pattern) if m.is_file())
                    expanded.extend(matches[: max_files - len(expanded)])
            except (OSError, ValueError):
                expanded.append(p)  # Treat invalid pattern as literal
        else:
            expanded.append(p)
        if len(expanded) >= max_files:
            break
    return expanded[:max_files]


@mcp.tool()
def batch_read(
    ctx: Context,
    paths: str,
    max_total_tokens: int = 50000,
    priority: str = "",
) -> str:
    """Read multiple files under a token budget.

    Timing guidance:
    - Prefer over repeated read calls when working with 2+ files.
    - Start with a tight max_total_tokens budget, then increase only if needed.
    - Use this early to seed cache before search/similar operations.

    Args:
        paths: Comma-separated paths or JSON array
        max_total_tokens: Token budget (default: 50000, max: 200000)
        priority: Comma-separated or JSON array of paths to read first (order preserved)
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

        result = batch_smart_read(
            cache, path_list, max_total_tokens=max_total_tokens, priority=priority_list
        )

        # Build restructured response — separate unchanged, skipped, and content files
        summary: dict[str, Any] = {
            "files_read": result.files_read,
            "files_skipped": result.files_skipped,
            "total_tokens": result.total_tokens,
            "tokens_saved": result.tokens_saved,
        }
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
    except Exception as e:
        logger.exception("Error in batch_read")
        return _render_error("batch_read", str(e), max_response_tokens)


@mcp.tool()
def similar(
    ctx: Context,
    path: str,
    k: int = 5,
) -> str:
    """Find cached files semantically similar to given file.

    Timing guidance:
    - Use after reading source and likely neighbors into cache.
    - Start with small k (3-5) and increase only if needed.
    - Prefer this after a focused read to find adjacent implementation/test files.

    Args:
        path: Source file path
        k: Max results (default: 5, max: 50)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

    try:
        result = find_similar_files(cache, path, k=k)

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
    except Exception as e:
        logger.exception("Error in similar")
        return _render_error("similar", str(e), max_response_tokens)


@mcp.tool()
def glob(
    ctx: Context,
    pattern: str,
    directory: str = ".",
) -> str:
    """Find files by pattern with cache status. Max 1000 matches, 5s timeout.

    Timing guidance:
    - Use early to shortlist candidate files before reading content.
    - Follow with batch_read on selected files instead of reading all matches.
    - Keep patterns specific to avoid large low-value result lists.

    Args:
        pattern: Glob pattern (e.g., "**/*.py")
        directory: Base directory (default: current)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    mode = _response_mode()
    max_response_tokens = _response_token_cap()

    try:
        result = glob_with_cache_status(cache, pattern, directory=directory)
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


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
