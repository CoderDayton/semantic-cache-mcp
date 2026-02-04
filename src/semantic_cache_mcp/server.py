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

from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan

from .cache import SemanticCache, smart_edit, smart_read, smart_write
from .config import MAX_CONTENT_SIZE
from .core.embeddings import get_model_info, warmup

logger = logging.getLogger(__name__)


@lifespan
async def app_lifespan(server: FastMCP):
    """Initialize cache and embedding model on startup."""
    logger.info("Semantic cache MCP server starting...")

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
    force_full: bool = False,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    """Read files with 80%+ token reduction. Use INSTEAD of Read tool.

    Returns diffs for changed files, minimal response for unchanged files,
    and can reference semantically similar cached files.

    Args:
        path: Path to the file to read
        max_size: Maximum content size to return (default: 100000)
        diff_mode: Return diff if file was previously read (default: true)
        force_full: Force full content even if cached (default: false)
        offset: Line number to start reading from (1-based). Only provide if file is too large.
        limit: Number of lines to read. Only provide if file is too large.
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]

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

            return f"{content}\n// [lines:{start + 1}-{min(end, len(lines))} of {len(lines)}]"

        result = smart_read(
            cache=cache,
            path=path,
            max_size=max_size,
            diff_mode=diff_mode,
            force_full=force_full,
        )
        meta = f"[cache:{result.from_cache} diff:{result.is_diff} saved:{result.tokens_saved}]"
        return f"{result.content}\n// {meta}"

    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def stats(ctx: Context) -> str:
    """Retrieve comprehensive semantic cache statistics including:

    - Number of files tracked in the cache
    - Total and unique token counts
    - Compression and deduplication ratios for stored file content
    - Current cache size versus disk usage
    - Embedding model status and readiness
    - Performance metrics if available (e.g., hit/miss rates, recent cache savings)
    - Additional implementation-specific details, if exposed by the cache backend

    Use this tool to monitor cache health, efficiency, and performance over time.
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    cache_stats = cache.get_stats()

    # Add embedding model info
    model_info = get_model_info()
    result: dict[str, int | float | str | bool] = {
        **cache_stats,
        "embedding_model": model_info["model"],
        "embedding_ready": model_info["ready"],
    }

    return json.dumps(result, indent=2)


@mcp.tool()
def clear(ctx: Context) -> str:
    """Clear all cache entries including file content, diffs, and similarity indexes.

    Use this tool to reset cache state, purge cached tokens, or start fresh.
    Returns the number of entries removed.
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]
    count = cache.clear()
    return f"Cleared {count} cache entries"


@mcp.tool()
def write(
    ctx: Context,
    path: str,
    content: str,
    create_parents: bool = True,
    dry_run: bool = False,
) -> str:
    """Write file with semantic cache integration.

    Returns diff of changes (not full content) for massive token savings.
    Updates cache so subsequent reads are instant.

    Args:
        path: Absolute path to file
        content: Content to write
        create_parents: Create parent directories (default: true)
        dry_run: Preview changes without writing (default: false)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]

    try:
        result = smart_write(
            cache=cache,
            path=path,
            content=content,
            create_parents=create_parents,
            dry_run=dry_run,
        )

        # Format output
        lines: list[str] = []

        if result.created:
            lines.append(
                f"// Created: {result.path} "
                f"({result.bytes_written:,} bytes, {result.tokens_written:,} tokens)"
            )
        else:
            lines.append(f"// Updated: {result.path}")
            if result.diff_stats:
                stats = result.diff_stats
                lines.append(
                    f"// Stats: +{stats.get('insertions', 0)} "
                    f"-{stats.get('deletions', 0)} "
                    f"~{stats.get('modifications', 0)} lines"
                )
            if result.tokens_saved > 0:
                lines.append(f"// Tokens saved: {result.tokens_saved:,} (diff vs full content)")

        lines.append(f"// Hash: {result.content_hash[:16]}...")

        # Include diff for updates
        if result.diff_content:
            lines.append(result.diff_content)

        # Metadata footer
        meta_parts = [
            f"created:{result.created}",
            f"dry_run:{dry_run}",
            f"cached:{result.from_cache}",
        ]
        lines.append(f"// [{' '.join(meta_parts)}]")

        return "\n".join(lines)

    except FileNotFoundError as e:
        return f"Error: {e}"
    except PermissionError as e:
        return f"Error: Permission denied - {e}"
    except ValueError as e:
        return f"Error: {e}"
    except OSError as e:
        logger.warning(f"I/O error in write: {e}")
        return f"Error: I/O operation failed - {e}"
    except Exception:
        logger.exception("Unexpected error in write")
        return "Error: Internal error occurred while writing file"


@mcp.tool()
def edit(
    ctx: Context,
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    dry_run: bool = False,
) -> str:
    """Edit file using find/replace with cached reads.

    Uses semantic cache for reading - no token cost for the read phase!
    Returns diff showing exactly what changed.

    Args:
        path: Absolute path to file
        old_string: Exact string to find (whitespace-sensitive)
        new_string: Replacement string
        replace_all: Replace all occurrences (default: false)
        dry_run: Preview without writing (default: false)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]

    try:
        result = smart_edit(
            cache=cache,
            path=path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            dry_run=dry_run,
        )

        # Format output
        lines: list[str] = []

        lines.append(f"// Edited: {result.path}")

        # Match info
        if result.matches_found == 1:
            lines.append(f"// Replaced 1 of 1 match at line {result.line_numbers[0]}")
        else:
            lines.append(
                f"// Replaced {result.replacements_made} of {result.matches_found} "
                f"matches at lines {result.line_numbers}"
            )

        # Stats
        stats = result.diff_stats
        lines.append(
            f"// Stats: +{stats.get('insertions', 0)} "
            f"-{stats.get('deletions', 0)} "
            f"~{stats.get('modifications', 0)} lines"
        )

        # Token savings from cached read
        if result.tokens_saved > 0:
            lines.append(f"// Tokens saved: {result.tokens_saved:,} (cached read)")

        lines.append(f"// Hash: {result.content_hash[:16]}...")

        # Include diff
        lines.append(result.diff_content)

        # Metadata footer
        meta_parts = [
            f"replace_all:{replace_all}",
            f"dry_run:{dry_run}",
            f"cached:{result.from_cache}",
        ]
        lines.append(f"// [{' '.join(meta_parts)}]")

        return "\n".join(lines)

    except FileNotFoundError as e:
        return f"Error: {e}"
    except PermissionError as e:
        return f"Error: Permission denied - {e}"
    except ValueError as e:
        # ValueError messages are user-friendly (from smart_edit validation)
        return f"Error: {e}"
    except OSError as e:
        logger.warning(f"I/O error in edit: {e}")
        return f"Error: I/O operation failed - {e}"
    except Exception:
        logger.exception("Unexpected error in edit")
        return "Error: Internal error occurred while editing file"


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
