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
from .config import CACHE_DIR, MAX_CONTENT_SIZE
from .core.embeddings import configure as configure_embeddings
from .core.embeddings import get_model_info, warmup

logger = logging.getLogger(__name__)


@lifespan
async def app_lifespan(server: FastMCP):
    """Initialize cache and embedding model on startup."""
    logger.info("Semantic cache MCP server starting...")

    # Configure and warmup embedding model (downloads if needed, loads into memory)
    logger.info("Initializing embedding model...")
    configure_embeddings(cache_dir=CACHE_DIR / "models")
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
    """Read files with 80%+ token reduction.

    Returns diffs for changed files, minimal response for unchanged.

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
    """Get cache statistics: files tracked, tokens, compression ratios, embedding status."""
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
    """Clear all cache entries (content, embeddings, indexes). Returns count removed."""
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
    auto_format: bool = False,
) -> str:
    """Write file with semantic cache integration.

    Returns diff of changes (not full content) for massive token savings.
    Updates cache so subsequent reads are instant.

    Args:
        path: Absolute path to file
        content: Content to write
        create_parents: Create parent directories (default: true)
        dry_run: Preview changes without writing (default: false)
        auto_format: Run formatter after write (default: false)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]

    try:
        result = smart_write(
            cache=cache,
            path=path,
            content=content,
            create_parents=create_parents,
            dry_run=dry_run,
            auto_format=auto_format,
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
    auto_format: bool = False,
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
        auto_format: Run formatter after edit (default: false)
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
            auto_format=auto_format,
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


@mcp.tool()
def multi_edit(
    ctx: Context,
    path: str,
    edits: str,
    dry_run: bool = False,
    auto_format: bool = False,
) -> str:
    """Apply multiple independent edits to a file.

    Each edit is processed independently - some can succeed while others fail.
    Partial success: successful edits are applied even if some fail.

    Args:
        path: Absolute path to file
        edits: JSON array of [old, new] pairs or {"old": ..., "new": ...} objects
        dry_run: Preview without writing (default: false)
        auto_format: Run formatter after edits (default: false)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]

    try:
        # Parse edits JSON
        edits_str = edits.strip()
        if not edits_str.startswith("["):
            return "Error: edits must be a JSON array of [old, new] pairs"

        edit_list = json.loads(edits_str)
        if not isinstance(edit_list, list):
            return "Error: edits must be a JSON array"

        # Convert to list of tuples
        edit_tuples: list[tuple[str, str]] = []
        for item in edit_list:
            if isinstance(item, list) and len(item) == 2:
                edit_tuples.append((str(item[0]), str(item[1])))
            elif isinstance(item, dict) and "old" in item and "new" in item:
                edit_tuples.append((str(item["old"]), str(item["new"])))
            else:
                return "Error: Each edit must be [old, new] or {old, new}"

        result = smart_multi_edit(
            cache=cache,
            path=path,
            edits=edit_tuples,
            dry_run=dry_run,
            auto_format=auto_format,
        )

        # Format output
        lines: list[str] = []
        lines.append(f"// Multi-edit: {result.path}")
        lines.append(f"// Results: {result.succeeded} succeeded, {result.failed} failed")

        if result.tokens_saved > 0:
            lines.append(f"// Tokens saved: {result.tokens_saved:,} (cached read)")

        lines.append("")

        # Per-edit outcomes
        for o in result.outcomes:
            old_preview = o.old_string[:30].replace("\n", "\\n")
            new_preview = o.new_string[:30].replace("\n", "\\n")
            if o.success:
                lines.append(f'✓ Line {o.line_number}: "{old_preview}" → "{new_preview}"')
            else:
                lines.append(f'✗ "{old_preview}" → "{new_preview}" ({o.error})')

        # Diff
        if result.diff_content:
            lines.append("")
            lines.append(result.diff_content)

        # Stats
        stats = result.diff_stats
        lines.append(
            f"\n// Stats: +{stats.get('insertions', 0)} "
            f"-{stats.get('deletions', 0)} "
            f"~{stats.get('modifications', 0)} lines"
        )
        lines.append(f"// Hash: {result.content_hash[:16]}...")
        lines.append(
            f"// [succeeded:{result.succeeded} failed:{result.failed} "
            f"dry_run:{dry_run} cached:{result.from_cache}]"
        )

        return "\n".join(lines)

    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON in edits - {e}"
    except FileNotFoundError as e:
        return f"Error: {e}"
    except PermissionError as e:
        return f"Error: Permission denied - {e}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception:
        logger.exception("Unexpected error in multi_edit")
        return "Error: Internal error occurred while editing file"


@mcp.tool()
def search(
    ctx: Context,
    query: str,
    k: int = 10,
    directory: str | None = None,
) -> str:
    """Search cached files by semantic meaning. Only searches previously-read files.

    Better than grep for concepts - finds by meaning, not keywords.

    Args:
        query: Search query (what you're looking for)
        k: Max results (default: 10, max: 100)
        directory: Optional directory to limit search
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]

    try:
        result = semantic_search(cache, query, k=k, directory=directory)

        if not result.matches:
            return f"// No matches for: {query}\n// [files:0 cached:{result.cached_files}]"

        lines: list[str] = []
        lines.append(
            f'// Search: "{query}" ({len(result.matches)} matches in {result.cached_files} cached)'
        )

        for i, m in enumerate(result.matches, 1):
            lines.append(f"{i}. {m.path} ({m.similarity:.2f}) - {m.tokens:,} tokens")
            lines.append(f"   {m.preview}...")

        meta = f"k:{k}"
        if directory:
            meta += f" directory:{directory}"
        lines.append(f"// [{meta}]")

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in search")
        return f"Error: {e}"


@mcp.tool()
def diff(
    ctx: Context,
    path1: str,
    path2: str,
    context_lines: int = 3,
) -> str:
    """Compare two files using cache.

    Avoids reading both files if cached. Shows unified diff with stats.

    Args:
        path1: First file path
        path2: Second file path
        context_lines: Lines of context in diff (default: 3)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]

    try:
        result = compare_files(cache, path1, path2, context_lines=context_lines)

        lines: list[str] = []
        lines.append(f"// Diff: {result.path1} vs {result.path2}")

        stats = result.diff_stats
        lines.append(
            f"// Stats: +{stats.get('insertions', 0)} "
            f"-{stats.get('deletions', 0)} "
            f"~{stats.get('modifications', 0)} lines"
        )
        lines.append(f"// Similarity: {result.similarity:.2f}")

        if result.tokens_saved > 0:
            lines.append(f"// Tokens saved: {result.tokens_saved:,} (cached)")

        lines.append(result.diff_content)

        cached_str = f"{result.from_cache[0]},{result.from_cache[1]}"
        lines.append(f"// [context:{context_lines} cached:{cached_str}]")

        return "\n".join(lines)

    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.exception("Error in diff")
        return f"Error: {e}"


@mcp.tool()
def batch_read(
    ctx: Context,
    paths: str,
    max_total_tokens: int = 50000,
) -> str:
    """Read multiple files with token budget. Skips files if budget exceeded.

    Single call reduces overhead vs reading one-by-one.

    Args:
        paths: Comma-separated paths or JSON array
        max_total_tokens: Token budget (default: 50000, max: 200000)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]

    try:
        # Parse paths (comma-separated or JSON array)
        paths_str = paths.strip()
        if paths_str.startswith("["):
            path_list = json.loads(paths_str)
        else:
            path_list = [p.strip() for p in paths_str.split(",") if p.strip()]

        result = batch_smart_read(cache, path_list, max_total_tokens=max_total_tokens)

        lines: list[str] = []
        cached_count = sum(1 for f in result.files if f.from_cache)
        lines.append(
            f"// Batch read: {result.files_read} files "
            f"({cached_count} cached, {result.files_read - cached_count} new)"
        )
        lines.append(
            f"// Total: {result.total_tokens:,} tokens | Saved: {result.tokens_saved:,} tokens"
        )

        for f in result.files:
            if f.status == "skipped":
                lines.append(f"\n## {f.path} (skipped)")
            else:
                lines.append(f"\n## {f.path} ({f.status}, {f.tokens:,} tokens)")
                if f.path in result.contents:
                    lines.append(result.contents[f.path])

        lines.append(
            f"\n// [files:{len(result.files)} tokens:{result.total_tokens} "
            f"saved:{result.tokens_saved} skipped:{result.files_skipped}]"
        )

        return "\n".join(lines)

    except json.JSONDecodeError:
        return "Error: Invalid paths format. Use comma-separated or JSON array."
    except Exception as e:
        logger.exception("Error in batch_read")
        return f"Error: {e}"


@mcp.tool()
def similar(
    ctx: Context,
    path: str,
    k: int = 5,
) -> str:
    """Find cached files semantically similar to given file.

    Searches only cached files. Useful for finding related code, tests, or docs.

    Args:
        path: Source file path
        k: Max results (default: 5, max: 50)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]

    try:
        result = find_similar_files(cache, path, k=k)

        if not result.similar_files:
            return (
                f"// Similar to: {result.source_path}\n"
                f"// No similar files found in {result.files_searched} cached\n"
                f"// [k:{k}]"
            )

        lines: list[str] = []
        lines.append(f"// Similar to: {result.source_path} ({result.source_tokens:,} tokens)")
        lines.append(
            f"// Found {len(result.similar_files)} similar in {result.files_searched} cached"
        )

        for i, f in enumerate(result.similar_files, 1):
            lines.append(f"{i}. {f.path} ({f.similarity:.2f}) - {f.tokens:,} tokens")

        lines.append(f"// [k:{k} searched:{result.files_searched}]")

        return "\n".join(lines)

    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        logger.exception("Error in similar")
        return f"Error: {e}"


@mcp.tool()
def glob(
    ctx: Context,
    pattern: str,
    directory: str = ".",
) -> str:
    """Find files by pattern with cache status. Max 1000 matches, 5s timeout.

    Shows which files are cached and their token counts.

    Args:
        pattern: Glob pattern (e.g., "**/*.py")
        directory: Base directory (default: current)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]

    try:
        result = glob_with_cache_status(cache, pattern, directory=directory)

        if not result.matches:
            return (
                f"// Glob: {pattern} in {result.directory}\n// No matches\n// [pattern:{pattern}]"
            )

        lines: list[str] = []
        lines.append(f"// Glob: {pattern} in {result.directory}")
        lines.append(
            f"// Found {result.total_matches} files "
            f"({result.cached_count} cached, {result.total_matches - result.cached_count} new)"
        )

        # Group by cached status
        cached = [m for m in result.matches if m.cached]
        not_cached = [m for m in result.matches if not m.cached]

        if cached:
            lines.append(f"\nCached ({len(cached)} files, {result.total_cached_tokens:,} tokens):")
            for m in cached[:20]:  # Limit display
                lines.append(f"  {m.path} ({m.tokens:,} tokens)")
            if len(cached) > 20:
                lines.append(f"  ... and {len(cached) - 20} more")

        if not_cached:
            lines.append(f"\nNot cached ({len(not_cached)} files):")
            for m in not_cached[:20]:
                lines.append(f"  {m.path}")
            if len(not_cached) > 20:
                lines.append(f"  ... and {len(not_cached) - 20} more")

        lines.append(
            f"\n// [pattern:{pattern} matches:{result.total_matches} cached:{result.cached_count}]"
        )

        return "\n".join(lines)

    except Exception as e:
        logger.exception("Error in glob")
        return f"Error: {e}"


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
