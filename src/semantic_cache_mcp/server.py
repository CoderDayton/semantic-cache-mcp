"""FastMCP server for semantic file caching.

Provides smart_read tool that achieves 80%+ token reduction through:
- Content-addressable storage with deduplication
- Semantic similarity for related file detection
- Diff-based updates for changed files
- LRU-K eviction for optimal cache utilization
"""

from __future__ import annotations

import json

from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan
from openai import OpenAI

from .cache import SemanticCache, smart_read
from .config import EMBEDDINGS_BASE_URL, MAX_CONTENT_SIZE


@lifespan
async def app_lifespan(server: FastMCP):
    """Initialize cache and OpenAI client on startup."""
    client = OpenAI(base_url=EMBEDDINGS_BASE_URL, api_key="not-needed")
    cache = SemanticCache(client=client)

    try:
        yield {"cache": cache, "client": client}
    finally:
        client.close()


mcp = FastMCP("semantic-cache-mcp", lifespan=app_lifespan)


@mcp.tool()
def read(
    ctx: Context,
    path: str,
    max_size: int = MAX_CONTENT_SIZE,
    diff_mode: bool = True,
    force_full: bool = False,
) -> str:
    """Read files with 80%+ token reduction. Use INSTEAD of Read tool.

    Returns diffs for changed files, minimal response for unchanged files,
    and can reference semantically similar cached files.

    Args:
        path: Path to the file to read
        max_size: Maximum content size to return (default: 100000)
        diff_mode: Return diff if file was previously read (default: true)
        force_full: Force full content even if cached (default: false)
    """
    cache: SemanticCache = ctx.lifespan_context["cache"]

    try:
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
    """Get semantic cache statistics: files, tokens, compression, deduplication ratios."""
    cache: SemanticCache = ctx.lifespan_context["cache"]
    return json.dumps(cache.get_stats(), indent=2)


@mcp.tool()
def clear(ctx: Context) -> str:
    """Clear the semantic cache."""
    cache: SemanticCache = ctx.lifespan_context["cache"]
    count = cache.clear()
    return f"Cleared {count} cache entries"


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
