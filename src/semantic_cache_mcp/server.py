"""MCP Server for semantic file caching.

Provides smart_read tool that achieves 80%+ token reduction through:
- Content-addressable storage with deduplication
- Semantic similarity for related file detection
- Diff-based updates for changed files
- LRU-K eviction for optimal cache utilization
"""

from __future__ import annotations

import json
from pathlib import Path

from fastmcp import Context, FastMCP
from fastmcp.server.lifespan import lifespan
from openai import OpenAI

from .cache import (
    EMBEDDINGS_BASE_URL,
    MAX_CONTENT_SIZE,
    ReadResult,
    SemanticCache,
    generate_diff,
    truncate_smart,
)


@lifespan
async def app_lifespan(server: FastMCP):
    """Manage application lifecycle - initialize cache and OpenAI client."""
    # Initialize OpenAI client pointing to local embeddings service
    client = OpenAI(base_url=EMBEDDINGS_BASE_URL, api_key="not-needed")
    cache = SemanticCache(client=client)

    try:
        yield {"cache": cache, "client": client}
    finally:
        client.close()


# Initialize FastMCP server with lifespan
mcp = FastMCP("semantic-cache-mcp", lifespan=app_lifespan)


def _smart_read(
    cache: SemanticCache,
    path: str,
    max_size: int = MAX_CONTENT_SIZE,
    diff_mode: bool = True,
    force_full: bool = False,
) -> ReadResult:
    """Read file with intelligent caching and optimization.

    Strategies (in order of token savings):
    1. File unchanged (mtime match) -> "// No changes" (99% reduction)
    2. File changed -> unified diff (80-95% reduction)
    3. Similar file in cache -> reference + diff (70-90% reduction)
    4. Large file -> smart truncation (50-80% reduction)
    5. New file -> full content with caching for future reads
    """
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Read current file
    content = file_path.read_text()
    mtime = file_path.stat().st_mtime
    tokens_original = cache._count_tokens(content)

    # Check cache
    cached = cache.get(str(file_path))

    if cached and diff_mode and not force_full:
        # File in cache - check if changed
        if cached.mtime >= mtime:
            # File unchanged - return minimal response only if it saves tokens
            cache.record_access(str(file_path))
            unchanged_msg = f"// File unchanged: {path} ({cached.tokens} tokens cached)"
            msg_tokens = cache._count_tokens(unchanged_msg)

            # Only use message if it actually saves tokens
            if msg_tokens < tokens_original:
                return ReadResult(
                    content=unchanged_msg,
                    from_cache=True,
                    is_diff=False,
                    tokens_original=tokens_original,
                    tokens_returned=msg_tokens,
                    tokens_saved=tokens_original - msg_tokens,
                    truncated=False,
                    compression_ratio=len(unchanged_msg) / len(content) if content else 1.0,
                )
            # Small file - return content from cache (no savings but confirms cache works)
            cached_content = cache.get_content(cached)
            return ReadResult(
                content=cached_content,
                from_cache=True,
                is_diff=False,
                tokens_original=tokens_original,
                tokens_returned=tokens_original,
                tokens_saved=0,
                truncated=False,
                compression_ratio=1.0,
            )

        # File changed - generate diff
        old_content = cache.get_content(cached)
        diff_content = generate_diff(old_content, content)
        diff_tokens = cache._count_tokens(diff_content)

        # Only use diff if it's significantly smaller (< 60% of original)
        if diff_tokens < tokens_original * 0.6:
            result_content = f"// Diff for {path} (changed since cache):\n{diff_content}"

            # Update cache with new content
            embedding = cache._get_embedding(content)
            cache.put(str(file_path), content, mtime, embedding)

            tokens_returned = cache._count_tokens(result_content)
            return ReadResult(
                content=result_content,
                from_cache=True,
                is_diff=True,
                tokens_original=tokens_original,
                tokens_returned=tokens_returned,
                tokens_saved=tokens_original - tokens_returned,
                truncated=False,
                compression_ratio=len(result_content) / len(content),
            )

    # Check for semantically similar file in cache
    if not cached and diff_mode and not force_full:
        embedding = cache._get_embedding(content)
        if embedding:
            similar_path = cache.find_similar(embedding, str(file_path))
            if similar_path:
                similar_entry = cache.get(similar_path)
                if similar_entry:
                    similar_content = cache.get_content(similar_entry)
                    diff_content = generate_diff(similar_content, content)
                    diff_tokens = cache._count_tokens(diff_content)

                    # Use semantic diff if it saves tokens
                    if diff_tokens < tokens_original * 0.7:
                        result_content = (
                            f"// Similar to cached: {similar_path}\n"
                            f"// Diff from similar file:\n{diff_content}"
                        )

                        # Cache this file too
                        cache.put(str(file_path), content, mtime, embedding)

                        tokens_returned = cache._count_tokens(result_content)
                        return ReadResult(
                            content=result_content,
                            from_cache=True,
                            is_diff=True,
                            tokens_original=tokens_original,
                            tokens_returned=tokens_returned,
                            tokens_saved=tokens_original - tokens_returned,
                            truncated=False,
                            compression_ratio=len(result_content) / len(content),
                            semantic_match=similar_path,
                        )

    # Full read (cache miss or diff not efficient)
    truncated = False
    final_content = content

    if len(content) > max_size:
        final_content = truncate_smart(content, max_size)
        truncated = True

    # Store in cache with embedding
    embedding = cache._get_embedding(content)
    cache.put(str(file_path), content, mtime, embedding)

    tokens_returned = cache._count_tokens(final_content)
    return ReadResult(
        content=final_content,
        from_cache=False,
        is_diff=False,
        tokens_original=tokens_original,
        tokens_returned=tokens_returned,
        tokens_saved=tokens_original - tokens_returned if truncated else 0,
        truncated=truncated,
        compression_ratio=len(final_content) / len(content) if content else 1.0,
    )


@mcp.tool()
def smart_read(
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
        result = _smart_read(
            cache=cache,
            path=path,
            max_size=max_size,
            diff_mode=diff_mode,
            force_full=force_full,
        )
        # Compact metadata footer
        meta = f"[cache:{result.from_cache} diff:{result.is_diff} saved:{result.tokens_saved}]"
        return f"{result.content}\n// {meta}"

    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def cache_stats(ctx: Context) -> str:
    """Get semantic cache statistics: files, tokens, compression, deduplication ratios."""
    cache: SemanticCache = ctx.lifespan_context["cache"]
    stats = cache.get_stats()
    return json.dumps(stats, indent=2)


@mcp.tool()
def cache_clear(ctx: Context) -> str:
    """Clear the semantic cache."""
    cache: SemanticCache = ctx.lifespan_context["cache"]
    count = cache.clear()
    return f"Cleared {count} cache entries"


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
