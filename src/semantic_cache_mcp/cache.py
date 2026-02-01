"""SemanticCache facade - orchestrates core algorithms and storage."""

from __future__ import annotations

import array
from pathlib import Path
from typing import TYPE_CHECKING

from .config import DB_PATH, MAX_CONTENT_SIZE
from .core import count_tokens, generate_diff, truncate_smart
from .storage import SQLiteStorage
from .types import CacheEntry, EmbeddingVector, ReadResult

if TYPE_CHECKING:
    from openai import OpenAI


class SemanticCache:
    """High-level cache interface with semantic similarity support.

    This facade coordinates:
    - Storage backend (SQLite with content-addressable chunks)
    - Embedding service (OpenAI-compatible API)
    - Caching strategies (diff, truncate, semantic match)
    """

    __slots__ = ("_storage", "_client")

    def __init__(
        self, db_path: Path = DB_PATH, client: OpenAI | None = None
    ) -> None:
        """Initialize cache.

        Args:
            db_path: Path to SQLite database
            client: OpenAI client for embeddings (optional)
        """
        self._storage = SQLiteStorage(db_path)
        self._client = client

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    def get_embedding(self, text: str) -> EmbeddingVector | None:
        """Get embedding vector for text.

        Args:
            text: Text to embed (first 512 chars used)

        Returns:
            Embedding as array.array or None if unavailable
        """
        if self._client is None:
            return None

        try:
            response = self._client.embeddings.create(
                input=[text[:512]],
                model="text-embedding",
            )
            return array.array("f", response.data[0].embedding)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Delegated operations
    # -------------------------------------------------------------------------

    def get(self, path: str) -> CacheEntry | None:
        """Get cached entry for path."""
        return self._storage.get(path)

    def put(
        self,
        path: str,
        content: str,
        mtime: float,
        embedding: EmbeddingVector | None = None,
    ) -> None:
        """Store file in cache."""
        self._storage.put(path, content, mtime, embedding)

    def get_content(self, entry: CacheEntry) -> str:
        """Get full content from cache entry."""
        return self._storage.get_content(entry)

    def record_access(self, path: str) -> None:
        """Record access for LRU-K tracking."""
        self._storage.record_access(path)

    def find_similar(
        self, embedding: EmbeddingVector, exclude_path: str | None = None
    ) -> str | None:
        """Find semantically similar cached file."""
        return self._storage.find_similar(embedding, exclude_path)

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        return self._storage.get_stats()

    def clear(self) -> int:
        """Clear all cache entries."""
        return self._storage.clear()


def smart_read(
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

    Args:
        cache: SemanticCache instance
        path: Path to file
        max_size: Maximum content size to return
        diff_mode: Enable diff-based responses
        force_full: Force full content even if cached

    Returns:
        ReadResult with content and metadata
    """
    file_path = Path(path).expanduser().resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    content = file_path.read_text()
    mtime = file_path.stat().st_mtime
    tokens_original = count_tokens(content)

    cached = cache.get(str(file_path))

    # Strategy 1 & 2: Cached file (unchanged or diff)
    if cached and diff_mode and not force_full:
        if cached.mtime >= mtime:
            # File unchanged
            cache.record_access(str(file_path))
            unchanged_msg = f"// File unchanged: {path} ({cached.tokens} tokens cached)"
            msg_tokens = count_tokens(unchanged_msg)

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

            # Small file - return full content
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
        diff_tokens = count_tokens(diff_content)

        if diff_tokens < tokens_original * 0.6:
            result_content = f"// Diff for {path} (changed since cache):\n{diff_content}"
            embedding = cache.get_embedding(content)
            cache.put(str(file_path), content, mtime, embedding)

            tokens_returned = count_tokens(result_content)
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

    # Strategy 3: Semantic similarity
    if not cached and diff_mode and not force_full:
        embedding = cache.get_embedding(content)
        if embedding:
            similar_path = cache.find_similar(embedding, str(file_path))
            if similar_path:
                similar_entry = cache.get(similar_path)
                if similar_entry:
                    similar_content = cache.get_content(similar_entry)
                    diff_content = generate_diff(similar_content, content)
                    diff_tokens = count_tokens(diff_content)

                    if diff_tokens < tokens_original * 0.7:
                        result_content = (
                            f"// Similar to cached: {similar_path}\n"
                            f"// Diff from similar file:\n{diff_content}"
                        )
                        cache.put(str(file_path), content, mtime, embedding)

                        tokens_returned = count_tokens(result_content)
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

    # Strategy 4 & 5: Full read (with optional truncation)
    truncated = False
    final_content = content

    if len(content) > max_size:
        final_content = truncate_smart(content, max_size)
        truncated = True

    embedding = cache.get_embedding(content)
    cache.put(str(file_path), content, mtime, embedding)

    tokens_returned = count_tokens(final_content)
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
