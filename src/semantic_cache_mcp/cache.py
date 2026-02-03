"""SemanticCache facade - orchestrates core algorithms and storage."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import DB_PATH, MAX_CONTENT_SIZE
from .core import (
    count_tokens,
    diff_stats,
    generate_diff,
    get_optimal_chunker,
    summarize_semantic,
    truncate_semantic,
)
from .core.embeddings import embed
from .storage import SQLiteStorage
from .types import CacheEntry, EmbeddingVector, ReadResult

logger = logging.getLogger(__name__)


class SemanticCache:
    """High-level cache interface with semantic similarity support.

    This facade coordinates:
    - Storage backend (SQLite with content-addressable chunks)
    - Local embedding generation (FastEmbed)
    - Caching strategies (diff, truncate, semantic match)
    """

    __slots__ = ("_storage",)

    def __init__(self, db_path: Path = DB_PATH) -> None:
        """Initialize cache.

        Args:
            db_path: Path to SQLite database
        """
        self._storage = SQLiteStorage(db_path)

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    def get_embedding(self, text: str) -> EmbeddingVector | None:
        """Get embedding vector for text using local FastEmbed model.

        Args:
            text: Text to embed

        Returns:
            Embedding as array.array or None if unavailable
        """
        try:
            result = embed(text)
            if result:
                logger.debug(f"Embedding generated for {text[:50]}...")
            return result
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return None

    # -------------------------------------------------------------------------
    # Delegated operations
    # -------------------------------------------------------------------------

    def get(self, path: str) -> CacheEntry | None:
        """Get cached entry for path."""
        entry = self._storage.get(path)
        if entry:
            logger.debug(f"Cache hit: {path}")
        return entry

    def put(
        self,
        path: str,
        content: str,
        mtime: float,
        embedding: EmbeddingVector | None = None,
    ) -> None:
        """Store file in cache."""
        tokens = count_tokens(content)
        content_bytes = content.encode()

        # Use optimal chunker (SIMD if available, otherwise Gear hash)
        chunker = get_optimal_chunker(prefer_simd=True)
        chunks = sum(1 for _ in chunker(content_bytes))

        self._storage.put(path, content, mtime, embedding)
        logger.info(f"Cached file: {path} ({tokens} tokens, {chunks} chunks)")

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

    if not file_path.is_file():
        raise ValueError(f"Not a regular file: {path}")

    # Log symlink resolution for debugging
    original = Path(path).expanduser()
    if original.is_symlink():
        logger.debug(f"Following symlink: {path} -> {file_path}")

    # Check for binary file by reading first 8KB and looking for null bytes
    try:
        sample = file_path.read_bytes()[:8192]
        if b"\x00" in sample:
            raise ValueError(
                f"Binary file not supported: {path}. Semantic cache only handles text files."
            )
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with error replacement for files with mixed encoding
        content = file_path.read_text(encoding="utf-8", errors="replace")
        logger.warning(f"File {path} contains non-UTF-8 characters, using replacement")

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

        # File changed - generate diff with stats
        old_content = cache.get_content(cached)
        diff_content = generate_diff(old_content, content)
        stats = diff_stats(old_content, content)
        diff_tokens = count_tokens(diff_content)

        if diff_tokens < tokens_original * 0.6:
            stats_msg = (
                f"// Stats: +{stats['insertions']} -{stats['deletions']} "
                f"~{stats['modifications']} lines, "
                f"{stats['compression_ratio']:.1%} size\n"
            )
            result_content = f"// Diff for {path} (changed since cache):\n{stats_msg}{diff_content}"
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
                    stats = diff_stats(similar_content, content)
                    diff_tokens = count_tokens(diff_content)

                    if diff_tokens < tokens_original * 0.7:
                        stats_msg = (
                            f"// Stats: +{stats['insertions']} -{stats['deletions']} "
                            f"~{stats['modifications']} lines, "
                            f"{stats['compression_ratio']:.1%} size\n"
                        )
                        result_content = (
                            f"// Similar to cached: {similar_path}\n"
                            f"{stats_msg}"
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

    # Strategy 4 & 5: Full read (with optional semantic summarization)
    truncated = False
    final_content = content

    if len(content) > max_size:
        # Use semantic summarization to preserve important content
        # Falls back to simple truncation for very small limits
        try:
            # Convert EmbeddingVector to NDArray for summarization
            def embed_fn(text: str):
                emb = cache.get_embedding(text)
                if emb is None:
                    return None
                # Convert array.array or list to numpy array
                return np.asarray(emb, dtype=np.float32)

            final_content = summarize_semantic(content, max_size, embed_fn=embed_fn)
            truncated = True
        except Exception as e:
            logger.warning(f"Semantic summarization failed: {e}, using fallback truncation")
            final_content = truncate_semantic(content, max_size)
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
