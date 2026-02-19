"""SemanticCache class - high-level cache interface with semantic similarity support."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import DB_PATH
from ..core import count_tokens, get_optimal_chunker
from ..storage import SQLiteStorage
from ..types import CacheEntry, EmbeddingVector

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
            # Late import from package so patch("semantic_cache_mcp.cache.embed") in tests works.
            # By call time the cache package is fully initialized in sys.modules.
            from . import embed as _embed  # noqa: PLC0415

            result = _embed(text)
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

    def get_stats(self) -> dict[str, int | float | str | bool]:
        """Get cache statistics including memory usage."""
        stats: dict[str, int | float | str | bool] = {**self._storage.get_stats()}

        # Add process memory stats
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        stats["process_rss_mb"] = round(int(line.split()[1]) / 1024, 1)
                        break
        except OSError:
            pass

        # Add merge cache stats
        from ..core.tokenizer import _tokenizer

        if _tokenizer is not None:
            stats["merge_cache_entries"] = len(_tokenizer._merge_cache)
            stats["merge_cache_maxsize"] = _tokenizer._merge_cache_maxsize

        # Add embedding model readiness
        from ..core.embeddings import _execution_provider, _model_ready

        stats["embedding_ready"] = _model_ready
        stats["embedding_provider"] = _execution_provider

        return stats

    def clear(self) -> int:
        """Clear all cache entries."""
        return self._storage.clear()
