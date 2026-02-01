"""SQLite storage backend for content-addressable caching."""

from __future__ import annotations

import array
import json
import logging
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

from ..config import (
    ACCESS_HISTORY_SIZE,
    DB_PATH,
    LRU_K,
    MAX_CACHE_ENTRIES,
    NEAR_DUPLICATE_THRESHOLD,
    SIMILARITY_THRESHOLD,
)
from ..core import (
    compress_adaptive,
    hypercdc_chunks,
    top_k_similarities,
    count_tokens,
    decompress,
    hash_chunk,
    hash_content,
)
from ..types import CacheEntry, ChunkData, ChunkHash, EmbeddingVector


class SQLiteStorage:
    """SQLite-backed content-addressable storage with semantic similarity.

    Architecture:
    - Content-addressable chunks stored by hash (deduplication)
    - File entries reference chunk hashes (like git)
    - Embeddings enable semantic similarity search
    - LRU-K eviction for frequency-aware cache management
    """

    __slots__ = ("db_path",)

    def __init__(self, db_path: Path = DB_PATH) -> None:
        """Initialize storage.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    hash TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    size INTEGER NOT NULL,
                    ref_count INTEGER DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    chunk_hashes TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    tokens INTEGER NOT NULL,
                    embedding BLOB,
                    created_at REAL NOT NULL,
                    access_history TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_created ON files(created_at);
            """)

    # -------------------------------------------------------------------------
    # Chunk operations
    # -------------------------------------------------------------------------

    def store_chunks(self, content: bytes) -> list[ChunkHash]:
        """Store content as content-addressable chunks.

        Args:
            content: Raw bytes to store

        Returns:
            List of chunk hashes in order
        """
        chunks_data: list[tuple[str, bytes, int]] = []

        for chunk in hypercdc_chunks(content):
            chunk_hash = hash_chunk(chunk)
            compressed = compress_adaptive(chunk)
            chunks_data.append((chunk_hash, compressed, len(chunk)))

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO chunks (hash, data, size, ref_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(hash) DO UPDATE SET ref_count = ref_count + 1
                """,
                chunks_data,
            )

        logger.debug(f"Stored {len(chunks_data)} chunks")
        return [h for h, _, _ in chunks_data]

    def load_chunks(self, chunk_hashes: list[ChunkHash]) -> bytes:
        """Reassemble content from chunk hashes.

        Args:
            chunk_hashes: Ordered list of chunk hashes

        Returns:
            Reassembled content bytes
        """
        if not chunk_hashes:
            return b""

        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(chunk_hashes))
            rows = conn.execute(
                f"SELECT hash, data FROM chunks WHERE hash IN ({placeholders})",
                chunk_hashes,
            ).fetchall()

        chunk_data = {h: data for h, data in rows}

        return b"".join(
            decompress(chunk_data[h]) for h in chunk_hashes if h in chunk_data
        )

    def release_chunks(self, chunk_hashes: list[ChunkHash]) -> None:
        """Decrement ref_count for chunks, delete if zero.

        Args:
            chunk_hashes: Chunks to release
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "UPDATE chunks SET ref_count = ref_count - 1 WHERE hash = ?",
                ((h,) for h in chunk_hashes),
            )
            conn.execute("DELETE FROM chunks WHERE ref_count <= 0")

    # -------------------------------------------------------------------------
    # File operations
    # -------------------------------------------------------------------------

    def get(self, path: str) -> CacheEntry | None:
        """Get cached file entry.

        Args:
            path: Absolute file path

        Returns:
            CacheEntry if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT path, content_hash, chunk_hashes, mtime, tokens, embedding, "
                "created_at, access_history FROM files WHERE path = ?",
                (path,),
            ).fetchone()

        if not row:
            return None

        embedding_data = row[5]
        embedding = (
            array.array("f", json.loads(embedding_data)) if embedding_data else None
        )

        return CacheEntry(
            path=row[0],
            content_hash=row[1],
            chunks=json.loads(row[2]),
            mtime=row[3],
            tokens=row[4],
            embedding=embedding,
            created_at=row[6],
            access_history=json.loads(row[7]),
        )

    def put(
        self,
        path: str,
        content: str,
        mtime: float,
        embedding: EmbeddingVector | None = None,
    ) -> None:
        """Store file in cache.

        Args:
            path: Absolute file path
            content: File content
            mtime: Modification time
            embedding: Optional embedding vector
        """
        content_hash = hash_content(content)
        content_bytes = content.encode()

        # Release old chunks if updating
        old_entry = self.get(path)
        if old_entry:
            self.release_chunks(old_entry.chunks)

        chunk_hashes = self.store_chunks(content_bytes)
        tokens = count_tokens(content)
        embedding_json = json.dumps(list(embedding)) if embedding else None

        logger.debug(f"Stored {len(chunk_hashes)} chunks for {path}")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO files
                (path, content_hash, chunk_hashes, mtime, tokens, embedding, created_at, access_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path,
                    content_hash,
                    json.dumps(chunk_hashes),
                    mtime,
                    tokens,
                    embedding_json,
                    time.time(),
                    json.dumps([time.time()]),
                ),
            )

        self._evict_if_needed()

    def get_content(self, entry: CacheEntry) -> str:
        """Get full content for a cache entry.

        Args:
            entry: Cache entry with chunk references

        Returns:
            Decoded file content
        """
        return self.load_chunks(entry.chunks).decode()

    def record_access(self, path: str) -> None:
        """Record access for LRU-K tracking.

        Args:
            path: File path accessed
        """
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT access_history FROM files WHERE path = ?", (path,)
            ).fetchone()

            if row:
                history = json.loads(row[0])
                history.append(time.time())
                conn.execute(
                    "UPDATE files SET access_history = ? WHERE path = ?",
                    (json.dumps(history[-ACCESS_HISTORY_SIZE:]), path),
                )

    # -------------------------------------------------------------------------
    # Similarity search
    # -------------------------------------------------------------------------

    def find_similar(
        self, embedding: EmbeddingVector, exclude_path: str | None = None
    ) -> str | None:
        """Find semantically similar cached file using batch similarity.

        Uses SIMD-optimized batch operations for 8-14x faster search
        compared to per-vector cosine similarity.

        Args:
            embedding: Query embedding vector
            exclude_path: Path to exclude from results

        Returns:
            Path of similar file or None
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT path, embedding FROM files WHERE embedding IS NOT NULL"
            params: tuple = ()
            if exclude_path:
                query += " AND path != ?"
                params = (exclude_path,)
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return None

        # Parse embeddings and collect paths
        paths = []
        vectors = []
        for path, emb_json in rows:
            paths.append(path)
            vectors.append(json.loads(emb_json))

        # Use top_k_similarities for efficient single-best retrieval
        top_results = top_k_similarities(embedding, vectors, k=1)

        if not top_results:
            return None

        best_idx, best_sim = top_results[0]

        if best_sim > SIMILARITY_THRESHOLD:
            return paths[best_idx]

        return None

    # -------------------------------------------------------------------------
    # Eviction
    # -------------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict entries using LRU-K policy if over limit."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

            if count <= MAX_CACHE_ENTRIES:
                return

            rows = conn.execute(
                "SELECT path, chunk_hashes, access_history FROM files"
            ).fetchall()

            entries_with_score: list[tuple[float, str, list[str]]] = []
            for path, chunks_json, history_json in rows:
                history = json.loads(history_json)
                score = history[-LRU_K] if len(history) >= LRU_K else history[0]
                entries_with_score.append((score, path, json.loads(chunks_json)))

            entries_with_score.sort()

            evict_count = max(1, count // 10)
            for _, path, chunk_hashes in entries_with_score[:evict_count]:
                conn.execute("DELETE FROM files WHERE path = ?", (path,))
                # Release chunks inline to avoid nested connection
                conn.executemany(
                    "UPDATE chunks SET ref_count = ref_count - 1 WHERE hash = ?",
                    ((h,) for h in chunk_hashes),
                )

            # Clean up chunks with zero references
            conn.execute("DELETE FROM chunks WHERE ref_count <= 0")
            logger.info(f"Cache eviction: removed {evict_count} entries")

    # -------------------------------------------------------------------------
    # Statistics and management
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            file_stats = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM files"
            ).fetchone()

            chunk_stats = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size), 0), COALESCE(SUM(LENGTH(data)), 0) FROM chunks"
            ).fetchone()

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        original_size = chunk_stats[1]
        compressed_size = chunk_stats[2]

        return {
            "files_cached": file_stats[0],
            "total_tokens_cached": file_stats[1],
            "unique_chunks": chunk_stats[0],
            "original_bytes": original_size,
            "compressed_bytes": compressed_size,
            "compression_ratio": round(compressed_size / original_size, 3)
            if original_size > 0
            else 0,
            "dedup_ratio": round(original_size / db_size, 2) if db_size > 0 else 1,
            "db_size_mb": round(db_size / 1024 / 1024, 2),
        }

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            conn.executescript("DELETE FROM files; DELETE FROM chunks;")
        return count
