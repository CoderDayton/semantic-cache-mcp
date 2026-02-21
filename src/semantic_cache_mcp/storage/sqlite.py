"""SQLite storage backend for content-addressable caching."""

from __future__ import annotations

import atexit
import json
import logging
import queue
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from ..config import (
    ACCESS_HISTORY_SIZE,
    DB_PATH,
    LRU_K,
    MAX_CACHE_ENTRIES,
    SIMILARITY_THRESHOLD,
)
from ..core import (
    compress_adaptive,
    count_tokens,
    decompress,
    dequantize_embedding,
    hash_chunk,
    hash_content,
    hypercdc_chunks,
    quantize_embedding,
    top_k_from_quantized,
)
from ..types import CacheEntry, ChunkHash, EmbeddingVector

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Queue-based SQLite connection pool.

    Maintains a pool of reusable connections with explicit checkout/checkin.
    Thread-safe using queue.Queue for connection management.

    Benefits:
    - Avoids connection creation overhead
    - Bounded pool size prevents resource exhaustion
    - Thread-safe by design (queue handles locking)
    - Connections properly closed on shutdown
    - No thread affinity (any thread can use any connection)
    """

    __slots__ = ("db_path", "_pool", "_max_size", "_created", "_closed")

    def __init__(self, db_path: Path, max_size: int = 5) -> None:
        """Initialize connection pool.

        Args:
            db_path: Path to SQLite database
            max_size: Maximum number of connections in pool
        """
        self.db_path = db_path
        self._max_size = max_size
        self._pool: queue.Queue[sqlite3.Connection] = queue.Queue(maxsize=max_size)
        self._created = 0
        self._closed = False
        atexit.register(self.close_all)

    def _create_connection(self) -> sqlite3.Connection:
        """Create and configure a new connection."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False,  # Allow connection to be used by any thread
        )
        conn.executescript("""
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA cache_size = -64000;
            PRAGMA temp_store = MEMORY;
            PRAGMA mmap_size = 268435456;
            PRAGMA busy_timeout = 5000;
        """)
        self._created += 1
        logger.debug(f"Created connection {self._created}/{self._max_size}")
        return conn

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get connection from pool (creates new if pool not full).

        Yields:
            SQLite connection from pool
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        # Try to get existing connection, or create new if pool not full
        conn = None
        try:
            conn = self._pool.get_nowait()
        except queue.Empty:
            if self._created < self._max_size:
                conn = self._create_connection()
            else:
                # Pool full, wait for available connection
                conn = self._pool.get(timeout=10.0)

        try:
            yield conn
            # Commit on success (matches sqlite3.connect context manager)
            conn.commit()
        except sqlite3.Error:
            # Rollback on database error
            try:
                conn.rollback()
            except sqlite3.Error:
                logger.debug("Rollback failed during error recovery")
            raise
        finally:
            # Return connection to pool if not closed
            if conn and not self._closed:
                try:
                    self._pool.put_nowait(conn)
                except queue.Full:
                    # Pool full (shouldn't happen), close connection
                    try:
                        conn.close()
                    except sqlite3.Error:
                        logger.debug("Failed to close overflow connection")

    def close_all(self) -> None:
        """Close all connections in pool."""
        if self._closed:
            return

        self._closed = True
        closed_count = 0

        # Drain queue and close all connections
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
                closed_count += 1
            except (queue.Empty, Exception):
                break

        logger.debug(f"Closed {closed_count} pooled connections")


class SQLiteStorage:
    """SQLite-backed content-addressable storage with semantic similarity.

    Architecture:
    - Content-addressable chunks stored by hash (deduplication)
    - File entries reference chunk hashes (like git)
    - Embeddings enable semantic similarity search
    - LRU-K eviction for frequency-aware cache management
    - Queue-based connection pooling for concurrent access
    """

    __slots__ = ("db_path", "_pool")

    def __init__(self, db_path: Path = DB_PATH, pool_size: int = 5) -> None:
        """Initialize storage with connection pool.

        Args:
            db_path: Path to SQLite database file
            pool_size: Maximum connections in pool (default: 5)
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._pool = ConnectionPool(db_path, max_size=pool_size)
        self._init_schema()

    def __del__(self) -> None:
        """Clean up connection pool on deletion."""
        if hasattr(self, "_pool"):
            self._pool.close_all()

    def _init_schema(self) -> None:
        """Initialize database schema with performance indexes."""
        with self._pool.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    hash TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    size INTEGER NOT NULL,
                    ref_count INTEGER DEFAULT 1
                ) WITHOUT ROWID;

                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    chunk_hashes TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    tokens INTEGER NOT NULL,
                    embedding BLOB,
                    created_at REAL NOT NULL,
                    access_history TEXT NOT NULL
                ) WITHOUT ROWID;

                CREATE INDEX IF NOT EXISTS idx_created ON files(created_at);
                CREATE INDEX IF NOT EXISTS idx_embedding ON files(embedding)
                    WHERE embedding IS NOT NULL;

                CREATE TABLE IF NOT EXISTS lsh_index (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    data BLOB NOT NULL,
                    blob_count INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                ) WITHOUT ROWID;
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

        with self._pool.get_connection() as conn:
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

        with self._pool.get_connection() as conn:
            placeholders = ",".join("?" * len(chunk_hashes))
            rows = conn.execute(
                f"SELECT hash, data FROM chunks WHERE hash IN ({placeholders})",  # nosec B608 — placeholders is "?,?,?" built from count, data is parameterized
                chunk_hashes,
            ).fetchall()

        chunk_data = dict(rows)

        return b"".join(decompress(chunk_data[h]) for h in chunk_hashes if h in chunk_data)

    def release_chunks(self, chunk_hashes: list[ChunkHash]) -> None:
        """Decrement ref_count for chunks, delete if zero.

        Args:
            chunk_hashes: Chunks to release
        """
        with self._pool.get_connection() as conn:
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
        with self._pool.get_connection() as conn:
            row = conn.execute(
                "SELECT path, content_hash, chunk_hashes, mtime, tokens, embedding, "
                "created_at, access_history FROM files WHERE path = ?",
                (path,),
            ).fetchone()

        if not row:
            return None

        embedding_data = row[5]
        embedding = dequantize_embedding(embedding_data) if embedding_data else None

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

        Single-transaction put: chunks, file entry, and eviction all happen
        in one commit to avoid 3-5 separate fsync calls.

        Args:
            path: Absolute file path
            content: File content
            mtime: Modification time
            embedding: Optional embedding vector
        """
        content_hash = hash_content(content)
        content_bytes = content.encode()
        tokens = count_tokens(content)
        embedding_blob = quantize_embedding(embedding) if embedding else None

        # Prepare chunks outside transaction (CPU-bound, no I/O)
        chunks_data: list[tuple[str, bytes, int]] = []
        for chunk in hypercdc_chunks(content_bytes):
            chunk_hash = hash_chunk(chunk)
            compressed = compress_adaptive(chunk)
            chunks_data.append((chunk_hash, compressed, len(chunk)))

        chunk_hashes = [h for h, _, _ in chunks_data]
        now = time.time()

        with self._pool.get_connection() as conn:
            # Release old chunks if updating
            old_row = conn.execute(
                "SELECT chunk_hashes FROM files WHERE path = ?", (path,)
            ).fetchone()
            if old_row:
                old_hashes = json.loads(old_row[0])
                conn.executemany(
                    "UPDATE chunks SET ref_count = ref_count - 1 WHERE hash = ?",
                    ((h,) for h in old_hashes),
                )
                conn.execute("DELETE FROM chunks WHERE ref_count <= 0")

            # Store chunks
            conn.executemany(
                """
                INSERT INTO chunks (hash, data, size, ref_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(hash) DO UPDATE SET ref_count = ref_count + 1
                """,
                chunks_data,
            )

            # Store file entry
            conn.execute(
                """
                INSERT OR REPLACE INTO files
                (path, content_hash, chunk_hashes, mtime, tokens, embedding,
                    created_at, access_history)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    path,
                    content_hash,
                    json.dumps(chunk_hashes),
                    mtime,
                    tokens,
                    embedding_blob,
                    now,
                    json.dumps([now]),
                ),
            )

            # Inline eviction check (avoids separate transaction)
            count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            if count > MAX_CACHE_ENTRIES:
                evict_count = max(1, count // 10)
                rows = conn.execute(
                    "SELECT path, chunk_hashes, access_history FROM files"
                ).fetchall()

                entries_with_score: list[tuple[float, str, list[str]]] = []
                for p, chunks_json, history_json in rows:
                    history = json.loads(history_json)
                    score = history[-LRU_K] if len(history) >= LRU_K else history[0]
                    entries_with_score.append((score, p, json.loads(chunks_json)))

                entries_with_score.sort()
                evict_paths = [p for _, p, _ in entries_with_score[:evict_count]]
                evict_chunks = [c for _, _, cs in entries_with_score[:evict_count] for c in cs]

                placeholders = ",".join("?" * len(evict_paths))
                conn.execute(
                    f"DELETE FROM files WHERE path IN ({placeholders})",
                    evict_paths,  # nosec B608 — placeholders is "?,?,?" built from count, data is parameterized
                )
                conn.executemany(
                    "UPDATE chunks SET ref_count = ref_count - 1 WHERE hash = ?",
                    ((h,) for h in evict_chunks),
                )
                conn.execute("DELETE FROM chunks WHERE ref_count <= 0")
                logger.info(f"Cache eviction: removed {evict_count} entries")

        logger.debug(f"Stored {len(chunks_data)} chunks for {path}")

    def get_content(self, entry: CacheEntry) -> str:
        """Get full content for a cache entry.

        Args:
            entry: Cache entry with chunk references

        Returns:
            Decoded file content

        Raises:
            ValueError: If content cannot be decoded as UTF-8
        """
        data = self.load_chunks(entry.chunks)
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.error(f"Failed to decode cached content: {e}")
            # Try with replacement characters for partial recovery
            return data.decode("utf-8", errors="replace")

    def record_access(self, path: str) -> None:
        """Record access for LRU-K tracking.

        Args:
            path: File path accessed
        """
        with self._pool.get_connection() as conn:
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
        """Find semantically similar cached file using pre-quantized vectors.

        Uses pre-quantized binary storage for 3x faster search than
        quantizing at query time. 22x smaller storage than JSON.

        Args:
            embedding: Query embedding vector
            exclude_path: Path to exclude from results

        Returns:
            Path of similar file or None
        """
        with self._pool.get_connection() as conn:
            query = """
                SELECT path, embedding FROM files
                WHERE embedding IS NOT NULL
            """
            params: list = []
            if exclude_path:
                query += " AND path != ?"
                params.append(exclude_path)

            query += " ORDER BY created_at DESC"
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return None

        paths = [row[0] for row in rows]
        blobs = [row[1] for row in rows]

        top_results = top_k_from_quantized(embedding, blobs, k=1)

        if not top_results:
            return None

        best_idx, best_sim = top_results[0]

        if best_sim > SIMILARITY_THRESHOLD:
            return paths[best_idx]

        return None

    # -------------------------------------------------------------------------
    # LSH index persistence
    # -------------------------------------------------------------------------

    def get_lsh_index(self) -> tuple[bytes, int] | None:
        """Load persisted LSH index blob and expected blob_count.

        Returns:
            (data, blob_count) if present, None otherwise
        """
        with self._pool.get_connection() as conn:
            row = conn.execute("SELECT data, blob_count FROM lsh_index WHERE id = 1").fetchone()
        return (row[0], row[1]) if row else None

    def set_lsh_index(self, data: bytes, blob_count: int) -> None:
        """Persist LSH index blob.

        Args:
            data: Serialized LSHIndex bytes
            blob_count: Number of embeddings indexed (used to detect staleness)
        """
        with self._pool.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO lsh_index (id, data, blob_count, updated_at)
                VALUES (1, ?, ?, ?)
                """,
                (data, blob_count, time.time()),
            )

    def clear_lsh_index(self) -> None:
        """Delete persisted LSH index (called on any cache mutation)."""
        with self._pool.get_connection() as conn:
            conn.execute("DELETE FROM lsh_index WHERE id = 1")

    # -------------------------------------------------------------------------
    # Eviction
    # -------------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict entries using LRU-K policy if over limit.

        Optimizations:
        - Fast count check with early return
        - Batch deletes and updates
        - Single pass for chunk cleanup
        """
        with self._pool.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

            if count <= MAX_CACHE_ENTRIES:
                return

            evict_count = max(1, count // 10)

            # Get candidates for eviction
            rows = conn.execute("SELECT path, chunk_hashes, access_history FROM files").fetchall()

            entries_with_score: list[tuple[float, str, list[str]]] = []
            for path, chunks_json, history_json in rows:
                history = json.loads(history_json)
                score = history[-LRU_K] if len(history) >= LRU_K else history[0]
                entries_with_score.append((score, path, json.loads(chunks_json)))

            entries_with_score.sort()

            # Collect paths and chunks for batch operations
            evict_paths = [path for _, path, _ in entries_with_score[:evict_count]]
            evict_chunks = [
                chunk for _, _, chunks in entries_with_score[:evict_count] for chunk in chunks
            ]

            # Batch delete files
            placeholders = ",".join("?" * len(evict_paths))
            conn.execute(f"DELETE FROM files WHERE path IN ({placeholders})", evict_paths)  # nosec B608 — placeholders is "?,?,?" built from count, data is parameterized

            # Batch update chunk ref counts
            conn.executemany(
                "UPDATE chunks SET ref_count = ref_count - 1 WHERE hash = ?",
                ((h,) for h in evict_chunks),
            )

            # Clean up unreferenced chunks
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
        with self._pool.get_connection() as conn:
            file_stats = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(tokens), 0) FROM files"
            ).fetchone()

            chunk_stats = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(size), 0), "
                "COALESCE(SUM(LENGTH(data)), 0) FROM chunks"
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
        with self._pool.get_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            conn.executescript("DELETE FROM files; DELETE FROM chunks; DELETE FROM lsh_index;")
        return count
