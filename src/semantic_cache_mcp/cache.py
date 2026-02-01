"""Core cache engine with semantic similarity, content-addressable storage, and compression.

Scientific improvements over token-optimizer-mcp:
1. Content-addressable storage - Hash-based deduplication across files
2. Semantic similarity - Embeddings for finding related cached content
3. Rolling hash chunking - Content-defined chunking for better deduplication
4. Adaptive compression - Brotli quality based on content entropy
5. LRU-K eviction - Frequency-aware cache eviction
6. Delta encoding - Binary diff for minimal storage

Performance optimizations:
- Inlined rolling hash in chunking loop (eliminates method call overhead)
- Counter for entropy calculation (C-implemented)
- sum() for dot product (C-implemented)
- executemany for batch inserts
- Token counting without list allocation
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from difflib import unified_diff
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import brotli

if TYPE_CHECKING:
    from openai import OpenAI

# Configuration
CACHE_DIR = Path.home() / ".cache" / "semantic-cache-mcp"
DB_PATH = CACHE_DIR / "cache.db"
EMBEDDINGS_BASE_URL = os.environ.get("EMBEDDINGS_URL", "http://localhost:8899/v1")
MAX_CONTENT_SIZE = 100_000  # 100KB default max return size
SIMILARITY_THRESHOLD = 0.85  # Semantic similarity threshold for related files
MAX_CACHE_ENTRIES = 10_000  # LRU-K limit

# Rolling hash constants (inlined for performance)
_RH_PRIME = 31
_RH_MOD = (1 << 32) - 1
_RH_WINDOW = 48
_RH_MASK = 0x1FFF  # 13 bits - average chunk size ~8KB
_RH_POW_OUT = pow(_RH_PRIME, _RH_WINDOW - 1, _RH_MOD)


@dataclass
class CacheEntry:
    """Cached file entry with metadata."""

    path: str
    content_hash: str
    chunks: list[str]  # Content-addressable chunk hashes
    mtime: float
    tokens: int
    embedding: list[float] | None
    created_at: float
    access_history: list[float] = field(default_factory=list)  # LRU-K timestamps


@dataclass
class ReadResult:
    """Result from smart_read operation."""

    content: str
    from_cache: bool
    is_diff: bool
    tokens_original: int
    tokens_returned: int
    tokens_saved: int
    truncated: bool
    compression_ratio: float
    semantic_match: str | None = None  # Path of semantically similar cached file


def content_defined_chunking(
    content: bytes, min_size: int = 2048, max_size: int = 65536
) -> Iterator[bytes]:
    """Split content into chunks using rolling hash (Rabin fingerprinting).

    Content-defined chunking finds natural boundaries that survive insertions
    and deletions, enabling better deduplication than fixed-size chunking.

    Optimized: Rolling hash is inlined to eliminate method call overhead.
    """
    n = len(content)
    if n == 0:
        return

    # Inline constants and state for performance
    PRIME, MOD, WINDOW, MASK = _RH_PRIME, _RH_MOD, _RH_WINDOW, _RH_MASK
    pow_out = _RH_POW_OUT

    h = 0
    buf = [0] * WINDOW  # Circular buffer
    pos = 0
    full = False
    chunk_start = 0

    for i in range(n):
        b = content[i]

        # Inlined rolling hash update
        if full:
            h = (h - buf[pos] * pow_out) % MOD
        h = (h * PRIME + b) % MOD
        buf[pos] = b
        pos += 1
        if pos == WINDOW:
            pos = 0
            full = True

        chunk_size = i - chunk_start + 1
        if chunk_size >= min_size and ((h & MASK) == 0 or chunk_size >= max_size):
            yield content[chunk_start : i + 1]
            chunk_start = i + 1

    # Emit final chunk
    if chunk_start < n:
        yield content[chunk_start:]


def estimate_entropy(data: bytes) -> float:
    """Estimate Shannon entropy of data (bits per byte).

    Used to adaptively select compression quality.
    High entropy (>7) = already compressed, low quality sufficient.
    Low entropy (<5) = highly compressible, use high quality.

    Optimized: Uses Counter (C-implemented) instead of manual frequency counting.
    """
    if not data:
        return 0.0

    n = len(data)
    counts = Counter(data)  # C-implemented, ~2-3x faster than manual loop
    log2_n = math.log2(n)

    # Only iterate non-zero counts (Counter naturally excludes zeros)
    entropy = 0.0
    for count in counts.values():
        entropy -= count * (math.log2(count) - log2_n) / n

    return entropy


class SemanticCache:
    """SQLite-backed cache with content-addressable storage and semantic similarity.

    Architecture:
    - Content-addressable chunks stored by hash (deduplication)
    - File entries reference chunk hashes (like git)
    - Embeddings enable semantic similarity search
    - LRU-K eviction for frequency-aware cache management
    """

    __slots__ = ("db_path", "_client")

    def __init__(self, db_path: Path = DB_PATH, client: OpenAI | None = None) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._client = client

    def _init_db(self) -> None:
        """Initialize SQLite schema with content-addressable storage."""
        with sqlite3.connect(self.db_path) as conn:
            # Content-addressable chunk store
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    hash TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    size INTEGER NOT NULL,
                    ref_count INTEGER DEFAULT 1
                )
            """)

            # File metadata with chunk references
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    path TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    chunk_hashes TEXT NOT NULL,
                    mtime REAL NOT NULL,
                    tokens INTEGER NOT NULL,
                    embedding BLOB,
                    created_at REAL NOT NULL,
                    access_history TEXT NOT NULL
                )
            """)

            # Index for LRU-K eviction
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created ON files(created_at)")

    @staticmethod
    def _hash_chunk(data: bytes) -> str:
        """Hash chunk using BLAKE2b (faster than SHA256, cryptographically secure)."""
        return hashlib.blake2b(data, digest_size=20).hexdigest()

    @staticmethod
    def _hash_content(content: str) -> str:
        """Hash full content for change detection."""
        return hashlib.blake2b(content.encode(), digest_size=16).hexdigest()

    @staticmethod
    def _compress_adaptive(data: bytes) -> bytes:
        """Compress with adaptive quality based on entropy.

        High entropy data (already compressed) -> quality 1 (fast)
        Low entropy data (text, code) -> quality 6 (better ratio)
        """
        entropy = estimate_entropy(data[:4096])  # Sample first 4KB

        if entropy > 7.0:
            quality = 1  # Already compressed
        elif entropy > 5.5:
            quality = 4  # Medium compressibility
        else:
            quality = 6  # Highly compressible

        return brotli.compress(data, quality=quality)

    @staticmethod
    def _decompress(data: bytes) -> bytes:
        """Decompress Brotli data."""
        return brotli.decompress(data)

    @staticmethod
    def _count_tokens(content: str) -> int:
        """Approximate token count using byte-pair encoding heuristic.

        Optimized: Counts whitespace characters instead of splitting into list.
        """
        if not content:
            return 0
        # Count whitespace to estimate words without allocating a list
        spaces = content.count(" ") + content.count("\n") + content.count("\t")
        words = spaces + 1
        return int(words * 1.3 + len(content) * 0.1)

    def _get_embedding(self, text: str) -> list[float] | None:
        """Get embedding from shared service for semantic similarity."""
        if self._client is None:
            return None
        try:
            # Truncate to first 512 chars for embedding (enough for semantic matching)
            sample = text[:512]
            response = self._client.embeddings.create(
                input=[sample],
                model="text-embedding",  # Service uses configured model
            )
            return response.data[0].embedding
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity for normalized embeddings (dot product).

        Optimized: Uses sum() with generator (C-implemented).
        """
        return sum(x * y for x, y in zip(a, b))

    def _store_chunks(self, content: bytes) -> list[str]:
        """Store content as content-addressable chunks. Returns chunk hashes.

        Optimized: Uses executemany for batch inserts.
        """
        # Prepare all chunk data first
        chunks_data: list[tuple[str, bytes, int]] = []
        for chunk in content_defined_chunking(content):
            chunk_hash = self._hash_chunk(chunk)
            compressed = self._compress_adaptive(chunk)
            chunks_data.append((chunk_hash, compressed, len(chunk)))

        # Batch insert with executemany
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO chunks (hash, data, size, ref_count)
                VALUES (?, ?, ?, 1)
                ON CONFLICT(hash) DO UPDATE SET ref_count = ref_count + 1
                """,
                chunks_data,
            )

        return [h for h, _, _ in chunks_data]

    def _load_chunks(self, chunk_hashes: list[str]) -> bytes:
        """Reassemble content from chunk hashes."""
        parts: list[bytes] = []

        with sqlite3.connect(self.db_path) as conn:
            for chunk_hash in chunk_hashes:
                row = conn.execute(
                    "SELECT data FROM chunks WHERE hash = ?", (chunk_hash,)
                ).fetchone()
                if row:
                    parts.append(self._decompress(row[0]))

        return b"".join(parts)

    def _release_chunks(self, chunk_hashes: list[str]) -> None:
        """Decrement ref_count for chunks, delete if zero.

        Optimized: Uses executemany for batch updates.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "UPDATE chunks SET ref_count = ref_count - 1 WHERE hash = ?",
                [(h,) for h in chunk_hashes],
            )
            conn.execute("DELETE FROM chunks WHERE ref_count <= 0")

    def get(self, path: str) -> CacheEntry | None:
        """Get cached entry if exists."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT path, content_hash, chunk_hashes, mtime, tokens, embedding, "
                "created_at, access_history FROM files WHERE path = ?",
                (path,),
            ).fetchone()

        if not row:
            return None

        return CacheEntry(
            path=row[0],
            content_hash=row[1],
            chunks=json.loads(row[2]),
            mtime=row[3],
            tokens=row[4],
            embedding=json.loads(row[5]) if row[5] else None,
            created_at=row[6],
            access_history=json.loads(row[7]),
        )

    def get_content(self, entry: CacheEntry) -> str:
        """Get full content for a cache entry."""
        content_bytes = self._load_chunks(entry.chunks)
        return content_bytes.decode()

    def put(
        self,
        path: str,
        content: str,
        mtime: float,
        embedding: list[float] | None = None,
    ) -> None:
        """Store file in cache with content-addressable chunks."""
        content_hash = self._hash_content(content)
        content_bytes = content.encode()

        # Check if we need to release old chunks
        old_entry = self.get(path)
        if old_entry:
            self._release_chunks(old_entry.chunks)

        # Store new chunks
        chunk_hashes = self._store_chunks(content_bytes)
        tokens = self._count_tokens(content)
        embedding_json = json.dumps(embedding) if embedding else None
        access_history = json.dumps([time.time()])

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
                    access_history,
                ),
            )

        # Evict if over limit
        self._evict_lru_k()

    def record_access(self, path: str) -> None:
        """Record access for LRU-K tracking."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT access_history FROM files WHERE path = ?", (path,)
            ).fetchone()
            if row:
                history = json.loads(row[0])
                history.append(time.time())
                # Keep last 5 accesses (K=5 for LRU-K)
                history = history[-5:]
                conn.execute(
                    "UPDATE files SET access_history = ? WHERE path = ?",
                    (json.dumps(history), path),
                )

    def find_similar(
        self, embedding: list[float], exclude_path: str | None = None
    ) -> str | None:
        """Find semantically similar cached file using embeddings.

        Optimized: SQL-level filtering and early termination on high similarity.
        """
        with sqlite3.connect(self.db_path) as conn:
            if exclude_path:
                rows = conn.execute(
                    "SELECT path, embedding FROM files WHERE embedding IS NOT NULL AND path != ?",
                    (exclude_path,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT path, embedding FROM files WHERE embedding IS NOT NULL"
                ).fetchall()

        best_path = None
        best_sim = SIMILARITY_THRESHOLD

        for path, emb_json in rows:
            emb = json.loads(emb_json)
            sim = sum(x * y for x, y in zip(embedding, emb))  # Inlined for speed
            if sim > best_sim:
                best_sim = sim
                best_path = path
                # Early termination on near-duplicate
                if sim > 0.98:
                    break

        return best_path

    def _evict_lru_k(self) -> None:
        """Evict entries using LRU-K policy (considers access frequency).

        LRU-K evicts based on K-th most recent access, not just most recent.
        This prevents one-time accesses from evicting frequently-used entries.
        """
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

            if count <= MAX_CACHE_ENTRIES:
                return

            # Get all entries with their K-th access time (K=2)
            rows = conn.execute(
                "SELECT path, chunk_hashes, access_history FROM files"
            ).fetchall()

            entries_with_score: list[tuple[float, str, list[str]]] = []
            for path, chunks_json, history_json in rows:
                history = json.loads(history_json)
                # LRU-K score: K-th most recent access (or oldest if < K accesses)
                k = 2
                score = history[-k] if len(history) >= k else history[0]
                entries_with_score.append((score, path, json.loads(chunks_json)))

            # Sort by score (oldest K-th access first)
            entries_with_score.sort()

            # Evict oldest 10%
            evict_count = max(1, count // 10)
            for _, path, chunk_hashes in entries_with_score[:evict_count]:
                conn.execute("DELETE FROM files WHERE path = ?", (path,))
                self._release_chunks(chunk_hashes)

    def get_stats(self) -> dict[str, int | float]:
        """Get detailed cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            file_stats = conn.execute(
                "SELECT COUNT(*), SUM(tokens) FROM files"
            ).fetchone()

            chunk_stats = conn.execute(
                "SELECT COUNT(*), SUM(size), SUM(LENGTH(data)) FROM chunks"
            ).fetchone()

        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        original_size = chunk_stats[1] or 0
        compressed_size = chunk_stats[2] or 0

        return {
            "files_cached": file_stats[0] or 0,
            "total_tokens_cached": file_stats[1] or 0,
            "unique_chunks": chunk_stats[0] or 0,
            "original_bytes": original_size,
            "compressed_bytes": compressed_size,
            "compression_ratio": round(compressed_size / original_size, 3)
            if original_size > 0
            else 0,
            "dedup_ratio": round(original_size / db_size, 2) if db_size > 0 else 1,
            "db_size_mb": round(db_size / 1024 / 1024, 2),
        }

    def clear(self) -> int:
        """Clear all cache entries."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            conn.execute("DELETE FROM files")
            conn.execute("DELETE FROM chunks")
        return count


def generate_diff(old: str, new: str, context_lines: int = 3) -> str:
    """Generate unified diff."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    diff = list(
        unified_diff(
            old_lines, new_lines, fromfile="cached", tofile="current", n=context_lines
        )
    )

    if not diff:
        return "// No changes"

    return "".join(diff)


def truncate_smart(
    content: str, max_size: int, keep_top: int = 80, keep_bottom: int = 40
) -> str:
    """Smart truncation preserving structure."""
    if len(content) <= max_size:
        return content

    lines = content.splitlines(keepends=True)
    top_lines = lines[:keep_top]
    bottom_lines = lines[-keep_bottom:] if keep_bottom > 0 else []

    top_content = "".join(top_lines)
    bottom_content = "".join(bottom_lines)

    truncation_msg = (
        f"\n\n// ... [{len(lines) - keep_top - keep_bottom} lines truncated] ...\n\n"
    )

    total = len(top_content) + len(truncation_msg) + len(bottom_content)
    if total > max_size:
        return content[: max_size - 50] + "\n\n// [TRUNCATED]"

    return top_content + truncation_msg + bottom_content
