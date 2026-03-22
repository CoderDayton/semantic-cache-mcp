"""simplevecdb-backed storage: raw text + HNSW embeddings, HyperCDC chunking for large files."""

from __future__ import annotations

import array
import asyncio
import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

from simplevecdb import AsyncVectorDB, DistanceStrategy, Quantization

from ...config import (
    CACHE_DIR,
    CHUNK_MIN_SIZE,
    MAX_CACHE_ENTRIES,
    SIMILARITY_THRESHOLD,
    STARTUP_SENTINEL,
)
from ...core.chunking import get_optimal_chunker
from ...core.hashing import hash_content
from ...core.tokenizer import count_tokens
from ...types import CacheEntry, EmbeddingVector

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

VECDB_PATH = CACHE_DIR / "vecdb.db"

# Metadata keys stored with each document in simplevecdb
_META_PATH = "path"
_META_CHUNK_INDEX = "chunk_index"
_META_TOTAL_CHUNKS = "total_chunks"
_META_CONTENT_HASH = "content_hash"
_META_MTIME = "mtime"
_META_TOKENS = "tokens"
_META_CREATED_AT = "created_at"
_META_ACCESS_HISTORY = "access_history"
_META_IS_PARENT = "is_parent"
_META_HAS_EMBEDDING = "has_embedding"

# Files larger than this (bytes) are split via HyperCDC into multiple chunks,
# each stored as a child document with its own vector. Smaller files are stored
# as a single document. CHUNK_MIN_SIZE (2KB) is the CDC minimum chunk size.
CHUNK_THRESHOLD = CHUNK_MIN_SIZE * 4  # 8KB — files below this stay as one doc


class VectorStorage:
    """simplevecdb-backed storage for file content and embeddings.

    Architecture:
    - Files stored as Documents with raw text (no compression)
    - HNSW index for O(log N) semantic similarity search
    - Metadata-based filtering for path lookups
    - LRU-K eviction via access_history metadata
    """

    __slots__ = ("_db", "_collection", "_db_path", "_closed", "_io_executor")

    async def _run_sync(self, fn: Callable[[], _T]) -> _T:
        """Run a blocking function in the shared IO executor.

        Uses the single-threaded executor set by SemanticCache to prevent
        segfaults from ONNX/usearch allocator conflicts.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._io_executor, fn)

    # Quantization currently in use — stored in sidecar to detect future changes.
    _QUANTIZATION = Quantization.INT8
    _COLLECTION_NAME = "files"

    def __init__(
        self,
        db_path: Path = VECDB_PATH,
        *,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._clear_if_quantization_changed(db_path)
        self._recover_if_crashed(db_path)
        self._db = AsyncVectorDB(
            path=str(db_path),
            distance_strategy=DistanceStrategy.COSINE,
            quantization=self._QUANTIZATION,
            executor=executor,
        )
        self._collection = self._db.collection(self._COLLECTION_NAME)
        self._closed = False
        self._io_executor = self._db._executor
        # Write sentinel — removed on clean shutdown by _remove_sentinel().
        STARTUP_SENTINEL.touch()
        logger.info(f"VectorStorage initialized at {db_path}")

    @staticmethod
    def _remove_sentinel() -> None:
        """Remove crash sentinel on clean shutdown."""
        STARTUP_SENTINEL.unlink(missing_ok=True)

    @classmethod
    def _recover_if_crashed(cls, db_path: Path) -> None:
        """Wipe vecdb if the previous run crashed (sentinel still present).

        A C-level crash (SIGABRT from malloc corruption, SIGSEGV) kills the
        process before Python cleanup runs, leaving the sentinel behind.
        On next startup we detect this and delete the potentially corrupted
        usearch index + SQLite WAL so the DB rebuilds cleanly.
        """
        if not STARTUP_SENTINEL.exists():
            return

        logger.warning(
            "Crash sentinel detected — previous run did not shut down cleanly. "
            "Clearing vecdb to prevent heap corruption from stale index files."
        )
        # Delete usearch index, SQLite DB, WAL, SHM, and metadata sidecar.
        for pattern in (
            f".{cls._COLLECTION_NAME}.usearch",
            "",
            "-wal",
            "-shm",
            ".meta.json",
        ):
            target = db_path.parent / (db_path.name + pattern) if pattern else db_path
            if target.exists():
                target.unlink()
                logger.info(f"Deleted stale file: {target}")

        STARTUP_SENTINEL.unlink(missing_ok=True)

    @classmethod
    def _clear_if_quantization_changed(cls, db_path: Path) -> None:
        """Delete stale usearch index when quantization setting changes.

        Opening a usearch index built with one quantization type (e.g. FLOAT16)
        using a different type (e.g. INT8) causes heap corruption (free():
        corrupted unsorted chunks). We track the active quantization in a JSON
        sidecar and wipe the index files on mismatch so usearch rebuilds clean.
        """
        meta_path = db_path.with_suffix(".meta.json")
        current_quant = cls._QUANTIZATION.value
        stored_quant: str | None = None

        import contextlib

        if meta_path.exists():
            with contextlib.suppress(Exception):
                stored_quant = json.loads(meta_path.read_text()).get("quantization")

        if stored_quant != current_quant:
            if stored_quant is not None:
                logger.warning(
                    f"Quantization changed {stored_quant!r} → {current_quant!r}; "
                    "clearing stale usearch index (cache content preserved in SQLite)"
                )
            # Delete usearch index files so VectorDB rebuilds with correct quantization.
            # Pattern: {db_path}.{collection}.usearch
            for suffix in (f".{cls._COLLECTION_NAME}.usearch",):
                stale = db_path.parent / (db_path.name + suffix)
                if stale.exists():
                    stale.unlink()
                    logger.info(f"Deleted stale index: {stale}")
            meta_path.write_text(json.dumps({"quantization": current_quant}))

    def clear_if_model_changed(self, model_name: str, dim: int) -> None:
        """Wipe vector index and stored embeddings when the embedding model changes.

        Different models produce incompatible embeddings (different dimensions,
        different vector spaces). Detect via sidecar metadata and rebuild clean.
        """
        meta_path = self._db_path.with_suffix(".meta.json")
        meta: dict = {}

        import contextlib

        if meta_path.exists():
            with contextlib.suppress(Exception):
                meta = json.loads(meta_path.read_text())

        stored_model = meta.get("embedding_model")
        if stored_model is not None and stored_model != model_name:
            logger.warning(
                f"Embedding model changed {stored_model!r} -> {model_name!r} "
                f"(dim {meta.get('embedding_dim', '?')} -> {dim}); "
                "rebuilding vector index and clearing cached embeddings"
            )
            # Delete usearch index files
            for suffix in (f".{self._COLLECTION_NAME}.usearch",):
                stale = self._db_path.parent / (self._db_path.name + suffix)
                if stale.exists():
                    stale.unlink()
                    logger.info(f"Deleted stale index: {stale}")
            # Clear all documents from SQLite so stale embeddings don't persist.
            self._clear_sync()
            logger.info("Cleared all cached embeddings")

        meta["embedding_model"] = model_name
        meta["embedding_dim"] = dim
        meta_path.write_text(json.dumps(meta))

    def close(self, *, timeout: float = 5.0) -> None:
        """Save and close the database with a deadline.

        Uses a background thread so a hung usearch/SQLite save cannot block
        the asyncio event loop or delay process exit past *timeout* seconds.
        Idempotent — safe to call multiple times.
        """
        if self._closed:
            return

        import threading  # noqa: PLC0415

        self._closed = True
        close_error: BaseException | None = None

        def _do_close() -> None:
            nonlocal close_error
            try:
                # Access the sync VectorDB via AsyncVectorDB._db.
                # Guard with getattr so a simplevecdb API change logs a
                # warning instead of crashing during shutdown.
                sync_db = getattr(self._db, "_db", None)
                if sync_db is None:
                    logger.warning("simplevecdb internal API changed — cannot access sync VectorDB")
                    return
                sync_db.save()
                sync_db.close()
            except Exception as exc:
                close_error = exc

        # Use a daemon thread so a hung save doesn't prevent process exit.
        # ThreadPoolExecutor.__exit__ calls shutdown(wait=True) which would
        # block indefinitely if _do_close hangs — daemon thread avoids this.
        t = threading.Thread(target=_do_close, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            logger.warning(
                f"VectorStorage close timed out after {timeout}s — "
                "index may need recovery on next startup"
            )
        elif close_error is not None:
            logger.warning(f"VectorStorage close error: {close_error}")
        else:
            logger.info("VectorStorage closed cleanly")

    # -------------------------------------------------------------------------
    # File operations
    # -------------------------------------------------------------------------

    async def get(self, path: str) -> CacheEntry | None:
        """Return metadata for path. For chunked files, uses the parent document."""
        if self._closed:
            return None
        results = await self._find_docs_by_path(path)
        if not results:
            return None

        # Prefer parent doc metadata (has file-level tokens, hash, etc.)
        # For single-doc files there is no parent, so use the only doc.
        selected_id = results[0][0]
        meta = results[0][1]
        for doc_id, m, _text in results:
            if m.get(_META_IS_PARENT, False):
                selected_id = doc_id
                meta = m
                break

        access_history = json.loads(meta.get(_META_ACCESS_HISTORY, "[]"))

        # Retrieve embedding from catalog so compare_files() can compute similarity.
        # Must run in executor — catalog is not thread-safe.
        embedding: EmbeddingVector | None = None
        try:
            emb_map = await self._collection.get_embeddings_by_ids([selected_id])
            raw = emb_map.get(selected_id)
            if raw is not None:
                embedding = array.array("f", raw)
        except Exception:  # nosec B110 — best-effort embedding retrieval
            pass

        return CacheEntry(
            path=meta[_META_PATH],
            content_hash=meta[_META_CONTENT_HASH],
            mtime=meta[_META_MTIME],
            tokens=meta[_META_TOKENS],
            embedding=embedding,
            created_at=meta[_META_CREATED_AT],
            access_history=access_history,
        )

    async def put(
        self,
        path: str,
        content: str,
        mtime: float,
        embedding: EmbeddingVector | None = None,
    ) -> None:
        """Store file as raw text + embedding. Large files (>8KB) are HyperCDC-chunked."""
        if self._closed:
            return
        content_hash = hash_content(content)
        tokens = count_tokens(content)
        now = time.time()
        content_bytes = content.encode("utf-8")

        emb_list = self._resolve_embedding(embedding)
        has_embedding = embedding is not None

        base_meta = {
            _META_PATH: path,
            _META_CONTENT_HASH: content_hash,
            _META_MTIME: mtime,
            _META_TOKENS: tokens,
            _META_CREATED_AT: now,
            _META_ACCESS_HISTORY: json.dumps([now]),
            _META_HAS_EMBEDDING: has_embedding,
        }

        # Remove old entry for this path
        await self._delete_by_path(path)

        if len(content_bytes) < CHUNK_THRESHOLD:
            # Small file: single document
            meta = {**base_meta, _META_CHUNK_INDEX: 0, _META_TOTAL_CHUNKS: 1}
            await self._collection.add_texts(
                texts=[content],
                metadatas=[meta],
                embeddings=[emb_list],
            )
            logger.debug(f"Stored {path} as single doc ({tokens} tokens)")
        else:
            # Large file: parent + chunked children
            await self._put_chunked(path, content, content_bytes, base_meta, emb_list)

        await self._evict_if_needed()

    async def _put_chunked(
        self,
        path: str,
        content: str,
        content_bytes: bytes,
        base_meta: dict,
        file_embedding: list[float],
    ) -> None:
        """Store large file as parent doc (file metadata) + CDC-chunked children (raw text)."""
        chunker = get_optimal_chunker(prefer_simd=True)
        chunks_bytes = list(chunker(content_bytes))
        total_chunks = len(chunks_bytes)

        # Decode each chunk back to str, tracking line offsets
        chunk_texts: list[str] = []
        line_offset = 0
        chunk_line_starts: list[int] = []

        for chunk_b in chunks_bytes:
            text = chunk_b.decode("utf-8", errors="replace")
            chunk_texts.append(text)
            chunk_line_starts.append(line_offset)
            line_offset += text.count("\n")

        # Store parent document (holds file-level metadata, searchable by file embedding)
        parent_meta = {
            **base_meta,
            _META_IS_PARENT: True,
            _META_CHUNK_INDEX: -1,
            _META_TOTAL_CHUNKS: total_chunks,
        }
        parent_ids = await self._collection.add_texts(
            texts=[""],  # Parent has no content — children hold raw text
            metadatas=[parent_meta],
            embeddings=[file_embedding],
        )
        parent_id = parent_ids[0]

        # Store children — each chunk is a child of the parent.
        # AsyncVectorCollection.add_texts does not expose parent_ids, so we call
        # the underlying sync collection directly via run_in_executor.
        child_metas: list[dict] = []
        child_embeddings: list[list[float]] = []
        zero_emb = self._resolve_embedding(None)

        for i, text in enumerate(chunk_texts):
            child_meta = {
                _META_PATH: path,
                _META_CHUNK_INDEX: i,
                _META_TOTAL_CHUNKS: total_chunks,
                _META_CONTENT_HASH: base_meta[_META_CONTENT_HASH],
                _META_MTIME: base_meta[_META_MTIME],
                _META_TOKENS: count_tokens(text),
                _META_CREATED_AT: base_meta[_META_CREATED_AT],
                _META_ACCESS_HISTORY: base_meta[_META_ACCESS_HISTORY],
                "line_start": chunk_line_starts[i],
            }
            child_metas.append(child_meta)
            # Chunks use zero embedding — similarity search uses the parent's
            # file-level embedding. Per-chunk embeddings would require N model
            # calls which is too expensive for the cache hot path.
            child_embeddings.append(zero_emb)

        await self._collection.add_texts(
            texts=chunk_texts,
            metadatas=child_metas,
            embeddings=child_embeddings,
            parent_ids=[parent_id] * total_chunks,
        )

        logger.debug(
            f"Stored {path} as {total_chunks} chunks "
            f"(parent_id={parent_id}, {base_meta[_META_TOKENS]} tokens)"
        )

    def _resolve_embedding(self, embedding: EmbeddingVector | None) -> list[float]:
        if embedding is not None:
            return list(embedding)
        dim = self._collection.dim
        if dim is None:
            # Query the actual embedding model dimension instead of assuming 384.
            # Falls back to 384 (bge-small-en-v1.5) only if model isn't loaded yet.
            from ...core.embeddings import get_embedding_dim  # noqa: PLC0415

            dim = get_embedding_dim() or 384
        return [0.0] * dim

    async def get_content(self, entry: CacheEntry) -> str:
        """Reassemble full text from simplevecdb chunks (sorted by chunk_index)."""
        if self._closed:
            raise ValueError(f"Storage closed, cannot read content for {entry.path}")
        results = await self._find_docs_by_path(entry.path)
        if not results:
            raise ValueError(f"No cached content found for {entry.path}")

        # Filter out parent docs (they have is_parent=True and empty content)
        children = [r for r in results if not r[1].get(_META_IS_PARENT, False)]

        if not children:
            # Shouldn't happen — but fall back to all results
            children = results

        # Sort by chunk_index for correct reassembly
        children.sort(key=lambda r: r[1].get(_META_CHUNK_INDEX, 0))

        # Concatenate raw text from all chunks
        return "".join(r[2] for r in children)

    async def record_access(self, path: str) -> None:
        if self._closed:
            return
        results = await self._find_docs_by_path(path)
        if not results:
            return

        now = time.time()
        updates: list[tuple[int, dict]] = []

        for doc_id, meta, _text in results:
            history = json.loads(meta.get(_META_ACCESS_HISTORY, "[]"))
            history.append(now)
            history = history[-5:]  # Keep last 5 accesses
            updates.append((doc_id, {_META_ACCESS_HISTORY: json.dumps(history)}))

        if updates:
            await self._collection.update_metadata(updates)

    async def update_mtime(self, path: str, new_mtime: float) -> None:
        """Update cached mtime without re-storing content or re-embedding.

        Used when content hash matches but mtime changed (touch, git checkout).
        Prevents repeated hash checks on subsequent reads.
        """
        if self._closed:
            return
        results = await self._find_docs_by_path(path)
        if not results:
            return

        updates: list[tuple[int, dict]] = []
        for doc_id, _meta, _text in results:
            updates.append((doc_id, {_META_MTIME: new_mtime}))

        if updates:
            await self._collection.update_metadata(updates)

    # -------------------------------------------------------------------------
    # Similarity search
    # -------------------------------------------------------------------------

    async def find_similar(
        self, embedding: EmbeddingVector, exclude_path: str | None = None
    ) -> str | None:
        """Return the most similar cached file path above SIMILARITY_THRESHOLD, or None."""
        if self._closed:
            return None
        emb_list = list(embedding)

        try:
            results = await self._collection.similarity_search(
                query=emb_list,
                k=10,  # Get candidates, then filter
            )
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")
            return None

        if not results:
            return None

        for doc, distance in results:
            # COSINE distance in simplevecdb: 0 = identical, 2 = opposite
            # Convert to similarity: 1 - (distance / 2) maps [0,2] → [1,0]
            similarity = 1.0 - (distance / 2.0)
            candidate_path = doc.metadata.get(_META_PATH)
            has_emb = doc.metadata.get(_META_HAS_EMBEDDING, False)

            is_match = (
                has_emb
                and candidate_path
                and candidate_path != exclude_path
                and similarity >= SIMILARITY_THRESHOLD
            )
            if is_match:
                logger.debug(f"Similar file found: {candidate_path} (similarity={similarity:.3f})")
                return candidate_path

        return None

    async def find_similar_multi(
        self,
        embedding: EmbeddingVector,
        exclude_path: str | None = None,
        k: int = 5,
        threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """Return up to k (path, similarity) tuples above threshold."""
        if self._closed:
            return []
        if threshold is None:
            threshold = SIMILARITY_THRESHOLD

        emb_list = list(embedding)

        try:
            results = await self._collection.similarity_search(
                query=emb_list,
                k=k * 3,  # Over-fetch to account for filtering
            )
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")
            return []

        matches: list[tuple[str, float]] = []
        seen_paths: set[str] = set()

        for doc, distance in results:
            similarity = 1.0 - (distance / 2.0)
            candidate_path = doc.metadata.get(_META_PATH)
            has_emb = doc.metadata.get(_META_HAS_EMBEDDING, False)

            if (
                has_emb
                and candidate_path
                and candidate_path != exclude_path
                and candidate_path not in seen_paths
                and similarity >= threshold
            ):
                seen_paths.add(candidate_path)
                matches.append((candidate_path, similarity))

                if len(matches) >= k:
                    break

        return matches

    # -------------------------------------------------------------------------
    # Query-based search
    # -------------------------------------------------------------------------

    async def search_by_query(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
    ) -> list[tuple[str, str, float]]:
        """BM25 keyword search over cached content. FTS5 syntax supported."""
        if self._closed:
            return []
        try:
            results = await self._collection.keyword_search(query, k=k * 2, filter=filter)
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return []

        return self._dedupe_search_results(results, k)

    async def search_hybrid(
        self,
        query: str,
        embedding: EmbeddingVector | None = None,
        k: int = 5,
        filter: dict | None = None,
    ) -> list[tuple[str, str, float]]:
        """Hybrid BM25 + vector search with RRF fusion.

        Falls back to keyword-only if no embedding provided.
        """
        if self._closed:
            return []
        query_vector = list(embedding) if embedding is not None else None

        try:
            results = await self._collection.hybrid_search(
                query,
                k=k * 2,
                filter=filter,
                query_vector=query_vector,
            )
        except Exception as e:
            logger.warning(f"Hybrid search failed: {e}")
            return []

        return self._dedupe_search_results(results, k)

    def _dedupe_search_results(
        self,
        results: list[tuple],
        k: int,
    ) -> list[tuple[str, str, float]]:
        """Deduplicate by path, keeping best score. Skips parent (empty) docs."""
        seen_paths: set[str] = set()
        matches: list[tuple[str, str, float]] = []

        for doc, score in results:
            path = doc.metadata.get(_META_PATH, "")
            if doc.metadata.get(_META_IS_PARENT, False):
                continue  # Skip parent docs (empty content)
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            preview = doc.page_content[:200]
            matches.append((path, preview, float(score)))
            if len(matches) >= k:
                break

        return matches

    # -------------------------------------------------------------------------
    # Grep — regex/literal content search across cached files
    # -------------------------------------------------------------------------

    # Upper bounds for grep parameters — prevent excessive memory/CPU usage
    _GREP_MAX_CONTEXT_LINES = 20
    _GREP_MAX_MATCHES = 10_000
    _GREP_MAX_FILES = 500

    async def grep(
        self,
        pattern: str,
        *,
        fixed_string: bool = False,
        case_sensitive: bool = True,
        context_lines: int = 0,
        max_matches: int = 100,
        max_files: int = 50,
    ) -> list[dict]:
        """Exact pattern matching across cached files — like ripgrep on the cache.

        Unlike search/search_hybrid, returns line numbers and context, not ranked scores.
        """
        if self._closed:
            return []
        import re  # noqa: PLC0415

        # Clamp inputs to prevent excessive memory/CPU usage
        context_lines = max(0, min(context_lines, self._GREP_MAX_CONTEXT_LINES))
        max_matches = max(1, min(max_matches, self._GREP_MAX_MATCHES))
        max_files = max(1, min(max_files, self._GREP_MAX_FILES))

        flags = 0 if case_sensitive else re.IGNORECASE
        if fixed_string:
            compiled = re.compile(re.escape(pattern), flags)
        else:
            try:
                compiled = re.compile(pattern, flags)
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {e}")
                return []

        # Collect all cached file content grouped by path.
        all_docs = await self._collection.get_documents()
        files: dict[str, list[tuple[int, str]]] = {}  # path -> [(chunk_index, text)]
        for _doc_id, text, meta in all_docs:
            if meta.get(_META_IS_PARENT, False):
                continue  # Parent docs have empty content
            path = meta.get(_META_PATH, "")
            if not path:
                continue
            chunk_idx = meta.get(_META_CHUNK_INDEX, 0)
            if path not in files:
                files[path] = []
            files[path].append((chunk_idx, text))

        # Search each file's content
        results: list[dict] = []
        total_matches = 0

        for path, chunks in files.items():
            if total_matches >= max_matches or len(results) >= max_files:
                break

            # Reconstruct file content from sorted chunks
            chunks.sort(key=lambda c: c[0])
            content = "".join(text for _, text in chunks)
            lines = content.splitlines()

            file_matches: list[dict] = []
            for i, line in enumerate(lines):
                if total_matches >= max_matches:
                    break
                if compiled.search(line):
                    match_info: dict[str, object] = {
                        "line_number": i + 1,
                        "line": line,
                    }
                    if context_lines > 0:
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        match_info["before"] = lines[start:i]
                        match_info["after"] = lines[i + 1 : end]
                    file_matches.append(match_info)
                    total_matches += 1

            if file_matches:
                results.append({"path": path, "matches": file_matches})

        return results

    # -------------------------------------------------------------------------
    # Statistics and management
    # -------------------------------------------------------------------------

    async def is_healthy(self) -> bool:
        """Lightweight probe: True when the catalog and index are accessible.

        Intended for post-startup validation and observability. Does not
        perform I/O beyond a single count query on the in-memory catalog.
        """
        if self._closed:
            return False
        try:
            await self._collection.count()
            return True
        except Exception:
            return False

    async def get_stats(self) -> dict[str, int | float]:
        if self._closed:
            return {
                "files_cached": 0,
                "total_tokens_cached": 0,
                "total_documents": 0,
                "db_size_mb": 0,
            }
        count = await self._collection.count()
        db_size = self._db_path.stat().st_size if self._db_path.exists() else 0

        # Count unique files (not chunks)
        unique_paths: set[str] = set()
        total_tokens = 0

        all_docs = await self._collection.get_documents()
        for _doc_id, _text, meta in all_docs:
            path = meta.get(_META_PATH, "")
            if path:
                unique_paths.add(path)
            # Only count tokens from parent docs or single-chunk files.
            # Child chunks also store token counts, so summing all docs
            # would double-count chunked files.
            if meta.get(_META_IS_PARENT, False) or meta.get(_META_TOTAL_CHUNKS, 1) == 1:
                total_tokens += meta.get(_META_TOKENS, 0)

        return {
            "files_cached": len(unique_paths),
            "total_tokens_cached": total_tokens,
            "total_documents": count,
            "db_size_mb": round(db_size / 1024 / 1024, 2),
        }

    def _clear_sync(self) -> int:
        """Synchronous clear — safe to call without an event loop.

        Used by clear_if_model_changed() which runs before the async
        server starts. Avoids the fragile get_event_loop().run_until_complete()
        pattern that breaks when called from within a running loop.
        """
        sync_coll = self._collection._collection
        count = sync_coll.count()
        if count > 0:
            all_docs = sync_coll.get_documents()
            doc_ids = [doc_id for doc_id, _, _ in all_docs]
            if doc_ids:
                sync_coll.delete_by_ids(doc_ids)
                sync_coll.save()
        return count

    async def clear(self) -> int:
        """Clear all cache entries. Returns count of files removed."""
        return await self._run_sync(self._clear_sync)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _find_docs_by_path(self, path: str) -> list[tuple[int, dict, str]]:
        """Return [(doc_id, metadata, text)] for all documents at path."""
        try:
            docs = await self._collection.get_documents(
                filter_dict={_META_PATH: path},
            )
        except Exception as e:
            logger.debug(f"Filter lookup failed for {path}: {e}")
            return []

        # docs is list of (doc_id, text, metadata_dict)
        return [(doc_id, meta, text) for doc_id, text, meta in docs]

    async def _delete_by_path(self, path: str) -> int:
        results = await self._find_docs_by_path(path)
        if not results:
            return 0

        doc_ids = [r[0] for r in results]
        await self._collection.delete_by_ids(doc_ids)
        return len(doc_ids)

    async def _evict_if_needed(self) -> None:
        """Evict oldest-K-th-access files (LRU-K) when over MAX_CACHE_ENTRIES."""
        # Get all docs once — used for both the file-count check and LRU scoring.
        all_docs = await self._collection.get_documents()

        # Count unique files, not documents. Chunked files span multiple
        # documents, so using doc count fires eviction too early.
        unique_file_count = len(
            {meta.get(_META_PATH) for _, _, meta in all_docs if meta.get(_META_PATH)}
        )
        if unique_file_count <= MAX_CACHE_ENTRIES:
            return

        evict_count = max(1, unique_file_count // 10)

        # Score each unique path by its K-th most recent access
        path_scores: dict[str, tuple[float, list[int]]] = {}
        for doc_id, _text, meta in all_docs:
            path = meta.get(_META_PATH, "")
            history = json.loads(meta.get(_META_ACCESS_HISTORY, "[]"))

            # LRU-K score: use 2nd most recent access (or oldest if < K accesses)
            k = 2
            if len(history) >= k:
                score = history[-k]
            elif history:
                score = history[0]
            else:
                score = 0.0

            if path not in path_scores:
                path_scores[path] = (score, [])
            path_scores[path][1].append(doc_id)

        # Sort by score ascending (oldest K-th access first)
        sorted_paths = sorted(path_scores.items(), key=lambda x: x[1][0])

        # Evict until we've removed enough
        evicted = 0
        ids_to_delete: list[int] = []
        for _path, (_, doc_ids) in sorted_paths:
            if evicted >= evict_count:
                break
            ids_to_delete.extend(doc_ids)
            evicted += 1

        if ids_to_delete:
            await self._collection.delete_by_ids(ids_to_delete)
            await self._collection.save()
            logger.info(f"Cache eviction: removed {evicted} documents")

    def save(self) -> None:
        """Persist index to disk."""
        self._collection._collection.save()
        sync_db = getattr(self._db, "_db", None)
        if sync_db is not None:
            sync_db.save()
