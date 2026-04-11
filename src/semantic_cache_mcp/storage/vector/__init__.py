"""simplevecdb-backed storage: raw text + HNSW embeddings, HyperCDC chunking for large files."""

from __future__ import annotations

import array
import asyncio
import fnmatch
import json
import logging
import time
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

from simplevecdb import AsyncVectorCollection, DistanceStrategy, Quantization, VectorDB

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
from ...logger import log_marker
from ...types import CacheEntry, EmbeddingVector

logger = logging.getLogger(__name__)

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

    __slots__ = (
        "_db",
        "_collection",
        "_sync_collection",
        "_db_path",
        "_closed",
        "_io_executor",
        "_owns_executor",
    )

    # Quantization currently in use — stored in sidecar to detect future changes.
    _QUANTIZATION = Quantization.INT8
    _COLLECTION_NAME = "files"

    def __init__(
        self,
        db_path: Path = VECDB_PATH,
        *,
        executor: Executor | None = None,
    ) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._clear_if_quantization_changed(db_path)
        self._recover_if_crashed(db_path)
        # Use the sync VectorDB directly (not AsyncVectorDB) so we can pass
        # store_embeddings=True to collection() — that parameter is missing
        # from AsyncVectorDB.collection() in simplevecdb 2.5.0. Async behavior
        # is provided by wrapping the sync collection in AsyncVectorCollection
        # below, which serializes ops on our injected executor.
        self._db = VectorDB(
            path=str(db_path),
            distance_strategy=DistanceStrategy.COSINE,
            quantization=self._QUANTIZATION,
        )
        self._closed = False
        # Sync VectorDB has no executor, so we own one when none is injected.
        # Single-thread by default — ONNX/usearch are not thread-safe, and
        # SemanticCache always passes its own DetachedExecutor in production.
        # Annotate explicitly so mypy widens to Executor — both branches must
        # be assignable, including the rebind_executor seam below.
        self._io_executor: Executor
        if executor is None:
            self._io_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="vecstorage-io"
            )
            self._owns_executor = True
        else:
            self._io_executor = executor
            self._owns_executor = False
        self._collection = self._build_collection()
        # Write sentinel — removed on clean shutdown by _remove_sentinel().
        STARTUP_SENTINEL.touch()
        logger.info(f"VectorStorage initialized at {db_path}")

    def _reset_collection_sync(self, *, reason: str) -> None:
        """Drop and recreate the collection synchronously.

        Uses simplevecdb 2.5.0's ``delete_collection`` which atomically drops
        the SQLite tables, FTS index, and usearch file in one call. The new
        wrapper for the recreated collection is reattached to
        ``self._collection`` so further async ops keep working. Safe to call
        from sync startup paths — no event loop required.
        """
        try:
            self._db.delete_collection(self._COLLECTION_NAME)
        except KeyError:
            # Collection didn't exist yet — nothing to drop.
            pass
        except Exception as exc:
            logger.warning(f"delete_collection failed during {reason}: {exc}")
        self._collection = self._build_collection()
        logger.info(f"Collection reset complete ({reason})")

    def rebind_executor(self, new_executor: Executor) -> None:
        """Swap the IO executor used by the async collection wrapper.

        Called by ``SemanticCache.reset_executor`` after a hung worker is
        abandoned. Re-creates the AsyncVectorCollection wrapper around the
        same sync collection so the new executor takes effect on subsequent
        calls. Sync VectorDB has no executor of its own, so nothing else
        needs to be touched.
        """
        self._io_executor = new_executor
        # Reuse the existing sync collection — we only need a fresh async
        # wrapper bound to the new executor.
        self._collection = AsyncVectorCollection(self._sync_collection, new_executor)

    def _build_collection(self) -> AsyncVectorCollection:
        """Open the ``files`` collection with ``store_embeddings=True`` and
        wrap it for async use.

        ``store_embeddings=True`` is required so ``get_embeddings_by_ids``
        can return the vectors used by ``SemanticCache.get()`` /
        ``compare_files()``. simplevecdb 2.5.0 changed the default to False.
        Stores a direct reference to the sync collection for sync code paths
        (``save``, ``_clear_sync``) so they don't have to reach through the
        async wrapper.
        """
        sync_coll = self._db.collection(
            self._COLLECTION_NAME,
            distance_strategy=DistanceStrategy.COSINE,
            quantization=self._QUANTIZATION,
            store_embeddings=True,
        )
        self._sync_collection = sync_coll
        return AsyncVectorCollection(sync_coll, self._io_executor)

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
            self._reset_collection_sync(reason="embedding model change")

        # Runtime dim check: even if the sidecar matches, verify the live
        # index dimension agrees with the current model. A stale or missing
        # sidecar could leave a dim-mismatched index that segfaults on add.
        index_dim = self._collection.dim
        if index_dim is not None and index_dim > 0 and index_dim != dim:
            logger.warning(
                f"Index dimension {index_dim} != model dimension {dim}; "
                "clearing index to prevent usearch segfault"
            )
            self._reset_collection_sync(reason="dimension mismatch")

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
                self._db.save()
                self._db.close()
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
            logger.debug("VectorStorage closed cleanly")

        # Shut down the executor only when we own it. SemanticCache injects
        # its own DetachedExecutor and manages its lifecycle separately.
        if self._owns_executor:
            try:
                self._io_executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                logger.debug(f"Owned executor shutdown failed: {exc}")

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
        started = time.perf_counter()
        content_hash = hash_content(content)
        tokens = count_tokens(content)
        now = time.time()
        content_bytes = content.encode("utf-8")
        chunked = len(content_bytes) >= CHUNK_THRESHOLD

        emb_list = self._resolve_embedding(embedding)
        has_embedding = embedding is not None
        log_marker(
            logger,
            "vector.put.begin",
            path=path,
            tokens=tokens,
            bytes=len(content_bytes),
            chunked=chunked,
            has_embedding=has_embedding,
        )

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
        delete_started = time.perf_counter()
        log_marker(logger, "vector.put.delete.begin", path=path)
        await self._delete_by_path(path)
        log_marker(
            logger,
            "vector.put.delete.end",
            path=path,
            elapsed_ms=round((time.perf_counter() - delete_started) * 1000, 1),
        )

        add_started = time.perf_counter()
        log_marker(logger, "vector.put.add.begin", path=path, chunked=chunked)
        if not chunked:
            # Small file: single document
            meta = {**base_meta, _META_CHUNK_INDEX: 0, _META_TOTAL_CHUNKS: 1}
            await self._collection.add_texts(
                texts=[content],
                metadatas=[meta],
                embeddings=[emb_list],
            )
            logger.debug(f"Stored {path} as single doc ({tokens} tokens)")
        else:
            # Large file: try chunked storage, fall back to single-doc if too many chunks
            stored = await self._put_chunked(path, content, content_bytes, base_meta, emb_list)
            if not stored:
                meta = {**base_meta, _META_CHUNK_INDEX: 0, _META_TOTAL_CHUNKS: 1}
                await self._collection.add_texts(
                    texts=[content],
                    metadatas=[meta],
                    embeddings=[emb_list],
                )
                logger.debug(f"Stored {path} as single doc (chunk fallback, {tokens} tokens)")
        log_marker(
            logger,
            "vector.put.add.end",
            path=path,
            chunked=chunked,
            elapsed_ms=round((time.perf_counter() - add_started) * 1000, 1),
        )

        evict_started = time.perf_counter()
        log_marker(logger, "vector.put.evict.begin", path=path)
        await self._evict_if_needed()
        log_marker(
            logger,
            "vector.put.evict.end",
            path=path,
            elapsed_ms=round((time.perf_counter() - evict_started) * 1000, 1),
        )
        log_marker(
            logger,
            "vector.put.end",
            path=path,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def _put_chunked(
        self,
        path: str,
        content: str,
        content_bytes: bytes,
        base_meta: dict,
        file_embedding: list[float],
    ) -> bool:
        """Store large file as parent doc + CDC-chunked children.

        Returns False if the file produced too many chunks (caller should
        fall back to single-doc storage to preserve full content).
        """
        chunker = get_optimal_chunker(prefer_simd=True)
        chunks_bytes = list(chunker(content_bytes))
        total_chunks = len(chunks_bytes)

        # Safety cap: if CDC produces too many chunks, bail out and let
        # the caller store the file as a single unchunked document instead
        # of silently truncating (which would cause data loss).
        _MAX_CHUNKS = 500  # noqa: N806
        if total_chunks > _MAX_CHUNKS:
            logger.warning(
                f"File {path} produced {total_chunks} chunks (max {_MAX_CHUNKS}); "
                "falling back to single-doc storage to preserve full content"
            )
            return False

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
        return True

    def _expected_dim(self) -> int:
        """Return the expected embedding dimension, querying the model if needed."""
        dim = self._collection.dim
        if dim is not None and dim > 0:
            return dim
        from ...core.embeddings import get_embedding_dim  # noqa: PLC0415

        dim = get_embedding_dim()
        if dim == 0:
            raise RuntimeError(
                "Cannot determine embedding dimension — model not loaded yet. "
                "Ensure warmup() completes before storing documents."
            )
        return dim

    def _resolve_embedding(self, embedding: EmbeddingVector | None) -> list[float]:
        expected = self._expected_dim()
        if embedding is not None:
            actual = len(embedding)
            if actual != expected:
                raise ValueError(
                    f"Embedding dimension mismatch: got {actual}, expected {expected}. "
                    "This usually means the embedding model changed mid-session."
                )
            return list(embedding)
        return [0.0] * expected

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
        path: str | None = None,
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
            # Cap pattern length to mitigate ReDoS from pathological regexes.
            if len(pattern) > 1000:
                logger.warning(f"Regex pattern too long ({len(pattern)} chars), rejecting")
                return []
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
            doc_path = meta.get(_META_PATH, "")
            if not doc_path or not self._grep_path_matches(doc_path, path_filter=path):
                continue
            chunk_idx = meta.get(_META_CHUNK_INDEX, 0)
            if doc_path not in files:
                files[doc_path] = []
            files[doc_path].append((chunk_idx, text))

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

    @staticmethod
    def _grep_path_matches(path: str, *, path_filter: str | None) -> bool:
        """Match exact paths, relative suffixes, basenames, and glob filters."""
        if not path_filter:
            return True

        normalized_path = path.replace("\\", "/")
        normalized_filter = path_filter.replace("\\", "/")
        has_glob = any(ch in normalized_filter for ch in "*?[")

        if has_glob:
            return any(
                fnmatch.fnmatchcase(normalized_path, candidate)
                for candidate in (
                    normalized_filter,
                    f"*/{normalized_filter}",
                )
            )

        return (
            normalized_path == normalized_filter
            or normalized_path.endswith(f"/{normalized_filter}")
            or Path(normalized_path).name == normalized_filter
        )

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
        try:
            db_size = self._db_path.stat().st_size if self._db_path.exists() else 0
        except OSError:
            db_size = 0

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

        Avoids the fragile get_event_loop().run_until_complete() pattern that
        breaks when called from within a running loop. Kept for tests and
        callers that need to clear without dropping the collection's tables.
        """
        sync_coll = self._sync_collection
        count = sync_coll.count()
        if count > 0:
            all_docs = sync_coll.get_documents()
            doc_ids = [doc_id for doc_id, _, _ in all_docs]
            if doc_ids:
                sync_coll.delete_by_ids(doc_ids)
                sync_coll.save()
        return count

    async def clear(self) -> int:
        """Clear all cache entries. Returns count of documents removed.

        Uses simplevecdb 2.5.0's ``delete_collection`` to atomically drop the
        SQLite tables, FTS index, and usearch file in one call, then rebuilds
        an empty collection. Faster and safer than the previous per-id loop.
        Runs the (sync) ``delete_collection`` on the shared IO executor so it
        does not block the event loop.
        """
        import contextlib  # noqa: PLC0415

        count = await self._collection.count()
        if count > 0:
            loop = asyncio.get_running_loop()
            with contextlib.suppress(KeyError):
                await loop.run_in_executor(
                    self._io_executor,
                    lambda: self._db.delete_collection(self._COLLECTION_NAME),
                )
            self._collection = self._build_collection()
        return count

    async def delete_path(self, path: str) -> int:
        """Delete cached documents for one path. Returns docs removed."""
        removed = await self._delete_by_path(path)
        if removed > 0:
            await self._collection.save()
        return removed

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
        """Persist index to disk.

        Guarded: if close() is already running on a daemon thread, skip
        to avoid concurrent usearch save (not thread-safe → heap corruption).
        """
        if self._closed:
            return
        self._sync_collection.save()
        self._db.save()
