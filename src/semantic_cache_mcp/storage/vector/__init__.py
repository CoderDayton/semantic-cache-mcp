"""simplevecdb-backed storage: raw text + HNSW embeddings, HyperCDC chunking for large files."""

from __future__ import annotations

import array
import asyncio
import enum
import fnmatch
import json
import logging
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

from simplevecdb import AsyncVectorCollection, DistanceStrategy, Quantization, VectorDB

from ... import config
from ...config import (
    ACCESS_HISTORY_SIZE,
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
from ._tinylfu import TinyLFUIndex

logger = logging.getLogger(__name__)

__all__ = ["CHUNK_THRESHOLD", "StorageMode", "VECDB_PATH", "VectorStorage"]

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
_META_PREVIEW = "preview"
_META_STORAGE_MODE = "storage_mode"
_PREVIEW_CHARS = 200


class StorageMode(enum.StrEnum):
    """How a file's content was stored. Surfaces retrieval-quality signals.

    SINGLE_DOC          — file fits in one doc, embedding represents it well.
    CHUNKED             — file split via CDC; per-chunk text + parent embedding.
    SINGLE_DOC_FALLBACK — chunk count exceeded the cap; whole file stored as
                          one doc with one embedding. Embedding quality and
                          retrieval granularity are degraded vs CHUNKED.
    """

    SINGLE_DOC = "single_doc"
    CHUNKED = "chunked"
    SINGLE_DOC_FALLBACK = "single_doc_fallback"


# Files larger than this (bytes) are split via HyperCDC into multiple chunks,
# each stored as a child document with its own vector. Smaller files are stored
# as a single document. CHUNK_MIN_SIZE (2KB) is the CDC minimum chunk size.
CHUNK_THRESHOLD = CHUNK_MIN_SIZE * 4  # 8KB — files below this stay as one doc


# Per-dimension cached zero embeddings used by chunk children (which carry no
# real embedding — similarity uses the parent's vector). Populating this once
# avoids re-allocating `[0.0] * dim` on every chunked write.
_ZERO_EMB_CACHE: dict[int, list[float]] = {}


def _zero_embedding(dim: int) -> list[float]:
    # Returns a shared list — callers MUST NOT mutate. simplevecdb copies via
    # np.asarray on ingest, so the cache stays clean in practice.
    cached = _ZERO_EMB_CACHE.get(dim)
    if cached is None:
        cached = [0.0] * dim
        _ZERO_EMB_CACHE[dim] = cached
    return cached


class VectorStorage:
    """simplevecdb-backed storage for file content and embeddings.

    Architecture:
    - Files stored as Documents with raw text (no compression)
    - HNSW index for O(log N) semantic similarity search
    - Metadata-based filtering for path lookups
    - TinyLFU eviction (frequency + recency) backed by an in-memory index
      that bootstraps from access_history metadata on first need
    """

    __slots__ = (
        "_db",
        "_collection",
        "_sync_collection",
        "_db_path",
        "_closed",
        "_io_executor",
        "_owns_executor",
        "_save_lock",
        "_index",
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
        # Use the sync VectorDB directly (not AsyncVectorDB) so we own the
        # executor: ONNX/usearch require a single-threaded executor to avoid
        # segfaults, and SemanticCache needs rebind_executor() to swap the
        # IO executor after a hung worker. AsyncVectorDB manages its own
        # internal pool and exposes neither hook. Async behavior is provided
        # by wrapping the sync collection in AsyncVectorCollection below.
        self._db = VectorDB(
            path=str(db_path),
            distance_strategy=DistanceStrategy.COSINE,
            quantization=self._QUANTIZATION,
        )
        self._closed = False
        # Mutex between save() (called from the IO executor during eviction)
        # and the close() daemon thread's final save. usearch's save is not
        # thread-safe, so the two cannot run concurrently.
        self._save_lock = threading.Lock()
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
        # In-memory eviction index. Bootstraps lazily on first eviction-need
        # so __init__ stays sync; the first scan after threshold is the only
        # O(N) read against the collection.
        self._index = TinyLFUIndex(
            capacity=MAX_CACHE_ENTRIES,
            history_size=ACCESS_HISTORY_SIZE,
        )
        # Write sentinel — removed on clean shutdown by _remove_sentinel().
        STARTUP_SENTINEL.touch()
        logger.info(f"VectorStorage initialized at {db_path}")

    def _reset_collection_sync(self, *, reason: str) -> None:
        """Drop and recreate the collection synchronously.

        Uses ``delete_collection`` which atomically drops the SQLite tables,
        FTS index, and usearch file in one call. The new wrapper for the
        recreated collection is reattached to
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
        # Index may not be initialized yet on cold-path resets (constructor
        # calls _reset_collection_sync before assigning _index). Tolerate
        # that case — the constructor builds a fresh index right after.
        index = getattr(self, "_index", None)
        if index is not None:
            index.clear()
        logger.info(f"Collection reset complete ({reason})")

    def rebind_executor(self, new_executor: Executor) -> None:
        """Swap the IO executor used by the async collection wrapper.

        Called by ``SemanticCache.reset_executor`` after a hung worker is
        abandoned. Re-creates the AsyncVectorCollection wrapper around the
        same sync collection so the new executor takes effect on subsequent
        calls. Sync VectorDB has no executor of its own, so nothing else
        needs to be touched.

        Ownership transfers OUT: the caller passing a replacement executor
        owns its lifecycle. If we previously owned the executor (no executor
        was injected at construction), we shut it down here so the worker
        thread isn't leaked, and flip ``_owns_executor`` to False so a later
        ``close()`` doesn't shut down the caller's replacement.
        """
        if self._owns_executor and self._io_executor is not new_executor:
            try:
                self._io_executor.shutdown(wait=False, cancel_futures=True)
            except Exception as exc:
                logger.debug(f"Old owned executor shutdown failed: {exc}")
        self._io_executor = new_executor
        self._owns_executor = False
        # Reuse the existing sync collection — we only need a fresh async
        # wrapper bound to the new executor.
        self._collection = AsyncVectorCollection(self._sync_collection, new_executor)

    def _build_collection(self) -> AsyncVectorCollection:
        """Open the ``files`` collection with ``store_embeddings=True`` and
        wrap it for async use.

        ``store_embeddings=True`` is required so ``get_embeddings_by_ids``
        can return the vectors used by ``SemanticCache.get()`` /
        ``compare_files()``. simplevecdb defaults this to False.
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

        self._closed = True
        close_error: BaseException | None = None

        def _do_close() -> None:
            nonlocal close_error
            with self._save_lock:
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
        content_bytes = content.encode("utf-8")
        content_hash = hash_content(content_bytes)
        tokens = count_tokens(content)
        now = time.time()
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

        # Pre-compute a stable preview from the start of the file so that
        # search results don't re-slice chunk content at query time (which
        # may yield partial-word or mid-comment slices for chunked docs).
        base_meta = {
            _META_PATH: path,
            _META_CONTENT_HASH: content_hash,
            _META_MTIME: mtime,
            _META_TOKENS: tokens,
            _META_CREATED_AT: now,
            _META_ACCESS_HISTORY: json.dumps([now]),
            _META_HAS_EMBEDDING: has_embedding,
            _META_PREVIEW: content[:_PREVIEW_CHARS],
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
        new_doc_ids: list[int] = []
        if not chunked:
            # Small file: single document
            meta = {
                **base_meta,
                _META_CHUNK_INDEX: 0,
                _META_TOTAL_CHUNKS: 1,
                _META_STORAGE_MODE: StorageMode.SINGLE_DOC.value,
            }
            ids = await self._collection.add_texts(
                texts=[content],
                metadatas=[meta],
                embeddings=[emb_list],
            )
            new_doc_ids = list(ids) if ids else []
            logger.debug(f"Stored {path} as single doc ({tokens} tokens)")
        else:
            # Large file: try chunked storage, fall back to single-doc if too many chunks
            stored_ids = await self._put_chunked(path, content, content_bytes, base_meta, emb_list)
            if not stored_ids:
                meta = {
                    **base_meta,
                    _META_CHUNK_INDEX: 0,
                    _META_TOTAL_CHUNKS: 1,
                    _META_STORAGE_MODE: StorageMode.SINGLE_DOC_FALLBACK.value,
                }
                ids = await self._collection.add_texts(
                    texts=[content],
                    metadatas=[meta],
                    embeddings=[emb_list],
                )
                new_doc_ids = list(ids) if ids else []
                logger.debug(f"Stored {path} as single doc (chunk fallback, {tokens} tokens)")
            else:
                new_doc_ids = list(stored_ids)
        # Mirror the new doc IDs into the in-memory eviction index. Only do
        # this once the index has been bootstrapped, to keep the very first
        # put cheap; once bootstrap fires, every subsequent put updates the
        # index synchronously without a DB scan.
        if self._index.loaded and new_doc_ids:
            self._index.upsert(path, new_doc_ids, now)
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
        file_embedding: Sequence[float],
    ) -> list[int]:
        """Store large file as parent doc + CDC-chunked children.

        Returns:
            list[int]: Stored doc IDs in insertion order — ``[parent_id,
            child_id_0, child_id_1, ...]`` on success. **Empty list signals
            fallback**: CDC produced more than ``_MAX_CHUNKS`` (500) chunks,
            so nothing was written and the caller MUST retry as single-doc
            storage to preserve full content. Treat the empty case as a
            silent contract — no exception is raised — so callers must not
            interpret ``[]`` as "stored zero docs".
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
            return []

        # Decode each chunk back to str, tracking line offsets and byte
        # lengths (used below for proportional per-chunk token estimates).
        chunk_texts: list[str] = []
        chunk_byte_lens: list[int] = []
        line_offset = 0
        chunk_line_starts: list[int] = []

        for chunk_b in chunks_bytes:
            text = chunk_b.decode("utf-8", errors="replace")
            chunk_texts.append(text)
            chunk_byte_lens.append(len(chunk_b))
            chunk_line_starts.append(line_offset)
            line_offset += text.count("\n")

        # Per-chunk token counts: estimate proportionally from the parent's
        # exact total instead of running the BPE encoder N times serially on
        # the single-threaded executor. The encoder dominates a chunked write
        # for large files. We preserve the sum invariant
        # (sum(chunk_tokens) == parent_tokens) by giving the final chunk the
        # remainder. Estimates remain stable for display/sort heuristics.
        parent_tokens = int(base_meta[_META_TOKENS])
        total_bytes_sum = sum(chunk_byte_lens) or 1
        chunk_token_estimates: list[int] = []
        running = 0
        for i, byte_len in enumerate(chunk_byte_lens):
            if i == total_chunks - 1:
                est = max(0, parent_tokens - running)
            else:
                est = (parent_tokens * byte_len) // total_bytes_sum
                running += est
            chunk_token_estimates.append(est)

        # Store parent document (holds file-level metadata, searchable by file embedding)
        parent_meta = {
            **base_meta,
            _META_IS_PARENT: True,
            _META_CHUNK_INDEX: -1,
            _META_TOTAL_CHUNKS: total_chunks,
            _META_STORAGE_MODE: StorageMode.CHUNKED.value,
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
        child_embeddings: list[Sequence[float]] = []
        # Module-level cache; same list object is shared across all child
        # rows. add_texts does not mutate, so this is safe.
        zero_emb = self._resolve_embedding(None)

        for i, _text in enumerate(chunk_texts):
            child_meta = {
                _META_PATH: path,
                _META_CHUNK_INDEX: i,
                _META_TOTAL_CHUNKS: total_chunks,
                _META_CONTENT_HASH: base_meta[_META_CONTENT_HASH],
                _META_MTIME: base_meta[_META_MTIME],
                _META_TOKENS: chunk_token_estimates[i],
                _META_CREATED_AT: base_meta[_META_CREATED_AT],
                _META_ACCESS_HISTORY: base_meta[_META_ACCESS_HISTORY],
                _META_STORAGE_MODE: StorageMode.CHUNKED.value,
                "line_start": chunk_line_starts[i],
            }
            child_metas.append(child_meta)
            # Chunks use zero embedding — similarity search uses the parent's
            # file-level embedding. Per-chunk embeddings would require N model
            # calls which is too expensive for the cache hot path.
            child_embeddings.append(zero_emb)

        child_ids = await self._collection.add_texts(
            texts=chunk_texts,
            metadatas=child_metas,
            embeddings=child_embeddings,
            parent_ids=[parent_id] * total_chunks,
        )

        logger.debug(
            f"Stored {path} as {total_chunks} chunks "
            f"(parent_id={parent_id}, {base_meta[_META_TOKENS]} tokens)"
        )
        return [parent_id, *(list(child_ids) if child_ids else [])]

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

    def _resolve_embedding(self, embedding: EmbeddingVector | None) -> Sequence[float]:
        """Validate dim and return a Sequence[float] suitable for simplevecdb.

        Skips the defensive ``list(embedding)`` copy: simplevecdb's
        ``add_texts`` / ``similarity_search`` accept any ``Sequence[float]``,
        and the call sites do not mutate the buffer. For the zero-vector path
        (``embedding is None``), the result is reused from a module-level
        per-dim cache to avoid re-allocating ``[0.0] * dim`` on every chunked
        write or unembedded put.
        """
        expected = self._expected_dim()
        if embedding is not None:
            actual = len(embedding)
            if actual != expected:
                raise ValueError(
                    f"Embedding dimension mismatch: got {actual}, expected {expected}. "
                    "This usually means the embedding model changed mid-session."
                )
            return embedding
        return _zero_embedding(expected)

    async def get_content(self, entry: CacheEntry, *, max_bytes: int | None = None) -> str:
        """Reassemble full text from simplevecdb chunks (sorted by chunk_index).

        Args:
            entry: Cache entry whose path identifies the stored content.
            max_bytes: Optional UTF-8 byte cap. ``None`` (default) returns the
                full reassembled text. When set, accumulation stops once the
                next whole chunk would push the running UTF-8 byte total past
                the cap; the partial chunk is then truncated on a UTF-8
                code-point boundary using ``errors="ignore"``.

        Notes:
            - The cap is in **bytes**, not characters or tokens. Multi-byte
              code points may make the returned string shorter than the cap
              even when more content exists.
            - Defense-in-depth for direct callers that bypass the
              ``MAX_CONTENT_SIZE`` gate enforced by ``cache/read.py``.
            - Parent docs (empty placeholders for chunked files) are filtered
              out before reassembly; only child chunks contribute text.
        """
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

        if max_bytes is None:
            return "".join(r[2] for r in children)

        parts: list[str] = []
        total = 0
        for _, _, text in children:
            chunk_bytes = len(text.encode("utf-8"))
            if total + chunk_bytes > max_bytes:
                # Truncate this chunk on a UTF-8 boundary, then stop.
                remaining = max_bytes - total
                if remaining > 0:
                    encoded = text.encode("utf-8")[:remaining]
                    parts.append(encoded.decode("utf-8", errors="ignore"))
                break
            parts.append(text)
            total += chunk_bytes
        return "".join(parts)

    async def record_access(self, path: str) -> None:
        if self._closed:
            return
        results = await self._find_docs_by_path(path)
        if not results:
            return

        now = time.time()

        # All chunks of a single path share one access history (set together
        # in put() and updated together here). Parse once, reserialize once,
        # then fan the same string out to every chunk doc — avoids N×
        # json.loads/dumps round-trips on every cached read.
        first_meta = results[0][1]
        history = json.loads(first_meta.get(_META_ACCESS_HISTORY, "[]"))
        history.append(now)
        history = history[-ACCESS_HISTORY_SIZE:]
        history_json = json.dumps(history)
        updates: list[tuple[int, dict]] = [
            (doc_id, {_META_ACCESS_HISTORY: history_json}) for doc_id, _meta, _text in results
        ]

        if updates:
            await self._collection.update_metadata(updates)
        # Mirror the access into the in-memory index so eviction has up-to-
        # date frequency/recency without re-reading metadata. Skip while the
        # index is unloaded — the bootstrap will replay history from the DB.
        if self._index.loaded:
            self._index.add_access(path, now)

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

        try:
            results = await self._collection.similarity_search(
                query=embedding,
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
            # Defense-in-depth: chunk children carry zero embeddings; a query
            # with near-zero magnitude could match them as false positives if
            # `has_embedding` is ever stale (schema migration, partial write).
            is_chunk_child = (
                doc.metadata.get(_META_CHUNK_INDEX, -1) >= 0
                and not doc.metadata.get(_META_IS_PARENT, False)
                and doc.metadata.get(_META_TOTAL_CHUNKS, 1) > 1
            )

            is_match = (
                has_emb
                and not is_chunk_child
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

        try:
            results = await self._collection.similarity_search(
                query=embedding,
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
            is_chunk_child = (
                doc.metadata.get(_META_CHUNK_INDEX, -1) >= 0
                and not doc.metadata.get(_META_IS_PARENT, False)
                and doc.metadata.get(_META_TOTAL_CHUNKS, 1) > 1
            )

            if (
                has_emb
                and not is_chunk_child
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

        try:
            results = await self._collection.hybrid_search(
                query,
                k=k * 2,
                filter=filter,
                query_vector=embedding,
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
            preview = doc.metadata.get(_META_PREVIEW) or doc.page_content[:_PREVIEW_CHARS]
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

        # BM25 prefilter: avoid the full-collection scan when the pattern
        # has usable literal tokens. Only when no tokens are extractable
        # (e.g., regex like `[A-Z][a-z]+`) or the BM25 lookup fails do we
        # fall back to scanning every doc. The need_full_scan flag is
        # explicit because the prefilter has *three* terminal states
        # (success / confidently empty / errored), and the dispatcher must
        # not conflate "BM25 errored" with "BM25 confident no match".
        keyword_query = self._grep_prefilter_query(pattern, fixed_string=fixed_string)
        files: dict[str, list[tuple[int, str]]] = {}
        need_full_scan = True
        if keyword_query is not None:
            candidates = await self._grep_candidate_paths(
                keyword_query, max_files=max_files, path_filter=path
            )
            if candidates is not None:
                if candidates:
                    # Positive matches: regex post-filter on the targeted set
                    # is exact, so no full scan is needed.
                    need_full_scan = False
                    files = await self._grep_load_files(candidates)
                elif self._grep_empty_is_authoritative(pattern):
                    # Pattern's longest token is long enough that BM25's empty
                    # answer cannot hide a substring inside a larger document
                    # token; trust it and short-circuit.
                    return []
                # Otherwise (empty + not authoritative): full scan certifies.
            # candidates is None ⇒ BM25 errored; full scan stays armed.
        if need_full_scan:
            all_docs = await self._collection.get_documents()
            for _doc_id, text, meta in all_docs:
                if meta.get(_META_IS_PARENT, False):
                    continue  # Parent docs have empty content
                doc_path = meta.get(_META_PATH, "")
                if not doc_path or not self._grep_path_matches(doc_path, path_filter=path):
                    continue
                chunk_idx = meta.get(_META_CHUNK_INDEX, 0)
                files.setdefault(doc_path, []).append((chunk_idx, text))

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

    # Two thresholds govern the BM25 prefilter, both tuned to preserve
    # grep's substring semantics over FTS5's whole-token semantics:
    #
    #   _GREP_PREFILTER_MIN_TOKEN_LEN = minimum token length to *engage*
    #     the prefilter at all. Below this, short tokens like "foo" are
    #     too likely to be substrings of larger document tokens (e.g.,
    #     "needlefoo") for the FTS5 query to be safely informative.
    #
    #   _GREP_PREFILTER_AUTHORITATIVE_LEN = the longest token in the pattern
    #     must be ≥ this for an *empty* BM25 result to be trusted as a true
    #     "no match anywhere". For shorter tokens BM25 returning 0 may still
    #     hide a substring inside a longer token, so we fall back to the
    #     full scan to certify the empty answer.
    #
    # Net behavior:
    #   * pattern with tokens of length < 4 → never engage prefilter
    #   * pattern with longest token ≥ 8 and BM25 returns candidates → use them
    #   * pattern with longest token ≥ 8 and BM25 returns []     → return [] fast
    #   * pattern with all tokens in [4, 7] and BM25 returns []  → full-scan to
    #     check for substrings inside longer document tokens
    _GREP_MIN_TOKEN_LEN = 2
    _GREP_PREFILTER_MIN_TOKEN_LEN = 4
    _GREP_PREFILTER_AUTHORITATIVE_LEN = 8

    @staticmethod
    def _grep_prefilter_query(pattern: str, *, fixed_string: bool) -> str | None:
        """Build an FTS5 query from a grep pattern, or None if not feasible.

        Extracts contiguous alphanumeric runs (length ≥ ``_GREP_MIN_TOKEN_LEN``)
        from the raw pattern — works for both literal strings and regex bodies
        because regex metacharacters are non-alphanumeric and break up the runs.
        Each token is wrapped as an FTS5 phrase so reserved words (``NOT``,
        ``OR``, ``AND``) and double-quote chars are not interpreted as syntax.

        Returns ``None`` when:
          * no alphanumeric run of length ≥ ``_GREP_MIN_TOKEN_LEN`` exists, or
          * the *shortest* extracted token is below
            ``_GREP_PREFILTER_MIN_TOKEN_LEN`` — short tokens may legitimately
            appear as substrings inside longer document tokens that BM25
            would not return, breaking grep's substring semantics.

        In both cases the caller falls back to the full collection scan.
        """
        import re as _re  # noqa: PLC0415

        del fixed_string  # extraction is the same: pull literal alphanumerics
        tokens = _re.findall(rf"[A-Za-z0-9]{{{VectorStorage._GREP_MIN_TOKEN_LEN},}}", pattern)
        if not tokens:
            return None
        if min(len(t) for t in tokens) < VectorStorage._GREP_PREFILTER_MIN_TOKEN_LEN:
            return None
        # Phrase-quote each token; FTS5 implicit AND combines them.
        return " ".join(f'"{t}"' for t in tokens)

    @staticmethod
    def _grep_empty_is_authoritative(pattern: str) -> bool:
        """Whether a 0-result BM25 reply can be trusted as a true 'no match'.

        A long token is implausible as a substring of an even longer
        alphanumeric run in real-world documents, so BM25 returning empty
        is reliably "no occurrences anywhere". For shorter tokens we cannot
        rule out a substring hidden inside a larger document token (e.g.,
        ``"nicel"`` inside ``"mynicelongword"``) so the caller must full-scan.
        """
        import re as _re  # noqa: PLC0415

        tokens = _re.findall(rf"[A-Za-z0-9]{{{VectorStorage._GREP_MIN_TOKEN_LEN},}}", pattern)
        if not tokens:
            return False
        return max(len(t) for t in tokens) >= VectorStorage._GREP_PREFILTER_AUTHORITATIVE_LEN

    # Hard cap on the BM25 over-fetch so an unusually large `max_files`
    # (clamped at _GREP_MAX_FILES = 500) doesn't translate into a 2k-result
    # FTS5 query. 4× headroom on max_files is plenty to absorb regex
    # post-filter false positives.
    _GREP_PREFILTER_FETCH_CAP = 1000

    async def _grep_candidate_paths(
        self,
        keyword_query: str,
        *,
        max_files: int,
        path_filter: str | None,
    ) -> list[str] | None:
        """BM25 lookup returning unique candidate paths for grep.

        Returns ``None`` when the keyword search itself errors (signals the
        caller to fall back to a full scan). Returns a possibly-empty list
        otherwise; an empty list means BM25 is confident no file contains
        the tokens and grep can return immediately.
        """
        fetch_k = min(max_files * 4, self._GREP_PREFILTER_FETCH_CAP)
        try:
            results = await self._collection.keyword_search(keyword_query, k=fetch_k)
        except Exception as exc:
            logger.debug(f"grep BM25 prefilter failed: {exc}; falling back to scan")
            return None
        seen: set[str] = set()
        candidates: list[str] = []
        for doc, _score in results:
            meta = doc.metadata
            if meta.get(_META_IS_PARENT, False):
                continue
            doc_path = meta.get(_META_PATH, "")
            if not doc_path or doc_path in seen:
                continue
            if not self._grep_path_matches(doc_path, path_filter=path_filter):
                continue
            seen.add(doc_path)
            candidates.append(doc_path)
        return candidates

    async def _grep_load_files(self, paths: list[str]) -> dict[str, list[tuple[int, str]]]:
        """Load chunk text for a set of paths in a single batched lookup.

        Uses simplevecdb's list-value filter, which compiles to
        ``json_extract(metadata, '$.path') IN (?, ?, ...)`` — one round trip
        through the executor instead of N. Falls back to per-path lookups
        only if the batch query itself errors.
        """
        files: dict[str, list[tuple[int, str]]] = {}
        if not paths:
            return files

        try:
            docs = await self._collection.get_documents(
                filter_dict={_META_PATH: list(paths)},
            )
        except Exception as e:
            logger.debug(f"Batched grep lookup failed ({len(paths)} paths): {e}")
            for path in paths:
                results = await self._find_docs_by_path(path)
                for _doc_id, meta, text in results:
                    if meta.get(_META_IS_PARENT, False):
                        continue
                    chunk_idx = meta.get(_META_CHUNK_INDEX, 0)
                    files.setdefault(path, []).append((chunk_idx, text))
            return files

        for _doc_id, text, meta in docs:
            if meta.get(_META_IS_PARENT, False):
                continue
            doc_path = meta.get(_META_PATH)
            if doc_path is None:
                continue
            chunk_idx = meta.get(_META_CHUNK_INDEX, 0)
            files.setdefault(doc_path, []).append((chunk_idx, text))
        return files

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
        index = getattr(self, "_index", None)
        if index is not None:
            index.clear()
        return count

    async def clear(self) -> int:
        """Clear all cache entries. Returns count of documents removed.

        Uses ``delete_collection`` to atomically drop the SQLite tables, FTS
        index, and usearch file in one call, then rebuilds
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
        self._index.clear()
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
            # Index may still hold a stale entry if a prior put failed mid-way.
            self._index.remove(path)
            return 0

        doc_ids = [r[0] for r in results]
        # Symmetric race-closing as in _evict_if_needed: drop from the index
        # before the DB delete so a concurrent put on the same path cannot
        # have its fresh entry wiped by our trailing remove. mark_dirty here
        # is meaningful — the index has actually been mutated, and a failed
        # DB delete leaves the two genuinely divergent.
        self._index.remove(path)
        try:
            await self._collection.delete_by_ids(doc_ids)
        except Exception:
            self._index.mark_dirty()
            raise
        return len(doc_ids)

    async def _evict_if_needed(self) -> None:
        """Evict via TinyLFU when unique-file count exceeds MAX_CACHE_ENTRIES.

        Cheap doc-count gate first (no scan when under cap). When over, the
        in-memory `TinyLFUIndex` drives the choice — it bootstraps once from
        the collection on first use, then every subsequent put updates it
        synchronously with no DB scan. Frequency comes from a 4-bit Count-Min
        sketch with periodic halving; recency comes from the OrderedDict.
        """
        # Read the cap via explicit module attribute access so tests can
        # patch `config.MAX_CACHE_ENTRIES` and have it observed live (the
        # module-top `from ...config import MAX_CACHE_ENTRIES` snapshot is
        # only used by the constructor at instance-creation time).
        cap = config.MAX_CACHE_ENTRIES

        doc_count = await self._collection.count()
        if doc_count <= cap:
            return

        # Lazy bootstrap. Captures the executor closure here so the index
        # module stays decoupled from AsyncVectorCollection's exact API.
        if not self._index.loaded:
            await self._index.ensure_loaded(self._collection.get_documents)

        if self._index.total_paths() <= cap:
            return

        evict_count = max(1, self._index.total_paths() // 10)
        victims = self._index.select_evictions(evict_count)
        if not victims:
            return

        ids_to_delete: list[int] = []
        for _path, doc_ids in victims:
            ids_to_delete.extend(doc_ids)

        # Remove from the index BEFORE the DB delete to close a race window:
        # if the order were inverted, a concurrent `put(victim_path)` between
        # the DB delete and the index removal would re-insert the path with
        # fresh doc IDs, and the trailing index removal would then wipe the
        # *freshly-inserted* entry. With the order below, the worst-case
        # interleaving has a concurrent put delete the same docs we were
        # about to delete (idempotent at the DB level) and re-insert under
        # the new IDs — the index stays in sync with the DB.
        for path, _ in victims:
            self._index.remove(path)
        try:
            await self._collection.delete_by_ids(ids_to_delete)
        except Exception:
            # Index has dropped the victim entries but the DB still holds
            # their docs — re-bootstrap re-discovers them on the next
            # eviction-need rather than leaking them silently.
            self._index.mark_dirty()
            raise
        await self._collection.save()
        logger.info(
            f"Cache eviction: removed {len(victims)} files "
            f"({len(ids_to_delete)} documents) via TinyLFU"
        )

    def save(self) -> None:
        """Persist index to disk.

        Guarded by `_save_lock` against the close() daemon thread's final
        save: usearch's save is not thread-safe, and a concurrent save from
        eviction (running on the IO executor) racing with close() would
        corrupt the heap. The lock also covers a re-check of `_closed` so
        we never write after close() has begun tearing down the DB.
        """
        with self._save_lock:
            if self._closed:
                return
            self._sync_collection.save()
            self._db.save()
