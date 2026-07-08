"""SQLite + FTS5 storage: raw text + HyperCDC chunking for large files."""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ... import config
from ...config import (
    ACCESS_HISTORY_SIZE,
    CACHE_DIR,
    CHUNK_MIN_SIZE,
    MAX_CACHE_ENTRIES,
)
from ...core.chunking import get_optimal_chunker
from ...core.hashing import hash_chunk, hash_content
from ...core.tokenizer import count_tokens
from ...logger import log_marker
from ...types import CacheEntry
from ._docstore import AsyncDocStore, DocStore
from ._tinylfu import TinyLFUIndex

logger = logging.getLogger(__name__)

__all__ = ["CHUNK_THRESHOLD", "StorageMode", "CONTENT_DB_PATH", "ContentStorage"]

CONTENT_DB_PATH = CACHE_DIR / "docstore.db"

# Metadata keys stored with each document
_META_PATH = "path"
_META_CHUNK_INDEX = "chunk_index"
_META_TOTAL_CHUNKS = "total_chunks"
_META_CONTENT_HASH = "content_hash"
_META_MTIME = "mtime"
_META_TOKENS = "tokens"
_META_CREATED_AT = "created_at"
_META_ACCESS_HISTORY = "access_history"
_META_IS_PARENT = "is_parent"
_META_PREVIEW = "preview"
_META_STORAGE_MODE = "storage_mode"
_META_CHUNK_HASH = "chunk_hash"
_META_MANIFEST = "manifest"
_PREVIEW_CHARS = 200

# Safety cap: when CDC produces more than this many chunks, store the file as a
# single unchunked document instead of exploding the doc table.
_MAX_CHUNKS = 500


class StorageMode(enum.StrEnum):
    """How a file's content was stored. Surfaces retrieval-quality signals.

    SINGLE_DOC          — file fits in one doc.
    CHUNKED             — file split via CDC into per-chunk child docs.
    SINGLE_DOC_FALLBACK — chunk count exceeded the cap; whole file stored as
                          one doc. Retrieval granularity is degraded vs CHUNKED.
    """

    SINGLE_DOC = "single_doc"
    CHUNKED = "chunked"
    SINGLE_DOC_FALLBACK = "single_doc_fallback"


@dataclass(slots=True)
class _ChunkPlan:
    """A file cut into content-addressed chunks, ready to store or reconcile.

    ``hashes`` is the file's manifest: the ordered list of per-chunk BLAKE3
    hashes. Two writes that share a chunk's bytes share its hash, which is what
    lets ``_reconcile_chunks`` reuse an existing row instead of re-inserting and
    re-indexing it.
    """

    texts: list[str]
    hashes: list[str]
    token_estimates: list[int]
    line_starts: list[int]
    total: int


# Files larger than this (bytes) are split via HyperCDC into multiple chunks,
# each stored as a child document. Smaller files are stored as a single
# document. CHUNK_MIN_SIZE (2KB) is the CDC minimum chunk size.
CHUNK_THRESHOLD = CHUNK_MIN_SIZE * 4  # 8KB — files below this stay as one doc


class ContentStorage:
    """SQLite + FTS5 backed storage for file content.

    Architecture:
    - Files stored as documents with raw text (no compression)
    - FTS5 BM25 keyword search + metadata filtering for path lookups
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
        "_has_cached_cache",
    )

    _COLLECTION_NAME = "files"

    def __init__(
        self,
        db_path: Path = CONTENT_DB_PATH,
        *,
        executor: Executor | None = None,
    ) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._migrate_legacy_db(db_path)
        # Sync DocStore (SQLite + FTS5). Async behaviour is provided by wrapping
        # it in AsyncDocStore (runs each call on the IO executor) below, so
        # SemanticCache.rebind_executor() can swap the executor after a hung
        # worker without recreating the store.
        self._db = DocStore(db_path)
        self._closed = False
        # TTL cache keyed by path_filter for has_cached_paths_under (see
        # comment there). Initialized here so the slot is always present.
        self._has_cached_cache: dict[str, tuple[float, bool]] = {}
        # Mutex between save() (called from the IO executor during eviction)
        # and the close() daemon thread's final save, so the two never run
        # concurrently on the shared connection.
        self._save_lock = threading.Lock()
        # DocStore has no executor, so we own one when none is injected.
        # Single-thread by default — all storage I/O serializes through it, and
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
        # One-time wipe of a pre-manifest chunked cache so the store is uniformly
        # in the content-addressed format (see method).
        self._migrate_chunk_manifest()
        # In-memory eviction index. Bootstraps lazily on first eviction-need
        # so __init__ stays sync; the first scan after threshold is the only
        # O(N) read against the collection.
        self._index = TinyLFUIndex(
            capacity=MAX_CACHE_ENTRIES,
            history_size=ACCESS_HISTORY_SIZE,
        )
        logger.info(f"ContentStorage initialized at {db_path}")

    def _reset_collection_sync(self, *, reason: str) -> None:
        """Drop all documents and rebuild the async wrapper synchronously.

        Safe to call from sync startup paths — no event loop required.
        """
        try:
            self._db.clear()
        except Exception as exc:
            logger.warning(f"clear failed during {reason}: {exc}")
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
        abandoned. Re-creates the AsyncDocStore wrapper around the same
        DocStore so the new executor takes effect on subsequent calls.

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
        # Reuse the existing sync store — we only need a fresh async wrapper
        # bound to the new executor.
        self._collection = AsyncDocStore(self._sync_collection, new_executor)

    def _build_collection(self) -> AsyncDocStore:
        """Wrap the DocStore for async use on the IO executor.

        Stores a direct reference to the sync store for sync code paths
        (``save``, ``_clear_sync``) so they don't reach through the async
        wrapper.
        """
        self._sync_collection = self._db
        return AsyncDocStore(self._db, self._io_executor)

    @classmethod
    def _migrate_legacy_db(cls, db_path: Path) -> None:
        """One-time wipe of the legacy ``vecdb.db`` cache on upgrade.

        Earlier releases stored the cache in ``vecdb.db`` (simplevecdb + usearch,
        then an interim FTS build). The current store is ``docstore.db`` with a
        different schema, so delete any stale ``vecdb.db`` files once, guarded by
        a sentinel. The cache is fully rebuildable and repopulates on demand;
        new installs just create the sentinel with nothing to delete.
        """
        marker = db_path.parent / ".docstore_v1"
        if marker.exists():
            return
        legacy_db = db_path.parent / "vecdb.db"
        legacy = [
            legacy_db,
            legacy_db.with_name("vecdb.db-wal"),
            legacy_db.with_name("vecdb.db-shm"),
            legacy_db.with_name(f"vecdb.db.{cls._COLLECTION_NAME}.usearch"),
            legacy_db.with_name("vecdb.db.meta.json"),
            db_path.parent / ".vectors_removed",
            db_path.parent / ".fts_store_v1",
        ]
        for target in legacy:
            if target.exists():
                target.unlink()
                logger.info(f"Migration: deleted legacy storage file {target}")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("docstore v1\n")

    def _has_premanifest_chunks(self) -> bool:
        """True if the store holds a chunked parent doc with no manifest — the
        pre-chunk-hash format that predates content-addressed reconciliation."""
        for _doc_id, _text, meta in self._sync_collection.get_documents():
            if meta.get(_META_IS_PARENT, False) and _META_MANIFEST not in meta:
                return True
        return False

    def _migrate_chunk_manifest(self) -> None:
        """One-time wipe of a pre-manifest docstore on upgrade.

        Chunked entries written before chunk-level content addressing carry no
        ``manifest`` on the parent and no ``chunk_hash`` on children. Reconcile
        now upgrades them correctly, but clearing once on first startup keeps the
        store uniformly in the new format and skips the one-time re-write churn.
        Guarded by a sentinel; the cache repopulates on demand. Single-doc caches
        have no parent doc, so they are left untouched.
        """
        marker = self._db_path.parent / ".docstore_manifest_v1"
        if marker.exists():
            return
        try:
            if self._has_premanifest_chunks():
                removed = self._db.clear()
                logger.info(f"Migration: cleared {removed} docs from a pre-manifest cache")
        except Exception as exc:
            logger.warning(f"chunk-manifest migration check failed: {exc}")
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("manifest v1\n")
        except OSError as exc:
            logger.debug(f"could not write manifest migration marker: {exc}")

    def close(self, *, timeout: float = 5.0) -> None:
        """Save and close the database with a deadline.

        Uses a background thread so a hung SQLite save cannot block the
        asyncio event loop or delay process exit past *timeout* seconds.
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
                f"ContentStorage close timed out after {timeout}s — "
                "index may need recovery on next startup"
            )
        elif close_error is not None:
            logger.warning(f"ContentStorage close error: {close_error}")
        else:
            logger.debug("ContentStorage closed cleanly")

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
        meta = results[0][1]
        for _doc_id, m, _text in results:
            if m.get(_META_IS_PARENT, False):
                meta = m
                break

        access_history = json.loads(meta.get(_META_ACCESS_HISTORY, "[]"))

        return CacheEntry(
            path=meta[_META_PATH],
            content_hash=meta[_META_CONTENT_HASH],
            mtime=meta[_META_MTIME],
            tokens=meta[_META_TOKENS],
            created_at=meta[_META_CREATED_AT],
            access_history=access_history,
        )

    async def put(
        self,
        path: str,
        content: str,
        mtime: float,
    ) -> None:
        """Store file as raw text. Large files (>8KB) are HyperCDC-chunked."""
        if self._closed:
            return
        started = time.perf_counter()
        content_bytes = content.encode("utf-8")
        content_hash = hash_content(content_bytes)
        tokens = count_tokens(content)
        now = time.time()
        chunked = len(content_bytes) >= CHUNK_THRESHOLD

        log_marker(
            logger,
            "docstore.put.begin",
            path=path,
            tokens=tokens,
            bytes=len(content_bytes),
            chunked=chunked,
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
            _META_PREVIEW: content[:_PREVIEW_CHARS],
        }

        # Load the current docs for this path once. A chunked re-write reuses
        # the rows whose chunk bytes are unchanged (addressed by chunk_hash)
        # rather than deleting and re-indexing every chunk — the manifest design.
        existing = await self._find_docs_by_path(path)
        parent_id: int | None = None
        for doc_id, meta, _text in existing:
            if meta.get(_META_IS_PARENT, False):
                parent_id = doc_id
                break

        add_started = time.perf_counter()
        log_marker(logger, "docstore.put.add.begin", path=path, chunked=chunked)
        new_doc_ids: list[int] = []
        try:
            if not chunked:
                new_doc_ids = await self._store_single(
                    path, content, base_meta, existing, fallback=False
                )
                logger.debug(f"Stored {path} as single doc ({tokens} tokens)")
            else:
                plan = self._plan_chunks(content_bytes, tokens)
                if plan is None:
                    logger.warning(
                        f"File {path} exceeded {_MAX_CHUNKS} chunks; "
                        "storing as a single doc to preserve full content"
                    )
                    new_doc_ids = await self._store_single(
                        path, content, base_meta, existing, fallback=True
                    )
                elif parent_id is not None:
                    new_doc_ids = await self._reconcile_chunks(
                        path, plan, base_meta, existing, parent_id
                    )
                else:
                    if existing:
                        await self._delete_by_path(path)
                    new_doc_ids = await self._insert_chunks_fresh(path, plan, base_meta)
        except Exception:
            # A store that fails partway can leave the DB and the in-memory
            # eviction index divergent (rows deleted/inserted, but the index not
            # yet upserted below). Flag the index dirty so the next eviction-need
            # re-bootstraps from the DB instead of trusting a stale path→doc_ids
            # map — the same recovery discipline _delete_by_path and
            # _evict_if_needed apply when their delete fails.
            logger.exception(f"docstore put failed for {path}; marking index dirty")
            self._index.mark_dirty()
            raise
        # Mirror the new doc IDs into the in-memory eviction index. Only do
        # this once the index has been bootstrapped, to keep the very first
        # put cheap; once bootstrap fires, every subsequent put updates the
        # index synchronously without a DB scan.
        if self._index.loaded and new_doc_ids:
            self._index.upsert(path, new_doc_ids, now)
        log_marker(
            logger,
            "docstore.put.add.end",
            path=path,
            chunked=chunked,
            elapsed_ms=round((time.perf_counter() - add_started) * 1000, 1),
        )

        evict_started = time.perf_counter()
        log_marker(logger, "docstore.put.evict.begin", path=path)
        await self._evict_if_needed()
        log_marker(
            logger,
            "docstore.put.evict.end",
            path=path,
            elapsed_ms=round((time.perf_counter() - evict_started) * 1000, 1),
        )
        log_marker(
            logger,
            "docstore.put.end",
            path=path,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    async def _store_single(
        self,
        path: str,
        content: str,
        base_meta: dict,
        existing: list[tuple[int, dict, str]],
        *,
        fallback: bool,
    ) -> list[int]:
        """Store the whole file as one document, clearing any prior rows first.

        Used for files below the chunk threshold and for the over-cap fallback
        (``fallback=True`` tags the doc ``single_doc_fallback`` so retrieval
        quality is observable).
        """
        if existing:
            await self._delete_by_path(path)
        mode = StorageMode.SINGLE_DOC_FALLBACK if fallback else StorageMode.SINGLE_DOC
        meta = {
            **base_meta,
            _META_CHUNK_INDEX: 0,
            _META_TOTAL_CHUNKS: 1,
            _META_STORAGE_MODE: mode.value,
        }
        ids = await self._collection.add_texts(texts=[content], metadatas=[meta])
        return list(ids) if ids else []

    def _plan_chunks(self, content_bytes: bytes, parent_tokens: int) -> _ChunkPlan | None:
        """Cut ``content_bytes`` into CDC chunks with per-chunk hashes and token
        estimates, or return ``None`` when the count exceeds ``_MAX_CHUNKS`` (the
        caller then stores the file as a single doc to preserve full content).
        """
        chunker = get_optimal_chunker(prefer_simd=True)
        chunks_bytes = list(chunker(content_bytes))
        total = len(chunks_bytes)
        if total > _MAX_CHUNKS:
            return None

        texts: list[str] = []
        hashes: list[str] = []
        byte_lens: list[int] = []
        line_starts: list[int] = []
        line_offset = 0
        for chunk_b in chunks_bytes:
            text = chunk_b.decode("utf-8", errors="replace")
            texts.append(text)
            hashes.append(hash_chunk(chunk_b))
            byte_lens.append(len(chunk_b))
            line_starts.append(line_offset)
            line_offset += text.count("\n")

        # Per-chunk token counts estimated proportionally from the parent's exact
        # total instead of running the BPE encoder N times on the single IO
        # thread. Preserves sum(chunk_tokens) == parent_tokens by giving the last
        # chunk the remainder.
        total_bytes_sum = sum(byte_lens) or 1
        token_estimates: list[int] = []
        running = 0
        for i, byte_len in enumerate(byte_lens):
            if i == total - 1:
                est = max(0, parent_tokens - running)
            else:
                est = (parent_tokens * byte_len) // total_bytes_sum
                running += est
            token_estimates.append(est)

        return _ChunkPlan(
            texts=texts,
            hashes=hashes,
            token_estimates=token_estimates,
            line_starts=line_starts,
            total=total,
        )

    def _child_meta(self, path: str, i: int, plan: _ChunkPlan, base_meta: dict) -> dict:
        """Metadata for chunk ``i`` — carries its own ``chunk_hash`` so a later
        re-write can content-address it for reuse."""
        return {
            _META_PATH: path,
            _META_CHUNK_INDEX: i,
            _META_TOTAL_CHUNKS: plan.total,
            _META_CHUNK_HASH: plan.hashes[i],
            _META_CONTENT_HASH: base_meta[_META_CONTENT_HASH],
            _META_MTIME: base_meta[_META_MTIME],
            _META_TOKENS: plan.token_estimates[i],
            _META_CREATED_AT: base_meta[_META_CREATED_AT],
            _META_ACCESS_HISTORY: base_meta[_META_ACCESS_HISTORY],
            _META_STORAGE_MODE: StorageMode.CHUNKED.value,
            "line_start": plan.line_starts[i],
        }

    def _parent_meta(self, plan: _ChunkPlan, base_meta: dict) -> dict:
        """File-level parent metadata, including the manifest (ordered chunk
        hashes) — the explicit recipe a reconcile diffs against."""
        return {
            **base_meta,
            _META_IS_PARENT: True,
            _META_CHUNK_INDEX: -1,
            _META_TOTAL_CHUNKS: plan.total,
            _META_STORAGE_MODE: StorageMode.CHUNKED.value,
            _META_MANIFEST: plan.hashes,
        }

    async def _insert_chunks_fresh(self, path: str, plan: _ChunkPlan, base_meta: dict) -> list[int]:
        """First-time chunked store: one empty parent doc + one child per chunk."""
        parent_ids = await self._collection.add_texts(
            texts=[""],  # Parent has no content — children hold raw text
            metadatas=[self._parent_meta(plan, base_meta)],
        )
        parent_id = parent_ids[0]
        child_metas = [self._child_meta(path, i, plan, base_meta) for i in range(plan.total)]
        child_ids = await self._collection.add_texts(
            texts=plan.texts,
            metadatas=child_metas,
            parent_ids=[parent_id] * plan.total,
        )
        logger.debug(
            f"Stored {path} as {plan.total} chunks "
            f"(parent_id={parent_id}, {base_meta[_META_TOKENS]} tokens)"
        )
        return [parent_id, *(list(child_ids) if child_ids else [])]

    async def _reconcile_chunks(
        self,
        path: str,
        plan: _ChunkPlan,
        base_meta: dict,
        existing: list[tuple[int, dict, str]],
        parent_id: int,
    ) -> list[int]:
        """Re-store a chunked file, reusing rows whose chunk bytes are unchanged.

        Pools the existing children by their stored ``chunk_hash``; each new
        chunk that hits the pool reuses that row via a cheap metadata refresh
        (no FTS re-index), each miss is inserted, and leftover rows are deleted.
        A small edit to a large file thus rewrites ~1 chunk instead of all of
        them — the write-amplification this design exists to remove.

        Non-atomic across its three store calls, which is fine for a cache: a
        crash mid-reconcile leaves an entry whose content_hash no longer matches
        disk, and the next read transparently re-puts it.
        """
        existing_child_ids: list[int] = []
        pool: defaultdict[str, deque[int]] = defaultdict(deque)
        for doc_id, meta, _text in existing:
            if meta.get(_META_IS_PARENT, False):
                continue
            existing_child_ids.append(doc_id)
            chunk_hash = meta.get(_META_CHUNK_HASH)
            if chunk_hash is not None:
                pool[chunk_hash].append(doc_id)

        reused_updates: list[tuple[int, dict]] = []
        reused_ids: list[int] = []
        insert_texts: list[str] = []
        insert_metas: list[dict] = []
        for i in range(plan.total):
            child_meta = self._child_meta(path, i, plan, base_meta)
            available = pool.get(plan.hashes[i])
            if available:
                reused_id = available.popleft()
                reused_ids.append(reused_id)
                reused_updates.append((reused_id, child_meta))
            else:
                insert_texts.append(plan.texts[i])
                insert_metas.append(child_meta)

        # Every existing child not reused must be deleted — including legacy
        # rows with no chunk_hash, which never entered the pool. A pool-only
        # leftover would orphan them and duplicate the file against the freshly
        # inserted chunks.
        reused_set = set(reused_ids)
        leftover = [doc_id for doc_id in existing_child_ids if doc_id not in reused_set]

        # Drop stale rows, insert only genuinely-new chunks (the sole FTS work),
        # then refresh metadata for the parent and every reused child (text
        # untouched → their FTS rows are never re-tokenized).
        if leftover:
            await self._collection.delete_by_ids(leftover)
        new_ids: list[int] = []
        if insert_texts:
            ids = await self._collection.add_texts(
                texts=insert_texts,
                metadatas=insert_metas,
                parent_ids=[parent_id] * len(insert_texts),
            )
            new_ids = list(ids) if ids else []
        await self._collection.update_metadata(
            [(parent_id, self._parent_meta(plan, base_meta)), *reused_updates]
        )
        logger.debug(
            f"Reconciled {path}: {plan.total} chunks "
            f"({len(reused_ids)} reused, {len(new_ids)} new, {len(leftover)} dropped)"
        )
        return [parent_id, *reused_ids, *new_ids]

    async def get_content(self, entry: CacheEntry, *, max_bytes: int | None = None) -> str:
        """Reassemble full text from stored chunks (sorted by chunk_index).

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
        # access_history is semi-trusted DB metadata: a corrupt or non-list
        # value must not crash a cache-hit read (record_access is awaited
        # bare on that path). Parse defensively and keep only numeric
        # entries, matching TinyLFUIndex._parse_history.
        try:
            parsed = json.loads(first_meta.get(_META_ACCESS_HISTORY, "[]"))
        except (ValueError, TypeError):
            parsed = []
        history: list[float] = (
            [t for t in parsed if isinstance(t, (int, float))] if isinstance(parsed, list) else []
        )
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
        """Update cached mtime without re-storing content.

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
    # Query-based search
    # -------------------------------------------------------------------------

    async def search_by_query(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
    ) -> list[tuple[str, str, float]]:
        """BM25 keyword search over cached content. FTS5 syntax supported."""
        return await _search.search_by_query(self, query, k=k, filter=filter)

    # -------------------------------------------------------------------------
    # Grep — regex/literal content search across cached files
    # -------------------------------------------------------------------------

    # Short TTL cache for has_cached_paths_under — the call is a fallback on
    # empty grep results, and a wrong `path` argument from a model can fire
    # it many times in a row. 5s is long enough to dedupe a burst, short
    # enough that newly-cached files appear quickly.
    _HAS_CACHED_TTL_S = 5.0

    async def has_cached_paths_under(self, path_filter: str | None) -> bool:
        """Return True if any cached document matches `path_filter`.

        Used by the grep tool to distinguish "no files cached under path"
        from "pattern produced no matches". `None`/empty filter is treated
        as "anything cached" — short-circuits on the first non-parent doc.
        """
        if self._closed:
            return False
        import time

        key = path_filter or ""
        ttl_cache: dict[str, tuple[float, bool]] = self._has_cached_cache
        now = time.monotonic()
        entry = ttl_cache.get(key)
        if entry is not None and now - entry[0] < self._HAS_CACHED_TTL_S:
            return entry[1]

        all_docs = await self._collection.get_documents()
        if not path_filter:
            result = any(not meta.get(_META_IS_PARENT, False) for _id, _text, meta in all_docs)
        else:
            matcher = _grep.path_matches
            result = False
            for _doc_id, _text, meta in all_docs:
                if meta.get(_META_IS_PARENT, False):
                    continue
                doc_path = meta.get(_META_PATH, "")
                if doc_path and matcher(doc_path, path_filter=path_filter):
                    result = True
                    break
        ttl_cache[key] = (now, result)
        # Bound memory: keep the map small — bursty wrong paths typically
        # share a small alphabet of distinct keys.
        if len(ttl_cache) > 64:
            oldest = min(ttl_cache, key=lambda k: ttl_cache[k][0])
            del ttl_cache[oldest]
        return result

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

        Unlike search, returns line numbers and context, not ranked scores.
        Implementation lives in ``_grep`` to keep this module from becoming a god-object.
        """
        return await _grep.grep(
            self,
            pattern,
            path=path,
            fixed_string=fixed_string,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            max_matches=max_matches,
            max_files=max_files,
        )

    @staticmethod
    def _grep_required_tokens(pattern: str, *, fixed_string: bool) -> list[str] | None:
        """Literal tokens that must appear in every match (see ``_grep.required_tokens``)."""
        return _grep.required_tokens(pattern, fixed_string=fixed_string)

    async def _grep_sound_candidates(
        self,
        pattern: str,
        *,
        fixed_string: bool,
        path_filter: str | None,
    ) -> list[str] | None:
        """Exact candidate paths for grep, or ``None`` to force a full scan."""
        return await _grep.sound_candidates(
            self, pattern, fixed_string=fixed_string, path_filter=path_filter
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

        Deletes all rows from the doc store (text + FTS) on the shared IO
        executor so it does not block the event loop.
        """
        count = await self._collection.count()
        if count > 0:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._io_executor, self._db.clear)
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

        # docs is (doc_id, text, metadata); reordered to (doc_id, metadata,
        # text) — callers branch on metadata and often ignore the text, so it
        # goes last per this helper's documented return type.
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

        # Once the index is in memory, total_paths() is the exact path count
        # — free, no DB round-trip. Only when it has not bootstrapped yet do
        # we fall back to a cheap doc-count gate: count() counts every chunk,
        # so it is an upper bound on the path count (>=1 doc per path), and
        # doc_count <= cap proves we are under the path cap without scanning.
        # This keeps a heavily-chunked collection from issuing a count()
        # query on every put once the index is warm.
        if not self._index.loaded:
            if await self._collection.count() <= cap:
                return
            # Lazy bootstrap. Captures the executor closure here so the index
            # module stays decoupled from the async store's exact API.
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
        """Commit + checkpoint the store to disk.

        Guarded by `_save_lock` against the close() daemon thread's final save
        so an eviction-triggered save (on the IO executor) never races close().
        The lock also covers a re-check of `_closed` so we never write after
        close() has begun tearing down the connection.
        """
        with self._save_lock:
            if self._closed:
                return
            self._db.save()


# Imported at end of module to break the import cycle: the _grep and _search
# submodules read this module's metadata-key constants, so they can only load
# once those constants exist.
from . import _grep, _search  # noqa: E402
