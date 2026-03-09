"""Vector storage backend using simplevecdb for raw text + embedding persistence.

Replaces compressed chunk storage with vector + raw text metadata pattern:
- Each file stored as one or more Documents (chunks for large files)
- page_content = raw text (no compression, 100% fidelity)
- Embedding vectors enable HNSW-based semantic similarity search
- Metadata tracks path, chunk ordering, mtime, content hash, tokens
"""

from __future__ import annotations

import array
import json
import logging
import threading
import time
from pathlib import Path

from simplevecdb import DistanceStrategy, Quantization, VectorDB

from ..config import (
    CACHE_DIR,
    CHUNK_MIN_SIZE,
    MAX_CACHE_ENTRIES,
    SIMILARITY_THRESHOLD,
)
from ..core.chunking import get_optimal_chunker
from ..core.hashing import hash_content
from ..core.tokenizer import count_tokens
from ..types import CacheEntry, EmbeddingVector

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

    __slots__ = ("_db", "_collection", "_db_path", "_lock")

    # Quantization currently in use — stored in sidecar to detect future changes.
    _QUANTIZATION = Quantization.INT8
    _COLLECTION_NAME = "files"

    def __init__(self, db_path: Path = VECDB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._clear_if_quantization_changed(db_path)
        self._db = VectorDB(
            path=str(db_path),
            distance_strategy=DistanceStrategy.COSINE,
            quantization=self._QUANTIZATION,
        )
        self._collection = self._db.collection(self._COLLECTION_NAME)
        self._lock = threading.RLock()  # Reentrant: public methods may call each other
        logger.info(f"VectorStorage initialized at {db_path}")

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
            with self._lock:
                # Delete usearch index files
                for suffix in (f".{self._COLLECTION_NAME}.usearch",):
                    stale = self._db_path.parent / (self._db_path.name + suffix)
                    if stale.exists():
                        stale.unlink()
                        logger.info(f"Deleted stale index: {stale}")
                # Clear all documents from SQLite so stale embeddings don't persist
                self.clear()
                logger.info("Cleared all cached embeddings")

        meta["embedding_model"] = model_name
        meta["embedding_dim"] = dim
        meta_path.write_text(json.dumps(meta))

    def __del__(self) -> None:
        if hasattr(self, "_db"):
            try:
                self._db.save()
                self._db.close()
            except Exception:  # nosec B110 — best-effort cleanup in __del__
                pass

    # -------------------------------------------------------------------------
    # File operations
    # -------------------------------------------------------------------------

    def get(self, path: str) -> CacheEntry | None:
        """Get cached file entry by path.

        Returns metadata only — call get_content() for full text.
        For chunked files, returns metadata from the parent document.
        """
        with self._lock:
            results = self._find_docs_by_path(path)
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
            embedding: EmbeddingVector | None = None
            try:
                emb_map = self._collection._catalog.get_embeddings_by_ids([selected_id])
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

    def put(
        self,
        path: str,
        content: str,
        mtime: float,
        embedding: EmbeddingVector | None = None,
    ) -> None:
        """Store file in cache as raw text with embedding.

        Small files (< CHUNK_THRESHOLD) stored as a single document.
        Large files are split via HyperCDC SIMD chunker into content-defined
        chunks, stored as parent (file metadata) + children (raw text chunks).
        Each chunk gets its own embedding for fine-grained similarity search.
        """
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

        with self._lock:
            # Remove old entry for this path
            self._delete_by_path(path)

            if len(content_bytes) < CHUNK_THRESHOLD:
                # Small file: single document
                meta = {**base_meta, _META_CHUNK_INDEX: 0, _META_TOTAL_CHUNKS: 1}
                self._collection.add_texts(
                    texts=[content],
                    metadatas=[meta],
                    embeddings=[emb_list],
                )
                logger.debug(f"Stored {path} as single doc ({tokens} tokens)")
            else:
                # Large file: parent + chunked children
                self._put_chunked(path, content, content_bytes, base_meta, emb_list)

            self._collection.save()
            self._evict_if_needed()

    def _put_chunked(
        self,
        path: str,
        content: str,
        content_bytes: bytes,
        base_meta: dict,
        file_embedding: list[float],
    ) -> None:
        """Store a large file as parent doc + CDC-chunked children."""
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
        parent_ids = self._collection.add_texts(
            texts=[""],  # Parent has no content — children hold raw text
            metadatas=[parent_meta],
            embeddings=[file_embedding],
        )
        parent_id = parent_ids[0]

        # Store children — each chunk is a child of the parent
        child_metas: list[dict] = []
        child_embeddings: list[list[float]] = []
        zero_emb = self._zero_embedding()

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

        self._collection.add_texts(
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
        """Convert embedding to list[float], falling back to zero vector."""
        if embedding is not None:
            return list(embedding)
        dim = self._collection._dim
        if dim is None:
            dim = 384  # Default: matches BAAI/bge-small-en-v1.5
        return [0.0] * dim

    def _zero_embedding(self) -> list[float]:
        """Return a zero vector matching the index dimension."""
        dim = self._collection._dim
        if dim is None:
            dim = 384
        return [0.0] * dim

    def get_content(self, entry: CacheEntry) -> str:
        """Get full content for a cache entry.

        Retrieves raw text from simplevecdb — no decompression needed.
        For chunked files, skips the parent doc and reassembles children
        in chunk_index order.
        """
        with self._lock:
            results = self._find_docs_by_path(entry.path)
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

    def record_access(self, path: str) -> None:
        """Record access for LRU-K tracking."""
        with self._lock:
            results = self._find_docs_by_path(path)
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
                self._collection._catalog.update_metadata_batch(updates)

    def update_mtime(self, path: str, new_mtime: float) -> None:
        """Update cached mtime without re-storing content or re-embedding.

        Used when content hash matches but mtime changed (touch, git checkout).
        Prevents repeated hash checks on subsequent reads.
        """
        with self._lock:
            results = self._find_docs_by_path(path)
            if not results:
                return

            updates: list[tuple[int, dict]] = []
            for doc_id, _meta, _text in results:
                updates.append((doc_id, {_META_MTIME: new_mtime}))

            if updates:
                self._collection._catalog.update_metadata_batch(updates)

    # -------------------------------------------------------------------------
    # Similarity search
    # -------------------------------------------------------------------------

    def find_similar(
        self, embedding: EmbeddingVector, exclude_path: str | None = None
    ) -> str | None:
        """Find semantically similar cached file using HNSW search.

        Returns the path of the most similar file above SIMILARITY_THRESHOLD.
        """
        emb_list = list(embedding)

        with self._lock:
            try:
                results = self._collection.similarity_search(
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
                    logger.debug(
                        f"Similar file found: {candidate_path} (similarity={similarity:.3f})"
                    )
                    return candidate_path

            return None

    def find_similar_multi(
        self,
        embedding: EmbeddingVector,
        exclude_path: str | None = None,
        k: int = 5,
        threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """Find multiple similar files with scores.

        Returns list of (path, similarity) tuples above threshold.
        """
        if threshold is None:
            threshold = SIMILARITY_THRESHOLD

        emb_list = list(embedding)

        with self._lock:
            try:
                results = self._collection.similarity_search(
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

    def search_by_query(
        self,
        query: str,
        k: int = 5,
        filter: dict | None = None,
    ) -> list[tuple[str, str, float]]:
        """Search cached content by text query using BM25 keyword ranking.

        Uses FTS5 full-text search for exact/partial keyword matching.

        Args:
            query: Text query (FTS5 syntax supported)
            k: Maximum results to return
            filter: Optional metadata filter (e.g. {path: "/src/foo.py"})

        Returns:
            List of (path, preview, score) tuples sorted by relevance
        """
        with self._lock:
            try:
                results = self._collection.keyword_search(query, k=k * 2, filter=filter)
            except Exception as e:
                logger.warning(f"Keyword search failed: {e}")
                return []

            return self._dedupe_search_results(results, k)

    def search_hybrid(
        self,
        query: str,
        embedding: EmbeddingVector | None = None,
        k: int = 5,
        filter: dict | None = None,
    ) -> list[tuple[str, str, float]]:
        """Search cached content using hybrid BM25 + vector similarity (RRF).

        Combines keyword matching with semantic similarity for best results.
        Falls back to keyword-only if no embedding provided.

        Args:
            query: Text query for keyword component
            embedding: Optional query embedding for vector component
            k: Maximum results to return
            filter: Optional metadata filter

        Returns:
            List of (path, preview, score) tuples sorted by fused relevance
        """
        query_vector = list(embedding) if embedding is not None else None

        with self._lock:
            try:
                results = self._collection.hybrid_search(
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
        """Deduplicate search results by path, keeping best score per file."""
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

    def grep(
        self,
        pattern: str,
        *,
        fixed_string: bool = False,
        case_sensitive: bool = True,
        context_lines: int = 0,
        max_matches: int = 100,
        max_files: int = 50,
    ) -> list[dict]:
        """Search cached file content by regex or literal pattern.

        Unlike search/search_hybrid (ranked BM25+vector), this does exact
        pattern matching with line numbers and context — like ripgrep on the
        cache.

        Args:
            pattern: Regex pattern (or literal string if fixed_string=True)
            fixed_string: Treat pattern as literal, not regex
            case_sensitive: Case-sensitive matching (default: True)
            context_lines: Lines of context before/after each match
            max_matches: Total match limit across all files
            max_files: Maximum files to return

        Returns:
            List of dicts: {path, matches: [{line_number, line, before, after}]}
        """
        import re  # noqa: PLC0415

        flags = 0 if case_sensitive else re.IGNORECASE
        if fixed_string:
            compiled = re.compile(re.escape(pattern), flags)
        else:
            try:
                compiled = re.compile(pattern, flags)
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {e}")
                return []

        # Collect all cached file content grouped by path (lock for DB access)
        with self._lock:
            all_docs = self._collection._catalog.get_all_docs_with_text()
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

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        with self._lock:
            count = self._collection.count()
            db_size = self._db_path.stat().st_size if self._db_path.exists() else 0

            # Count unique files (not chunks)
            unique_paths: set[str] = set()
            total_tokens = 0

            all_docs = self._collection._catalog.get_all_docs_with_text()
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

    def clear(self) -> int:
        """Clear all cache entries. Returns count of files removed."""
        with self._lock:
            count = self._collection.count()
            if count > 0:
                all_docs = self._collection._catalog.get_all_docs_with_text()
                doc_ids = [doc_id for doc_id, _, _ in all_docs]
                if doc_ids:
                    self._collection.delete_by_ids(doc_ids)
                    self._collection.save()
            return count

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _find_docs_by_path(self, path: str) -> list[tuple[int, dict, str]]:
        """Find all documents for a given file path.

        Returns list of (doc_id, metadata_dict, text) tuples.
        """
        catalog = self._collection._catalog
        try:
            docs = catalog.get_all_docs_with_text(
                filter_dict={_META_PATH: path},
                filter_builder=catalog.build_filter_clause,
            )
        except Exception as e:
            logger.debug(f"Filter lookup failed for {path}: {e}")
            return []

        # docs is list of (doc_id, text, metadata_dict)
        return [(doc_id, meta, text) for doc_id, text, meta in docs]

    def _delete_by_path(self, path: str) -> int:
        """Delete all documents for a given file path. Returns count deleted."""
        results = self._find_docs_by_path(path)
        if not results:
            return 0

        doc_ids = [r[0] for r in results]
        self._collection.delete_by_ids(doc_ids)
        return len(doc_ids)

    def _evict_if_needed(self) -> None:
        """Evict entries using LRU-K policy if over limit."""
        # Get all docs once — used for both the file-count check and LRU scoring.
        all_docs = self._collection._catalog.get_all_docs_with_text()

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
            evicted += len(doc_ids)

        if ids_to_delete:
            self._collection.delete_by_ids(ids_to_delete)
            self._collection.save()
            logger.info(f"Cache eviction: removed {evicted} documents")

    def save(self) -> None:
        """Persist index to disk."""
        self._collection.save()
        self._db.save()

    def close(self) -> None:
        """Close the database."""
        try:
            self._db.save()
            self._db.close()
        except Exception as e:
            logger.warning(f"Error closing VectorStorage: {e}")
