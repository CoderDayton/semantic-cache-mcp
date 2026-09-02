"""SQLite + FTS5 document store backing :class:`ContentStorage`.

A focused, self-contained replacement for the ``simplevecdb`` dependency. The
cache only needs a text + JSON-metadata table with BM25 keyword search and
metadata filtering — no vectors, no usearch index, no encryption.

The fiddly parts (the FTS5 ``MATCH`` query, ``bm25()`` ranking, the
malformed-query guard, and the JSON-extract metadata-filter SQL) are lifted
verbatim from SimpleVecDB's ``CatalogManager``
(github.com/CoderDayton/simplevecdb, ``engine/catalog.py``); the vector /
clustering / TTL / edges machinery is dropped. Single SQLite file, WAL mode,
all access serialized through a re-entrant lock (the storage layer already
funnels every call through one IO thread).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TABLE = "documents"
_FTS = "documents_fts"

# PRAGMA auto_vacuum mode 2. Freed pages go on a free list that
# ``PRAGMA incremental_vacuum`` can hand back to the filesystem, instead of
# being retained and reused (mode 0), which pins the file at its high-water
# mark forever.
_AUTO_VACUUM_INCREMENTAL = 2

# Defense-in-depth: table names are module constants, never user input, but
# validate anyway since they are interpolated into SQL.
_SAFE_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


# The ESCAPE character for a LIKE prefix match. Backslash is a plain character
# to SQLite — it has no string escapes — so it needs no doubling in the SQL.
_LIKE_ESCAPE = "\\"


def _like_prefix(prefix: str) -> str:
    """A LIKE pattern matching exactly the strings starting with *prefix*.

    ``%`` and ``_`` are wildcards, and a directory is free to contain either,
    so both are escaped along with the escape character itself.
    """
    for char in (_LIKE_ESCAPE, "%", "_"):
        prefix = prefix.replace(char, _LIKE_ESCAPE + char)
    return prefix + "%"


def _validate_table_name(name: str) -> None:
    if not _SAFE_TABLE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid table name {name!r}. Must be alphanumeric + underscores, "
            "starting with a letter or underscore."
        )


@dataclass(slots=True)
class Document:
    """Minimal document shape consumed by ``_search.py`` / ``_grep.py``."""

    page_content: str
    metadata: dict[str, Any]


class DocStore:
    """Synchronous SQLite + FTS5 store. Every method serializes on ``_lock``."""

    def __init__(self, db_path: Path) -> None:
        _validate_table_name(_TABLE)
        _validate_table_name(_FTS)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = db_path
        # Re-entrant lock guards Python-level access to the shared connection.
        # The connection is opened check_same_thread=False; SQLite is safe under
        # WAL, but Python's `with conn:` transaction context is not thread-safe.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._closed = False
        # Before the WAL pragma, which writes a header page: while the database
        # is still genuinely empty the mode is adopted with no rewrite at all,
        # so only a store created before 0.5.3 ever pays for the VACUUM.
        self._enable_incremental_vacuum()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _enable_incremental_vacuum(self) -> None:
        """Make freed pages returnable to the filesystem, once per store.

        Without this SQLite keeps freed pages inside the file and reuses them,
        so a cache that has evicted and re-indexed for weeks sits at its
        high-water mark: 139 MB of file for 4 MB of text, measured. The mode
        lives in the database header, so this check is self-describing and
        needs no migration marker. Changing it on a database that already has
        pages only takes effect after a full rewrite, which is what the VACUUM
        is for — 38.6 ms on that 139 MB store, and never again.
        """
        try:
            if int(self._conn.execute("PRAGMA auto_vacuum").fetchone()[0]) == (
                _AUTO_VACUUM_INCREMENTAL
            ):
                return
            # Read the page count before setting the mode, not after: recording
            # the mode is itself a write, so it creates the first header page
            # and would make every new store look like one needing a rewrite.
            populated = int(self._conn.execute("PRAGMA page_count").fetchone()[0]) > 0
            self._conn.execute(f"PRAGMA auto_vacuum = {_AUTO_VACUUM_INCREMENTAL}")
            if populated:
                started = time.perf_counter()
                self._conn.execute("VACUUM")
                logger.info(
                    "enabled incremental auto-vacuum on %s in %.1f ms",
                    self._path.name,
                    (time.perf_counter() - started) * 1000,
                )
        except sqlite3.Error as exc:
            # A cache that cannot shrink still works; never fail startup for it.
            logger.warning(f"could not enable incremental auto-vacuum: {exc}")

    def _create_tables(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    parent_id INTEGER
                )
                """
            )
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{_TABLE}_parent "
                f"ON {_TABLE}(parent_id) WHERE parent_id IS NOT NULL"
            )
            self._conn.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS} USING fts5(text)")

    @property
    def conn(self) -> sqlite3.Connection:
        """Raw SQLite connection for the grep BM25 prefilter (FTS vocab).

        Callers MUST hold ``self._lock`` while using it (grep's ``vocab_expand``
        does), so raw-connection access serializes with every other store op.
        """
        return self._conn

    # ------------------------------------------------------------------ writes
    #
    # Each ``_*_locked`` helper runs the statements alone, assuming the caller
    # already holds ``self._lock`` and has a transaction open. That split is
    # what lets several of them share one transaction: ``with self._conn:``
    # does not nest — the inner exit commits — so a composite write cannot be
    # built by calling the public methods.
    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict],
        parent_ids: Sequence[int | None] | None = None,
    ) -> list[int]:
        """Insert documents (text + JSON metadata + optional parent). Returns ids."""
        if not texts:
            return []
        with self._lock, self._conn:
            return self._add_texts_locked(texts, metadatas, parent_ids)

    def _add_texts_locked(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict],
        parent_ids: Sequence[int | None] | None = None,
    ) -> list[int]:
        if not texts:
            return []
        parents = list(parent_ids) if parent_ids else [None] * len(texts)
        meta_strs = [json.dumps(m, separators=(",", ":")) for m in metadatas]
        rows = list(zip(texts, meta_strs, parents, strict=True))
        placeholders = ",".join(["(?, ?, ?)"] * len(rows))
        flat = [v for row in rows for v in row]
        cursor = self._conn.execute(
            f"INSERT INTO {_TABLE}(text, metadata, parent_id) VALUES {placeholders} RETURNING id",
            flat,
        )
        # RETURNING row order is unspecified in SQLite; ids are AUTOINCREMENT,
        # so sorting recovers insertion order and keeps ids aligned with texts
        # for the FTS rowid sync below.
        ids = sorted(int(r[0]) for r in cursor.fetchall())
        self._upsert_fts_rows(ids, texts)
        return ids

    def _upsert_fts_rows(self, ids: Sequence[int], texts: Sequence[str]) -> None:
        if not ids:
            return
        placeholders = ",".join(["?"] * len(ids))
        self._conn.execute(f"DELETE FROM {_FTS} WHERE rowid IN ({placeholders})", tuple(ids))
        self._conn.executemany(
            f"INSERT INTO {_FTS}(rowid, text) VALUES (?, ?)",
            list(zip(ids, texts, strict=True)),
        )

    def _delete_fts_rows(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        placeholders = ",".join(["?"] * len(ids))
        self._conn.execute(f"DELETE FROM {_FTS} WHERE rowid IN ({placeholders})", tuple(ids))

    def delete_by_ids(self, ids: Sequence[int]) -> list[int]:
        ids = list(ids)
        if not ids:
            return []
        with self._lock, self._conn:
            return self._delete_by_ids_locked(ids)

    def _delete_by_ids_locked(self, ids: Sequence[int]) -> list[int]:
        ids = list(ids)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        existing = [
            r[0]
            for r in self._conn.execute(
                f"SELECT id FROM {_TABLE} WHERE id IN ({placeholders})", tuple(ids)
            ).fetchall()
        ]
        if existing:
            eph = ",".join("?" for _ in existing)
            self._conn.execute(f"DELETE FROM {_TABLE} WHERE id IN ({eph})", tuple(existing))
            self._delete_fts_rows(existing)
        return existing

    def update_metadata(self, updates: list[tuple[int, dict[str, Any]]]) -> int:
        """Shallow-merge metadata updates for the given doc ids."""
        if not updates:
            return 0
        with self._lock, self._conn:
            return self._update_metadata_locked(updates)

    def _update_metadata_locked(self, updates: list[tuple[int, dict[str, Any]]]) -> int:
        if not updates:
            return 0
        ids = [u[0] for u in updates]
        placeholders = ",".join(["?"] * len(ids))
        rows = self._conn.execute(
            f"SELECT id, metadata FROM {_TABLE} WHERE id IN ({placeholders})", ids
        ).fetchall()
        current = {r[0]: (json.loads(r[1]) if r[1] else {}) for r in rows}
        data: list[tuple[str, int]] = []
        for doc_id, meta_updates in updates:
            if doc_id in current:
                meta = current[doc_id]
                meta.update(meta_updates)
                data.append((json.dumps(meta, separators=(",", ":")), doc_id))
        if data:
            self._conn.executemany(f"UPDATE {_TABLE} SET metadata = ? WHERE id = ?", data)
        return len(data)

    def reconcile_chunks(
        self,
        *,
        delete_ids: Sequence[int],
        insert_texts: Sequence[str],
        insert_metas: Sequence[dict],
        parent_id: int,
        meta_updates: list[tuple[int, dict[str, Any]]],
    ) -> list[int]:
        """Drop, insert and re-tag a file's chunk rows in one transaction.

        Rewriting a chunked file is three statements that are only meaningful
        together. Run apart, an interruption between them commits a chunk set
        that is a blend of two versions — and one that looks entirely healthy,
        since the surviving rows still number 0..n-1 with no gaps or
        duplicates. Only the file's own hash gives it away. One transaction
        means the rewrite either lands whole or leaves the previous version
        exactly as it was, and it costs one hop through the IO thread instead
        of three.
        """
        with self._lock, self._conn:
            self._delete_by_ids_locked(delete_ids)
            new_ids = self._add_texts_locked(
                insert_texts, insert_metas, [parent_id] * len(insert_texts)
            )
            self._update_metadata_locked(meta_updates)
        return new_ids

    # ------------------------------------------------------------------- reads
    def keyword_search(
        self,
        query: str,
        k: int,
        filter_dict: dict[str, Any] | None = None,
        path_prefix: str | None = None,
    ) -> list[tuple[Document, float]]:
        """BM25 keyword search via FTS5. Returns ``(Document, score)`` best-first.

        ``path_prefix`` restricts the search to documents under one directory.
        It belongs in SQL rather than in the caller: ``LIMIT`` applies after the
        ``WHERE``, so a caller filtering afterwards has to over-fetch and guess
        how far, and still comes up short when another project dominates the
        store. Here the ranking never spends a slot on a file the caller cannot
        use.
        """
        if not query.strip():
            return []
        filter_clause = ""
        filter_params: list[Any] = []
        if filter_dict:
            filter_clause, filter_params = self.build_filter_clause(filter_dict, "ti.metadata")
        prefix_clause = ""
        prefix_params: list[Any] = []
        if path_prefix:
            # LIKE, not a bare comparison, because SQLite has no prefix operator
            # — so the directory's own `%` and `_` have to be escaped or they
            # act as wildcards and match a sibling project. The trailing
            # separator is what makes it a directory match: without it
            # `/project` also matches `/project_evil`.
            prefix_clause = f"AND json_extract(ti.metadata, ?) LIKE ? ESCAPE '{_LIKE_ESCAPE}'"
            prefix_params = [
                '$."path"',
                _like_prefix(path_prefix.rstrip(os.sep) + os.sep),
            ]
        # FTS5 MATCH + bm25() ranking. The FROM aliases the FTS table as `f`
        # (used for the rowid JOIN) while bm25()/MATCH reference it by name —
        # this exact shape is lifted from simplevecdb and is FTS5-correct.
        sql = f"""
            SELECT ti.id, ti.text, ti.metadata, bm25({_FTS}) AS score
            FROM {_FTS} f
            JOIN {_TABLE} ti ON ti.id = f.rowid
            WHERE {_FTS} MATCH ?
            {filter_clause}
            {prefix_clause}
            ORDER BY score ASC
            LIMIT ?
        """
        params = (query,) + tuple(filter_params) + tuple(prefix_params) + (k,)
        try:
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            # FTS5 raises OperationalError on a malformed MATCH query (unbalanced
            # quotes, a bare operator, ...). Surface a clear caller-facing error
            # instead of the raw SQLite message; re-raise unrelated op errors.
            msg = str(exc).lower()
            if any(
                s in msg
                for s in ("fts5", "unterminated", "malformed", "special query", "syntax error")
            ):
                raise ValueError(f"Invalid full-text search query {query!r}: {exc}") from exc
            raise
        out: list[tuple[Document, float]] = []
        for _doc_id, text, meta_json, score in rows:
            meta = json.loads(meta_json) if meta_json else {}
            out.append((Document(page_content=text, metadata=meta), float(score)))
        return out

    def get_documents(
        self, filter_dict: dict[str, Any] | None = None
    ) -> list[tuple[int, str, dict[str, Any]]]:
        """Return ``(id, text, metadata)`` for all docs (optionally filtered)."""
        filter_clause = ""
        filter_params: list[Any] = []
        if filter_dict:
            filter_clause, filter_params = self.build_filter_clause(filter_dict, "metadata")
        sql = f"SELECT id, text, metadata FROM {_TABLE} WHERE 1=1 {filter_clause} ORDER BY id"
        with self._lock:
            rows = self._conn.execute(sql, tuple(filter_params)).fetchall()
        return [(int(r[0]), r[1], json.loads(r[2]) if r[2] else {}) for r in rows]

    def get_metadata(
        self, filter_dict: dict[str, Any] | None = None
    ) -> list[tuple[int, dict[str, Any]]]:
        """Return ``(id, metadata)`` for all docs (optionally filtered), no text.

        The text column holds the entire cached corpus, so selecting it just to
        read a metadata field re-materializes every cached file in memory.
        Callers that never look at the text use this projection instead.
        """
        filter_clause = ""
        filter_params: list[Any] = []
        if filter_dict:
            filter_clause, filter_params = self.build_filter_clause(filter_dict, "metadata")
        sql = f"SELECT id, metadata FROM {_TABLE} WHERE 1=1 {filter_clause} ORDER BY id"
        with self._lock:
            rows = self._conn.execute(sql, tuple(filter_params)).fetchall()
        return [(int(r[0]), json.loads(r[1]) if r[1] else {}) for r in rows]

    def distinct_paths(self, *, path_key: str, is_parent_key: str) -> list[str]:
        """Distinct non-parent document paths, extracted and de-duplicated in SQL.

        Returns bare strings. Callers that only match paths were decoding every
        document's whole metadata JSON in Python to reach one field, which cost
        more than the query itself.
        """
        with self._lock:
            rows = self._conn.execute(
                f"SELECT DISTINCT json_extract(metadata, ?) FROM {_TABLE} "
                "WHERE COALESCE(json_extract(metadata, ?), 0) != 1",
                (f'$."{path_key}"', f'$."{is_parent_key}"'),
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    def get_document_ids(self) -> list[int]:
        """Every document id — the cheapest projection there is."""
        with self._lock:
            rows = self._conn.execute(f"SELECT id FROM {_TABLE} ORDER BY id").fetchall()
        return [int(r[0]) for r in rows]

    def file_stats(
        self,
        *,
        path_key: str,
        tokens_key: str,
        is_parent_key: str,
        total_chunks_key: str,
    ) -> tuple[int, int]:
        """``(distinct files, summed file-level tokens)``, aggregated in SQL.

        Mirrors the per-document accounting exactly: a chunked file contributes
        its tokens once through its parent doc and an unchunked one through its
        only doc, so summing every row would double-count the chunked files.
        The metadata key names belong to the caller's schema, so they arrive as
        arguments and are bound as JSON paths rather than duplicated here.
        """

        def _path(key: str) -> str:
            return f'$."{key}"'

        with self._lock:
            files_row = self._conn.execute(
                f"SELECT COUNT(DISTINCT json_extract(metadata, ?)) FROM {_TABLE} "
                "WHERE json_extract(metadata, ?) IS NOT NULL "
                "AND json_extract(metadata, ?) != ''",
                (_path(path_key),) * 3,
            ).fetchone()
            tokens_row = self._conn.execute(
                f"SELECT COALESCE(SUM(json_extract(metadata, ?)), 0) FROM {_TABLE} "
                "WHERE json_extract(metadata, ?) = 1 "
                "OR COALESCE(json_extract(metadata, ?), 1) = 1",
                (_path(tokens_key), _path(is_parent_key), _path(total_chunks_key)),
            ).fetchone()
        files = int(files_row[0]) if files_row and files_row[0] is not None else 0
        tokens = int(tokens_row[0]) if tokens_row and tokens_row[0] is not None else 0
        return files, tokens

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------ metadata filter (SQL)
    # Builds a JSON-extract WHERE clause from a ``{key: scalar|list}`` filter.
    # The json_extract path + IN forms follow simplevecdb's catalog, trimmed to
    # the scalar-equality and list-IN cases the cache actually uses.
    def build_filter_clause(
        self, filter_dict: dict[str, Any] | None, metadata_column: str = "metadata"
    ) -> tuple[str, list[Any]]:
        if not filter_dict:
            return "", []

        # metadata_column is interpolated into the SQL below (identifiers
        # cannot be bound parameters). Every call site passes a fixed literal
        # ("metadata" / "ti.metadata"); this allowlist keeps the seam safe if
        # a caller-supplied value ever reaches it.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", metadata_column):
            raise ValueError(f"Invalid metadata column identifier: {metadata_column!r}")

        clauses: list[str] = []
        params: list[Any] = []
        for key, value in filter_dict.items():
            if '"' in key:
                # The key is interpolated into the JSON path label below, so a
                # double-quote would produce a malformed path. (Values are always
                # bound as parameters — this is robustness, not injection; every
                # current key is a fixed metadata constant.)
                raise ValueError(f"Filter key must not contain a double-quote: {key!r}")
            # Quote the path label so a literal key like "a.b" matches the
            # top-level member, not the nested path a -> b. The path string is
            # passed as a bound parameter, never interpolated into SQL.
            json_path = f'$."{key}"'
            text_extract = f"json_extract({metadata_column}, ?)"

            if isinstance(value, bool):
                clauses.append(f"{text_extract} = ?")
                params.extend([json_path, 1 if value else 0])
            elif isinstance(value, (int, float, str)):
                clauses.append(f"{text_extract} = ?")
                params.extend([json_path, value])
            elif isinstance(value, list):
                placeholders = ",".join("?" for _ in value)
                clauses.append(f"{text_extract} IN ({placeholders})")
                params.append(json_path)
                params.extend(value)
            else:
                raise ValueError(f"Unsupported filter value type for {key}: {type(value).__name__}")

        where = " AND ".join(clauses)
        return (f"AND ({where})" if where else ""), params

    # --------------------------------------------------------------- lifecycle
    def clear(self) -> int:
        """Delete all documents. Returns the count removed."""
        with self._lock, self._conn:
            count = self._conn.execute(f"SELECT COUNT(*) FROM {_TABLE}").fetchone()[0]
            self._conn.execute(f"DELETE FROM {_TABLE}")
            self._conn.execute(f"DELETE FROM {_FTS}")
        return int(count)

    def optimize(self) -> None:
        """Merge the FTS5 index into one segment, discarding delete markers.

        FTS5 records a deletion as an index entry rather than removing one, so
        a store that has evicted many files keeps the vocabulary of every file
        it ever held. Only a full merge discards them — the incremental
        ``merge`` command leaves them in place. Measured on a 39 MB store:
        24.7 MB of index down to 1.2 MB in 144 ms. Called at shutdown, where a
        one-off cost is affordable and an interrupted merge is simply rolled
        back.
        """
        with self._lock:
            if self._closed:
                return
            try:
                with self._conn:
                    self._conn.execute(f"INSERT INTO {_FTS}({_FTS}) VALUES('optimize')")
            except sqlite3.Error as exc:
                logger.debug(f"fts optimize failed: {exc}")

    def reclaim_free_pages(self) -> None:
        """Return freed pages to the filesystem.

        A no-op until :meth:`_enable_incremental_vacuum` has run, and cheap
        when there is nothing to give back. Paired with :meth:`optimize` at
        shutdown: the merge is what frees the pages, and this is what hands
        them over — on their own, neither shrinks the file.
        """
        with self._lock:
            if self._closed:
                return
            try:
                # Each step of this pragma hands back one page and yields, so a
                # bare execute() reclaims exactly one. It has to be driven to
                # completion or the file never measurably shrinks.
                self._conn.execute("PRAGMA incremental_vacuum").fetchall()
                self._conn.commit()
            except sqlite3.Error as exc:
                logger.debug(f"incremental_vacuum failed: {exc}")

    def save(self) -> None:
        """Commit and checkpoint the WAL."""
        with self._lock:
            if self._closed:
                return
            self._conn.commit()
            try:
                self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            except sqlite3.Error as exc:
                logger.debug(f"wal_checkpoint failed: {exc}")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.commit()
                self._conn.close()
            except sqlite3.Error as exc:
                logger.debug(f"DocStore close error: {exc}")


class AsyncDocStore:
    """Async adapter: runs each :class:`DocStore` call on the IO executor.

    Mirrors the method surface ``ContentStorage`` previously called on
    ``AsyncVectorCollection`` so the storage layer swap is local.
    """

    def __init__(self, store: DocStore, executor: Executor) -> None:
        self._store = store
        self._executor = executor

    async def _run(self, fn: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args))

    async def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict],
        parent_ids: Sequence[int | None] | None = None,
    ) -> list[int]:
        return await self._run(self._store.add_texts, texts, metadatas, parent_ids)

    async def keyword_search(
        self,
        query: str,
        k: int,
        filter: dict[str, Any] | None = None,
        path_prefix: str | None = None,
    ) -> list[tuple[Document, float]]:
        return await self._run(self._store.keyword_search, query, k, filter, path_prefix)

    async def get_documents(
        self, filter_dict: dict[str, Any] | None = None
    ) -> list[tuple[int, str, dict[str, Any]]]:
        return await self._run(self._store.get_documents, filter_dict)

    async def get_metadata(
        self, filter_dict: dict[str, Any] | None = None
    ) -> list[tuple[int, dict[str, Any]]]:
        return await self._run(self._store.get_metadata, filter_dict)

    async def get_document_ids(self) -> list[int]:
        return await self._run(self._store.get_document_ids)

    async def distinct_paths(self, *, path_key: str, is_parent_key: str) -> list[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._store.distinct_paths(path_key=path_key, is_parent_key=is_parent_key),
        )

    async def file_stats(
        self,
        *,
        path_key: str,
        tokens_key: str,
        is_parent_key: str,
        total_chunks_key: str,
    ) -> tuple[int, int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._store.file_stats(
                path_key=path_key,
                tokens_key=tokens_key,
                is_parent_key=is_parent_key,
                total_chunks_key=total_chunks_key,
            ),
        )

    async def count(self) -> int:
        return await self._run(self._store.count)

    async def delete_by_ids(self, ids: Sequence[int]) -> list[int]:
        return await self._run(self._store.delete_by_ids, ids)

    async def update_metadata(self, updates: list[tuple[int, dict[str, Any]]]) -> int:
        return await self._run(self._store.update_metadata, updates)

    async def reconcile_chunks(
        self,
        *,
        delete_ids: Sequence[int],
        insert_texts: Sequence[str],
        insert_metas: Sequence[dict],
        parent_id: int,
        meta_updates: list[tuple[int, dict[str, Any]]],
    ) -> list[int]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._store.reconcile_chunks(
                delete_ids=delete_ids,
                insert_texts=insert_texts,
                insert_metas=insert_metas,
                parent_id=parent_id,
                meta_updates=meta_updates,
            ),
        )

    async def save(self) -> None:
        return await self._run(self._store.save)
