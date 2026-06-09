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
import re
import sqlite3
import threading
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TABLE = "documents"
_FTS = "documents_fts"

# Defense-in-depth: table names are module constants, never user input, but
# validate anyway since they are interpolated into SQL.
_SAFE_TABLE_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


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
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._closed = False
        self._create_tables()

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
    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict],
        parent_ids: Sequence[int | None] | None = None,
    ) -> list[int]:
        """Insert documents (text + JSON metadata + optional parent). Returns ids."""
        if not texts:
            return []
        parents = list(parent_ids) if parent_ids else [None] * len(texts)
        meta_strs = [json.dumps(m, separators=(",", ":")) for m in metadatas]
        rows = list(zip(texts, meta_strs, parents, strict=True))
        placeholders = ",".join(["(?, ?, ?)"] * len(rows))
        flat = [v for row in rows for v in row]
        with self._lock, self._conn:
            cursor = self._conn.execute(
                f"INSERT INTO {_TABLE}(text, metadata, parent_id) "
                f"VALUES {placeholders} RETURNING id",
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
        placeholders = ",".join("?" for _ in ids)
        with self._lock, self._conn:
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

    # ------------------------------------------------------------------- reads
    def keyword_search(
        self, query: str, k: int, filter_dict: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """BM25 keyword search via FTS5. Returns ``(Document, score)`` best-first."""
        if not query.strip():
            return []
        filter_clause = ""
        filter_params: list[Any] = []
        if filter_dict:
            filter_clause, filter_params = self.build_filter_clause(filter_dict, "ti.metadata")
        # FTS5 MATCH + bm25() ranking. The FROM aliases the FTS table as `f`
        # (used for the rowid JOIN) while bm25()/MATCH reference it by name —
        # this exact shape is lifted from simplevecdb and is FTS5-correct.
        sql = f"""
            SELECT ti.id, ti.text, ti.metadata, bm25({_FTS}) AS score
            FROM {_FTS} f
            JOIN {_TABLE} ti ON ti.id = f.rowid
            WHERE {_FTS} MATCH ?
            {filter_clause}
            ORDER BY score ASC
            LIMIT ?
        """
        params = (query,) + tuple(filter_params) + (k,)
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
        self, query: str, k: int, filter: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        return await self._run(self._store.keyword_search, query, k, filter)

    async def get_documents(
        self, filter_dict: dict[str, Any] | None = None
    ) -> list[tuple[int, str, dict[str, Any]]]:
        return await self._run(self._store.get_documents, filter_dict)

    async def count(self) -> int:
        return await self._run(self._store.count)

    async def delete_by_ids(self, ids: Sequence[int]) -> list[int]:
        return await self._run(self._store.delete_by_ids, ids)

    async def update_metadata(self, updates: list[tuple[int, dict[str, Any]]]) -> int:
        return await self._run(self._store.update_metadata, updates)

    async def save(self) -> None:
        return await self._run(self._store.save)
