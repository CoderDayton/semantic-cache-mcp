"""The indexed ``path`` column: lookups by path never scan the table, and the
per-file metadata paths (get, record_access, update_mtime) touch only the
file's head row, never its chunk text."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from semantic_cache_mcp.storage.docstore import (
    _META_ACCESS_HISTORY,
    _META_IS_PARENT,
    _META_MTIME,
    ContentStorage,
)


def _big_text(marker: str = "origg", lines: int = 2000) -> str:
    body = "\n".join(f"line {i:05d} {marker} " + "abcdefgh" * 8 for i in range(lines))
    return body + "\n"


@contextmanager
def _captured_sql(storage: ContentStorage) -> Iterator[list[str]]:
    statements: list[str] = []
    conn = storage._db.conn
    conn.set_trace_callback(statements.append)
    try:
        yield statements
    finally:
        conn.set_trace_callback(None)


def _scans_documents(conn: sqlite3.Connection, statement: str) -> bool:
    plan = conn.execute(f"EXPLAIN QUERY PLAN {statement}").fetchall()
    return any("SCAN documents" in str(row[-1]) for row in plan)


def _rows(storage: ContentStorage) -> dict[int, dict]:
    rows = storage._db.conn.execute("SELECT id, metadata FROM documents").fetchall()
    return {int(i): json.loads(m) for i, m in rows}


async def test_path_lookups_use_the_index(tmp_path: Path) -> None:
    storage = ContentStorage(tmp_path / "c.db")
    await storage.put("/a.py", _big_text(), time.time())
    await storage.put("/b.py", "small file\n", time.time())
    conn = storage._db.conn
    with _captured_sql(storage) as statements:
        assert await storage.get("/a.py") is not None
        await storage.record_access("/a.py")
        await storage.update_mtime("/a.py", time.time())
        await storage.get_content(await storage.get("/b.py"))  # type: ignore[arg-type]
    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    assert selects
    assert not [s for s in selects if _scans_documents(conn, s)]


async def test_get_loads_no_chunk_text(tmp_path: Path) -> None:
    storage = ContentStorage(tmp_path / "c.db")
    await storage.put("/a.py", _big_text(), time.time())
    with _captured_sql(storage) as statements:
        entry = await storage.get("/a.py")
    assert entry is not None
    assert not [s for s in statements if "text" in s.split("FROM")[0]]


async def test_record_access_and_mtime_touch_only_the_head_row(tmp_path: Path) -> None:
    storage = ContentStorage(tmp_path / "c.db")
    await storage.put("/a.py", _big_text(), time.time())
    before = _rows(storage)
    await storage.record_access("/a.py")
    await storage.update_mtime("/a.py", 123.0)
    after = _rows(storage)
    changed = [i for i in before if before[i] != after[i]]
    assert len(changed) == 1
    head = after[changed[0]]
    assert head.get(_META_IS_PARENT) is True
    assert head[_META_MTIME] == 123.0
    assert len(json.loads(head[_META_ACCESS_HISTORY])) == 2
    entry = await storage.get("/a.py")
    assert entry is not None
    assert entry.mtime == 123.0
    assert len(entry.access_history) == 2


async def test_existing_store_gains_the_path_column(tmp_path: Path) -> None:
    db = tmp_path / "c.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE documents (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "text TEXT NOT NULL, metadata TEXT, parent_id INTEGER)"
    )
    conn.execute("CREATE VIRTUAL TABLE documents_fts USING fts5(text)")
    meta = {
        "path": "/old.py",
        "content_hash": "h",
        "mtime": 1.0,
        "tokens": 2,
        "created_at": 1.0,
        "access_history": "[1.0]",
        "chunk_index": 0,
        "total_chunks": 1,
    }
    conn.execute(
        "INSERT INTO documents(text, metadata) VALUES (?, ?)", ("old text\n", json.dumps(meta))
    )
    conn.execute("INSERT INTO documents_fts(rowid, text) VALUES (1, 'old text')")
    conn.commit()
    conn.close()

    storage = ContentStorage(db)
    columns = {r[1] for r in storage._db.conn.execute("PRAGMA table_info(documents)")}
    assert "path" in columns
    entry = await storage.get("/old.py")
    assert entry is not None
    assert entry.content_hash == "h"
