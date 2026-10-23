"""The write-ahead log stays bounded while other connections share the store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from semantic_cache_mcp.storage.docstore import ContentStorage
from semantic_cache_mcp.storage.docstore._docstore import WAL_SIZE_LIMIT


def _wal_size(db: Path) -> int:
    wal = db.with_name(db.name + "-wal")
    return wal.stat().st_size if wal.exists() else 0


async def _write_while_another_reader_holds_a_snapshot(
    vs: ContentStorage, other: sqlite3.Connection
) -> None:
    other.execute("BEGIN")
    other.execute("SELECT 1 FROM documents LIMIT 1").fetchall()
    for i in range(400):
        await vs.put(f"/f{i}.txt", f"file {i}\n" + "payload line\n" * 2_000, mtime=1.0)
    other.execute("COMMIT")


async def test_close_truncates_the_wal_while_another_connection_is_open(tmp_path: Path) -> None:
    db = tmp_path / "cache.db"
    vs = ContentStorage(db_path=db)
    other = sqlite3.connect(db)
    try:
        await _write_while_another_reader_holds_a_snapshot(vs, other)
        assert _wal_size(db) > WAL_SIZE_LIMIT
        vs.close()
        assert _wal_size(db) == 0
    finally:
        other.close()


async def test_a_checkpoint_resets_the_wal_to_the_size_limit(tmp_path: Path) -> None:
    db = tmp_path / "cache.db"
    vs = ContentStorage(db_path=db)
    other = sqlite3.connect(db)
    try:
        await _write_while_another_reader_holds_a_snapshot(vs, other)
        assert _wal_size(db) > WAL_SIZE_LIMIT
        vs.save()
        await vs.put("/after.txt", "one more write\n", mtime=1.0)
        assert _wal_size(db) <= WAL_SIZE_LIMIT
    finally:
        vs.close()
        other.close()
