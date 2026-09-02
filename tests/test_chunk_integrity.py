"""Reassembled chunks are checked against the hash they are served under.

A chunked file lives as one parent row plus N child rows, joined back together
on read. The reconcile that rewrites those rows on an edit is deliberately not
atomic — a crash between its delete, insert and metadata steps is accepted,
because "the next read re-puts it" — but nothing downstream was actually
checking. `get_content` sorted whatever children it found by `chunk_index` and
joined them, so a lost, duplicated, or half-renumbered child produced text that
is neither the old version nor the new one, handed back under the parent's
unchanged `content_hash`. That is the one failure this cache must never have:
wrong bytes wearing a hash the caller can redeem.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from semantic_cache_mcp.core.hashing import hash_content
from semantic_cache_mcp.storage.docstore import (
    _META_IS_PARENT,
    CHUNK_THRESHOLD,
    ContentStorage,
)


def _big_text(marker: str = "origg", lines: int = 2000) -> str:
    body = "\n".join(f"line {i:05d} {marker} " + "abcdefgh" * 8 for i in range(lines))
    return body + "\n"


async def _chunked_store(tmp_path: Path, path: str = "/big.py") -> tuple[ContentStorage, str]:
    storage = ContentStorage(tmp_path / "c.db")
    text = _big_text()
    assert len(text.encode()) >= CHUNK_THRESHOLD
    await storage.put(path, text, time.time())
    return storage, text


async def test_a_coherent_chunk_set_still_reassembles(tmp_path: Path) -> None:
    """The guard must not fire on the normal case."""
    storage, text = await _chunked_store(tmp_path)
    entry = await storage.get("/big.py")
    assert entry is not None
    assert await storage.get_content(entry) == text
    assert entry.content_hash == hash_content(text)


async def test_a_lost_child_row_is_not_served_as_the_file(tmp_path: Path) -> None:
    """A missing chunk must be an error, not a shorter file under the same hash."""
    storage, text = await _chunked_store(tmp_path)
    docs = await storage._find_docs_by_path("/big.py")
    children = [d for d in docs if not d[1].get(_META_IS_PARENT, False)]
    assert len(children) >= 3
    await storage._collection.delete_by_ids([children[1][0]])

    entry = await storage.get("/big.py")
    assert entry is not None
    with pytest.raises(ValueError, match="does not match"):
        await storage.get_content(entry)


async def test_a_duplicated_child_row_is_not_served_as_the_file(tmp_path: Path) -> None:
    """Two rows at one index reassemble to text that was never written."""
    storage, _text = await _chunked_store(tmp_path)
    docs = await storage._find_docs_by_path("/big.py")
    children = [d for d in docs if not d[1].get(_META_IS_PARENT, False)]
    doc_id, meta, chunk_text = children[1]
    await storage._collection.add_texts(texts=[chunk_text], metadatas=[dict(meta)])

    entry = await storage.get("/big.py")
    assert entry is not None
    with pytest.raises(ValueError, match="does not match"):
        await storage.get_content(entry)


async def test_a_capped_read_is_not_checked_against_the_whole_file_hash(
    tmp_path: Path,
) -> None:
    """`max_bytes` returns a prefix on purpose, so the whole-file hash cannot apply."""
    storage, text = await _chunked_store(tmp_path)
    entry = await storage.get("/big.py")
    assert entry is not None

    prefix = await storage.get_content(entry, max_bytes=1000)
    assert prefix and text.startswith(prefix)


async def test_an_interrupted_rewrite_leaves_the_previous_version_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rewrite is one transaction, so a failure inside it changes nothing.

    Before it was one, an interruption between the insert and the metadata
    refresh committed a chunk set blended from both versions — and a blend that
    looks entirely healthy: 80 children numbered 0..79, no gaps, no duplicates,
    166,000 bytes where the file is 164,000. Only the file's own hash disagreed.
    """
    storage, v1 = await _chunked_store(tmp_path)

    def boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated crash mid-rewrite")

    # The last of the three statements the transaction now covers.
    monkeypatch.setattr(storage._sync_collection, "_update_metadata_locked", boom)

    with pytest.raises(RuntimeError, match="simulated crash"):
        await storage.put("/big.py", _big_text("edited"), time.time())

    monkeypatch.undo()
    docs = await storage._find_docs_by_path("/big.py")
    children = [d for d in docs if not d[1].get(_META_IS_PARENT, False)]
    joined = "".join(d[2] for d in sorted(children, key=lambda r: r[1].get("chunk_index", 0)))
    assert joined == v1, "the rewrite left a blend of two versions behind"
    assert hash_content(joined) == hash_content(v1)


async def test_the_cache_recovers_from_an_incoherent_chunk_set(tmp_path: Path) -> None:
    """Storage refuses the bad bytes; the cache falls back to the file itself.

    An incoherent chunk set is a cache problem, not a caller problem — the
    file is still on disk — so the recovery is a re-read, not a failed tool
    call.
    """
    from semantic_cache_mcp.cache import SemanticCache

    target = tmp_path / "big.py"
    text = _big_text()
    target.write_text(text)

    cache = SemanticCache(db_path=tmp_path / "cache.db")
    await cache.put(str(target), text, target.stat().st_mtime)

    docs = await cache._storage._find_docs_by_path(str(target))
    children = [d for d in docs if not d[1].get(_META_IS_PARENT, False)]
    assert len(children) >= 3
    await cache._storage._collection.delete_by_ids([children[1][0]])

    entry = await cache.get(str(target))
    assert entry is not None
    assert await cache.get_content(entry) == text


async def test_an_incoherent_entry_for_a_deleted_file_still_errors(tmp_path: Path) -> None:
    """With no disk copy to fall back on there is no honest answer to give."""
    from semantic_cache_mcp.cache import SemanticCache

    target = tmp_path / "gone.py"
    text = _big_text()
    target.write_text(text)

    cache = SemanticCache(db_path=tmp_path / "cache.db")
    await cache.put(str(target), text, target.stat().st_mtime)
    docs = await cache._storage._find_docs_by_path(str(target))
    children = [d for d in docs if not d[1].get(_META_IS_PARENT, False)]
    await cache._storage._collection.delete_by_ids([children[1][0]])
    target.unlink()

    entry = await cache.get(str(target))
    assert entry is not None
    with pytest.raises(OSError):
        await cache.get_content(entry)
