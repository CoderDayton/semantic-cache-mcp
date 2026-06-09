"""Content-addressed chunk reconciliation on re-write.

These tests pin the whole point of the manifest design: when a chunked file is
re-written, only the chunks whose bytes actually changed are deleted/inserted
and re-indexed; every unchanged chunk keeps its existing row (proved by stable
doc ids). They also cover content integrity, the manifest/chunk_hash metadata,
and the single-doc/chunked transitions.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from semantic_cache_mcp.storage.docstore import (
    _META_CHUNK_HASH,
    _META_CHUNK_INDEX,
    _META_IS_PARENT,
    _META_MANIFEST,
    CHUNK_THRESHOLD,
    ContentStorage,
)


def _big_text(marker: str = "origg", lines: int = 2000) -> str:
    """~160 KB of line-structured content — HyperCDC cuts it into many chunks."""
    body = "\n".join(f"line {i:05d} {marker} " + "abcdefgh" * 8 for i in range(lines))
    return body + "\n"


async def _docs(storage: ContentStorage, path: str) -> list[tuple[int, dict, str]]:
    return await storage._find_docs_by_path(path)


def _children(docs: list[tuple[int, dict, str]]) -> list[tuple[int, dict, str]]:
    return [d for d in docs if not d[1].get(_META_IS_PARENT, False)]


def _parent(docs: list[tuple[int, dict, str]]) -> tuple[int, dict, str] | None:
    return next((d for d in docs if d[1].get(_META_IS_PARENT, False)), None)


def _child_ids(docs: list[tuple[int, dict, str]]) -> set[int]:
    return {d[0] for d in _children(docs)}


async def test_large_file_is_chunked_into_many(tmp_path: Path) -> None:
    storage = ContentStorage(tmp_path / "c.db")
    text = _big_text()
    assert len(text.encode()) >= CHUNK_THRESHOLD
    await storage.put("/f.py", text, time.time())
    docs = await _docs(storage, "/f.py")
    assert _parent(docs) is not None
    assert len(_children(docs)) >= 3


async def test_substitution_edit_reuses_unchanged_rows(tmp_path: Path) -> None:
    """Equal-length edit in the middle: only the edited chunk's row changes."""
    storage = ContentStorage(tmp_path / "c.db")
    path = "/big.py"
    orig = _big_text("origg")
    await storage.put(path, orig, time.time())
    before = await _docs(storage, path)
    before_children = _child_ids(before)
    before_parent_id = _parent(before)[0]  # type: ignore[index]
    total = len(before_children)
    assert total >= 3

    # "origg" -> "EDITd": same length, so this is a pure local substitution.
    edited = orig.replace("line 01000 origg ", "line 01000 EDITd ", 1)
    assert edited != orig
    await storage.put(path, edited, time.time())
    after = await _docs(storage, path)
    after_children = _child_ids(after)

    # Parent row is updated in place, not deleted + reinserted.
    assert _parent(after)[0] == before_parent_id  # type: ignore[index]
    # Almost every child row survives; only the edited chunk is a new row.
    assert len(before_children & after_children) >= total - 2
    assert len(after_children - before_children) <= 2
    # Content round-trips exactly.
    entry = await storage.get(path)
    assert entry is not None
    assert await storage.get_content(entry) == edited


async def test_insertion_edit_resyncs_and_reuses_tail(tmp_path: Path) -> None:
    """Length-changing edit shifts the byte stream; CDC resyncs so tail rows
    are still reused, not rewritten."""
    storage = ContentStorage(tmp_path / "c.db")
    path = "/shift.py"
    orig = _big_text("orig")
    await storage.put(path, orig, time.time())
    before = _child_ids(await _docs(storage, path))
    total = len(before)

    edited = orig.replace("line 01000 orig ", "line 01000 orig EXTRA-INSERTED-RUN ", 1)
    assert len(edited) != len(orig)
    await storage.put(path, edited, time.time())
    after = _child_ids(await _docs(storage, path))

    # The tail re-synchronizes, so most rows are reused despite the shift.
    assert len(before & after) >= total - 3
    entry = await storage.get(path)
    assert entry is not None
    assert await storage.get_content(entry) == edited


async def test_identical_reput_reuses_every_chunk(tmp_path: Path) -> None:
    storage = ContentStorage(tmp_path / "c.db")
    path = "/same.py"
    text = _big_text()
    await storage.put(path, text, time.time())
    before = _child_ids(await _docs(storage, path))
    # Same bytes, newer mtime — no chunk hash changes, so no row should move.
    await storage.put(path, text, time.time() + 10)
    after = _child_ids(await _docs(storage, path))
    assert before == after


async def test_manifest_matches_child_hashes(tmp_path: Path) -> None:
    storage = ContentStorage(tmp_path / "c.db")
    path = "/m.py"
    await storage.put(path, _big_text(), time.time())
    docs = await _docs(storage, path)
    parent = _parent(docs)
    assert parent is not None
    manifest = parent[1].get(_META_MANIFEST)
    children = sorted(_children(docs), key=lambda d: d[1][_META_CHUNK_INDEX])
    child_hashes = [c[1][_META_CHUNK_HASH] for c in children]
    assert manifest == child_hashes
    assert all(isinstance(h, str) and len(h) == 64 for h in manifest)


async def test_transition_chunked_to_single(tmp_path: Path) -> None:
    storage = ContentStorage(tmp_path / "c.db")
    path = "/t.py"
    await storage.put(path, _big_text(), time.time())
    assert len(_children(await _docs(storage, path))) >= 3
    await storage.put(path, "tiny", time.time())  # now below the chunk threshold
    docs = await _docs(storage, path)
    assert _parent(docs) is None
    assert len(_children(docs)) == 1
    entry = await storage.get(path)
    assert entry is not None
    assert await storage.get_content(entry) == "tiny"


async def test_transition_single_to_chunked(tmp_path: Path) -> None:
    storage = ContentStorage(tmp_path / "c.db")
    path = "/t2.py"
    await storage.put(path, "tiny", time.time())
    assert _parent(await _docs(storage, path)) is None
    big = _big_text()
    await storage.put(path, big, time.time())
    docs = await _docs(storage, path)
    assert _parent(docs) is not None
    entry = await storage.get(path)
    assert entry is not None
    assert await storage.get_content(entry) == big


async def test_grep_finds_reused_and_edited_chunks(tmp_path: Path) -> None:
    storage = ContentStorage(tmp_path / "c.db")
    path = "/g.py"
    orig = _big_text("orig")
    await storage.put(path, orig, time.time())
    edited = orig.replace("line 01000 orig ", "line 01000 EDITd ", 1)
    await storage.put(path, edited, time.time())

    # A line far from the edit lives in a reused chunk — still searchable.
    far = await storage.grep("line 01900 orig", fixed_string=True)
    assert far
    # The edited content lives in a freshly inserted chunk — also searchable.
    near = await storage.grep("line 01000 EDITd", fixed_string=True)
    assert near


async def test_reconcile_upgrades_legacy_children_without_duplication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry written by an older build has children with no chunk_hash. A
    re-write must DELETE those legacy rows, not orphan them — otherwise the new
    chunks are inserted alongside the old ones and the file is duplicated."""
    storage = ContentStorage(tmp_path / "c.db")
    path = "/legacy.py"
    orig = _big_text("orig")

    # First store mimics the old code: strip chunk_hash off every child.
    # ContentStorage uses __slots__, so patch the class method (capturing the
    # original to avoid recursion), then undo before the reconcile.
    original = ContentStorage._child_meta

    def legacy_child_meta(
        self: ContentStorage, p: str, i: int, plan: object, base_meta: dict
    ) -> dict:
        meta = original(self, p, i, plan, base_meta)  # type: ignore[arg-type]
        meta.pop(_META_CHUNK_HASH, None)
        return meta

    monkeypatch.setattr(ContentStorage, "_child_meta", legacy_child_meta)
    await storage.put(path, orig, time.time())
    monkeypatch.undo()  # restore the real metadata builder for the reconcile

    before_children = len(_children(await _docs(storage, path)))
    assert before_children >= 3

    # Re-store changed content — reconcile runs against the hash-less children.
    edited = orig.replace("line 01000 orig ", "line 01000 EDITd ", 1)
    await storage.put(path, edited, time.time())

    after_children = _children(await _docs(storage, path))
    # Definitive anti-duplication check: content round-trips exactly.
    entry = await storage.get(path)
    assert entry is not None
    assert await storage.get_content(entry) == edited
    # Row count reflects the new file only — not the old + new sets doubled up.
    assert len(after_children) <= before_children + 1
    # Every surviving child is now upgraded to the hash-bearing shape.
    assert all(_META_CHUNK_HASH in c[1] for c in after_children)


async def test_migration_clears_premanifest_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On upgrade, a chunked cache written before the manifest format is wiped
    once so the store doesn't mix old and new chunk shapes."""
    storage = ContentStorage(tmp_path / "c.db")

    # Store a legacy-shaped chunked entry: no manifest on the parent, no
    # chunk_hash on the children (ContentStorage uses __slots__ → patch the
    # class methods, capturing the originals to avoid recursion).
    orig_parent = ContentStorage._parent_meta
    orig_child = ContentStorage._child_meta

    def legacy_parent(self: ContentStorage, plan: object, base_meta: dict) -> dict:
        meta = orig_parent(self, plan, base_meta)  # type: ignore[arg-type]
        meta.pop(_META_MANIFEST, None)
        return meta

    def legacy_child(self: ContentStorage, p: str, i: int, plan: object, base_meta: dict) -> dict:
        meta = orig_child(self, p, i, plan, base_meta)  # type: ignore[arg-type]
        meta.pop(_META_CHUNK_HASH, None)
        return meta

    monkeypatch.setattr(ContentStorage, "_parent_meta", legacy_parent)
    monkeypatch.setattr(ContentStorage, "_child_meta", legacy_child)
    await storage.put("/legacy.py", _big_text(), time.time())
    monkeypatch.undo()

    assert storage._has_premanifest_chunks() is True
    marker = tmp_path / ".docstore_manifest_v1"
    marker.unlink()  # simulate the pre-upgrade state (marker not yet written)

    storage._migrate_chunk_manifest()

    assert marker.exists()
    assert storage._has_premanifest_chunks() is False
    assert (await storage.get_stats())["total_documents"] == 0


async def test_migration_keeps_new_format_cache(tmp_path: Path) -> None:
    """A cache already in the manifest format is left untouched by the wipe."""
    storage = ContentStorage(tmp_path / "c.db")
    await storage.put("/new.py", _big_text(), time.time())
    before = (await storage.get_stats())["total_documents"]
    assert before > 0

    marker = tmp_path / ".docstore_manifest_v1"
    if marker.exists():
        marker.unlink()
    storage._migrate_chunk_manifest()

    assert marker.exists()
    assert (await storage.get_stats())["total_documents"] == before


async def test_failed_store_marks_index_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that fails partway must flag the eviction index for re-bootstrap,
    so it never trusts a path->doc_ids map that diverged from the DB."""
    storage = ContentStorage(tmp_path / "c.db")
    path = "/big.py"
    await storage.put(path, _big_text("orig"), time.time())
    # Force the index to load so `loaded` reflects real state, not lazy-unloaded.
    await storage._index.ensure_loaded(storage._collection.get_documents)
    assert storage._index.loaded is True

    async def boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated DB failure")

    # update_metadata runs last in reconcile, after rows are already mutated —
    # the exact moment the index can diverge from the DB.
    monkeypatch.setattr(storage._collection, "update_metadata", boom)

    with pytest.raises(RuntimeError, match="simulated DB failure"):
        await storage.put(path, _big_text("edit2"), time.time())
    assert storage._index.loaded is False
