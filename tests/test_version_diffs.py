"""Diffs from an older version the caller still holds.

A caller's ``known_hash`` names the bytes it holds. When the cache has moved
past them (another read refreshed the entry, or an edit the caller never saw
the result of), the server keeps the last few versions of the path, so it can
still answer with a diff from exactly those bytes instead of the whole file.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastmcp import Context

from semantic_cache_mcp.cache import SemanticCache
from semantic_cache_mcp.cache.read import batch_smart_read
from semantic_cache_mcp.config import MAX_CONTENT_SIZE
from semantic_cache_mcp.core.hashing import hash_content, short_hash
from semantic_cache_mcp.server.tools import batch_read, read
from semantic_cache_mcp.storage.docstore import ContentStorage


def _source(tag: str, lines: int = 80) -> str:
    return "".join(
        f"def fn_{i}():\n    return {i}  # {tag if i == 30 else 'same'}\n" for i in range(lines)
    )


class TestVersionStore:
    async def test_replacing_content_keeps_the_old_version(self, tmp_path: Path) -> None:
        storage = ContentStorage(tmp_path / "c.db")
        old, new = _source("A"), _source("B")
        await storage.put("/f.py", old, 1.0)
        await storage.put("/f.py", new, 2.0)
        assert await storage.version_text("/f.py", short_hash(hash_content(old))) == old
        assert await storage.version_text("/f.py", hash_content(old)) == old

    async def test_versions_are_capped_per_path(self, tmp_path: Path) -> None:
        storage = ContentStorage(tmp_path / "c.db")
        texts = [_source(f"v{i}") for i in range(7)]
        for i, text in enumerate(texts):
            await storage.put("/f.py", text, float(i))
        assert await storage.version_text("/f.py", hash_content(texts[0])) is None
        assert await storage.version_text("/f.py", hash_content(texts[5])) == texts[5]

    async def test_deleting_or_clearing_drops_versions(self, tmp_path: Path) -> None:
        storage = ContentStorage(tmp_path / "c.db")
        for path in ("/a.py", "/b.py"):
            await storage.put(path, _source("A"), 1.0)
            await storage.put(path, _source("B"), 2.0)
        old = hash_content(_source("A"))
        await storage.delete_path("/a.py")
        assert await storage.version_text("/a.py", old) is None
        assert await storage.version_text("/b.py", old) is not None
        await storage.clear()
        assert await storage.version_text("/b.py", old) is None

    async def test_oversized_content_is_not_kept(self, tmp_path: Path) -> None:
        storage = ContentStorage(tmp_path / "c.db")
        big = _source("A", lines=MAX_CONTENT_SIZE // 30)
        assert len(big.encode()) > MAX_CONTENT_SIZE
        await storage.put("/f.py", big, 1.0)
        await storage.put("/f.py", _source("B"), 2.0)
        assert await storage.version_text("/f.py", hash_content(big)) is None

    async def test_a_corrupt_version_is_refused(self, tmp_path: Path) -> None:
        db = tmp_path / "c.db"
        storage = ContentStorage(db)
        old = _source("A")
        await storage.put("/f.py", old, 1.0)
        await storage.put("/f.py", _source("B"), 2.0)
        storage._db.conn.execute("UPDATE versions SET body = ?", (b"not zlib",))
        assert await storage.version_text("/f.py", hash_content(old)) is None

    async def test_unchanged_content_keeps_no_version(self, tmp_path: Path) -> None:
        storage = ContentStorage(tmp_path / "c.db")
        await storage.put("/f.py", _source("A"), 1.0)
        await storage.put("/f.py", _source("A"), 2.0)
        conn: sqlite3.Connection = storage._db.conn
        assert conn.execute("SELECT COUNT(*) FROM versions").fetchone()[0] == 0


@pytest.fixture
def cache(tmp_path: Path) -> SemanticCache:
    return SemanticCache(db_path=tmp_path / "cache.db")


@pytest.fixture
def ctx(cache: SemanticCache) -> MagicMock:
    context = MagicMock(spec=Context)
    context.lifespan_context = {"cache": cache}
    return context


def _rewrite(path: Path, text: str) -> None:
    path.write_text(text)
    stamp = time.time() + 5
    os.utime(path, (stamp, stamp))


class TestReadFromAnOlderVersion:
    async def test_read_diffs_from_the_held_version(self, ctx: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "mod.py"
        source.write_text(_source("A"))
        held = (await read(ctx, str(source)))["content_hash"]
        _rewrite(source, _source("B"))
        await read(ctx, str(source))

        payload = await read(ctx, str(source), known_hash=held)

        assert payload.get("is_diff") is True
        assert "+    return 30  # B" in payload["content"]
        assert "-    return 30  # A" in payload["content"]
        assert payload["content_hash"] == short_hash(hash_content(_source("B")))

    async def test_batch_read_diffs_from_the_held_version(
        self, ctx: MagicMock, tmp_path: Path
    ) -> None:
        source = tmp_path / "mod.py"
        source.write_text(_source("A"))
        held = (await read(ctx, str(source)))["content_hash"]
        _rewrite(source, _source("B"))
        await read(ctx, str(source))

        payload = await batch_read(ctx, str(source), known_hashes=f'{{"{source}": "{held}"}}')

        text = str(payload)
        assert "+    return 30  # B" in text
        assert "def fn_0" not in text

    async def test_an_unknown_hash_gets_the_full_file(self, ctx: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "mod.py"
        source.write_text(_source("A"))
        await read(ctx, str(source))
        _rewrite(source, _source("B"))
        await read(ctx, str(source))

        payload = await read(ctx, str(source), known_hash="0" * 16)

        assert payload.get("is_diff") is not True
        assert payload["content"] == _source("B")

    async def test_a_diff_over_max_size_is_not_sent(self, ctx: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "mod.py"
        source.write_text(_source("A"))
        held = (await read(ctx, str(source)))["content_hash"]
        _rewrite(source, _source("B"))
        await read(ctx, str(source))

        payload = await read(ctx, str(source), known_hash=held, max_size=100)

        assert payload.get("is_diff") is not True
        assert payload.get("truncated") is True

    async def test_batch_without_diff_mode_sends_full_content(
        self, cache: SemanticCache, ctx: MagicMock, tmp_path: Path
    ) -> None:
        source = tmp_path / "mod.py"
        source.write_text(_source("A"))
        held = (await read(ctx, str(source)))["content_hash"]
        _rewrite(source, _source("B"))
        await read(ctx, str(source))

        result = await batch_smart_read(
            cache, [str(source)], diff_mode=False, known_hashes={str(source): held}
        )

        assert result.contents[str(source)] == _source("B")

    async def test_clear_drops_versions_when_no_documents_remain(self, tmp_path: Path) -> None:
        storage = ContentStorage(tmp_path / "c.db")
        old = _source("A")
        await storage.put("/f.py", old, 1.0)
        await storage.put("/f.py", _source("B"), 2.0)
        storage._db.conn.execute("DELETE FROM documents")
        await storage.clear()
        assert await storage.version_text("/f.py", hash_content(old)) is None


class TestRevertToTheHeldVersion:
    async def test_read_answers_unchanged(self, ctx: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "mod.py"
        source.write_text(_source("A"))
        held = (await read(ctx, str(source)))["content_hash"]
        _rewrite(source, _source("B"))
        await read(ctx, str(source))
        _rewrite(source, _source("A"))

        payload = await read(ctx, str(source), known_hash=held)

        assert payload.get("unchanged") is True
        assert "content" not in payload
        assert payload["content_hash"] == held

    async def test_batch_read_answers_unchanged(self, ctx: MagicMock, tmp_path: Path) -> None:
        source = tmp_path / "mod.py"
        source.write_text(_source("A"))
        held = (await read(ctx, str(source)))["content_hash"]
        _rewrite(source, _source("B"))
        await read(ctx, str(source))
        _rewrite(source, _source("A"))

        payload = await batch_read(ctx, str(source), known_hashes=f'{{"{source}": "{held}"}}')

        assert "def fn_0" not in str(payload)
        assert "unchanged" in str(payload)


class TestVersionUpkeep:
    async def test_previous_text_skips_reassembly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        storage = ContentStorage(tmp_path / "c.db")
        old = _source("A")
        await storage.put("/f.py", old, 1.0)

        async def _no_reassembly(*_args: object, **_kwargs: object) -> str:
            raise AssertionError("reassembled the previous version")

        monkeypatch.setattr(ContentStorage, "get_content", _no_reassembly)
        await storage.put("/f.py", _source("B"), 2.0, previous=old)
        monkeypatch.undo()
        assert await storage.version_text("/f.py", hash_content(old)) == old

    async def test_a_previous_text_that_is_not_the_cached_one_is_ignored(
        self, tmp_path: Path
    ) -> None:
        storage = ContentStorage(tmp_path / "c.db")
        cached = _source("A")
        await storage.put("/f.py", cached, 1.0)
        await storage.put("/f.py", _source("B"), 2.0, previous=_source("disk"))
        assert await storage.version_text("/f.py", hash_content(cached)) == cached
        assert await storage.version_text("/f.py", hash_content(_source("disk"))) is None

    async def test_orphan_pruning_keeps_versions_of_a_live_path(self, tmp_path: Path) -> None:
        storage = ContentStorage(tmp_path / "c.db")
        old = _source("A")
        await storage.put("/f.py", old, 1.0)
        await storage.put("/f.py", _source("B"), 2.0)
        await storage._collection.delete_versions(["/f.py"], orphaned_only=True)
        assert await storage.version_text("/f.py", hash_content(old)) == old
        await storage._delete_by_path("/f.py")
        await storage._collection.delete_versions(["/f.py"], orphaned_only=True)
        assert await storage.version_text("/f.py", hash_content(old)) is None

    async def test_a_path_change_moves_the_indexed_column(self, tmp_path: Path) -> None:
        storage = ContentStorage(tmp_path / "c.db")
        await storage.put("/old.py", "text\n", 1.0)
        head = await storage._collection.head_metadata("/old.py")
        assert head is not None
        await storage._collection.update_metadata([(head[0], {"path": "/new.py"})])
        assert await storage._collection.head_metadata("/old.py") is None
        assert await storage._collection.head_metadata("/new.py") is not None
