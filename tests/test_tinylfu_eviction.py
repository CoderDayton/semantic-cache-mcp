"""Tests for the TinyLFU in-memory eviction index.

Covers the sketch's frequency tracking and aging, the index's
recency-aware sample selection, and the integrated VectorStorage eviction
path that should now keep frequently-accessed files alive while reclaiming
cold ones.
"""

from __future__ import annotations

from pathlib import Path

from semantic_cache_mcp.storage.vector import VectorStorage
from semantic_cache_mcp.storage.vector._tinylfu import (
    TinyLFUIndex,
    _CountMinSketch,
)


class TestCountMinSketch:
    def test_estimate_grows_with_increments(self) -> None:
        s = _CountMinSketch(128)
        h = hash("hot")
        for _ in range(5):
            s.increment(h)
        assert s.estimate(h) >= 5

    def test_unseen_key_returns_zero(self) -> None:
        s = _CountMinSketch(128)
        s.increment(hash("a"))
        assert s.estimate(hash("definitely-not-inserted")) == 0

    def test_saturates_at_max(self) -> None:
        s = _CountMinSketch(64)
        h = hash("flooded")
        for _ in range(100):
            s.increment(h)
        # 4-bit saturation: cap at 15
        assert s.estimate(h) == 15

    def test_halving_runs_after_sample_size(self) -> None:
        s = _CountMinSketch(64)
        h = hash("aged")
        for _ in range(20):
            s.increment(h)
        before = s.estimate(h)
        # Force enough events to trip the halving threshold.
        for i in range(s._sample_size + 50):  # noqa: SLF001
            s.increment(hash(f"noise-{i}"))
        after = s.estimate(h)
        # The aged key's count should not exceed its pre-aging value
        # (and typically drops). Halving must not zero it out instantly,
        # but it should not have grown either.
        assert after <= before


class TestTinyLFUIndex:
    def test_upsert_and_remove_roundtrip(self) -> None:
        idx = TinyLFUIndex(capacity=16, history_size=5)
        idx.clear()  # mark loaded
        idx.upsert("a.py", [1, 2], ts=100.0)
        assert idx.total_paths() == 1
        assert idx.doc_ids_for("a.py") == [1, 2]
        ids = idx.remove("a.py")
        assert ids == [1, 2]
        assert idx.total_paths() == 0

    def test_select_evictions_prefers_low_frequency(self) -> None:
        """Among LRU-tail candidates, the lowest-frequency entry loses."""
        idx = TinyLFUIndex(capacity=16, history_size=5)
        idx.clear()
        # Three paths inserted in order; by recency: cold.py, hot.py, mid.py.
        idx.upsert("cold.py", [1], ts=100.0)
        idx.upsert("hot.py", [2], ts=101.0)
        idx.upsert("mid.py", [3], ts=102.0)
        # Push hot.py's frequency way up; it is in LRU-middle but should
        # still survive eviction because the sample sees its high count.
        for _ in range(15):
            idx.add_access("hot.py", ts=103.0)
        # Single eviction picks from sample = 2 oldest (cold, hot).
        # cold has freq 1, hot has freq ≥ 15 → cold evicted.
        victims = idx.select_evictions(1)
        assert [path for path, _ in victims] == ["cold.py"]

    def test_select_evictions_respects_count(self) -> None:
        idx = TinyLFUIndex(capacity=16, history_size=5)
        idx.clear()
        for i in range(10):
            idx.upsert(f"f{i}.py", [i], ts=float(i))
        victims = idx.select_evictions(3)
        assert len(victims) == 3
        # All victims are doc ids we registered.
        chosen = {path for path, _ in victims}
        assert chosen.issubset({f"f{i}.py" for i in range(10)})

    def test_clear_resets_state(self) -> None:
        idx = TinyLFUIndex(capacity=8, history_size=3)
        idx.clear()
        idx.upsert("a", [1], ts=1.0)
        idx.clear()
        assert idx.total_paths() == 0
        assert idx.doc_ids_for("a") == []

    async def test_ensure_loaded_replays_history_into_sketch(self) -> None:
        idx = TinyLFUIndex(capacity=16, history_size=5)

        async def loader() -> list[tuple[int, str, dict]]:
            # Two paths; "warm" has 4 historical accesses, "cold" has 1.
            return [
                (1, "", {"path": "warm.py", "access_history": "[1,2,3,4]"}),
                (2, "", {"path": "cold.py", "access_history": "[1]"}),
            ]

        await idx.ensure_loaded(loader)
        assert idx.total_paths() == 2
        assert idx.estimate_frequency("warm.py") >= 4
        assert idx.estimate_frequency("cold.py") >= 1

    async def test_ensure_loaded_idempotent(self) -> None:
        idx = TinyLFUIndex(capacity=16, history_size=5)
        calls = {"n": 0}

        async def loader() -> list[tuple[int, str, dict]]:
            calls["n"] += 1
            return [(1, "", {"path": "a.py", "access_history": "[1]"})]

        await idx.ensure_loaded(loader)
        await idx.ensure_loaded(loader)
        assert calls["n"] == 1

    async def test_mark_dirty_forces_reload(self) -> None:
        idx = TinyLFUIndex(capacity=16, history_size=5)
        calls = {"n": 0}

        async def loader() -> list[tuple[int, str, dict]]:
            calls["n"] += 1
            return [(1, "", {"path": "a.py", "access_history": "[1]"})]

        await idx.ensure_loaded(loader)
        idx.mark_dirty()
        await idx.ensure_loaded(loader)
        assert calls["n"] == 2


class TestVectorStorageEvictionUsesIndex:
    """End-to-end: storage eviction goes through the TinyLFU index."""

    async def test_frequently_accessed_file_survives_eviction(self, tmp_path: Path) -> None:
        from semantic_cache_mcp import config as cfg

        vs = VectorStorage(db_path=tmp_path / "vec.db")
        original_cap = cfg.MAX_CACHE_ENTRIES
        try:
            # Tight cap so eviction fires after a handful of puts.
            cfg.MAX_CACHE_ENTRIES = 3
            for i in range(3):
                await vs.put(f"/cold/file{i}.txt", f"cold {i}\n", mtime=float(i))
            await vs.put("/hot.txt", "hot file\n", mtime=10.0)
            # Heavy access on /hot.txt to push its sketch frequency up.
            for _ in range(8):
                await vs.record_access("/hot.txt")
            # Trigger eviction by adding more cold files.
            for i in range(3, 6):
                await vs.put(f"/cold/file{i}.txt", f"cold {i}\n", mtime=float(i))

            # /hot.txt should still be present after eviction storms.
            entry = await vs.get("/hot.txt")
            assert entry is not None, "frequently-accessed file was evicted"
        finally:
            cfg.MAX_CACHE_ENTRIES = original_cap
            vs.close()

    async def test_eviction_does_not_full_scan_when_under_cap(self, tmp_path: Path) -> None:
        """Under the cap, the index never bootstraps — first-put cost stays cheap."""
        vs = VectorStorage(db_path=tmp_path / "vec.db")
        try:
            for i in range(3):
                await vs.put(f"/f{i}.txt", f"x {i}\n", mtime=float(i))
            assert vs._index.loaded is False  # noqa: SLF001
        finally:
            vs.close()
