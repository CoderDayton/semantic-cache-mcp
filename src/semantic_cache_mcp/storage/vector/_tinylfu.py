"""TinyLFU-inspired in-memory eviction index for VectorStorage.

Replaces the per-put O(N) `get_documents()` scan in `_evict_if_needed`:
tracks doc IDs per cached path, approximates per-path access frequency
with a 4-bit packed Count-Min sketch (periodic halving = TinyLFU "freshen"),
and orders paths by recency. Eviction samples the LRU tail and picks the
lowest-frequency victims (W-TinyLFU's main eviction rule, batched).

The DB stays the source of truth: `record_access` still persists history
into document metadata; the index replays that history into the sketch on
bootstrap so frequency carries across restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = ["TinyLFUIndex"]


# SplitMix64 constants — used to derive depth-independent hashes from a
# single key hash. Avoids constructing N hash functions; one mix per row is
# enough for Count-Min's accuracy guarantees in practice.
_HASH_SALT = (
    0x9E3779B97F4A7C15,
    0xBF58476D1CE4E5B9,
    0x94D049BB133111EB,
    0xD6E8FEB86659FD93,
)
_U64 = 0xFFFFFFFFFFFFFFFF


# Pre-built byte translation table for CountMinSketch._halve. Each byte packs
# two 4-bit counters; `(b >> 1) & 0x77` is equivalent to halving the high and
# low nibbles independently (the mask drops each counter's LSB, which is the
# bit that would otherwise cross the nibble boundary). bytes.translate is a
# single C call; verified byte-for-byte against the Python loop in tests.
_HALVE_TABLE = bytes((b >> 1) & 0x77 for b in range(256))


def _mix(key_hash: int, salt: int) -> int:
    x = (key_hash ^ salt) & _U64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _U64
    return x ^ (x >> 31)


class _CountMinSketch:
    """4-bit packed Count-Min sketch with halve-on-saturation aging.

    Width = next power of two ≥ capacity (min 64); 4 hash rows; counters
    saturate at 15. After `sample_size` (10× capacity) events every counter
    is halved — the TinyLFU "freshen" pass.
    """

    __slots__ = ("_width", "_mask", "_table", "_sample_size", "_events")

    _DEPTH = 4
    _MAX = 15

    def __init__(self, capacity: int) -> None:
        width = 1 << max(6, max(capacity, 1).bit_length())
        self._width = width
        self._mask = width - 1
        self._table = [bytearray(width // 2) for _ in range(self._DEPTH)]
        # 10× capacity is the W-TinyLFU sample size used by Caffeine.
        self._sample_size = max(width, capacity) * 10
        self._events = 0

    def _slot(self, key_hash: int, row: int) -> int:
        return _mix(key_hash, _HASH_SALT[row]) & self._mask

    def increment(self, key_hash: int) -> None:
        for row in range(self._DEPTH):
            slot = self._slot(key_hash, row)
            byte_idx = slot >> 1
            byte = self._table[row][byte_idx]
            if slot & 1:
                cur = byte >> 4
                if cur < self._MAX:
                    self._table[row][byte_idx] = (byte & 0x0F) | ((cur + 1) << 4)
            else:
                cur = byte & 0x0F
                if cur < self._MAX:
                    self._table[row][byte_idx] = (byte & 0xF0) | (cur + 1)
        self._events += 1
        if self._events >= self._sample_size:
            self._halve()

    def estimate(self, key_hash: int) -> int:
        best = self._MAX
        for row in range(self._DEPTH):
            slot = self._slot(key_hash, row)
            byte = self._table[row][slot >> 1]
            cur = (byte >> 4) if (slot & 1) else (byte & 0x0F)
            if cur < best:
                best = cur
        return best

    def _halve(self) -> None:
        # Halve every 4-bit counter via a precomputed translation table:
        # bytes.translate is a single C call vs. a Python-bytecode loop over
        # every byte. Equivalent to the prior `(b >> 1) & 0x77` per-byte op.
        for row in self._table:
            row[:] = bytes(row).translate(_HALVE_TABLE)
        self._events //= 2

    @property
    def events(self) -> int:
        return self._events


@dataclass(slots=True)
class _Entry:
    doc_ids: list[int]
    last_access: float
    history: list[float] = field(default_factory=list)


# Shape of one document tuple from AsyncVectorCollection.get_documents():
# (doc_id, text, metadata). We only use doc_id and metadata.
DocLoader = Callable[[], Awaitable[Iterable[tuple[int, str, dict]]]]


class TinyLFUIndex:
    """In-memory frequency+recency index used by VectorStorage eviction.

    Concurrency model: methods that mutate the index are sync — under
    cooperative asyncio they run to completion without yielding the loop,
    so dict mutations are safe. The async `ensure_loaded()` uses a lock to
    serialize concurrent first-time bootstraps.
    """

    __slots__ = (
        "_entries",
        "_lru",
        "_sketch",
        "_capacity",
        "_history_size",
        "_lock",
        "_loaded",
        "_dirty",
    )

    # When sampling LRU candidates for eviction, look at this multiple of
    # the eviction count so frequency has room to break ties.
    _TAIL_SAMPLE_FACTOR = 2

    def __init__(self, capacity: int, history_size: int) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._sketch = _CountMinSketch(capacity)
        self._capacity = capacity
        self._history_size = max(1, history_size)
        self._lock = asyncio.Lock()
        self._loaded = False
        self._dirty = False

    @staticmethod
    def _hash(path: str) -> int:
        # CPython hash() is salted per process — fine for an in-memory sketch.
        return hash(path)

    @property
    def loaded(self) -> bool:
        return self._loaded and not self._dirty

    def total_paths(self) -> int:
        return len(self._entries)

    def doc_ids_for(self, path: str) -> list[int]:
        entry = self._entries.get(path)
        return list(entry.doc_ids) if entry is not None else []

    def estimate_frequency(self, path: str) -> int:
        return self._sketch.estimate(self._hash(path))

    def mark_dirty(self) -> None:
        """Force a re-bootstrap on the next `ensure_loaded()` call.

        Use after a partial DB failure leaves the index potentially out of
        sync with persisted state. Cost: one full collection scan; worth
        it for safety.
        """
        self._dirty = True

    async def ensure_loaded(self, load_all: DocLoader) -> None:
        """Bootstrap from the underlying collection on first use (or after dirty).

        Replays each document's persisted access history into the sketch so
        frequency information survives process restarts.
        """
        if self._loaded and not self._dirty:
            return
        async with self._lock:
            if self._loaded and not self._dirty:
                return
            self._entries.clear()
            self._lru.clear()
            self._sketch = _CountMinSketch(self._capacity)
            for doc_id, _text, meta in await load_all():
                path = meta.get("path", "")
                if not path:
                    continue
                entry = self._entries.get(path)
                if entry is None:
                    raw = meta.get("access_history", "[]")
                    history = self._parse_history(raw)
                    last = max(history) if history else 0.0
                    entry = _Entry(doc_ids=[doc_id], last_access=last, history=history)
                    self._entries[path] = entry
                    h = self._hash(path)
                    for _ in history:
                        self._sketch.increment(h)
                else:
                    entry.doc_ids.append(doc_id)
            for path, _ in sorted(self._entries.items(), key=lambda kv: kv[1].last_access):
                self._lru[path] = None
            self._loaded = True
            self._dirty = False
            logger.debug(f"TinyLFUIndex bootstrap: {len(self._entries)} paths loaded")

    @staticmethod
    def _parse_history(raw: object) -> list[float]:
        if isinstance(raw, list):
            return [float(t) for t in raw]
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                return []
            return [float(t) for t in parsed] if isinstance(parsed, list) else []
        return []

    def upsert(self, path: str, doc_ids: Iterable[int], ts: float) -> None:
        """Register or replace doc_ids for a path; record an access at `ts`."""
        if not path:
            return
        existing = self._entries.get(path)
        history = list(existing.history) if existing is not None else []
        history.append(ts)
        if len(history) > self._history_size:
            history = history[-self._history_size :]
        self._entries[path] = _Entry(
            doc_ids=list(doc_ids),
            last_access=ts,
            history=history,
        )
        self._lru.pop(path, None)
        self._lru[path] = None
        self._sketch.increment(self._hash(path))

    def add_access(self, path: str, ts: float) -> None:
        entry = self._entries.get(path)
        if entry is None:
            return
        entry.last_access = ts
        entry.history.append(ts)
        if len(entry.history) > self._history_size:
            entry.history = entry.history[-self._history_size :]
        self._lru.move_to_end(path)
        self._sketch.increment(self._hash(path))

    def remove(self, path: str) -> list[int]:
        entry = self._entries.pop(path, None)
        self._lru.pop(path, None)
        if entry is None:
            return []
        return entry.doc_ids

    def clear(self) -> None:
        self._entries.clear()
        self._lru.clear()
        self._sketch = _CountMinSketch(self._capacity)
        self._loaded = True
        self._dirty = False

    def select_evictions(self, evict_count: int) -> list[tuple[str, list[int]]]:
        """Return up to `evict_count` (path, doc_ids) pairs to delete.

        Sample the oldest `_TAIL_SAMPLE_FACTOR × evict_count` paths from the
        LRU; rank by (frequency asc, last_access asc); take the top
        `evict_count`. Lowest-frequency oldest entries lose first.
        """
        if evict_count <= 0 or not self._lru:
            return []
        sample_size = min(len(self._lru), evict_count * self._TAIL_SAMPLE_FACTOR)
        sample: list[tuple[str, int, float]] = []
        for path in self._lru:
            entry = self._entries.get(path)
            if entry is None:
                continue
            freq = self._sketch.estimate(self._hash(path))
            sample.append((path, freq, entry.last_access))
            if len(sample) >= sample_size:
                break
        sample.sort(key=lambda t: (t[1], t[2]))
        out: list[tuple[str, list[int]]] = []
        for path, _, _ in sample[:evict_count]:
            entry = self._entries.get(path)
            if entry is not None:
                out.append((path, list(entry.doc_ids)))
        return out
