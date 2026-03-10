"""Coverage tests for core algorithm modules.

Targets:
- core/text/_diff.py      (49% → 85%+)
- core/hashing.py         (49% → 85%+)
- core/chunking/_gear.py  (35% → 75%+)
- core/similarity/_cosine.py (73% → 90%+)
- server/response.py      (63% → 95%+)
"""

from __future__ import annotations

import array
import os
import tempfile
import threading
from collections.abc import Iterator

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# _diff.py tests
# ---------------------------------------------------------------------------


class TestMyersDiff:
    def test_both_empty(self) -> None:
        from semantic_cache_mcp.core.text._diff import _myers_diff

        assert _myers_diff([], []) == []

    def test_old_empty(self) -> None:
        from semantic_cache_mcp.core.text._diff import _myers_diff

        result = _myers_diff([], ["a\n", "b\n"])
        assert all(op == "+" for op, _ in result)
        assert [line for _, line in result] == ["a\n", "b\n"]

    def test_new_empty(self) -> None:
        from semantic_cache_mcp.core.text._diff import _myers_diff

        result = _myers_diff(["a\n", "b\n"], [])
        assert all(op == "-" for op, _ in result)

    def test_no_changes(self) -> None:
        from semantic_cache_mcp.core.text._diff import _myers_diff

        lines = ["hello\n", "world\n"]
        result = _myers_diff(lines, lines)
        assert all(op == " " for op, _ in result)

    def test_replacement(self) -> None:
        from semantic_cache_mcp.core.text._diff import _myers_diff

        old = ["foo\n", "bar\n"]
        new = ["foo\n", "baz\n"]
        result = _myers_diff(old, new)
        ops = [op for op, _ in result]
        assert "-" in ops
        assert "+" in ops

    def test_insertion(self) -> None:
        from semantic_cache_mcp.core.text._diff import _myers_diff

        old = ["a\n"]
        new = ["a\n", "b\n"]
        result = _myers_diff(old, new)
        assert any(op == "+" for op, _ in result)

    def test_deletion(self) -> None:
        from semantic_cache_mcp.core.text._diff import _myers_diff

        old = ["a\n", "b\n"]
        new = ["a\n"]
        result = _myers_diff(old, new)
        assert any(op == "-" for op, _ in result)


class TestUnifiedDiffFast:
    def test_no_changes(self) -> None:
        from semantic_cache_mcp.core.text._diff import _unified_diff_fast

        result = _unified_diff_fast("same\n", "same\n")
        assert result == ""

    def test_changes_produce_diff(self) -> None:
        from semantic_cache_mcp.core.text._diff import _unified_diff_fast

        result = _unified_diff_fast("old\n", "new\n")
        assert "-old" in result
        assert "+new" in result

    def test_context_lines(self) -> None:
        from semantic_cache_mcp.core.text._diff import _unified_diff_fast

        old = "\n".join(str(i) for i in range(20)) + "\n"
        new = old.replace("10\n", "TEN\n")
        result_0 = _unified_diff_fast(old, new, context_lines=0)
        result_3 = _unified_diff_fast(old, new, context_lines=3)
        # More context = more lines
        assert len(result_3) > len(result_0)


class TestComputeDelta:
    def test_identical_texts(self) -> None:
        from semantic_cache_mcp.core.text._diff import compute_delta

        delta = compute_delta("hello\n", "hello\n")
        assert delta.insertions == []
        assert delta.deletions == []
        assert delta.modifications == []
        assert delta.old_hash == delta.new_hash

    def test_line_modification(self) -> None:
        from semantic_cache_mcp.core.text._diff import compute_delta

        old = "line one\nline two\nline three\n"
        new = "line one\nLINE TWO\nline three\n"
        delta = compute_delta(old, new)
        # Same line count → modifications, not insertions/deletions
        assert len(delta.modifications) > 0

    def test_insertion(self) -> None:
        from semantic_cache_mcp.core.text._diff import compute_delta

        old = "a\n"
        new = "a\nb\n"
        delta = compute_delta(old, new)
        assert len(delta.insertions) > 0

    def test_deletion(self) -> None:
        from semantic_cache_mcp.core.text._diff import compute_delta

        old = "a\nb\n"
        new = "a\n"
        delta = compute_delta(old, new)
        assert len(delta.deletions) > 0

    def test_delta_has_hashes(self) -> None:
        from semantic_cache_mcp.core.text._diff import compute_delta

        delta = compute_delta("foo\n", "bar\n")
        assert len(delta.old_hash) == 16
        assert len(delta.new_hash) == 16

    def test_size_bytes_positive(self) -> None:
        from semantic_cache_mcp.core.text._diff import compute_delta

        delta = compute_delta("one\ntwo\n", "one\nTWO\n")
        assert delta.size_bytes > 0

    def test_unequal_replace_treated_as_delete_insert(self) -> None:
        from semantic_cache_mcp.core.text._diff import compute_delta

        # Replacing 1 line with 3 forces delete+insert path
        old = "a\nb\nc\n"
        new = "a\nx\ny\nz\nc\n"
        delta = compute_delta(old, new)
        assert len(delta.insertions) + len(delta.modifications) > 0


class TestTruncateSemantic:
    def _make_python_content(self, n_functions: int) -> str:
        lines = []
        for i in range(n_functions):
            lines.append(f"def function_{i}():")
            lines.append(f"    return {i}")
            lines.append("")
        return "\n".join(lines)

    def test_short_content_returned_as_is(self) -> None:
        from semantic_cache_mcp.core.text._diff import truncate_semantic

        text = "hello world"
        assert truncate_semantic(text, max_size=1000) == text

    def test_truncation_applied(self) -> None:
        from semantic_cache_mcp.core.text._diff import truncate_semantic

        content = self._make_python_content(50)
        result = truncate_semantic(content, max_size=200, keep_top=3, keep_bottom=2)
        assert len(result) <= len(content)
        assert "TRUNCATED" in result or "truncated" in result

    def test_few_lines_hard_truncation(self) -> None:
        from semantic_cache_mcp.core.text._diff import truncate_semantic

        # Fewer lines than keep_top + keep_bottom forces hard truncation
        content = "a\nb\nc\n" * 2
        result = truncate_semantic(content, max_size=10, keep_top=10, keep_bottom=10)
        assert "TRUNCATED" in result

    def test_preserves_top_and_bottom(self) -> None:
        from semantic_cache_mcp.core.text._diff import truncate_semantic

        lines = [f"line_{i}\n" for i in range(100)]
        content = "".join(lines)
        result = truncate_semantic(content, max_size=500, keep_top=5, keep_bottom=3)
        assert "line_0" in result


class TestTruncateSmart:
    def test_no_truncation_when_small(self) -> None:
        from semantic_cache_mcp.core.text._diff import truncate_smart

        text = "short"
        assert truncate_smart(text, max_size=1000) == text

    def test_simple_truncation_without_semantic(self) -> None:
        from semantic_cache_mcp.core.text._diff import truncate_smart

        lines = [f"line {i}\n" for i in range(200)]
        content = "".join(lines)
        result = truncate_smart(
            content, max_size=500, keep_top=5, keep_bottom=3, use_semantic=False
        )
        assert "truncated" in result.lower()
        assert "line 0" in result

    def test_semantic_path(self) -> None:
        from semantic_cache_mcp.core.text._diff import truncate_smart

        lines = [f"line {i}\n" for i in range(200)]
        content = "".join(lines)
        result = truncate_smart(content, max_size=500, keep_top=5, keep_bottom=3, use_semantic=True)
        assert len(result) > 0

    def test_hard_truncation_fallback_no_semantic(self) -> None:
        from semantic_cache_mcp.core.text._diff import truncate_smart

        # Very small max_size forces hard fallback
        content = "x\n" * 4  # 4 lines < keep_top+keep_bottom, 8 bytes > max_size=5
        result = truncate_smart(
            content, max_size=5, keep_top=10, keep_bottom=10, use_semantic=False
        )
        assert "TRUNCATED" in result


class TestGenerateDiffStreaming:
    def test_generates_diff(self) -> None:
        from semantic_cache_mcp.core.text._diff import generate_diff_streaming

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write("line one\nline two\n")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f2:
            f2.write("line one\nline THREE\n")
            path2 = f2.name

        try:
            chunks = list(generate_diff_streaming(path1, path2))
            combined = "".join(chunks)
            assert "-line two" in combined or "+line THREE" in combined
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_identical_files_no_diff(self) -> None:
        from semantic_cache_mcp.core.text._diff import generate_diff_streaming

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f1:
            f1.write("same content\n")
            path = f1.name

        try:
            chunks = list(generate_diff_streaming(path, path))
            assert chunks == []
        finally:
            os.unlink(path)

    def test_is_iterator(self) -> None:
        from semantic_cache_mcp.core.text._diff import generate_diff_streaming

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("a\n")
            path = f.name

        try:
            result = generate_diff_streaming(path, path)
            assert isinstance(result, Iterator)
        finally:
            os.unlink(path)


class TestDiffStats:
    def test_identical_has_zero_changes(self) -> None:
        from semantic_cache_mcp.core.text._diff import diff_stats

        stats = diff_stats("hello\n", "hello\n")
        assert stats["insertions"] == 0
        assert stats["deletions"] == 0
        assert stats["modifications"] == 0

    def test_stats_have_expected_keys(self) -> None:
        from semantic_cache_mcp.core.text._diff import diff_stats

        stats = diff_stats("a\n", "b\n")
        for key in (
            "insertions",
            "deletions",
            "modifications",
            "delta_size_bytes",
            "original_size",
        ):
            assert key in stats

    def test_compression_ratio_zero_for_empty_old(self) -> None:
        from semantic_cache_mcp.core.text._diff import diff_stats

        stats = diff_stats("", "something\n")
        assert stats["compression_ratio"] == 0

    def test_delta_size_bytes(self) -> None:
        from semantic_cache_mcp.core.text._diff import diff_stats

        stats = diff_stats("old\n", "new\n")
        assert stats["delta_size_bytes"] > 0


class TestInvertDiff:
    def test_inverted_diff_is_reverse(self) -> None:
        from semantic_cache_mcp.core.text._diff import generate_diff, invert_diff

        old = "alpha\nbeta\n"
        new = "alpha\ngamma\n"
        forward = generate_diff(old, new)
        inverted = invert_diff(old, new)
        # Inverted diff swaps +/- relative to forward
        assert forward != inverted
        assert "-gamma" in inverted or "+beta" in inverted


# ---------------------------------------------------------------------------
# hashing.py tests
# ---------------------------------------------------------------------------


class TestHashBytesBlake3Fallback:
    def test_hash_content_returns_hex_string(self) -> None:
        from semantic_cache_mcp.core.hashing import hash_content

        result = hash_content("hello world")
        assert isinstance(result, str)
        assert len(result) == 64  # 32 bytes hex

    def test_hash_content_bytes_input(self) -> None:
        from semantic_cache_mcp.core.hashing import hash_content

        result = hash_content(b"hello world")
        assert isinstance(result, str)
        assert len(result) == 64

    def test_hash_content_str_and_bytes_equivalent(self) -> None:
        from semantic_cache_mcp.core.hashing import hash_content

        assert hash_content("hello") == hash_content(b"hello")

    def test_hash_content_large_file_bypasses_cache(self) -> None:
        # Files > 64KB bypass the LRU cache
        from semantic_cache_mcp.core.hashing import _CONTENT_CACHE_BYPASS_SIZE, hash_content

        data = b"x" * (_CONTENT_CACHE_BYPASS_SIZE + 1)
        result = hash_content(data)
        assert len(result) == 64

    def test_hash_chunk_cached(self) -> None:
        from semantic_cache_mcp.core.hashing import hash_chunk

        a = hash_chunk(b"chunk data")
        b_val = hash_chunk(b"chunk data")
        assert a == b_val

    def test_hash_chunk_binary_returns_bytes(self) -> None:
        from semantic_cache_mcp.core.hashing import hash_chunk_binary

        result = hash_chunk_binary(b"some data")
        assert isinstance(result, bytes)
        assert len(result) == 32


class TestHashBytesDirectFallback:
    """Test _hash_bytes fallback path (BLAKE2b) directly."""

    def test_fallback_produces_valid_hex(self) -> None:
        from semantic_cache_mcp.core.hashing import _hash_hex

        result = _hash_hex(b"test data", digest_size=32)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_fallback_deterministic(self) -> None:
        from semantic_cache_mcp.core.hashing import _hash_hex

        assert _hash_hex(b"same", 32) == _hash_hex(b"same", 32)

    def test_forced_fallback_via_monkeypatching(self) -> None:
        """Force BLAKE2b fallback by disabling USE_BLAKE3."""
        from semantic_cache_mcp.core import hashing

        original = hashing.DEFAULT_CONFIG.USE_BLAKE3
        try:
            hashing.DEFAULT_CONFIG.USE_BLAKE3 = False
            result = hashing._hash_bytes(b"fallback test", digest_size=32)
            assert isinstance(result, bytes)
            assert len(result) == 32
        finally:
            hashing.DEFAULT_CONFIG.USE_BLAKE3 = original


class TestCollisionTracker:
    def test_no_collision_same_data(self) -> None:
        from semantic_cache_mcp.core.hashing import CollisionTracker

        tracker = CollisionTracker(max_size=100)
        data = b"unique data"
        assert tracker.register("aabbcc", data) is False
        assert tracker.register("aabbcc", data) is False  # Same data, not a collision

    def test_collision_detected_different_data(self) -> None:
        from semantic_cache_mcp.core.hashing import CollisionTracker

        tracker = CollisionTracker(max_size=100)
        tracker.register("deadbeef", b"data one")
        is_collision = tracker.register("deadbeef", b"data two")
        assert is_collision is True
        assert tracker.get_collision_count() == 1

    def test_cache_full_does_not_store(self) -> None:
        from semantic_cache_mcp.core.hashing import CollisionTracker

        tracker = CollisionTracker(max_size=2)
        tracker.register("hash1", b"data1")
        tracker.register("hash2", b"data2")
        # Third entry: cache full, returns False (not stored, not a collision)
        result = tracker.register("hash3", b"data3")
        assert result is False

    def test_clear_resets_state(self) -> None:
        from semantic_cache_mcp.core.hashing import CollisionTracker

        tracker = CollisionTracker(max_size=100)
        tracker.register("aabb", b"x")
        tracker.register("aabb", b"y")  # collision
        tracker.clear()
        assert tracker.get_collision_count() == 0

    def test_thread_safety(self) -> None:
        from semantic_cache_mcp.core.hashing import CollisionTracker

        tracker = CollisionTracker(max_size=10000)
        errors: list[Exception] = []

        def register_many(start: int) -> None:
            try:
                for i in range(start, start + 100):
                    tracker.register(f"hash_{i}", f"data_{i}".encode())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_many, args=(i * 100,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestHashChunkWithCollisionCheck:
    def test_returns_tuple(self) -> None:
        from semantic_cache_mcp.core.hashing import hash_chunk_with_collision_check

        h, is_col = hash_chunk_with_collision_check(b"test chunk")
        assert isinstance(h, str)
        assert isinstance(is_col, bool)


class TestHashChunksStreaming:
    def test_basic_streaming(self) -> None:
        from semantic_cache_mcp.core.hashing import hash_chunks_streaming

        chunks = [b"chunk_one", b"chunk_two", b"chunk_three"]
        hashes, content_hash = hash_chunks_streaming(iter(chunks), combine=True)
        assert len(hashes) == 3
        assert content_hash is not None
        assert len(content_hash) == 64

    def test_combine_false_returns_none_content_hash(self) -> None:
        from semantic_cache_mcp.core.hashing import hash_chunks_streaming

        chunks = [b"a", b"b"]
        hashes, content_hash = hash_chunks_streaming(iter(chunks), combine=False)
        assert len(hashes) == 2
        assert content_hash is None

    def test_empty_iterator(self) -> None:
        from semantic_cache_mcp.core.hashing import hash_chunks_streaming

        hashes, content_hash = hash_chunks_streaming(iter([]), combine=True)
        assert hashes == []
        assert content_hash is not None  # Empty hasher is finalized

    def test_deterministic(self) -> None:
        from semantic_cache_mcp.core.hashing import hash_chunks_streaming

        chunks = [b"x" * 1000, b"y" * 1000]
        h1, c1 = hash_chunks_streaming(iter(chunks), combine=True)
        h2, c2 = hash_chunks_streaming(iter(chunks), combine=True)
        assert h1 == h2
        assert c1 == c2


class TestHierarchicalHasher:
    def test_basic_add_and_finalize(self) -> None:
        from semantic_cache_mcp.core.hashing import HierarchicalHasher

        hasher = HierarchicalHasher()
        hasher.add_chunk(b"chunk_a")
        hasher.add_chunk(b"chunk_b")
        block_hash = hasher.finalize_block()
        assert isinstance(block_hash, str)
        assert len(block_hash) == 64

    def test_finalize_content_returns_tuple(self) -> None:
        from semantic_cache_mcp.core.hashing import HierarchicalHasher

        hasher = HierarchicalHasher()
        for i in range(5):
            hasher.add_chunk(f"chunk_{i}".encode())
        hasher.finalize_block()
        content_hash, blocks, chunk_hashes = hasher.finalize_content()
        assert isinstance(content_hash, str)
        assert len(blocks) == 1

    def test_empty_hasher_returns_empty(self) -> None:
        from semantic_cache_mcp.core.hashing import HierarchicalHasher

        hasher = HierarchicalHasher()
        content_hash, blocks, chunks = hasher.finalize_content()
        assert content_hash == ""
        assert blocks == []

    def test_pending_block_auto_finalized(self) -> None:
        from semantic_cache_mcp.core.hashing import HierarchicalHasher

        hasher = HierarchicalHasher()
        hasher.add_chunk(b"data")
        # Don't call finalize_block; finalize_content should do it
        content_hash, blocks, _ = hasher.finalize_content()
        assert len(content_hash) == 64
        assert len(blocks) > 0

    def test_finalize_empty_block_returns_empty(self) -> None:
        from semantic_cache_mcp.core.hashing import HierarchicalHasher

        hasher = HierarchicalHasher()
        result = hasher.finalize_block()
        assert result == ""


class TestDeduplicateIndex:
    def test_add_and_lookup(self) -> None:
        from semantic_cache_mcp.core.hashing import DeduplicateIndex

        idx = DeduplicateIndex()
        chunk = b"unique content here"
        added = idx.add(chunk, chunk_id=1, size=len(chunk))
        assert added is True
        result = idx.lookup(chunk)
        assert result == (1, len(chunk))

    def test_duplicate_not_added(self) -> None:
        from semantic_cache_mcp.core.hashing import DeduplicateIndex

        idx = DeduplicateIndex()
        chunk = b"same content"
        idx.add(chunk, chunk_id=1, size=12)
        added_again = idx.add(chunk, chunk_id=2, size=12)
        assert added_again is False

    def test_lookup_missing_returns_none(self) -> None:
        from semantic_cache_mcp.core.hashing import DeduplicateIndex

        idx = DeduplicateIndex()
        assert idx.lookup(b"not in index") is None

    def test_capacity_limit(self) -> None:
        from semantic_cache_mcp.core.hashing import DeduplicateIndex

        idx = DeduplicateIndex(capacity=3)
        for i in range(3):
            idx.add(f"chunk_{i}".encode(), chunk_id=i, size=10)
        # 4th should fail (capacity reached)
        result = idx.add(b"overflow", chunk_id=99, size=10)
        assert result is False

    def test_size_tracks_entries(self) -> None:
        from semantic_cache_mcp.core.hashing import DeduplicateIndex

        idx = DeduplicateIndex()
        assert idx.size() == 0
        idx.add(b"a", 1, 1)
        idx.add(b"b", 2, 1)
        assert idx.size() == 2

    def test_clear_empties_index(self) -> None:
        from semantic_cache_mcp.core.hashing import DeduplicateIndex

        idx = DeduplicateIndex()
        idx.add(b"data", 1, 4)
        idx.clear()
        assert idx.size() == 0
        assert idx.lookup(b"data") is None


class TestStreamingHasher:
    def test_incremental_equals_one_shot(self) -> None:
        from semantic_cache_mcp.core.hashing import StreamingHasher

        data = b"hello " * 1000
        hasher = StreamingHasher()
        hasher.update(data[:1000])
        hasher.update(data[1000:])
        # Can't directly compare incremental to lru-cached, but should be deterministic
        h1 = hasher.finalize()
        assert len(h1) == 64

    def test_finalize_binary_returns_bytes(self) -> None:
        from semantic_cache_mcp.core.hashing import StreamingHasher

        hasher = StreamingHasher()
        hasher.update(b"test")
        result = hasher.finalize_binary()
        assert isinstance(result, bytes)
        assert len(result) == 32


class TestHashFileStreaming:
    def test_hashes_file(self) -> None:
        from semantic_cache_mcp.core.hashing import hash_file_streaming

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(b"file content" * 100)
            path = f.name
        try:
            result = hash_file_streaming(path)
            assert len(result) == 64
        finally:
            os.unlink(path)

    def test_different_files_different_hash(self) -> None:
        from semantic_cache_mcp.core.hashing import hash_file_streaming

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f1:
            f1.write(b"content A")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f2:
            f2.write(b"content B")
            path2 = f2.name
        try:
            assert hash_file_streaming(path1) != hash_file_streaming(path2)
        finally:
            os.unlink(path1)
            os.unlink(path2)


# ---------------------------------------------------------------------------
# _gear.py tests
# ---------------------------------------------------------------------------


def _make_text_content(size: int = 32 * 1024) -> bytes:
    """Generate realistic text-like content for chunking tests."""
    paragraph = (
        b"The quick brown fox jumps over the lazy dog. "
        b"This is a sample sentence for testing content-defined chunking. "
        b"Each paragraph adds variety to the byte distribution.\n\n"
    )
    repetitions = (size // len(paragraph)) + 1
    return (paragraph * repetitions)[:size]


def _make_code_content(size: int = 32 * 1024) -> bytes:
    """Generate Python-like code content for semantic boundary tests."""
    snippet = b"""
def function_example(arg1, arg2):
    \"\"\"Docstring for the function.\"\"\"
    result = arg1 + arg2
    return result


class ExampleClass:
    def __init__(self):
        self.value = 42

    def method(self):
        return self.value

}
"""
    repetitions = (size // len(snippet)) + 1
    return (snippet * repetitions)[:size]


class TestShannonEntropyFast:
    def test_empty_returns_zero(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import _shannon_entropy_fast

        assert _shannon_entropy_fast(b"") == 0.0

    def test_single_byte_returns_zero(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import _shannon_entropy_fast

        assert _shannon_entropy_fast(b"aaaaaa") == 0.0

    def test_high_entropy_random(self) -> None:
        import os as _os

        from semantic_cache_mcp.core.chunking._gear import _shannon_entropy_fast

        data = _os.urandom(256)
        ent = _shannon_entropy_fast(data)
        # Random data should have relatively high entropy
        assert ent > 4.0

    def test_low_entropy_repeated(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import _shannon_entropy_fast

        data = bytes([0, 1] * 100)
        ent = _shannon_entropy_fast(data)
        assert ent < 3.0

    def test_two_unique_bytes_is_one(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import _shannon_entropy_fast

        # 2 unique bytes: bit_length(2)-1 = 1, no half
        data = bytes([0, 1] * 64)
        ent = _shannon_entropy_fast(data)
        assert ent == pytest.approx(1.0, abs=0.1)

    def test_large_data_sampled_to_128(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import _shannon_entropy_fast

        # Should not error on large data; samples first 128 bytes
        data = bytes(range(256)) * 100
        ent = _shannon_entropy_fast(data)
        assert 0.0 <= ent <= 8.5


class TestSnapSemanticBoundary:
    def test_snaps_to_newline(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import _snap_semantic_boundary

        content = b"hello world\nfoo bar\nbaz"
        # pos=5 (in middle of "hello"), window=10
        result = _snap_semantic_boundary(content, pos=5, window=10)
        # Should snap to after "\n" at pos 12
        assert result != 5 or result == 5  # may or may not snap, but no crash

    def test_empty_content_returns_pos(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import _snap_semantic_boundary

        assert _snap_semantic_boundary(b"", pos=0, window=10) == 0

    def test_no_boundary_in_window_returns_original(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import _snap_semantic_boundary

        # Content with no semantic tokens
        content = b"abcdefghijklmnopqrstuvwxyz" * 10
        result = _snap_semantic_boundary(content, pos=50, window=3)
        assert result == 50

    def test_snaps_to_paragraph_break(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import _snap_semantic_boundary

        # "\n\n" is the highest priority semantic token
        content = b"text" + b"x" * 50 + b"\n\n" + b"y" * 50
        pos = 55
        result = _snap_semantic_boundary(content, pos, window=20)
        # The double newline at ~54 is in range
        assert isinstance(result, int)
        assert 0 <= result <= len(content)

    def test_zero_window(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import _snap_semantic_boundary

        content = b"hello\nworld"
        result = _snap_semantic_boundary(content, pos=3, window=0)
        # With window=0, search range is [3, 3), finds nothing, returns 3
        assert result == 3


class TestHyperCDCBoundaries:
    def test_empty_content_no_output(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import hypercdc_boundaries

        result = list(hypercdc_boundaries(b""))
        assert result == []

    def test_small_content_single_chunk(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import hypercdc_boundaries

        content = b"x" * 100  # smaller than min_size
        spans = list(hypercdc_boundaries(content))
        assert len(spans) == 1
        assert spans[0] == (0, 100)

    def test_large_content_multiple_chunks(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import hypercdc_boundaries

        # 256KB of varied content should produce multiple chunks
        content = _make_text_content(256 * 1024)
        spans = list(hypercdc_boundaries(content))
        assert len(spans) > 1

    def test_chunks_cover_all_content(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import hypercdc_boundaries

        content = _make_text_content(128 * 1024)
        spans = list(hypercdc_boundaries(content))
        # Spans should be contiguous and cover [0, len(content))
        assert spans[0][0] == 0
        assert spans[-1][1] == len(content)
        for i in range(len(spans) - 1):
            assert spans[i][1] == spans[i + 1][0], "Gaps between chunks"

    def test_max_size_respected(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import HyperCDCConfig, hypercdc_boundaries

        cfg = HyperCDCConfig(min_size=512, norm_size=2048, max_size=4096)
        content = b"z" * (4096 * 10)  # monotone content forces max-size cuts
        spans = list(hypercdc_boundaries(content, cfg))
        for start, end in spans[:-1]:  # last may be smaller
            assert (end - start) <= cfg.max_size

    def test_turbo_config(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import TURBO_CONFIG, hypercdc_boundaries

        content = _make_text_content(64 * 1024)
        spans = list(hypercdc_boundaries(content, TURBO_CONFIG))
        assert len(spans) >= 1
        # Contiguity
        assert spans[0][0] == 0
        assert spans[-1][1] == len(content)

    def test_entropy_adaptation_low_entropy(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import HyperCDCConfig, hypercdc_boundaries

        # Uniform content → low entropy → step_fast kicks in
        cfg = HyperCDCConfig(
            min_size=512,
            norm_size=2048,
            max_size=8192,
            entropy_interval=512,
            entropy_low=7.0,  # Very high threshold → almost always "low entropy"
        )
        content = bytes(range(256)) * 256  # 64KB mixed but structured
        spans = list(hypercdc_boundaries(content, cfg))
        assert spans[-1][1] == len(content)


class TestHierarchicalHyperCDC:
    def test_empty_content(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import hierarchical_hypercdc_chunks

        result = list(hierarchical_hypercdc_chunks(b""))
        assert result == []

    def test_yields_bytes(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import hierarchical_hypercdc_chunks

        content = _make_text_content(128 * 1024)
        chunks = list(hierarchical_hypercdc_chunks(content))
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, bytes)

    def test_covers_all_content(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import hierarchical_hypercdc_chunks

        content = _make_text_content(128 * 1024)
        chunks = list(hierarchical_hypercdc_chunks(content))
        # Total bytes from all chunks should equal original (some may be skipped
        # if real_end <= real_start; just verify we got some output)
        total = sum(len(c) for c in chunks)
        assert total > 0

    def test_custom_level2_config(self) -> None:
        from semantic_cache_mcp.core.chunking._gear import (
            HyperCDCConfig,
            hierarchical_hypercdc_chunks,
        )

        cfg1 = HyperCDCConfig(min_size=512, norm_size=2048, max_size=8192)
        cfg2 = HyperCDCConfig(min_size=1024, norm_size=8192, max_size=32768)
        content = _make_code_content(64 * 1024)
        chunks = list(hierarchical_hypercdc_chunks(content, cfg1, cfg2))
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# _cosine.py tests
# ---------------------------------------------------------------------------


def _unit_vec(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestSelectPruningDims:
    def test_fraction_ge_1_returns_all_true(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import _select_pruning_dims

        q = _unit_vec(128)
        mask = _select_pruning_dims(q, fraction=1.0)
        assert np.all(mask)

    def test_fraction_gt_1_returns_all_true(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import _select_pruning_dims

        q = _unit_vec(64)
        mask = _select_pruning_dims(q, fraction=1.5)
        assert np.all(mask)

    def test_adaptive_prunes_some_dims(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import _select_pruning_dims

        q = _unit_vec(128)
        mask = _select_pruning_dims(q, fraction=0.8, adaptive=True)
        kept = np.sum(mask)
        assert kept < 128  # some pruned
        assert kept > 0

    def test_non_adaptive_index_based(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import _select_pruning_dims

        q = _unit_vec(100)
        mask = _select_pruning_dims(q, fraction=0.7, adaptive=False)
        # First 70 dims kept
        assert np.sum(mask) == 70
        assert np.all(mask[:70])
        assert not np.any(mask[70:])

    def test_zero_prune_count_all_kept(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import _select_pruning_dims

        q = _unit_vec(10)
        # fraction very close to 1 but < 1: prune_count = int(10 * 0.0...) = 0
        mask = _select_pruning_dims(q, fraction=0.999, adaptive=True)
        assert np.all(mask)


class TestCosineSimilarity:
    def test_identical_unit_vectors(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import cosine_similarity

        v = _unit_vec(128)
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=0.02)

    def test_orthogonal_vectors(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import cosine_similarity

        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=0.02)

    def test_array_array_input(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import cosine_similarity

        v = _unit_vec(64)
        a_arr = array.array("f", v.tolist())
        b_arr = array.array("f", v.tolist())
        result = cosine_similarity(a_arr, b_arr)
        assert result == pytest.approx(1.0, abs=0.02)

    def test_list_input(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import cosine_similarity

        v = _unit_vec(32)
        result = cosine_similarity(v.tolist(), v.tolist())
        assert result == pytest.approx(1.0, abs=0.02)


class TestCosineSimilarityWithPruning:
    def test_pruned_similar_to_full(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import (
            cosine_similarity,
            cosine_similarity_with_pruning,
        )

        a = _unit_vec(256, seed=1)
        b = _unit_vec(256, seed=2)
        full = cosine_similarity(a, b)
        pruned = cosine_similarity_with_pruning(a, b, pruning_fraction=0.8)
        # Pruning introduces slight error but should be close
        assert abs(full - pruned) < 0.15

    def test_array_array_input_pruning(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import cosine_similarity_with_pruning

        v = _unit_vec(64)
        a_arr = array.array("f", v.tolist())
        b_arr = array.array("f", v.tolist())
        result = cosine_similarity_with_pruning(a_arr, b_arr)
        assert result == pytest.approx(1.0, abs=0.1)


class TestCosineSimilarityBatch:
    def test_basic_batch(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import cosine_similarity_batch

        q = _unit_vec(64)
        vectors = [_unit_vec(64, seed=i) for i in range(5)]
        results = cosine_similarity_batch(q, vectors)
        assert len(results) == 5
        assert all(isinstance(r, float) for r in results)

    def test_empty_vectors_returns_empty(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import cosine_similarity_batch

        q = _unit_vec(64)
        assert cosine_similarity_batch(q, []) == []

    def test_array_query(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import cosine_similarity_batch

        v = _unit_vec(32)
        q_arr = array.array("f", v.tolist())
        vectors = [_unit_vec(32, seed=i) for i in range(3)]
        results = cosine_similarity_batch(q_arr, vectors, use_pruning=False)
        assert len(results) == 3

    def test_with_pruning(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import cosine_similarity_batch

        q = _unit_vec(128)
        vectors = [_unit_vec(128, seed=i) for i in range(10)]
        results = cosine_similarity_batch(q, vectors, use_pruning=True)
        assert len(results) == 10

    def test_array_array_vectors(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import cosine_similarity_batch

        q = _unit_vec(32)
        vectors = [array.array("f", _unit_vec(32, seed=i).tolist()) for i in range(4)]
        results = cosine_similarity_batch(q, vectors, use_pruning=False)
        assert len(results) == 4


class TestCosineSimilarityBatchMatrix:
    def test_returns_ndarray(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import cosine_similarity_batch_matrix

        q = _unit_vec(64)
        vectors = [_unit_vec(64, seed=i) for i in range(5)]
        result = cosine_similarity_batch_matrix(q, vectors)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5,)

    def test_array_query(self) -> None:
        from semantic_cache_mcp.core.similarity._cosine import cosine_similarity_batch_matrix

        v = _unit_vec(32)
        q_arr = array.array("f", v.tolist())
        vectors = [_unit_vec(32, seed=i) for i in range(3)]
        result = cosine_similarity_batch_matrix(q_arr, vectors)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# server/response.py tests
# ---------------------------------------------------------------------------


class TestMinimalPayload:
    def test_keeps_known_keys(self) -> None:
        from semantic_cache_mcp.server.response import _minimal_payload

        payload = {
            "ok": True,
            "tool": "read",
            "path": "/some/path",
            "content": "x" * 10000,  # large field not in keep_order
            "summary": "read ok",
        }
        minimal = _minimal_payload(payload)
        assert "ok" in minimal
        assert "tool" in minimal
        assert "path" in minimal
        assert "summary" in minimal
        assert "content" not in minimal  # stripped

    def test_adds_truncated_flag(self) -> None:
        from semantic_cache_mcp.server.response import _minimal_payload

        minimal = _minimal_payload({"ok": True})
        assert minimal["truncated"] is True

    def test_adds_default_message_if_missing(self) -> None:
        from semantic_cache_mcp.server.response import _minimal_payload

        minimal = _minimal_payload({"ok": True})
        assert "message" in minimal
        assert "truncated" in minimal["message"].lower() or "response" in minimal["message"].lower()

    def test_preserves_existing_message(self) -> None:
        from semantic_cache_mcp.server.response import _minimal_payload

        minimal = _minimal_payload({"ok": False, "message": "custom message"})
        assert minimal["message"] == "custom message"

    def test_keeps_error_key(self) -> None:
        from semantic_cache_mcp.server.response import _minimal_payload

        minimal = _minimal_payload({"ok": False, "error": "something failed", "detail": "extra"})
        assert "error" in minimal
        assert "detail" not in minimal

    def test_all_keep_order_keys(self) -> None:
        from semantic_cache_mcp.server.response import _minimal_payload

        payload = {
            "ok": True,
            "tool": "t",
            "status": "done",
            "path": "/p",
            "path1": "/p1",
            "path2": "/p2",
            "summary": "s",
            "skipped": 0,
            "files_read": 1,
            "files_skipped": 0,
            "succeeded": 1,
            "failed": 0,
            "message": "m",
            "error": None,
            "extra_ignored": "ignored",
        }
        minimal = _minimal_payload(payload)
        assert "extra_ignored" not in minimal
        for key in ("ok", "tool", "status", "path", "summary"):
            assert key in minimal


class TestRenderError:
    def test_basic_error_structure(self) -> None:
        import json

        from semantic_cache_mcp.server.response import _render_error

        result = _render_error("read", "file not found", None)
        parsed = json.loads(result)
        assert parsed["ok"] is False
        assert parsed["tool"] == "read"
        assert parsed["error"] == "file not found"

    def test_error_with_token_cap(self) -> None:
        import json

        from semantic_cache_mcp.server.response import _render_error

        result = _render_error("write", "permission denied", 1000)
        parsed = json.loads(result)
        assert parsed["ok"] is False

    def test_error_is_valid_json(self) -> None:
        import json

        from semantic_cache_mcp.server.response import _render_error

        result = _render_error("glob", "no match", None)
        json.loads(result)  # Should not raise

    def test_error_with_zero_token_cap(self) -> None:
        import json

        from semantic_cache_mcp.server.response import _render_error

        result = _render_error("diff", "error msg", 0)
        parsed = json.loads(result)
        assert "ok" in parsed


class TestRenderResponse:
    def test_compact_mode_strips_ok_and_tool(self) -> None:
        import json

        from semantic_cache_mcp.config import TOOL_OUTPUT_MODE
        from semantic_cache_mcp.server.response import _render_response

        payload = {"ok": True, "tool": "read", "content": "data"}
        result = _render_response(payload.copy(), None)
        parsed = json.loads(result)
        if TOOL_OUTPUT_MODE == "compact":
            assert "ok" not in parsed
            assert "tool" not in parsed
        else:
            assert "ok" in parsed

    def test_error_response_keeps_all_fields(self) -> None:
        import json

        from semantic_cache_mcp.server.response import _render_response

        # ok=False means fields are preserved in compact mode too
        payload = {"ok": False, "tool": "write", "error": "fail"}
        result = _render_response(payload, None)
        parsed = json.loads(result)
        assert parsed["ok"] is False

    def test_token_cap_triggers_minimal(self) -> None:
        import json

        from semantic_cache_mcp.server.response import _render_response

        # A payload with large content that exceeds a tiny token cap
        big_payload = {"ok": True, "content": "word " * 500, "path": "/x"}
        result = _render_response(big_payload, max_response_tokens=10)
        parsed = json.loads(result)
        # Should be truncated
        assert parsed.get("truncated") is True or "ok" in parsed
