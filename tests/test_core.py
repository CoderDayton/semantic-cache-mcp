"""Tests for core algorithms: chunking, hashing, similarity."""

from __future__ import annotations

import re

import pytest

from semantic_cache_mcp.core.chunking import hypercdc_chunks
from semantic_cache_mcp.core.hashing import (
    KEYED_HASH_KEY_SIZE,
    hash_chunk,
    hash_content,
    keyed_hash,
)
from semantic_cache_mcp.core.text import (
    diff_stats,
    diff_with_stats,
    generate_diff,
    rebase_diff_hunks,
    truncate_smart,
)


class TestContentDefinedChunking:
    """Tests for content-defined chunking."""

    def test_empty_content_yields_nothing(self) -> None:
        """Empty content should yield no chunks."""
        chunks = list(hypercdc_chunks(b""))
        assert chunks == []

    def test_single_byte_yields_single_chunk(self) -> None:
        """Single byte should yield one chunk."""
        chunks = list(hypercdc_chunks(b"x"))
        assert len(chunks) == 1
        assert chunks[0] == b"x"

    def test_small_content_single_chunk(self) -> None:
        """Small content below min_size should be single chunk."""
        data = b"Hello, World!"
        chunks = list(hypercdc_chunks(data, min_size=100))
        assert len(chunks) == 1
        assert chunks[0] == data

    def test_deterministic_output(self) -> None:
        """Same input should always produce same chunks."""
        data = b"The quick brown fox jumps over the lazy dog. " * 100
        chunks1 = list(hypercdc_chunks(data))
        chunks2 = list(hypercdc_chunks(data))
        assert chunks1 == chunks2

    def test_reassembly_matches_original(self) -> None:
        """Reassembled chunks should match original content."""
        data = b"Test data for chunking. " * 500
        chunks = list(hypercdc_chunks(data))
        reassembled = b"".join(chunks)
        assert reassembled == data

    def test_respects_max_size(self) -> None:
        """All chunks should be at most max_size."""
        data = b"x" * 100000
        max_size = 8192
        chunks = list(hypercdc_chunks(data, max_size=max_size))
        for chunk in chunks:
            assert len(chunk) <= max_size

    def test_chunks_at_least_min_size(self) -> None:
        """Non-final chunks should be at least min_size."""
        data = b"y" * 50000
        min_size = 1024
        chunks = list(hypercdc_chunks(data, min_size=min_size))
        # All but last chunk should meet min_size
        for chunk in chunks[:-1]:
            assert len(chunk) >= min_size

    def test_binary_content(self) -> None:
        """Binary content should chunk correctly."""
        data = bytes(range(256)) * 100
        chunks = list(hypercdc_chunks(data))
        assert b"".join(chunks) == data


class TestHashing:
    """Tests for BLAKE2b hashing."""

    def test_hash_chunk_consistent(self) -> None:
        """Same data should produce same hash."""
        data = b"Test data"
        hash1 = hash_chunk(data)
        hash2 = hash_chunk(data)
        assert hash1 == hash2

    def test_hash_chunk_format(self) -> None:
        """Hash should be 64-character hex string (32-byte BLAKE3/BLAKE2b)."""
        data = b"Test"
        result = hash_chunk(data)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_chunk_different_data(self) -> None:
        """Different data should produce different hashes."""
        hash1 = hash_chunk(b"data1")
        hash2 = hash_chunk(b"data2")
        assert hash1 != hash2

    def test_hash_content_consistent(self) -> None:
        """Same content should produce same hash."""
        content = "Hello, World!"
        hash1 = hash_content(content)
        hash2 = hash_content(content)
        assert hash1 == hash2

    def test_hash_content_format(self) -> None:
        """Content hash should be 64-character hex string (32-byte BLAKE3/BLAKE2b)."""
        result = hash_content("Test")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_content_empty_string(self) -> None:
        """Empty string should have a valid hash."""
        result = hash_content("")
        assert len(result) == 64


class TestKeyedHashing:
    """Tests for keyed hashing (BLAKE3 keyed mode, BLAKE2b fallback)."""

    KEY = b"\x01" * KEYED_HASH_KEY_SIZE
    OTHER_KEY = b"\x02" * KEYED_HASH_KEY_SIZE

    def test_keyed_hash_is_deterministic(self) -> None:
        assert keyed_hash(b"payload", self.KEY, 8) == keyed_hash(b"payload", self.KEY, 8)

    def test_keyed_hash_depends_on_the_key(self) -> None:
        """Without this a token signed by one process verifies under another."""
        assert keyed_hash(b"payload", self.KEY, 8) != keyed_hash(b"payload", self.OTHER_KEY, 8)

    def test_keyed_hash_depends_on_the_data(self) -> None:
        """Without this a tampered token keeps its original tag."""
        assert keyed_hash(b"payload", self.KEY, 8) != keyed_hash(b"payloae", self.KEY, 8)

    def test_keyed_hash_honors_digest_size(self) -> None:
        assert len(keyed_hash(b"payload", self.KEY, 8)) == 16  # hex, so two chars a byte
        assert len(keyed_hash(b"payload", self.KEY, 32)) == 64

    @pytest.mark.parametrize("bad_key", [b"", b"short", b"\x00" * 31, b"\x00" * 64])
    def test_keyed_hash_rejects_a_wrong_sized_key(self, bad_key: bytes) -> None:
        """BLAKE3 requires exactly 32 bytes; reject early so both backends agree."""
        with pytest.raises(ValueError, match="32-byte key"):
            keyed_hash(b"payload", bad_key, 8)


class TestDiffGeneration:
    """Tests for unified diff generation."""

    def test_unchanged_file_no_diff(self) -> None:
        """Unchanged content should return no changes message."""
        text = "Same content\nLine 2\n"
        result = generate_diff(text, text)
        assert result == "// No changes"

    def test_changed_file_produces_diff(self) -> None:
        """Changed content should produce unified diff."""
        old = "Line 1\nLine 2\n"
        new = "Line 1\nLine 2 modified\n"
        result = generate_diff(old, new)
        assert "---" in result or "-Line 2" in result

    def test_new_file_diff(self) -> None:
        """Adding to empty file should show additions."""
        old = ""
        new = "New line 1\nNew line 2\n"
        result = generate_diff(old, new)
        assert "+New line" in result or "+" in result

    def test_empty_files_no_diff(self) -> None:
        """Two empty files should have no diff."""
        result = generate_diff("", "")
        assert result == "// No changes"

    def test_diff_context_lines(self) -> None:
        """Diff should include context lines."""
        old = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
        new = "Line 1\nLine 2\nLine 3 CHANGED\nLine 4\nLine 5\n"
        result = generate_diff(old, new, context_lines=1)
        assert "Line 2" in result or "Line 4" in result

    def test_diff_omits_file_headers_keeps_hunk_header(self) -> None:
        """The diff drops difflib's `--- `/`+++ ` file headers (pure token
        overhead) but keeps the `@@` hunk header that carries line numbers."""
        old = "Line 1\nLine 2\nLine 3\n"
        new = "Line 1\nLine 2 changed\nLine 3\n"
        result = generate_diff(old, new)
        assert "@@" in result  # line-number anchor preserved
        assert not result.startswith("--- ")
        assert "+++ " not in result
        assert "-Line 2\n" in result
        assert "+Line 2 changed\n" in result


class TestDiffRebasing:
    """Tests for rebasing a slice's diff onto whole-file line numbers."""

    def test_zero_offset_is_identity(self) -> None:
        diff = generate_diff("a\nb\nc\n", "a\nB\nc\n")
        assert rebase_diff_hunks(diff, 0) == diff

    def test_both_sides_shift_and_lengths_are_untouched(self) -> None:
        assert rebase_diff_hunks("@@ -8,7 +8,7 @@\n ctx\n", 19) == "@@ -27,7 +27,7 @@\n ctx\n"

    def test_single_line_range_has_no_length_suffix(self) -> None:
        assert rebase_diff_hunks("@@ -1 +1 @@\n", 10) == "@@ -11 +11 @@\n"

    def test_every_hunk_is_shifted(self) -> None:
        diff = "@@ -1,2 +1,2 @@\n x\n@@ -50,2 +50,2 @@\n y\n"
        assert rebase_diff_hunks(diff, 5) == "@@ -6,2 +6,2 @@\n x\n@@ -55,2 +55,2 @@\n y\n"

    def test_no_changes_marker_is_left_alone(self) -> None:
        assert rebase_diff_hunks("// No changes", 42) == "// No changes"

    def test_body_lines_are_never_rewritten(self) -> None:
        """Only headers move; content that looks like a header must not."""
        diff = "@@ -1,2 +1,2 @@\n-was @@ -9,9 +9,9 @@ inline\n+now\n"
        rebased = rebase_diff_hunks(diff, 100)
        assert rebased.startswith("@@ -101,2 +101,2 @@")
        assert "-was @@ -9,9 +9,9 @@ inline" in rebased

    def test_rebased_slice_matches_a_whole_file_diff(self) -> None:
        """The property that matters: same change, same reported line numbers."""
        old = [f"line_{i}\n" for i in range(60)]
        new = list(old)
        new[29] = "line_29_changed\n"

        whole = generate_diff("".join(old), "".join(new))
        # Diff only file lines 20-40, then rebase that slice onto the file.
        sliced = generate_diff("".join(old[19:40]), "".join(new[19:40]))
        rebased = rebase_diff_hunks(sliced, 19)

        whole_header = re.search(r"@@ -\d+", whole)
        rebased_header = re.search(r"@@ -\d+", rebased)
        assert whole_header is not None and rebased_header is not None
        assert whole_header.group(0) == rebased_header.group(0)


class TestDiffWithStats:
    """diff_with_stats must be bit-identical to the two separate calls."""

    CASES = [
        ("identical", "a\nb\nc\n", "a\nb\nc\n"),
        ("both empty", "", ""),
        ("insert only", "a\nb\n", "a\nx\nb\n"),
        ("delete only", "a\nx\nb\n", "a\nb\n"),
        ("equal-length replace", "a\nb\nc\n", "a\nB\nc\n"),
        ("unequal replace", "a\nb\nc\nd\n", "a\nX\nd\n"),
        ("no trailing newline", "a\nb", "a\nc"),
        ("from empty", "", "new\nlines\n"),
        ("to empty", "old\nlines\n", ""),
        (
            "multiple hunks",
            "\n".join(f"line {i}" for i in range(40)) + "\n",
            "\n".join("CHANGED" if i in (3, 30) else f"line {i}" for i in range(40)) + "\n",
        ),
    ]

    @pytest.mark.parametrize("label,old,new", CASES, ids=[c[0] for c in CASES])
    def test_matches_separate_calls(self, label: str, old: str, new: str) -> None:
        diff, stats = diff_with_stats(old, new)
        assert diff == generate_diff(old, new)
        assert stats == diff_stats(old, new)

    @pytest.mark.parametrize("context", [0, 1, 2, 3])
    def test_matches_at_all_context_widths(self, context: int) -> None:
        old = "\n".join(f"line {i}" for i in range(30)) + "\n"
        new = old.replace("line 12", "changed 12")
        diff, _ = diff_with_stats(old, new, context_lines=context)
        assert diff == generate_diff(old, new, context_lines=context)


class TestSmartTruncation:
    """Tests for smart truncation."""

    def test_small_content_unchanged(self) -> None:
        """Small content should not be truncated."""
        content = "Short content\n"
        result = truncate_smart(content, max_size=1000)
        assert result == content

    def test_large_content_truncated(self) -> None:
        """Large content should be truncated."""
        lines = [f"Line {i}\n" for i in range(200)]
        content = "".join(lines)
        result = truncate_smart(content, max_size=500)
        assert len(result) <= 500
        assert "truncated" in result.lower() or "TRUNCATED" in result

    def test_preserves_top_lines(self) -> None:
        """Truncation should preserve top lines."""
        lines = [f"Line {i}\n" for i in range(200)]
        content = "".join(lines)
        result = truncate_smart(content, max_size=2000, keep_top=10)
        assert "Line 0" in result
        assert "Line 9" in result

    def test_preserves_bottom_lines(self) -> None:
        """Truncation should preserve bottom lines."""
        lines = [f"Line {i}\n" for i in range(200)]
        content = "".join(lines)
        result = truncate_smart(content, max_size=2000, keep_top=10, keep_bottom=10)
        assert "Line 199" in result
        assert "Line 190" in result

    def test_empty_content(self) -> None:
        """Empty content should return empty."""
        result = truncate_smart("", max_size=1000)
        assert result == ""
