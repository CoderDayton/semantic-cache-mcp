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
    diff_with_stats,
    generate_diff,
    rebase_diff_hunks,
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
        diff, _ = diff_with_stats(old, new)
        assert diff == generate_diff(old, new)

    def test_stats_for_an_equal_length_replace(self) -> None:
        _, stats = diff_with_stats("a\nb\nc\n", "a\nB\nc\n")
        assert stats == {
            "insertions": 0,
            "deletions": 0,
            "modifications": 1,
            "delta_size_bytes": 34,
            "original_size": 6,
            "compression_ratio": 34 / 6,
        }

    def test_stats_for_an_insert(self) -> None:
        _, stats = diff_with_stats("a\nb\n", "a\nx\nb\n")
        assert stats["insertions"] == 1
        assert stats["deletions"] == 0
        assert stats["modifications"] == 0
        assert stats["delta_size_bytes"] == 34

    def test_stats_from_empty_has_zero_ratio(self) -> None:
        _, stats = diff_with_stats("", "something\n")
        assert stats["original_size"] == 0
        assert stats["compression_ratio"] == 0

    @pytest.mark.parametrize("context", [0, 1, 2, 3])
    def test_matches_at_all_context_widths(self, context: int) -> None:
        old = "\n".join(f"line {i}" for i in range(30)) + "\n"
        new = old.replace("line 12", "changed 12")
        diff, _ = diff_with_stats(old, new, context_lines=context)
        assert diff == generate_diff(old, new, context_lines=context)


class TestMissingTrailingNewline:
    """A last line without ``\\n`` must not run into the next diff line."""

    MARK = "\\ No newline at end of file\n"

    @pytest.mark.parametrize("fn", ["generate_diff", "diff_with_stats"])
    def test_changed_last_line_on_both_sides(self, fn: str) -> None:
        diff = _diff(fn, "x = 1", "x = 2")
        assert diff == f"@@ -1 +1 @@\n-x = 1\n{self.MARK}+x = 2\n{self.MARK}"

    @pytest.mark.parametrize("fn", ["generate_diff", "diff_with_stats"])
    def test_newline_added_after_the_old_last_line(self, fn: str) -> None:
        diff = _diff(fn, "a\nb", "a\nb\nc")
        assert diff == f"@@ -1,2 +1,3 @@\n a\n-b\n{self.MARK}+b\n+c\n{self.MARK}"

    @pytest.mark.parametrize("fn", ["generate_diff", "diff_with_stats"])
    def test_unchanged_last_line_as_context(self, fn: str) -> None:
        diff = _diff(fn, "a\nb", "x\nb")
        assert diff == f"@@ -1,2 +1,2 @@\n-a\n+x\n b\n{self.MARK}"

    @pytest.mark.parametrize("fn", ["generate_diff", "diff_with_stats"])
    def test_every_line_has_a_diff_prefix(self, fn: str) -> None:
        diff = _diff(fn, "one\ntwo", "one\nTWO\nthree")
        for line in diff.splitlines():
            assert line[:1] in {" ", "+", "-", "@", "\\"}, f"bad diff line {line!r}"
        assert diff.endswith("\n")


def _diff(fn: str, old: str, new: str) -> str:
    if fn == "generate_diff":
        return generate_diff(old, new)
    return diff_with_stats(old, new)[0]
