"""Tests for new MCP tools: search, diff, batch_read, similar, glob."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from semantic_cache_mcp.cache import (
    SemanticCache,
    batch_smart_read,
    compare_files,
    find_similar_files,
    glob_with_cache_status,
    semantic_search,
    smart_batch_edit,
    smart_read,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir).resolve()


@pytest.fixture
def cache(temp_dir: Path):
    """Create a cache instance with temp database."""
    db_path = temp_dir / "cache.db"
    return SemanticCache(db_path=db_path)


@pytest.fixture
def sample_files(temp_dir: Path):
    """Create sample files for testing."""
    # Python file
    py_file = temp_dir / "example.py"
    py_file.write_text("def hello():\n    return 'world'\n")

    # Another Python file
    py_file2 = temp_dir / "example2.py"
    py_file2.write_text("def goodbye():\n    return 'world'\n")

    # Text file
    txt_file = temp_dir / "readme.txt"
    txt_file.write_text("This is a readme file.\n")

    # Subdirectory with file
    subdir = temp_dir / "src"
    subdir.mkdir()
    sub_file = subdir / "main.py"
    sub_file.write_text("if __name__ == '__main__':\n    print('hello')\n")

    return {
        "py": py_file,
        "py2": py_file2,
        "txt": txt_file,
        "sub": sub_file,
    }


class TestSemanticSearch:
    """Tests for semantic_search function."""

    def test_search_empty_cache(self, cache: SemanticCache):
        """Search on empty cache returns no matches."""
        result = semantic_search(cache, "hello world")
        assert result.matches == []
        assert result.files_searched == 0

    def test_search_finds_cached_files(self, cache: SemanticCache, sample_files: dict):
        """Search finds files after they're cached."""
        # Cache files first
        smart_read(cache, str(sample_files["py"]))
        smart_read(cache, str(sample_files["txt"]))

        result = semantic_search(cache, "function definition", k=5)
        assert result.cached_files >= 1

    def test_search_respects_k_limit(self, cache: SemanticCache, sample_files: dict):
        """Search respects k parameter."""
        for f in sample_files.values():
            smart_read(cache, str(f))

        result = semantic_search(cache, "code", k=2)
        assert len(result.matches) <= 2

    def test_search_directory_filter(
        self, cache: SemanticCache, sample_files: dict, temp_dir: Path
    ):
        """Search filters by directory."""
        for f in sample_files.values():
            smart_read(cache, str(f))

        subdir = temp_dir / "src"
        result = semantic_search(cache, "code", directory=str(subdir))
        for match in result.matches:
            assert str(subdir) in match.path


class TestCompareFiles:
    """Tests for compare_files function."""

    def test_diff_identical_files(self, cache: SemanticCache, temp_dir: Path):
        """Diff of identical files shows no changes."""
        file1 = temp_dir / "a.txt"
        file2 = temp_dir / "b.txt"
        content = "same content\n"
        file1.write_text(content)
        file2.write_text(content)

        result = compare_files(cache, str(file1), str(file2))
        assert result.diff_stats["insertions"] == 0
        assert result.diff_stats["deletions"] == 0

    def test_diff_different_files(self, cache: SemanticCache, temp_dir: Path):
        """Diff shows changes between different files."""
        file1 = temp_dir / "a.txt"
        file2 = temp_dir / "b.txt"
        file1.write_text("line one\n")
        file2.write_text("line two\n")

        result = compare_files(cache, str(file1), str(file2))
        assert result.diff_content != ""

    def test_diff_uses_cache(self, cache: SemanticCache, sample_files: dict):
        """Diff uses cached content when available."""
        # Pre-cache the files
        smart_read(cache, str(sample_files["py"]))
        smart_read(cache, str(sample_files["py2"]))

        result = compare_files(cache, str(sample_files["py"]), str(sample_files["py2"]))
        assert result.from_cache == (True, True)

    def test_diff_computes_similarity(self, cache: SemanticCache, sample_files: dict):
        """Diff computes semantic similarity."""
        result = compare_files(cache, str(sample_files["py"]), str(sample_files["py2"]))
        assert 0.0 <= result.similarity <= 1.0


class TestBatchSmartRead:
    """Tests for batch_smart_read function."""

    def test_batch_read_multiple_files(self, cache: SemanticCache, sample_files: dict):
        """Batch read handles multiple files."""
        paths = [str(f) for f in sample_files.values()]
        result = batch_smart_read(cache, paths)

        assert result.files_read > 0
        assert len(result.contents) == result.files_read

    def test_batch_read_respects_token_budget(self, cache: SemanticCache, temp_dir: Path):
        """Batch read respects token budget."""
        # Create files that exceed budget
        for i in range(10):
            f = temp_dir / f"file{i}.txt"
            f.write_text("x" * 1000)

        paths = [str(temp_dir / f"file{i}.txt") for i in range(10)]
        result = batch_smart_read(cache, paths, max_total_tokens=100)

        assert result.total_tokens <= 100 or result.files_skipped > 0

    def test_batch_read_handles_missing_files(self, cache: SemanticCache, temp_dir: Path):
        """Batch read handles missing files gracefully."""
        existing = temp_dir / "exists.txt"
        existing.write_text("content")

        paths = [str(existing), str(temp_dir / "missing.txt")]
        result = batch_smart_read(cache, paths)

        assert result.files_skipped >= 1
        assert result.files_read >= 1


class TestFindSimilarFiles:
    """Tests for find_similar_files function."""

    def test_similar_empty_cache(self, cache: SemanticCache, sample_files: dict):
        """Similar files returns empty when cache is empty."""
        result = find_similar_files(cache, str(sample_files["py"]))
        assert result.similar_files == []

    def test_similar_finds_related_files(self, cache: SemanticCache, sample_files: dict):
        """Similar files finds related cached files."""
        # Cache all files first
        for f in sample_files.values():
            smart_read(cache, str(f))

        result = find_similar_files(cache, str(sample_files["py"]), k=3)
        # Should find at least the other Python file
        assert result.files_searched >= 1

    def test_similar_excludes_source(self, cache: SemanticCache, sample_files: dict):
        """Similar files excludes the source file."""
        for f in sample_files.values():
            smart_read(cache, str(f))

        source = str(sample_files["py"])
        result = find_similar_files(cache, source, k=10)

        for sf in result.similar_files:
            assert sf.path != source


class TestGlobWithCacheStatus:
    """Tests for glob_with_cache_status function."""

    def test_glob_finds_files(self, cache: SemanticCache, sample_files: dict, temp_dir: Path):
        """Glob finds matching files."""
        result = glob_with_cache_status(cache, "*.py", directory=str(temp_dir))
        assert result.total_matches >= 1

    def test_glob_recursive_pattern(self, cache: SemanticCache, sample_files: dict, temp_dir: Path):
        """Glob handles recursive patterns."""
        result = glob_with_cache_status(cache, "**/*.py", directory=str(temp_dir))
        # Should find files in subdirectory too
        assert result.total_matches >= 2

    def test_glob_shows_cache_status(
        self, cache: SemanticCache, sample_files: dict, temp_dir: Path
    ):
        """Glob shows which files are cached."""
        # Cache one file
        smart_read(cache, str(sample_files["py"]))

        result = glob_with_cache_status(cache, "*.py", directory=str(temp_dir))
        assert result.cached_count >= 1

    def test_glob_no_matches(self, cache: SemanticCache, temp_dir: Path):
        """Glob returns empty for no matches."""
        result = glob_with_cache_status(cache, "*.nonexistent", directory=str(temp_dir))
        assert result.total_matches == 0
        assert result.matches == []

    def test_glob_cached_only(self, cache: SemanticCache, sample_files: dict, temp_dir: Path):
        """cached_only=True only returns files in cache."""
        # Cache just the py file
        from semantic_cache_mcp.cache import smart_read

        py_path = str(sample_files["py"])
        smart_read(cache, py_path)

        # Glob for all files but only cached
        result = glob_with_cache_status(
            cache,
            "*",
            directory=str(temp_dir),
            cached_only=True,
        )

        # Only the cached file should appear
        result_paths = [m.path for m in result.matches]
        assert py_path in result_paths
        assert all(m.cached for m in result.matches)


class TestBatchReadOptimizations:
    """Tests for batch_read optimizations: unchanged collapse, priority, est_tokens, globs."""

    def test_batch_read_collapses_unchanged(self, cache: SemanticCache, sample_files: dict):
        """Unchanged files → unchanged_paths, status='unchanged', no content."""
        paths = [str(sample_files["py"]), str(sample_files["txt"])]

        # First read caches the files
        batch_smart_read(cache, paths)

        # Second read: files haven't changed → should be "unchanged"
        result = batch_smart_read(cache, paths)

        assert len(result.unchanged_paths) == 2
        # Unchanged files should NOT appear in contents
        assert len(result.contents) == 0
        # All files should have status "unchanged"
        for f in result.files:
            assert f.status == "unchanged"

    def test_batch_read_diff_mode_false_returns_content_after_cache(
        self, cache: SemanticCache, sample_files: dict
    ):
        """diff_mode=False returns full content even for cached unchanged files (post-compression recovery)."""
        paths = [str(sample_files["py"]), str(sample_files["txt"])]

        # Seed cache
        batch_smart_read(cache, paths)

        # Simulate post-compression: files are cached but diff_mode=False forces full content
        result = batch_smart_read(cache, paths, diff_mode=False)

        # No files should be suppressed as "unchanged"
        assert len(result.unchanged_paths) == 0
        # All files should have content returned
        assert len(result.contents) == 2
        for path in paths:
            assert path in result.contents
            assert len(result.contents[path]) > 0

    def test_batch_read_priority_order(self, cache: SemanticCache, sample_files: dict):
        """Priority paths read before sorted remainder."""
        paths = [str(f) for f in sample_files.values()]
        priority_path = str(sample_files["txt"])

        result = batch_smart_read(cache, paths, priority=[priority_path])

        # First file processed should be the priority file
        assert result.files[0].path == priority_path

    def test_batch_read_skipped_has_est_tokens(self, cache: SemanticCache, temp_dir: Path):
        """Skipped files have est_tokens > 0."""
        # Create files that exceed budget
        for i in range(5):
            f = temp_dir / f"big{i}.txt"
            f.write_text("x" * 4000)  # ~1000 tokens each

        paths = [str(temp_dir / f"big{i}.txt") for i in range(5)]
        result = batch_smart_read(cache, paths, max_total_tokens=500)

        # At least some files should be skipped
        skipped = [f for f in result.files if f.status == "skipped"]
        assert len(skipped) > 0
        for f in skipped:
            assert f.est_tokens is not None
            assert f.est_tokens > 0

    def test_batch_read_mixed_unchanged_and_new(
        self, cache: SemanticCache, sample_files: dict, temp_dir: Path
    ):
        """Mix of cached-unchanged and new files works correctly."""
        py_path = str(sample_files["py"])
        # Pre-cache one file
        smart_read(cache, py_path)

        # Create a new file
        new_file = temp_dir / "brand_new.py"
        new_file.write_text("x = 42\n")

        result = batch_smart_read(cache, [py_path, str(new_file)])

        # py_path should be unchanged (cached, not diff)
        assert py_path in result.unchanged_paths
        # new file should have content
        assert str(new_file) in result.contents

    def test_batch_read_priority_does_not_override_budget(
        self, cache: SemanticCache, temp_dir: Path
    ):
        """Priority controls order, not budget bypass."""
        big = temp_dir / "big.txt"
        big.write_text("x" * 40000)  # ~10000 tokens
        small = temp_dir / "small.txt"
        small.write_text("hi\n")

        result = batch_smart_read(
            cache,
            [str(big), str(small)],
            max_total_tokens=50,
            priority=[str(big)],
        )

        # big should be skipped (over budget), small should be read
        statuses = {f.path: f.status for f in result.files}
        assert statuses[str(big)] == "skipped"


class TestExpandGlobs:
    """Tests for _expand_globs helper."""

    def test_expand_globs_non_glob_passthrough(self, temp_dir: Path):
        """Non-glob paths pass through unchanged."""
        from semantic_cache_mcp.server import _expand_globs

        paths = [str(temp_dir / "a.py"), str(temp_dir / "b.py")]
        result = _expand_globs(paths)
        assert result == paths

    def test_expand_globs_expands_pattern(self, sample_files: dict, temp_dir: Path):
        """Glob patterns expand to matching files."""
        from semantic_cache_mcp.server import _expand_globs

        pattern = str(temp_dir / "*.py")
        result = _expand_globs([pattern])
        assert len(result) >= 2  # example.py, example2.py at minimum
        for p in result:
            assert p.endswith(".py")

    def test_expand_globs_respects_max_files(self, temp_dir: Path):
        """Expansion respects max_files cap."""
        from semantic_cache_mcp.server import _expand_globs

        for i in range(10):
            (temp_dir / f"f{i}.txt").write_text(f"file {i}")

        pattern = str(temp_dir / "*.txt")
        result = _expand_globs([pattern], max_files=3)
        assert len(result) <= 3

    def test_expand_globs_invalid_pattern_literal(self):
        """Invalid glob pattern treated as literal path."""
        from semantic_cache_mcp.server import _expand_globs

        result = _expand_globs(["/nonexistent/path/[invalid"])
        assert result == ["/nonexistent/path/[invalid"]


class TestSmartBatchEdit:
    """Tests for smart_batch_edit function."""

    def test_batch_edit_all_succeed(self, cache: SemanticCache, temp_dir: Path):
        """All edits succeed when all matches found."""
        f = temp_dir / "test.py"
        f.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")

        result = smart_batch_edit(
            cache,
            str(f),
            [
                ("def foo", "def new_foo"),
                ("def bar", "def new_bar"),
            ],
        )

        assert result.succeeded == 2
        assert result.failed == 0
        assert "def new_foo" in f.read_text()
        assert "def new_bar" in f.read_text()

    def test_batch_edit_partial_success(self, cache: SemanticCache, temp_dir: Path):
        """Some edits succeed, some fail."""
        f = temp_dir / "test.py"
        f.write_text("def foo():\n    pass\n")

        result = smart_batch_edit(
            cache,
            str(f),
            [
                ("def foo", "def new_foo"),  # Should succeed
                ("def bar", "def new_bar"),  # Should fail - not found
            ],
        )

        assert result.succeeded == 1
        assert result.failed == 1
        assert "def new_foo" in f.read_text()

    def test_batch_edit_all_fail(self, cache: SemanticCache, temp_dir: Path):
        """No edits apply when none match."""
        f = temp_dir / "test.py"
        f.write_text("def foo():\n    pass\n")

        result = smart_batch_edit(
            cache,
            str(f),
            [
                ("not_found_1", "replacement_1"),
                ("not_found_2", "replacement_2"),
            ],
        )

        assert result.succeeded == 0
        assert result.failed == 2
        # File should be unchanged
        assert f.read_text() == "def foo():\n    pass\n"

    def test_batch_edit_dry_run(self, cache: SemanticCache, temp_dir: Path):
        """Dry run shows changes without applying."""
        f = temp_dir / "test.py"
        original = "def foo():\n    pass\n"
        f.write_text(original)

        result = smart_batch_edit(
            cache,
            str(f),
            [
                ("def foo", "def new_foo"),
            ],
            dry_run=True,
        )

        assert result.succeeded == 1
        # File should be unchanged
        assert f.read_text() == original

    def test_batch_edit_uses_cache(self, cache: SemanticCache, temp_dir: Path):
        """Multi-edit uses cached content."""
        f = temp_dir / "test.py"
        f.write_text("def foo():\n    pass\n")

        # Pre-cache the file
        smart_read(cache, str(f))

        result = smart_batch_edit(
            cache,
            str(f),
            [
                ("def foo", "def new_foo"),
            ],
        )

        assert result.from_cache is True
        assert result.tokens_saved > 0

    def test_batch_edit_outcomes_have_line_numbers(self, cache: SemanticCache, temp_dir: Path):
        """Successful edits report line numbers."""
        f = temp_dir / "test.py"
        f.write_text("line1\ndef foo():\n    pass\n")

        result = smart_batch_edit(
            cache,
            str(f),
            [
                ("def foo", "def new_foo"),
            ],
        )

        assert result.outcomes[0].success is True
        assert result.outcomes[0].line_number == 2

    def test_batch_edit_reports_errors(self, cache: SemanticCache, temp_dir: Path):
        """Failed edits report error messages."""
        f = temp_dir / "test.py"
        f.write_text("def foo():\n    pass\n")

        result = smart_batch_edit(
            cache,
            str(f),
            [
                ("", "replacement"),  # Empty old_string
                ("same", "same"),  # Identical strings
                ("not_found", "x"),  # Not found
            ],
        )

        assert result.failed == 3
        assert "empty" in result.outcomes[0].error.lower()
        assert "identical" in result.outcomes[1].error.lower()
        assert "not found" in result.outcomes[2].error.lower()

    def test_batch_edit_preserves_line_positions(self, cache: SemanticCache, temp_dir: Path):
        """Edits at different positions don't interfere."""
        f = temp_dir / "test.py"
        f.write_text("AAA\nBBB\nCCC\nDDD\n")

        result = smart_batch_edit(
            cache,
            str(f),
            [
                ("AAA", "111"),
                ("CCC", "333"),
            ],
        )

        assert result.succeeded == 2
        content = f.read_text()
        assert content == "111\nBBB\n333\nDDD\n"
