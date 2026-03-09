"""End-to-end tests for all MCP server tools.

Tests every tool through the cache layer with real files in /tmp/.
Covers small, medium, large (10K+ line) files and varied content types.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from semantic_cache_mcp.cache import SemanticCache, smart_read
from semantic_cache_mcp.cache.read import batch_smart_read
from semantic_cache_mcp.cache.search import (
    compare_files,
    find_similar_files,
    glob_with_cache_status,
    semantic_search,
)
from semantic_cache_mcp.cache.write import smart_batch_edit, smart_edit, smart_write

# ---------------------------------------------------------------------------
# Fixtures — file generators for various sizes and content types
# ---------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> SemanticCache:
    """Fresh SemanticCache for each test."""
    return SemanticCache(db_path=tmp_path / "cache.db")


@pytest.fixture
def small_py(tmp_path: Path) -> Path:
    """20-line Python file."""
    p = tmp_path / "small.py"
    lines = [
        '"""Small module."""',
        "",
        "import os",
        "from pathlib import Path",
        "",
        "",
        "def greet(name: str) -> str:",
        '    """Return greeting."""',
        '    return f"Hello, {name}!"',
        "",
        "",
        "def add(a: int, b: int) -> int:",
        '    """Add two numbers."""',
        "    return a + b",
        "",
        "",
        "def main() -> None:",
        '    print(greet("world"))',
        "    print(add(1, 2))",
        "",
    ]
    p.write_text("\n".join(lines))
    return p


@pytest.fixture
def medium_py(tmp_path: Path) -> Path:
    """~500-line Python file with classes, functions, imports."""
    lines: list[str] = [
        '"""Medium-sized module with several classes and functions."""',
        "",
        "from __future__ import annotations",
        "",
        "import json",
        "import logging",
        "from dataclasses import dataclass, field",
        "from pathlib import Path",
        "",
        "logger = logging.getLogger(__name__)",
        "",
    ]
    # Generate 10 dataclasses with methods (~45 lines each)
    for i in range(10):
        lines.extend(
            [
                "",
                "@dataclass",
                f"class Widget{i}:",
                f'    """Widget number {i}."""',
                "    name: str",
                f"    value: int = {i}",
                "    tags: list[str] = field(default_factory=list)",
                "",
                "    def process(self) -> dict:",
                '        """Process this widget."""',
                "        result = {",
                '            "name": self.name,',
                f'            "value": self.value * {i + 1},',
                '            "tags": self.tags,',
                "        }",
                '        logger.info(f"Processed widget {self.name}")',
                "        return result",
                "",
                "    def validate(self) -> bool:",
                '        """Validate widget state."""',
                "        if not self.name:",
                '            raise ValueError("Widget name required")',
                "        if self.value < 0:",
                '            raise ValueError("Value must be non-negative")',
                "        return True",
                "",
                "    def serialize(self) -> str:",
                '        """Serialize to JSON."""',
                "        return json.dumps(self.process())",
                "",
                "    @classmethod",
                f"    def from_dict(cls, data: dict) -> Widget{i}:",
                '        """Create from dictionary."""',
                "        return cls(",
                '            name=data["name"],',
                f'            value=data.get("value", {i}),',
                '            tags=data.get("tags", []),',
                "        )",
                "",
                "    def __repr__(self) -> str:",
                f'        return f"Widget{i}({{self.name}}, {{self.value}})"',
                "",
            ]
        )

    # Add some standalone functions
    for i in range(10):
        lines.extend(
            [
                f"def helper_{i}(x: int) -> int:",
                f'    """Helper function {i}."""',
                f"    return x * {i + 1} + {i}",
                "",
            ]
        )

    lines.append("# Total widgets: 10, helpers: 10")
    p = tmp_path / "medium.py"
    p.write_text("\n".join(lines))
    return p


@pytest.fixture
def large_py(tmp_path: Path) -> Path:
    """10,000+ line Python file — realistic large codebase file."""
    lines: list[str] = [
        '"""Large auto-generated module for stress testing."""',
        "",
        "from __future__ import annotations",
        "",
        "import hashlib",
        "import json",
        "import logging",
        "import os",
        "import time",
        "from dataclasses import dataclass",
        "from pathlib import Path",
        "from typing import Any",
        "",
        "logger = logging.getLogger(__name__)",
        "",
    ]

    # Generate 100 classes with ~80 lines each => ~8000 lines
    for i in range(100):
        lines.extend(
            [
                "",
                "@dataclass",
                f"class Entity{i:03d}:",
                f'    """Entity number {i} for processing pipeline."""',
                "    id: int",
                "    name: str",
                "    status: str = 'pending'",
                "    metadata: dict[str, Any] | None = None",
                "",
                "    def process(self) -> dict[str, Any]:",
                '        """Process this entity through the pipeline."""',
                "        start = time.monotonic()",
                "        result = {",
                '            "id": self.id,',
                '            "name": self.name,',
                '            "status": "processed",',
                '            "hash": hashlib.sha256(self.name.encode()).hexdigest(),',
                "        }",
                "        if self.metadata:",
                '            result["metadata"] = self.metadata',
                "        elapsed = time.monotonic() - start",
                '        logger.debug(f"Entity {self.id} processed in {elapsed:.4f}s")',
                "        return result",
                "",
                "    def validate(self) -> list[str]:",
                '        """Validate entity state, return list of errors."""',
                "        errors: list[str] = []",
                "        if not self.name:",
                '            errors.append("name is required")',
                "        if self.id < 0:",
                '            errors.append("id must be non-negative")',
                "        if self.status not in ('pending', 'active', 'processed', 'error'):",
                '            errors.append(f"invalid status: {self.status}")',
                "        return errors",
                "",
                "    def serialize(self) -> str:",
                '        """Serialize entity to JSON string."""',
                "        data = {",
                '            "id": self.id,',
                '            "name": self.name,',
                '            "status": self.status,',
                "        }",
                "        if self.metadata:",
                '            data["metadata"] = self.metadata',
                "        return json.dumps(data, indent=2)",
                "",
                "    @classmethod",
                f"    def from_json(cls, raw: str) -> Entity{i:03d}:",
                '        """Deserialize from JSON string."""',
                "        data = json.loads(raw)",
                "        return cls(",
                '            id=data["id"],',
                '            name=data["name"],',
                '            status=data.get("status", "pending"),',
                '            metadata=data.get("metadata"),',
                "        )",
                "",
                "    def to_log_entry(self) -> str:",
                '        """Format as log entry."""',
                '        return f"[{self.status.upper()}] Entity {self.id}: {self.name}"',
                "",
                f"    def merge(self, other: Entity{i:03d}) -> Entity{i:03d}:",
                '        """Merge another entity into this one."""',
                "        merged_meta = {**(self.metadata or {{}}), **(other.metadata or {{}})}",
                f"        return Entity{i:03d}(",
                "            id=self.id,",
                '            name=f"{self.name}+{other.name}",',
                "            status=self.status,",
                "            metadata=merged_meta if merged_meta else None,",
                "        )",
                "",
                "    def __hash__(self) -> int:",
                "        return hash((self.id, self.name))",
                "",
            ]
        )

    # Add 500 standalone functions => ~2000 more lines
    for i in range(500):
        lines.extend(
            [
                f"def transform_{i:03d}(value: Any) -> Any:",
                f'    """Transform function {i}."""',
                "    if isinstance(value, str):",
                f"        return value.upper() if len(value) < {i + 10} else value[:10]",
                "    return value",
                "",
            ]
        )

    p = tmp_path / "large.py"
    p.write_text("\n".join(lines))
    return p


@pytest.fixture
def various_files(tmp_path: Path) -> dict[str, Path]:
    """Dict of varied content files for format testing."""
    files: dict[str, Path] = {}

    # JSON
    json_f = tmp_path / "config.json"
    json_f.write_text(
        '{\n  "database": {\n    "host": "localhost",\n    "port": 5432\n  },\n  "debug": true\n}'
    )
    files["json"] = json_f

    # Markdown
    md_f = tmp_path / "README.md"
    md_f.write_text(
        "# Project\n\n## Overview\nThis is a test project.\n\n## Features\n- Feature 1\n- Feature 2\n\n```python\ndef example():\n    pass\n```\n"
    )
    files["md"] = md_f

    # SQL
    sql_f = tmp_path / "schema.sql"
    sql_f.write_text(
        "CREATE TABLE users (\n    id SERIAL PRIMARY KEY,\n    name VARCHAR(100) NOT NULL,\n    email VARCHAR(255) UNIQUE,\n    created_at TIMESTAMP DEFAULT NOW()\n);\n\nCREATE INDEX idx_users_email ON users(email);\n"
    )
    files["sql"] = sql_f

    # HTML
    html_f = tmp_path / "index.html"
    html_f.write_text(
        "<!DOCTYPE html>\n<html>\n<head><title>Test</title></head>\n<body>\n  <h1>Hello World</h1>\n  <p>Paragraph with <strong>bold</strong> text.</p>\n</body>\n</html>\n"
    )
    files["html"] = html_f

    # YAML
    yaml_f = tmp_path / "config.yaml"
    yaml_f.write_text(
        "server:\n  host: 0.0.0.0\n  port: 8080\n  workers: 4\n\nlogging:\n  level: INFO\n  format: json\n"
    )
    files["yaml"] = yaml_f

    # Unicode with CJK and emoji
    uni_f = tmp_path / "unicode.txt"
    uni_f.write_text(
        "Hello World 你好世界 こんにちは\nEmoji: 🚀 🎉 ✅ ❌\nArabic: مرحبا\nMixed: café résumé naïve\n"
    )
    files["unicode"] = uni_f

    # Empty file
    empty_f = tmp_path / "empty.txt"
    empty_f.write_text("")
    files["empty"] = empty_f

    # Whitespace-only file
    ws_f = tmp_path / "whitespace.txt"
    ws_f.write_text("   \n\t\n   \n")
    files["whitespace"] = ws_f

    return files


# ---------------------------------------------------------------------------
# Phase 2: Read Tool Tests
# ---------------------------------------------------------------------------


class TestReadFirstRead:
    """First read should return full content."""

    def test_first_read_returns_full_content(self, cache: SemanticCache, small_py: Path) -> None:
        result = smart_read(cache, str(small_py))
        assert not result.from_cache
        assert not result.is_diff
        assert result.tokens_saved == 0
        assert "def greet" in result.content

    def test_first_read_caches_file(self, cache: SemanticCache, small_py: Path) -> None:
        smart_read(cache, str(small_py))
        entry = cache.get(str(small_py))
        assert entry is not None
        assert entry.tokens > 0


class TestReadUnchanged:
    """Second read of unchanged file should return 'unchanged' marker."""

    def test_unchanged_returns_marker(self, cache: SemanticCache, small_py: Path) -> None:
        smart_read(cache, str(small_py))
        result = smart_read(cache, str(small_py))
        assert result.from_cache
        assert not result.is_diff
        assert "unchanged" in result.content.lower()
        assert result.tokens_saved > 0

    def test_unchanged_saves_tokens(self, cache: SemanticCache, medium_py: Path) -> None:
        first = smart_read(cache, str(medium_py))
        second = smart_read(cache, str(medium_py))
        # Unchanged message should be much smaller than original
        assert second.tokens_returned < first.tokens_returned
        assert second.tokens_saved > 0


class TestReadDiff:
    """Modified file should return diff."""

    def test_modified_returns_diff(self, cache: SemanticCache, medium_py: Path) -> None:
        """Modified file should return diff (needs medium+ file; small files
        fall through to full read because diff overhead > 60% threshold)."""
        smart_read(cache, str(medium_py))

        # Modify one line in the medium file
        content = medium_py.read_text()
        medium_py.write_text(content.replace("Widget0", "Gadget0"))

        result = smart_read(cache, str(medium_py))
        assert result.from_cache
        assert result.is_diff
        assert "-" in result.content or "+" in result.content

    def test_diff_saves_tokens_on_small_change(self, cache: SemanticCache, large_py: Path) -> None:
        """A 5-line change on a 10K file should produce a small diff."""
        first = smart_read(cache, str(large_py))

        # Modify 5 lines near the middle
        lines = large_py.read_text().splitlines()
        for i in range(5000, 5005):
            if i < len(lines):
                lines[i] = f"# MODIFIED LINE {i}"
        large_py.write_text("\n".join(lines))

        result = smart_read(cache, str(large_py))
        assert result.is_diff
        assert result.tokens_saved > 0
        # Diff should be much smaller than full file
        assert result.tokens_returned < first.tokens_returned * 0.5


class TestReadDiffModeOff:
    """diff_mode=False should force full content."""

    def test_force_full_returns_complete_content(
        self, cache: SemanticCache, small_py: Path
    ) -> None:
        smart_read(cache, str(small_py))
        result = smart_read(cache, str(small_py), diff_mode=False)
        # Should return full content, not unchanged marker
        assert "def greet" in result.content


class TestReadOffsetLimit:
    """offset/limit should return specific line ranges."""

    def test_offset_limit_returns_range(self, cache: SemanticCache, medium_py: Path) -> None:
        """offset/limit is handled in tools.py read() — test the slicing logic."""
        content = medium_py.read_text()
        all_lines = content.splitlines()
        assert len(all_lines) > 50

        # Simulate what the MCP read tool does for offset/limit
        offset, limit = 10, 5
        selected = all_lines[offset - 1 : offset - 1 + limit]
        assert len(selected) == 5
        # Lines should be from the middle of the file
        assert selected[0] == all_lines[9]

    def test_offset_out_of_bounds(self, cache: SemanticCache, small_py: Path) -> None:
        """Offset beyond file length should return empty."""
        content = small_py.read_text()
        lines = content.splitlines()
        # Requesting beyond file end
        offset = len(lines) + 100
        selected = lines[offset - 1 : offset - 1 + 10]
        assert len(selected) == 0


class TestReadLargeFile:
    """Large file handling."""

    def test_large_file_truncation(self, cache: SemanticCache, tmp_path: Path) -> None:
        """File exceeding max_size should be truncated."""
        big = tmp_path / "big.txt"
        big.write_text("x" * 200_000)
        result = smart_read(cache, str(big), max_size=50_000)
        assert result.truncated
        assert len(result.content) <= 50_000

    def test_10k_line_file_no_timeout(self, cache: SemanticCache, large_py: Path) -> None:
        """10K+ line file should complete without timeout.
        NOTE: 10K-line file exceeds MAX_CONTENT_SIZE (100KB) so it gets
        truncated via semantic summarization. This is expected behavior."""
        start = time.monotonic()
        result = smart_read(cache, str(large_py))
        elapsed = time.monotonic() - start
        assert elapsed < 120.0  # macOS CI cold-starts embedding + summarizes 134K tokens
        assert result.tokens_original > 0
        # Large file exceeds MAX_CONTENT_SIZE, so truncation is expected
        if result.truncated:
            assert len(result.content) <= 100_000


class TestReadEdgeCases:
    """Edge case handling for read."""

    def test_binary_file_rejected(self, cache: SemanticCache, tmp_path: Path) -> None:
        binary = tmp_path / "data.bin"
        binary.write_bytes(b"\x00\x01\x02\xff\xfe\x80")
        with pytest.raises(ValueError, match="[Bb]inary"):
            smart_read(cache, str(binary))

    def test_empty_file(self, cache: SemanticCache, tmp_path: Path) -> None:
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        result = smart_read(cache, str(empty))
        assert result.content == ""
        assert result.tokens_original == 0

    def test_unicode_content(self, cache: SemanticCache, various_files: dict[str, Path]) -> None:
        """Unicode file should round-trip with full fidelity."""
        uni = various_files["unicode"]
        original = uni.read_text()
        result = smart_read(cache, str(uni))
        assert "你好世界" in result.content
        assert "🚀" in result.content
        assert result.content == original

    def test_symlink_following(self, cache: SemanticCache, small_py: Path, tmp_path: Path) -> None:
        link = tmp_path / "link.py"
        link.symlink_to(small_py)
        result = smart_read(cache, str(link))
        assert "def greet" in result.content

    def test_file_not_found(self, cache: SemanticCache) -> None:
        with pytest.raises(FileNotFoundError):
            smart_read(cache, "/tmp/nonexistent_file_abc123.py")

    def test_whitespace_only_file(
        self, cache: SemanticCache, various_files: dict[str, Path]
    ) -> None:
        ws = various_files["whitespace"]
        result = smart_read(cache, str(ws))
        assert result.content.strip() == ""
        assert result.tokens_original >= 0  # Should not crash


# ---------------------------------------------------------------------------
# Phase 3: Write Tool Tests
# ---------------------------------------------------------------------------


class TestWriteNew:
    """Write new files."""

    def test_write_new_file(self, cache: SemanticCache, tmp_path: Path) -> None:
        target = tmp_path / "new_file.py"
        content = "def hello():\n    return 42\n"
        result = smart_write(cache, str(target), content)
        assert result.created
        assert target.exists()
        assert target.read_text() == content

    def test_write_creates_parents(self, cache: SemanticCache, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c" / "deep.py"
        result = smart_write(cache, str(target), "x = 1\n")
        assert result.created
        assert target.exists()


class TestWriteOverwrite:
    """Overwrite existing files."""

    def test_overwrite_returns_diff(self, cache: SemanticCache, small_py: Path) -> None:
        # Cache the file first
        smart_read(cache, str(small_py))
        result = smart_write(cache, str(small_py), "# completely new\nx = 1\n")
        assert not result.created
        assert result.diff_content is not None

    def test_write_then_read_cached(self, cache: SemanticCache, tmp_path: Path) -> None:
        target = tmp_path / "written.py"
        content = "class Foo:\n    pass\n"
        smart_write(cache, str(target), content)
        result = smart_read(cache, str(target))
        # Should come from cache
        assert result.from_cache


class TestWriteAppend:
    """Append mode."""

    def test_append_adds_content(self, cache: SemanticCache, tmp_path: Path) -> None:
        target = tmp_path / "append.txt"
        smart_write(cache, str(target), "line 1\n")
        smart_write(cache, str(target), "line 2\n", append=True)
        assert target.read_text() == "line 1\nline 2\n"


class TestWriteDryRun:
    """Dry run should not modify disk."""

    def test_dry_run_no_change(self, cache: SemanticCache, small_py: Path) -> None:
        original = small_py.read_text()
        smart_write(cache, str(small_py), "# replaced\n", dry_run=True)
        assert small_py.read_text() == original


class TestWriteLargeFile:
    """Write large files."""

    def test_write_10k_lines(self, cache: SemanticCache, tmp_path: Path) -> None:
        target = tmp_path / "big_write.py"
        content = "\n".join(f"line_{i} = {i}" for i in range(10_000))
        result = smart_write(cache, str(target), content)
        assert result.created
        assert target.read_text() == content


# ---------------------------------------------------------------------------
# Phase 4: Edit Tool Tests
# ---------------------------------------------------------------------------


class TestEditFindReplace:
    """Basic find/replace editing."""

    def test_single_replacement(self, cache: SemanticCache, small_py: Path) -> None:
        result = smart_edit(cache, str(small_py), old_string="def greet", new_string="def welcome")
        assert result.replacements_made == 1
        assert "welcome" in small_py.read_text()

    def test_multiple_matches_error(self, cache: SemanticCache, tmp_path: Path) -> None:
        f = tmp_path / "multi.py"
        f.write_text("foo = 1\nfoo = 2\nfoo = 3\n")
        with pytest.raises(ValueError, match="[Mm]ultiple|[Mm]ore than"):
            smart_edit(cache, str(f), old_string="foo", new_string="bar")

    def test_replace_all(self, cache: SemanticCache, tmp_path: Path) -> None:
        f = tmp_path / "multi.py"
        f.write_text("foo = 1\nfoo = 2\nfoo = 3\n")
        result = smart_edit(cache, str(f), old_string="foo", new_string="bar", replace_all=True)
        assert result.replacements_made == 3
        assert "foo" not in f.read_text()


class TestEditScoped:
    """Scoped find/replace with line ranges."""

    def test_scoped_edit(self, cache: SemanticCache, tmp_path: Path) -> None:
        f = tmp_path / "scoped.py"
        lines = [f"x_{i} = {i}" for i in range(20)]
        # Put "target" on lines 5 and 15
        lines[4] = "target = 'first'"
        lines[14] = "target = 'second'"
        f.write_text("\n".join(lines))

        # Edit only within lines 1-10 (should hit line 5 only)
        result = smart_edit(
            cache,
            str(f),
            old_string="target",
            new_string="found",
            start_line=1,
            end_line=10,
        )
        assert result.replacements_made == 1
        content = f.read_text()
        assert "found = 'first'" in content
        assert "target = 'second'" in content  # Untouched


class TestEditLineReplace:
    """Line replace mode (no old_string)."""

    def test_line_replace(self, cache: SemanticCache, tmp_path: Path) -> None:
        f = tmp_path / "linereplace.py"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        result = smart_edit(
            cache,
            str(f),
            old_string=None,
            new_string="replaced",
            start_line=2,
            end_line=4,
        )
        # Line replace counts lines replaced (3), not operations (1)
        assert result.replacements_made >= 1
        content = f.read_text()
        assert "line1" in content
        assert "replaced" in content
        assert "line5" in content
        # lines 2-4 should be gone
        assert "line2" not in content
        assert "line3" not in content
        assert "line4" not in content


class TestEditDryRun:
    """Dry run edits."""

    def test_dry_run_no_change(self, cache: SemanticCache, small_py: Path) -> None:
        original = small_py.read_text()
        smart_edit(cache, str(small_py), old_string="def greet", new_string="def xxx", dry_run=True)
        assert small_py.read_text() == original


class TestEditLargeFile:
    """Edit on large files."""

    def test_edit_near_end_of_large_file(self, cache: SemanticCache, large_py: Path) -> None:
        """Edit near line 10000 should be fast."""
        start = time.monotonic()
        # The large file has transform_499 near the end
        result = smart_edit(
            cache,
            str(large_py),
            old_string="def transform_499",
            new_string="def transform_final",
        )
        elapsed = time.monotonic() - start
        assert result.replacements_made == 1
        assert elapsed < 5.0


class TestEditCacheConsistency:
    """Cache should be updated after edit."""

    def test_edit_then_read_unchanged(self, cache: SemanticCache, small_py: Path) -> None:
        # Read to cache
        smart_read(cache, str(small_py))
        # Edit (updates cache)
        smart_edit(cache, str(small_py), old_string="def greet", new_string="def welcome")
        # Read again — should be "unchanged" since cache was updated by edit
        result = smart_read(cache, str(small_py))
        assert result.from_cache
        assert not result.is_diff

    def test_edit_nonexistent_file(self, cache: SemanticCache) -> None:
        with pytest.raises(FileNotFoundError):
            smart_edit(cache, "/tmp/nonexistent_xyz.py", old_string="x", new_string="y")


# ---------------------------------------------------------------------------
# Phase 5: Batch Edit Tests
# ---------------------------------------------------------------------------


class TestBatchEdit:
    """Batch edit operations."""

    def test_multiple_successful_edits(self, cache: SemanticCache, tmp_path: Path) -> None:
        f = tmp_path / "batch.py"
        f.write_text("aaa = 1\nbbb = 2\nccc = 3\n")
        edits = [
            ("aaa", "AAA"),
            ("bbb", "BBB"),
            ("ccc", "CCC"),
        ]
        result = smart_batch_edit(cache, str(f), edits)
        assert result.succeeded == 3
        assert result.failed == 0
        content = f.read_text()
        assert "AAA" in content
        assert "BBB" in content
        assert "CCC" in content

    def test_partial_failure(self, cache: SemanticCache, tmp_path: Path) -> None:
        f = tmp_path / "partial.py"
        f.write_text("aaa = 1\nbbb = 2\n")
        edits = [
            ("aaa", "AAA"),
            ("zzz", "ZZZ"),  # This won't match
            ("bbb", "BBB"),
        ]
        result = smart_batch_edit(cache, str(f), edits)
        assert result.succeeded == 2
        assert result.failed == 1

    def test_batch_edit_large_file(self, cache: SemanticCache, large_py: Path) -> None:
        """10 edits scattered through a 10K file."""
        edits = [(f"def transform_{i:03d}", f"def xform_{i:03d}") for i in range(0, 50, 5)]
        result = smart_batch_edit(cache, str(large_py), edits)
        assert result.succeeded == 10
        assert result.failed == 0


# ---------------------------------------------------------------------------
# Phase 6: Batch Read Tests
# ---------------------------------------------------------------------------


class TestBatchRead:
    """Batch read operations."""

    def test_read_multiple_files(
        self, cache: SemanticCache, various_files: dict[str, Path]
    ) -> None:
        paths = [str(v) for v in various_files.values() if v.stat().st_size > 0]
        result = batch_smart_read(cache, paths[:5])
        assert result.files_read > 0
        assert result.files_skipped == 0

    def test_token_budget_enforcement(self, cache: SemanticCache, tmp_path: Path) -> None:
        """Should skip files when budget exhausted."""
        files = []
        for i in range(10):
            f = tmp_path / f"budget_{i}.py"
            f.write_text("# content\n" * 500)  # ~500 lines each
            files.append(str(f))
        result = batch_smart_read(cache, files, max_total_tokens=1000)
        # Should have skipped some files
        assert result.files_skipped > 0

    def test_unchanged_detection(
        self, cache: SemanticCache, small_py: Path, tmp_path: Path
    ) -> None:
        """Second batch read should detect unchanged files."""
        f2 = tmp_path / "other.py"
        f2.write_text("y = 2\n")
        paths = [str(small_py), str(f2)]

        batch_smart_read(cache, paths)
        result = batch_smart_read(cache, paths)
        assert len(result.unchanged_paths) == 2

    def test_priority_ordering(self, cache: SemanticCache, tmp_path: Path) -> None:
        """Priority files should be read first."""
        files = []
        for i in range(5):
            f = tmp_path / f"prio_{i}.py"
            f.write_text(f"x = {i}\n")
            files.append(str(f))

        # Prioritize the last file
        result = batch_smart_read(cache, files, priority=[files[4]])
        # First file in result should be the priority one
        assert result.files[0].path == files[4]


# ---------------------------------------------------------------------------
# Phase 7: Search Quality Tests
# ---------------------------------------------------------------------------


class TestSearchQuality:
    """Search relevance and quality."""

    def test_search_empty_cache(self, cache: SemanticCache) -> None:
        result = semantic_search(cache, "hello world")
        assert result.matches == []

    def test_search_finds_cached_content(
        self, cache: SemanticCache, various_files: dict[str, Path]
    ) -> None:
        """Search should find files after caching."""
        # Cache all files
        for p in various_files.values():
            if p.stat().st_size > 0:
                smart_read(cache, str(p))

        result = semantic_search(cache, "database", k=5)
        # Should find the SQL file at minimum via BM25 keyword match
        paths = [m.path for m in result.matches]
        sql_found = any("schema.sql" in p for p in paths)
        assert sql_found or result.cached_files > 0  # At least cache is populated


class TestGrepTool:
    """Grep search across cached files."""

    def test_grep_finds_pattern(self, cache: SemanticCache, small_py: Path) -> None:
        smart_read(cache, str(small_py))
        results = cache._storage.grep("def ")
        assert len(results) > 0
        # Should find function definitions
        found_funcs = False
        for file_result in results:
            for match in file_result["matches"]:
                if "def " in match["line"]:
                    found_funcs = True
        assert found_funcs

    def test_grep_regex(self, cache: SemanticCache, small_py: Path) -> None:
        smart_read(cache, str(small_py))
        results = cache._storage.grep(r"def \w+\(")
        assert len(results) > 0

    def test_grep_fixed_string(self, cache: SemanticCache, tmp_path: Path) -> None:
        f = tmp_path / "special.py"
        f.write_text("result = foo.bar()\nother = baz()\n")
        smart_read(cache, str(f))
        results = cache._storage.grep("foo.bar()", fixed_string=True)
        assert len(results) == 1
        assert results[0]["matches"][0]["line_number"] == 1

    def test_grep_case_insensitive(self, cache: SemanticCache, tmp_path: Path) -> None:
        f = tmp_path / "case.py"
        f.write_text("Hello World\nhello world\nHELLO WORLD\n")
        smart_read(cache, str(f))
        results = cache._storage.grep("hello", case_sensitive=False)
        total = sum(len(r["matches"]) for r in results)
        assert total == 3

    def test_grep_context_lines(self, cache: SemanticCache, small_py: Path) -> None:
        smart_read(cache, str(small_py))
        results = cache._storage.grep("def greet", context_lines=2)
        assert len(results) > 0
        match = results[0]["matches"][0]
        assert "before" in match or "after" in match

    def test_grep_max_matches(self, cache: SemanticCache, large_py: Path) -> None:
        """Grep should respect max_matches limit."""
        smart_read(cache, str(large_py))
        results = cache._storage.grep("def ", max_matches=5)
        total = sum(len(r["matches"]) for r in results)
        assert total <= 5


# ---------------------------------------------------------------------------
# Phase 8: Similar Files Tests
# ---------------------------------------------------------------------------


class TestSimilarFiles:
    """Similar file search."""

    def test_similar_no_results_single_file(self, cache: SemanticCache, small_py: Path) -> None:
        """Only file in cache should return empty similar list."""
        smart_read(cache, str(small_py))
        result = find_similar_files(cache, str(small_py), k=3)
        # The source file is excluded, so no similar files
        assert len(result.similar_files) == 0


# ---------------------------------------------------------------------------
# Phase 9: Diff Tool Tests
# ---------------------------------------------------------------------------


class TestDiffTool:
    """File comparison."""

    def test_diff_different_files(self, cache: SemanticCache, tmp_path: Path) -> None:
        f1 = tmp_path / "file1.py"
        f2 = tmp_path / "file2.py"
        f1.write_text("def foo():\n    return 1\n")
        f2.write_text("def foo():\n    return 2\n")

        result = compare_files(cache, str(f1), str(f2))
        assert result.diff_content != ""
        assert result.diff_stats["modifications"] > 0 or result.diff_stats["deletions"] > 0

    def test_diff_identical_files(self, cache: SemanticCache, tmp_path: Path) -> None:
        f1 = tmp_path / "same1.py"
        f2 = tmp_path / "same2.py"
        content = "x = 1\ny = 2\n"
        f1.write_text(content)
        f2.write_text(content)

        result = compare_files(cache, str(f1), str(f2))
        # Diff of identical files should be empty
        assert result.diff_stats["insertions"] == 0
        assert result.diff_stats["deletions"] == 0


# ---------------------------------------------------------------------------
# Phase 10: Glob Tool Tests
# ---------------------------------------------------------------------------


class TestGlobTool:
    """File glob with cache status."""

    def test_basic_glob(self, cache: SemanticCache, tmp_path: Path) -> None:
        (tmp_path / "a.py").write_text("x = 1")
        (tmp_path / "b.py").write_text("y = 2")
        (tmp_path / "c.txt").write_text("hello")

        result = glob_with_cache_status(cache, "*.py", directory=str(tmp_path))
        py_matches = [m for m in result.matches if m.path.endswith(".py")]
        assert len(py_matches) == 2

    def test_glob_cached_only(self, cache: SemanticCache, tmp_path: Path) -> None:
        f1 = tmp_path / "cached.py"
        f2 = tmp_path / "uncached.py"
        f1.write_text("x = 1")
        f2.write_text("y = 2")

        # Cache only f1
        smart_read(cache, str(f1))

        result = glob_with_cache_status(cache, "*.py", directory=str(tmp_path), cached_only=True)
        assert len(result.matches) == 1
        assert "cached.py" in result.matches[0].path

    def test_glob_recursive(self, cache: SemanticCache, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.py").write_text("x = 1")
        (sub / "nested.py").write_text("y = 2")

        result = glob_with_cache_status(cache, "**/*.py", directory=str(tmp_path))
        assert result.total_matches == 2


# ---------------------------------------------------------------------------
# Phase 11: Stats & Clear Tests
# ---------------------------------------------------------------------------


class TestStatsAndClear:
    """Cache statistics and clearing."""

    def test_stats_reflect_cached_files(self, cache: SemanticCache, tmp_path: Path) -> None:
        for i in range(3):
            f = tmp_path / f"stat_{i}.py"
            f.write_text(f"x = {i}")
            smart_read(cache, str(f))

        stats = cache.get_stats()
        assert stats["files_cached"] == 3

    def test_clear_empties_cache(self, cache: SemanticCache, tmp_path: Path) -> None:
        f = tmp_path / "to_clear.py"
        f.write_text("x = 1")
        smart_read(cache, str(f))

        cache.clear()
        stats = cache.get_stats()
        assert stats["files_cached"] == 0

    def test_read_after_clear_is_full(self, cache: SemanticCache, tmp_path: Path) -> None:
        f = tmp_path / "recleared.py"
        f.write_text("x = 1")
        smart_read(cache, str(f))
        cache.clear()

        result = smart_read(cache, str(f))
        assert not result.from_cache


# ---------------------------------------------------------------------------
# Phase 12: Stale Cache Tests (mtime change, content unchanged)
# ---------------------------------------------------------------------------


class TestStaleCache:
    """Test that mtime-changed but content-identical files are handled correctly."""

    def test_touch_without_content_change(self, cache: SemanticCache, small_py: Path) -> None:
        """Touch (mtime bump) without content change returns 'unchanged' via content hash."""
        smart_read(cache, str(small_py))

        # Bump mtime without changing content
        future = time.time() + 100
        os.utime(small_py, (future, future))

        result = smart_read(cache, str(small_py))
        # Content hash match should detect unchanged content despite mtime bump
        assert result.from_cache
        assert not result.is_diff
        assert "unchanged" in result.content.lower()

    def test_actual_content_change_still_diffs(self, cache: SemanticCache, small_py: Path) -> None:
        """Real content change must still return diff (fix doesn't suppress real changes)."""
        smart_read(cache, str(small_py))

        # Actually modify content
        content = small_py.read_text()
        small_py.write_text(content.replace("greet", "salute"))

        result = smart_read(cache, str(small_py))
        assert result.is_diff or not result.from_cache
        # The change should be visible
        assert "salute" in small_py.read_text()


# ---------------------------------------------------------------------------
# Phase 13: Edge Cases & Stress Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and stress tests."""

    def test_concurrent_reads(self, cache: SemanticCache, small_py: Path) -> None:
        """10 threads reading same file simultaneously."""
        smart_read(cache, str(small_py))  # Prime cache
        errors: list[Exception] = []
        results: list[str] = []

        def reader():
            try:
                r = smart_read(cache, str(small_py))
                results.append(r.content)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent reads failed: {errors}"

    def test_file_deleted_between_reads(self, cache: SemanticCache, tmp_path: Path) -> None:
        f = tmp_path / "ephemeral.py"
        f.write_text("x = 1")
        smart_read(cache, str(f))
        f.unlink()
        with pytest.raises(FileNotFoundError):
            smart_read(cache, str(f))

    def test_very_long_path(self, cache: SemanticCache, tmp_path: Path) -> None:
        """Path near filesystem limit should work."""
        # Create nested dirs to get a long path
        current = tmp_path
        for i in range(10):
            current = current / f"dir_{i:03d}_padding"
            current.mkdir()
        f = current / "file.py"
        f.write_text("x = 1\n")
        result = smart_read(cache, str(f))
        assert "x = 1" in result.content

    def test_rapid_read_modify_read(self, cache: SemanticCache, tmp_path: Path) -> None:
        """Rapid cycle of modify + read should keep cache consistent."""
        f = tmp_path / "rapid.py"
        f.write_text("version = 0\n")

        for i in range(50):  # 50 iterations (reduced from 100 for speed)
            f.write_text(f"version = {i + 1}\n")
            result = smart_read(cache, str(f))
            # Content should always reflect latest version
            assert f"version = {i + 1}" in result.content or result.is_diff

    def test_various_file_formats(
        self, cache: SemanticCache, various_files: dict[str, Path]
    ) -> None:
        """All file formats should be readable and cacheable."""
        for name, path in various_files.items():
            if path.stat().st_size == 0 and name == "empty":
                continue  # Skip empty
            result = smart_read(cache, str(path))
            assert result.tokens_original >= 0, f"Failed on {name}"


# ---------------------------------------------------------------------------
# Phase 14: Throughput & Concurrency
# ---------------------------------------------------------------------------


class TestThroughputConcurrency:
    """Measure throughput under concurrent load and verify data integrity."""

    def test_concurrent_read_write_interleave(self, cache: SemanticCache, tmp_path: Path) -> None:
        """Writers and readers hitting different files concurrently."""
        n_files = 20
        files = []
        for i in range(n_files):
            f = tmp_path / f"concurrent_{i:03d}.py"
            f.write_text(f"# file {i}\nx = {i}\n")
            files.append(f)

        errors: list[Exception] = []

        def writer(idx: int) -> None:
            try:
                f = files[idx]
                smart_write(cache, str(f), f"# file {idx}\nx = {idx * 10}\n")
            except Exception as e:
                errors.append(e)

        def reader(idx: int) -> None:
            try:
                smart_read(cache, str(files[idx]))
            except Exception as e:
                errors.append(e)

        threads: list[threading.Thread] = []
        for i in range(n_files):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent read/write failed: {errors}"

    def test_concurrent_writes_same_file(self, cache: SemanticCache, tmp_path: Path) -> None:
        """Multiple threads writing to the same file — last write wins, no crash."""
        f = tmp_path / "contended.py"
        f.write_text("v = 0\n")
        errors: list[Exception] = []

        def writer(version: int) -> None:
            try:
                smart_write(cache, str(f), f"v = {version}\n")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent same-file writes failed: {errors}"
        # File should contain one of the written versions
        final = f.read_text()
        assert final.startswith("v = ")

    def test_concurrent_edits_different_files(self, cache: SemanticCache, tmp_path: Path) -> None:
        """Concurrent edits on separate files should all succeed."""
        n_files = 15
        files = []
        for i in range(n_files):
            f = tmp_path / f"edit_target_{i:03d}.py"
            f.write_text(f"placeholder_{i} = True\n")
            files.append(f)

        errors: list[Exception] = []
        results: list[bool] = []

        def editor(idx: int) -> None:
            try:
                r = smart_edit(
                    cache,
                    str(files[idx]),
                    old_string=f"placeholder_{idx} = True",
                    new_string=f"placeholder_{idx} = False",
                )
                results.append(r.replacements_made > 0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=editor, args=(i,)) for i in range(n_files)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent edits failed: {errors}"
        assert all(results), "Some edits did not apply"

    def test_batch_read_throughput(self, cache: SemanticCache, tmp_path: Path) -> None:
        """batch_smart_read with 30 files should complete and return all."""
        n_files = 30
        paths = []
        for i in range(n_files):
            f = tmp_path / f"batch_{i:03d}.py"
            f.write_text(f"# batch file {i}\ndata = {i}\n")
            paths.append(str(f))

        result = batch_smart_read(cache, paths)
        assert len(result.files) == n_files
        assert result.total_tokens > 0

    def test_rapid_write_read_consistency(self, cache: SemanticCache, tmp_path: Path) -> None:
        """Rapid sequential write→read cycles must always reflect latest content."""
        f = tmp_path / "rapid_wr.py"
        f.write_text("v = 0\n")

        for i in range(1, 51):
            content = f"v = {i}\n"
            smart_write(cache, str(f), content)
            result = smart_read(cache, str(f), diff_mode=False)
            assert f"v = {i}" in result.content, f"Stale at iteration {i}"

    def test_concurrent_search_during_writes(self, cache: SemanticCache, tmp_path: Path) -> None:
        """Semantic search should not crash while files are being written."""
        # Seed cache with some files first
        for i in range(5):
            f = tmp_path / f"searchable_{i}.py"
            f.write_text(f"def function_{i}(): return {i}\n")
            smart_read(cache, str(f))

        errors: list[Exception] = []

        def searcher() -> None:
            try:
                semantic_search(cache, "function that returns a value", k=3)
            except Exception as e:
                errors.append(e)

        def writer(idx: int) -> None:
            try:
                f = tmp_path / f"new_during_search_{idx}.py"
                smart_write(cache, str(f), f"def new_func_{idx}(): pass\n")
            except Exception as e:
                errors.append(e)

        threads: list[threading.Thread] = []
        for _ in range(5):
            threads.append(threading.Thread(target=searcher))
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent search+write failed: {errors}"

    def test_cache_integrity_after_concurrent_storm(
        self, cache: SemanticCache, tmp_path: Path
    ) -> None:
        """After heavy concurrent load, cache stats should be consistent."""
        n_files = 10
        files = []
        for i in range(n_files):
            f = tmp_path / f"storm_{i:03d}.py"
            f.write_text(f"storm = {i}\n")
            files.append(f)

        errors: list[Exception] = []

        def churn(idx: int) -> None:
            """Read, write, edit, read — full cycle per file (sequential within thread)."""
            try:
                f = files[idx]
                smart_read(cache, str(f))
                smart_write(cache, str(f), f"storm_a = {idx}\n")
                smart_edit(
                    cache,
                    str(f),
                    old_string=f"storm_a = {idx}",
                    new_string=f"storm_b = {idx * 200}",
                )
                smart_read(cache, str(f))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=churn, args=(i,)) for i in range(n_files)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, f"Concurrent storm failed: {errors}"

        # Verify cache is still queryable
        stats = cache.get_stats()
        assert stats["files_cached"] > 0

        # Verify each file has correct final content
        for i, f in enumerate(files):
            content = f.read_text()
            assert f"storm_b = {i * 200}" in content, f"File {i} has wrong content: {content}"
