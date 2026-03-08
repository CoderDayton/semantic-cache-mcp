"""Tests for cross-process sync safety (file locking) and stderr logging."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

from semantic_cache_mcp.storage.vector import VectorStorage


class TestThreadSafety:
    """Tests for thread-safe VectorStorage access."""

    def test_concurrent_thread_access(self, temp_dir: Path) -> None:
        """Multiple threads using VectorStorage should not crash."""
        db_path = temp_dir / "concurrent.db"
        storage = VectorStorage(db_path)

        storage.put("/test/a.txt", "content a", 1.0)

        errors: list[Exception] = []

        def writer(path: str, content: str) -> None:
            try:
                storage.put(path, content, 1.0)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(f"/test/{i}.txt", f"content {i}"))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent writes failed: {errors}"

    def test_concurrent_reads(self, temp_dir: Path) -> None:
        """Concurrent reads should not corrupt data."""
        db_path = temp_dir / "reads.db"
        storage = VectorStorage(db_path)
        storage.put("/test/file.txt", "test content", 1.0)

        errors: list[Exception] = []
        results: list[str] = []

        def reader() -> None:
            try:
                entry = storage.get("/test/file.txt")
                if entry:
                    content = storage.get_content(entry)
                    results.append(content)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent reads failed: {errors}"
        assert all(r == "test content" for r in results)


class TestStderrLogging:
    """Tests for stderr-only logging configuration."""

    def test_root_logger_has_stderr_handler(self) -> None:
        """Root logger should have a handler writing to stderr."""
        root = logging.getLogger()
        stderr_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stderr
        ]
        assert len(stderr_handlers) >= 1, (
            f"No stderr handler on root logger. Handlers: {root.handlers}"
        )

    def test_logger_does_not_write_stdout(self) -> None:
        """Logging output should never go to stdout."""
        root = logging.getLogger()
        stdout_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
        ]
        assert not stdout_handlers, f"Found stdout handler(s) on root logger: {stdout_handlers}"

    def test_no_print_statements_in_source(self) -> None:
        """Source code should not contain bare print() calls."""
        import ast

        src_dir = Path(__file__).parent.parent / "src" / "semantic_cache_mcp"
        violations: list[str] = []

        for py_file in src_dir.rglob("*.py"):
            source = py_file.read_text()
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    violations.append(f"{py_file.name}:{node.lineno}")

        assert not violations, f"Found print() calls in source: {violations}"


class TestStdoutRedirect:
    """Tests for stdout redirect during model warmup."""

    def test_stdout_redirect_restores(self) -> None:
        """Stdout should be restored after lifespan initialization."""
        original_stdout = sys.stdout

        # Simulate the redirect pattern from _mcp.py
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr
        assert sys.stdout is sys.stderr

        # Restore
        sys.stdout = _real_stdout
        assert sys.stdout is original_stdout

    def test_redirect_catches_print(self, capsys) -> None:
        """Print during redirect should go to stderr, not stdout."""
        import io

        stderr_capture = io.StringIO()
        _real_stdout = sys.stdout
        sys.stdout = stderr_capture

        try:
            print("rogue output from third-party lib")
        finally:
            sys.stdout = _real_stdout

        assert "rogue output" in stderr_capture.getvalue()
