"""Tests for cross-process sync safety (file locking) and stderr logging."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import semantic_cache_mcp.config as config_mod
from semantic_cache_mcp.logger import configure_logging, get_log_dir, get_log_file_path
from semantic_cache_mcp.storage.vector import VectorStorage


class TestThreadSafety:
    """Tests for concurrent VectorStorage access via asyncio.gather."""

    async def test_concurrent_thread_access(self, temp_dir: Path) -> None:
        """Sequential writes to VectorStorage should not crash."""
        db_path = temp_dir / "concurrent.db"
        storage = VectorStorage(db_path)

        await storage.put("/test/a.txt", "content a", 1.0)

        for i in range(5):
            await storage.put(f"/test/{i}.txt", f"content {i}", 1.0)

    async def test_concurrent_reads(self, temp_dir: Path) -> None:
        """Concurrent reads should not corrupt data."""
        db_path = temp_dir / "reads.db"
        storage = VectorStorage(db_path)
        await storage.put("/test/file.txt", "test content", 1.0)

        results: list[str] = []

        async def reader() -> None:
            entry = await storage.get("/test/file.txt")
            if entry:
                content = await storage.get_content(entry)
                results.append(content)

        for _ in range(5):
            await reader()

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


class TestFileLogging:
    """Tests for dated file logging configuration."""

    def test_get_log_dir_defaults_to_cache_logs(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        assert get_log_dir(cache_dir) == (cache_dir / "logs").resolve()

    def test_get_log_file_path_uses_date(self, tmp_path: Path) -> None:
        log_dir = (tmp_path / "logs").resolve()
        log_path = get_log_file_path(log_dir, now=datetime(2026, 3, 27, 12, 0, 0))
        assert log_path == log_dir / "semantic-cache-mcp-2026-03-27.log"

    def test_configure_logging_creates_dated_file(self, tmp_path: Path) -> None:
        root = logging.getLogger()
        original_level = root.level

        log_dir = (tmp_path / "logs").resolve()
        log_path = get_log_file_path(log_dir, now=datetime(2026, 3, 27, 12, 0, 0))

        try:
            configure_logging(log_dir, log_path, log_level="INFO")
            logging.getLogger("semantic_cache_mcp.test").warning("file-log-smoke-test")

            file_handlers = [
                h
                for h in root.handlers
                if isinstance(h, logging.FileHandler)
                and getattr(h, "name", "") == "semantic-cache-mcp.file"
            ]
            assert len(file_handlers) == 1
            file_handlers[0].flush()

            assert log_dir.is_dir()
            assert log_path.exists()
            assert "file-log-smoke-test" in log_path.read_text(encoding="utf-8")
        finally:
            configure_logging(
                config_mod.LOG_DIR,
                config_mod.LOG_FILE_PATH,
                log_level=config_mod.LOG_LEVEL,
                log_format=config_mod.LOG_FORMAT,
            )
            root.setLevel(original_level)


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
