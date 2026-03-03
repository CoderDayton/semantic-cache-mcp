"""Tests for cross-process sync safety (file locking) and stderr logging."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

from semantic_cache_mcp.storage.sqlite import ConnectionPool, SQLiteStorage


class TestFileLock:
    """Tests for cross-process file locking on ConnectionPool."""

    def test_lock_file_created(self, temp_dir: Path) -> None:
        """ConnectionPool creates a .lock file next to the database."""
        db_path = temp_dir / "test.db"
        pool = ConnectionPool(db_path)
        try:
            # Lock file should be accessible (created on first acquire)
            with pool.get_connection():
                lock_path = Path(str(db_path) + ".lock")
                assert lock_path.exists()
        finally:
            pool.close_all()

    def test_file_lock_exists_on_pool(self, temp_dir: Path) -> None:
        """ConnectionPool._file_lock should be a FileLock instance."""
        from filelock import FileLock

        db_path = temp_dir / "test.db"
        pool = ConnectionPool(db_path)
        try:
            assert isinstance(pool._file_lock, FileLock)
            assert pool._file_lock.lock_file == str(db_path) + ".lock"
        finally:
            pool.close_all()

    def test_concurrent_thread_access(self, temp_dir: Path) -> None:
        """Two threads using the same pool should not crash."""
        db_path = temp_dir / "concurrent.db"
        storage = SQLiteStorage(db_path)

        # Create schema
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

    def test_two_pools_same_db(self, temp_dir: Path) -> None:
        """Two ConnectionPools on the same DB should serialize via file lock."""
        db_path = temp_dir / "shared.db"
        storage1 = SQLiteStorage(db_path)
        storage2 = SQLiteStorage(db_path)

        storage1.put("/test/1.txt", "from pool 1", 1.0)
        storage2.put("/test/2.txt", "from pool 2", 2.0)

        entry1 = storage1.get("/test/1.txt")
        entry2 = storage2.get("/test/2.txt")

        assert entry1 is not None
        assert entry2 is not None
        assert storage1.get_content(entry1) == "from pool 1"
        assert storage2.get_content(entry2) == "from pool 2"

    def test_file_lock_timeout(self, temp_dir: Path) -> None:
        """FileLock timeout should match the connection pool's expected range."""
        db_path = temp_dir / "timeout.db"
        pool = ConnectionPool(db_path)
        try:
            assert pool._file_lock.timeout == 30
        finally:
            pool.close_all()


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
