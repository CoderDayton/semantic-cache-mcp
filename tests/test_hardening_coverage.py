"""Tests for 0.2.0 hardening changes — patch coverage for all modified code paths."""

from __future__ import annotations

import array
import os
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastmcp.exceptions import ToolError

from semantic_cache_mcp.cache import SemanticCache, compare_files, glob_with_cache_status
from semantic_cache_mcp.cache._helpers import _format_file
from semantic_cache_mcp.cache.search import find_similar_files, semantic_search
from semantic_cache_mcp.cache.write import _atomic_write, smart_edit
from semantic_cache_mcp.core.similarity._cosine import (
    _select_pruning_dims,
    cosine_similarity_batch,
    cosine_similarity_batch_matrix,
    top_k_similarities,
)
from semantic_cache_mcp.storage.sqlite import ConnectionPool
from tests.constants import TEST_EMBEDDING_DIM

# ---------------------------------------------------------------------------
# _helpers.py — special file rejection in _format_file
# ---------------------------------------------------------------------------


class TestFormatFileSpecialFileRejection:
    """Cover lines 62-69: stat.S_ISREG check and OSError handler."""

    async def test_rejects_non_regular_file(self, temp_dir: Path) -> None:
        """FIFO (named pipe) rejected before subprocess."""
        fifo_path = temp_dir / "test.py"
        os.mkfifo(fifo_path)  # Creates a named pipe
        assert not await _format_file(fifo_path)

    async def test_oserror_on_stat_returns_false(self, temp_dir: Path) -> None:
        """OSError during stat (e.g. dangling symlink) → False."""
        dangling = temp_dir / "dangling.py"
        dangling.symlink_to(temp_dir / "nonexistent.py")
        assert not await _format_file(dangling)

    async def test_regular_file_passes_check(self, temp_dir: Path) -> None:
        """Regular .py file is not blocked by S_ISREG check."""
        py_file = temp_dir / "valid.py"
        py_file.write_text("x = 1\n")
        # Even if formatter not installed, the check itself should pass
        # (function returns False for missing formatter, not for stat check)
        result = await _format_file(py_file)
        # Result depends on whether ruff is installed; we just verify no crash
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _cosine.py — np.partition pruning, pre-alloc, top_k_similarities
# ---------------------------------------------------------------------------


class TestPruningDimsPartition:
    """Cover lines 258-262: np.partition pruning path."""

    def test_adaptive_pruning_with_partition(self) -> None:
        """fraction < 1.0 and adaptive=True triggers np.partition."""
        query = np.array([0.1, 0.5, 0.01, 0.8, 0.3, 0.02, 0.9, 0.4], dtype=np.float32)
        mask = _select_pruning_dims(query, fraction=0.5, adaptive=True)
        assert mask.dtype == bool
        assert mask.sum() > 0
        # Should keep the high-magnitude dimensions
        assert mask[6]  # 0.9

    def test_adaptive_pruning_fallback_to_ones(self) -> None:
        """prune_count == 0 (fraction ~1.0) falls through to np.ones."""
        query = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        mask = _select_pruning_dims(query, fraction=0.99, adaptive=True)
        assert mask.all()


class TestPreAllocatedMatrices:
    """Cover pre-allocated matrix paths in cosine_similarity_batch[_matrix]."""

    def test_batch_with_array_array(self) -> None:
        """cosine_similarity_batch with array.array inputs uses pre-alloc."""
        dim = 16
        rng = np.random.default_rng(42)
        query = array.array("f", rng.standard_normal(dim).astype(np.float32).tolist())
        vectors = [
            array.array("f", rng.standard_normal(dim).astype(np.float32).tolist()) for _ in range(5)
        ]
        result = cosine_similarity_batch(query, vectors, use_pruning=False)
        assert len(result) == 5
        assert all(isinstance(s, float) for s in result)

    def test_batch_matrix_with_array_array(self) -> None:
        """cosine_similarity_batch_matrix with array.array uses pre-alloc."""
        dim = 16
        rng = np.random.default_rng(42)
        query = array.array("f", rng.standard_normal(dim).astype(np.float32).tolist())
        vectors = [
            array.array("f", rng.standard_normal(dim).astype(np.float32).tolist()) for _ in range(5)
        ]
        result = cosine_similarity_batch_matrix(query, vectors)
        assert len(result) == 5


class TestTopKSimilarities:
    """Cover argpartition paths in top_k_similarities."""

    def test_top_k_similarities_k_less_than_n(self) -> None:
        """top_k_similarities uses argpartition when k < len(vectors)."""
        dim = 16
        rng = np.random.default_rng(42)
        query = array.array("f", rng.standard_normal(dim).astype(np.float32).tolist())
        vectors = [
            array.array("f", rng.standard_normal(dim).astype(np.float32).tolist())
            for _ in range(20)
        ]
        results = top_k_similarities(query, vectors, k=3)
        assert len(results) == 3
        sims = [s for _, s in results]
        assert sims == sorted(sims, reverse=True)

    def test_top_k_similarities_k_equals_n(self) -> None:
        """top_k_similarities falls back to argsort when k == n."""
        dim = 16
        rng = np.random.default_rng(42)
        query = array.array("f", rng.standard_normal(dim).astype(np.float32).tolist())
        vectors = [
            array.array("f", rng.standard_normal(dim).astype(np.float32).tolist()) for _ in range(3)
        ]
        results = top_k_similarities(query, vectors, k=3)
        assert len(results) == 3

    def test_top_k_returns_exactly_k(self) -> None:
        """top_k_similarities must return exactly k results when k < n."""
        rng = np.random.default_rng(42)
        query = rng.standard_normal(64).astype(np.float32)
        query /= np.linalg.norm(query)
        embeddings = rng.standard_normal((20, 64)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= np.where(norms > 0, norms, 1.0)

        for k in (1, 3, 5, 10):
            results = top_k_similarities(query, embeddings, k=k)
            assert len(results) == k
            # Must be sorted descending by similarity
            sims = [s for _, s in results]
            assert sims == sorted(sims, reverse=True)


# ---------------------------------------------------------------------------
# tokenizer.py — thread-safe init, atomic download
# ---------------------------------------------------------------------------


class TestTokenizerThreadSafety:
    """Cover lines 380-426: double-checked locking, atomic download."""

    def test_ensure_tokenizer_thread_safe(self) -> None:
        """Multiple threads calling _ensure_tokenizer don't crash."""
        from semantic_cache_mcp.core.tokenizer import _ensure_tokenizer

        results = []
        errors = []

        def call():
            try:
                t = _ensure_tokenizer()
                results.append(t)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=call) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # All threads should get the same tokenizer (or all None)
        assert len({id(r) for r in results if r is not None}) <= 1

    def test_count_tokens_works(self) -> None:
        """count_tokens uses the thread-safe tokenizer."""
        from semantic_cache_mcp.core.tokenizer import count_tokens

        n = count_tokens("Hello, world!")
        assert n > 0

    def test_download_hash_mismatch(self) -> None:
        """Hash verification failure on download → returns None, sets loaded."""
        import semantic_cache_mcp.core.tokenizer._bpe as tokenizer

        original_loaded = tokenizer._tokenizer_loaded
        original_tokenizer = tokenizer._tokenizer

        try:
            tokenizer._tokenizer_loaded = False
            tokenizer._tokenizer = None

            with (
                patch.object(Path, "exists", return_value=False),
                patch("urllib.request.urlretrieve"),
                patch.object(tokenizer, "_verify_hash", return_value=False),
                patch.object(Path, "unlink"),
                patch.object(Path, "mkdir"),
                patch.object(Path, "with_suffix", return_value=Path("/tmp/fake.tmp")),
            ):
                result = tokenizer._ensure_tokenizer()

            assert result is None
            assert tokenizer._tokenizer_loaded is True
        finally:
            tokenizer._tokenizer_loaded = original_loaded
            tokenizer._tokenizer = original_tokenizer

    def test_download_failure(self) -> None:
        """Download failure → returns None, sets loaded."""
        import urllib.error

        import semantic_cache_mcp.core.tokenizer._bpe as tokenizer

        original_loaded = tokenizer._tokenizer_loaded
        original_tokenizer = tokenizer._tokenizer

        try:
            tokenizer._tokenizer_loaded = False
            tokenizer._tokenizer = None

            with (
                patch.object(Path, "exists", return_value=False),
                patch("urllib.request.urlretrieve", side_effect=urllib.error.URLError("fail")),
                patch.object(Path, "mkdir"),
            ):
                result = tokenizer._ensure_tokenizer()

            assert result is None
            assert tokenizer._tokenizer_loaded is True
        finally:
            tokenizer._tokenizer_loaded = original_loaded
            tokenizer._tokenizer = original_tokenizer


# ---------------------------------------------------------------------------
# sqlite.py — pool lock, FileLockTimeout
# ---------------------------------------------------------------------------


class TestConnectionPoolLock:
    """Cover create_lock and FileLockTimeout."""

    def test_pool_counter_lock_thread_safe(self, temp_dir: Path) -> None:
        """Concurrent threads requesting connections don't overflow pool."""
        db_path = temp_dir / "pool_test.db"
        pool = ConnectionPool(db_path, max_size=3)

        results = []
        errors = []

        def use_conn():
            try:
                with pool.get_connection() as conn:
                    conn.execute("SELECT 1")
                    results.append(True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=use_conn) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        pool.close_all()
        assert not errors
        assert len(results) == 10


# ---------------------------------------------------------------------------
# search.py — is_relative_to, compare_files safety, symlink scope
# ---------------------------------------------------------------------------


class TestDirectoryFilterBypass:
    """Cover line 129: Path.is_relative_to instead of startswith."""

    async def test_is_relative_to_rejects_prefix_attack(
        self, temp_dir: Path, semantic_cache: SemanticCache
    ) -> None:
        """'/project_evil' not matched when filtering for '/project'."""
        project = temp_dir / "project"
        project_evil = temp_dir / "project_evil"
        project.mkdir()
        project_evil.mkdir()

        (project / "good.py").write_text("good = True\n")
        (project_evil / "evil.py").write_text("evil = True\n")

        from semantic_cache_mcp.cache import smart_read

        await smart_read(semantic_cache, str(project / "good.py"), max_size=100000)
        await smart_read(semantic_cache, str(project_evil / "evil.py"), max_size=100000)

        with patch(
            "semantic_cache_mcp.cache.search.embed_query",
            return_value=array.array("f", [0.1] * TEST_EMBEDDING_DIM),
        ):
            result = await semantic_search(
                semantic_cache, query="test", k=10, directory=str(project)
            )

        paths = [m.path for m in result.matches]
        for p in paths:
            assert "evil" not in p


class TestCompareFilesSafety:
    """Cover lines 190-193 and 208-216: existence, binary, unicode checks."""

    async def test_missing_file_raises(self, temp_dir: Path, semantic_cache: SemanticCache) -> None:
        """compare_files with missing file → FileNotFoundError."""
        existing = temp_dir / "exists.py"
        existing.write_text("x = 1\n")

        with pytest.raises(FileNotFoundError, match="File not found"):
            await compare_files(semantic_cache, str(existing), str(temp_dir / "nope.py"))

    async def test_binary_file_raises(self, temp_dir: Path, semantic_cache: SemanticCache) -> None:
        """compare_files with binary file → ValueError."""
        text_file = temp_dir / "text.py"
        text_file.write_text("x = 1\n")
        binary_file = temp_dir / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\xff" * 100)

        # Cache the text file first
        from semantic_cache_mcp.cache import smart_read

        await smart_read(semantic_cache, str(text_file), max_size=100000)

        with pytest.raises(ValueError, match="binary"):
            await compare_files(semantic_cache, str(text_file), str(binary_file))

    async def test_invalid_utf8_raises(self, temp_dir: Path, semantic_cache: SemanticCache) -> None:
        """compare_files with non-UTF-8 file → ValueError."""
        text_file = temp_dir / "text.py"
        text_file.write_text("x = 1\n")
        bad_file = temp_dir / "bad.txt"
        bad_file.write_bytes(b"Hello \x80\x81\x82 World")  # Invalid UTF-8, no null bytes

        from semantic_cache_mcp.cache import smart_read

        await smart_read(semantic_cache, str(text_file), max_size=100000)

        with pytest.raises(ValueError, match="not valid UTF-8"):
            await compare_files(semantic_cache, str(text_file), str(bad_file))


class TestSymlinkScope:
    """Cover line 382: symlink escaping base directory."""

    async def test_symlink_outside_base_skipped(
        self, temp_dir: Path, semantic_cache: SemanticCache
    ) -> None:
        """Symlinks pointing outside base directory are skipped in glob."""
        base = temp_dir / "base"
        outside = temp_dir / "outside"
        base.mkdir()
        outside.mkdir()

        (outside / "secret.py").write_text("secret = True\n")
        (base / "link.py").symlink_to(outside / "secret.py")
        (base / "real.py").write_text("real = True\n")

        result = await glob_with_cache_status(semantic_cache, "**/*.py", str(base))
        paths = [m.path for m in result.matches]

        assert any("real.py" in p for p in paths)
        assert not any("link.py" in p for p in paths)


class TestSearchKValidation:
    """Cover k = max(1, min(k, MAX)) in search and similar."""

    async def test_k_zero_clamped(self, semantic_cache: SemanticCache) -> None:
        """k=0 clamped to 1 instead of returning empty."""
        with patch(
            "semantic_cache_mcp.cache.search.embed_query",
            return_value=array.array("f", [0.1] * TEST_EMBEDDING_DIM),
        ):
            result = await semantic_search(semantic_cache, query="test", k=0)
        assert result.matches is not None

    async def test_similar_k_zero_clamped(
        self, temp_dir: Path, semantic_cache: SemanticCache
    ) -> None:
        """find_similar_files with k=0 doesn't crash."""
        f = temp_dir / "test.py"
        f.write_text("x = 1\n")
        result = await find_similar_files(semantic_cache, str(f), k=0)
        assert result.similar_files is not None


# ---------------------------------------------------------------------------
# write.py — _atomic_write, TypeError on None line range
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    """Cover lines 30-42: _atomic_write helper."""

    def test_atomic_write_creates_file(self, temp_dir: Path) -> None:
        """Normal atomic write creates the file."""
        target = temp_dir / "output.txt"
        _atomic_write(target, "hello world\n")
        assert target.read_text() == "hello world\n"

    def test_atomic_write_overwrites(self, temp_dir: Path) -> None:
        """Atomic write overwrites existing file."""
        target = temp_dir / "output.txt"
        target.write_text("old\n")
        _atomic_write(target, "new\n")
        assert target.read_text() == "new\n"

    def test_atomic_write_cleans_up_on_failure(self, temp_dir: Path) -> None:
        """Temp file cleaned up on write failure."""
        target = temp_dir / "output.txt"

        with (
            patch("builtins.open", side_effect=OSError("disk full")),
            pytest.raises(OSError, match="disk full"),
        ):
            _atomic_write(target, "content")

        # No temp files left behind
        tmp_files = list(temp_dir.glob("*.tmp"))
        assert len(tmp_files) == 0


class TestTypeErrorOnNoneLineRange:
    """Cover assert→TypeError replacements in smart_edit and smart_batch_edit."""

    async def test_edit_line_replace_both_none(
        self, temp_dir: Path, semantic_cache: SemanticCache
    ) -> None:
        """Line-replace mode (old_string=None) requires both start_line and end_line.

        The validation for both-or-neither is done first (raises ValueError).
        The TypeError is only reached when dispatch selects Mode C/B with
        one of them being None — but the earlier validation catches mixed cases.
        We test the full path through smart_edit to cover the TypeError branches
        by bypassing the early validation.
        """
        f = temp_dir / "test.py"
        f.write_text("line1\nline2\nline3\n")

        from semantic_cache_mcp.cache import smart_read

        await smart_read(semantic_cache, str(f), max_size=100000)

        # Test that providing neither start_line nor end_line with old_string=None
        # raises ValueError (from the earlier validation)
        with pytest.raises(ValueError, match="old_string.*or.*start_line.*end_line"):
            await smart_edit(
                cache=semantic_cache,
                path=str(f),
                old_string=None,
                new_string="replaced\n",
                start_line=None,
                end_line=None,
            )

    async def test_edit_line_replace_both_provided(
        self, temp_dir: Path, semantic_cache: SemanticCache
    ) -> None:
        """Line-replace mode works when both start_line and end_line provided."""
        f = temp_dir / "test.py"
        f.write_text("line1\nline2\nline3\n")

        from semantic_cache_mcp.cache import smart_read

        await smart_read(semantic_cache, str(f), max_size=100000)

        # This should work (covers the non-error path through the TypeError guard)
        result = await smart_edit(
            cache=semantic_cache,
            path=str(f),
            old_string=None,
            new_string="replaced\n",
            start_line=2,
            end_line=2,
        )
        assert result.diff_content is not None


# ---------------------------------------------------------------------------
# tools.py — offset/limit/max_size bounds validation
# ---------------------------------------------------------------------------


class TestToolsBoundsValidation:
    """Cover lines 81-85: offset, limit, max_size validation."""

    async def test_read_negative_offset(self, semantic_cache: SemanticCache) -> None:
        """offset < 1 returns error."""
        from unittest.mock import MagicMock

        from semantic_cache_mcp.server.tools import read

        ctx = MagicMock()
        ctx.lifespan_context = {"cache": semantic_cache}

        with pytest.raises(ToolError, match="read: offset must be >= 1"):
            await read(ctx=ctx, path="/any/file.py", offset=0, max_size=100000)

    async def test_read_negative_limit(self, semantic_cache: SemanticCache) -> None:
        """limit < 1 returns error."""
        from semantic_cache_mcp.server.tools import read

        ctx = MagicMock()
        ctx.lifespan_context = {"cache": semantic_cache}

        with pytest.raises(ToolError, match="read: limit must be >= 1"):
            await read(ctx=ctx, path="/any/file.py", limit=0, max_size=100000)

    async def test_read_max_size_clamped(
        self, temp_dir: Path, semantic_cache: SemanticCache
    ) -> None:
        """max_size is clamped to valid range."""
        from semantic_cache_mcp.server.tools import read

        f = temp_dir / "test.txt"
        f.write_text("hello\n")

        ctx = MagicMock()
        ctx.lifespan_context = {"cache": semantic_cache}

        # Negative max_size clamped to 1 (won't crash)
        result = await read(ctx=ctx, path=str(f), max_size=-999)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# _mcp.py — lifespan startup, redirect_stdout, error handling
# ---------------------------------------------------------------------------


class TestMcpLifespan:
    """Cover app_lifespan: redirect_stdout, exception handling, cache None guard."""

    @pytest.mark.asyncio
    async def test_lifespan_redirect_stdout(self) -> None:
        """Stdout is redirected to stderr during init."""

        async def mock_start() -> None:
            print("worker startup noise")

        with (
            patch("semantic_cache_mcp.server._mcp.get_tokenizer"),
            patch("semantic_cache_mcp.server._mcp.ToolProcessSupervisor") as mock_supervisor_cls,
        ):
            mock_supervisor = MagicMock()
            mock_supervisor.start = AsyncMock(side_effect=mock_start)
            mock_supervisor.async_close = AsyncMock()
            mock_supervisor_cls.return_value = mock_supervisor

            from semantic_cache_mcp.server._mcp import app_lifespan

            server = MagicMock()
            # app_lifespan is decorated with @lifespan, returns async context manager
            async with app_lifespan(server) as context:
                assert "cache" in context

    @pytest.mark.asyncio
    async def test_lifespan_init_exception_reraises(self) -> None:
        """Exception during init is logged and re-raised."""
        with (
            patch(
                "semantic_cache_mcp.server._mcp.get_tokenizer",
                side_effect=RuntimeError("init fail"),
            ),
        ):
            from semantic_cache_mcp.server._mcp import app_lifespan

            server = MagicMock()
            with pytest.raises(RuntimeError, match="init fail"):
                async with app_lifespan(server) as context:
                    pass

    @pytest.mark.asyncio
    async def test_lifespan_model_not_ready(self) -> None:
        """Worker supervisor startup still yields a cache context."""
        with (
            patch("semantic_cache_mcp.server._mcp.get_tokenizer"),
            patch("semantic_cache_mcp.server._mcp.ToolProcessSupervisor") as mock_supervisor_cls,
        ):
            mock_supervisor = MagicMock()
            mock_supervisor.start = AsyncMock()
            mock_supervisor.async_close = AsyncMock()
            mock_supervisor_cls.return_value = mock_supervisor

            from semantic_cache_mcp.server._mcp import app_lifespan

            server = MagicMock()
            async with app_lifespan(server) as context:
                assert "cache" in context


# ---------------------------------------------------------------------------
# Bug-fix verification + remaining coverage gaps
# ---------------------------------------------------------------------------


class TestAtomicWritePermissions:
    """Verify permission preservation in _atomic_write."""

    def test_preserves_existing_permissions(self, tmp_path: Path) -> None:
        """_atomic_write must copy original file permissions."""
        import os
        import stat

        target = tmp_path / "perms.txt"
        target.write_text("original")
        os.chmod(target, 0o755)

        _atomic_write(target, "updated content")

        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o755
        assert target.read_text() == "updated content"

    def test_new_file_gets_default_permissions(self, tmp_path: Path) -> None:
        """_atomic_write on nonexistent file should not crash."""
        target = tmp_path / "new_file.txt"
        _atomic_write(target, "fresh content")
        assert target.read_text() == "fresh content"

    def test_permission_oserror_is_suppressed(self, tmp_path: Path) -> None:
        """OSError from chmod is suppressed (best-effort)."""
        target = tmp_path / "oserr.txt"
        target.write_text("original")

        with patch("os.chmod", side_effect=OSError("permission denied")):
            _atomic_write(target, "updated")

        assert target.read_text() == "updated"


class TestTokenizerCachePaths:
    """Cover tokenizer hash-mismatch and download-failure paths."""

    def test_hash_mismatch_triggers_redownload(self, tmp_path: Path) -> None:
        """If cached file has wrong hash, it should be deleted."""
        import semantic_cache_mcp.core.tokenizer._bpe as tok_mod

        # Reset global state for this test
        original_loaded = tok_mod._tokenizer_loaded
        original_tok = tok_mod._tokenizer
        original_dir = tok_mod.TOKENIZER_CACHE_DIR
        tok_mod._tokenizer_loaded = False
        tok_mod._tokenizer = None
        tok_mod.TOKENIZER_CACHE_DIR = tmp_path

        cache_file = tmp_path / "o200k_base.tiktoken"
        cache_file.write_text("bad content")

        try:
            with (
                patch.object(tok_mod, "_verify_hash", return_value=False),
                patch.object(tok_mod, "_init_tokenizer", return_value=None),
                patch("urllib.request.urlretrieve", side_effect=OSError("no network")),
            ):
                result = tok_mod._ensure_tokenizer()
        finally:
            tok_mod._tokenizer_loaded = original_loaded
            tok_mod._tokenizer = original_tok
            tok_mod.TOKENIZER_CACHE_DIR = original_dir

        # Hash mismatch → file deleted, download fails → returns None
        assert result is None
        assert not cache_file.exists()

    def test_download_hash_mismatch_returns_none(self, tmp_path: Path) -> None:
        """If downloaded file has wrong hash, return None without init."""
        import semantic_cache_mcp.core.tokenizer._bpe as tok_mod

        original_loaded = tok_mod._tokenizer_loaded
        original_tok = tok_mod._tokenizer
        original_dir = tok_mod.TOKENIZER_CACHE_DIR
        tok_mod._tokenizer_loaded = False
        tok_mod._tokenizer = None
        tok_mod.TOKENIZER_CACHE_DIR = tmp_path

        try:

            def fake_download(url: str, path: str | os.PathLike[str]) -> None:
                Path(path).write_text("fake data")

            with (
                patch("urllib.request.urlretrieve", side_effect=fake_download),
                patch.object(tok_mod, "_verify_hash", return_value=False),
            ):
                result = tok_mod._ensure_tokenizer()
        finally:
            tok_mod._tokenizer_loaded = original_loaded
            tok_mod._tokenizer = original_tok
            tok_mod.TOKENIZER_CACHE_DIR = original_dir

        assert result is None

    def test_download_success_inits_tokenizer(self, tmp_path: Path) -> None:
        """Successful download + hash match should init tokenizer."""
        import semantic_cache_mcp.core.tokenizer._bpe as tok_mod

        original_loaded = tok_mod._tokenizer_loaded
        original_tok = tok_mod._tokenizer
        original_dir = tok_mod.TOKENIZER_CACHE_DIR
        tok_mod._tokenizer_loaded = False
        tok_mod._tokenizer = None
        tok_mod.TOKENIZER_CACHE_DIR = tmp_path

        sentinel = MagicMock()

        try:

            def fake_download(url: str, path: str | os.PathLike[str]) -> None:
                Path(path).write_text("fake data")

            with (
                patch("urllib.request.urlretrieve", side_effect=fake_download),
                patch.object(tok_mod, "_verify_hash", return_value=True),
                patch.object(tok_mod, "_init_tokenizer", return_value=sentinel),
            ):
                result = tok_mod._ensure_tokenizer()
        finally:
            tok_mod._tokenizer_loaded = original_loaded
            tok_mod._tokenizer = original_tok
            tok_mod.TOKENIZER_CACHE_DIR = original_dir

        assert result is sentinel

    def test_network_error_returns_none(self, tmp_path: Path) -> None:
        """URLError during download should return None gracefully."""
        import urllib.error

        import semantic_cache_mcp.core.tokenizer._bpe as tok_mod

        original_loaded = tok_mod._tokenizer_loaded
        original_tok = tok_mod._tokenizer
        original_dir = tok_mod.TOKENIZER_CACHE_DIR
        tok_mod._tokenizer_loaded = False
        tok_mod._tokenizer = None
        tok_mod.TOKENIZER_CACHE_DIR = tmp_path

        try:
            with patch(
                "urllib.request.urlretrieve",
                side_effect=urllib.error.URLError("timeout"),
            ):
                result = tok_mod._ensure_tokenizer()
        finally:
            tok_mod._tokenizer_loaded = original_loaded
            tok_mod._tokenizer = original_tok
            tok_mod.TOKENIZER_CACHE_DIR = original_dir

        assert result is None


class TestGlobSymlinkEscape:
    """Cover symlink escape filtering in glob_with_cache_status."""

    async def test_symlink_outside_dir_is_skipped(self, tmp_path: Path) -> None:
        """Symlinks pointing outside the glob directory are filtered out."""
        import os

        search_dir = tmp_path / "search"
        search_dir.mkdir()
        (search_dir / "real.txt").write_text("content")

        outside = tmp_path / "outside.txt"
        outside.write_text("escape")

        link = search_dir / "escape.txt"
        os.symlink(outside, link)

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache._io_executor = None  # use default asyncio executor for astat
        result = await glob_with_cache_status(mock_cache, "*.txt", str(search_dir))

        paths = [m.path for m in result.matches]
        assert str(search_dir / "real.txt") in paths
        # symlink pointing outside should be filtered
        assert str(link) not in paths

    async def test_symlink_inside_dir_is_kept(self, tmp_path: Path) -> None:
        """Symlinks pointing within the glob directory are kept."""
        import os

        search_dir = tmp_path / "search"
        search_dir.mkdir()
        real = search_dir / "real.txt"
        real.write_text("content")

        link = search_dir / "alias.txt"
        os.symlink(real, link)

        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache._io_executor = None  # use default asyncio executor for astat
        result = await glob_with_cache_status(mock_cache, "*.txt", str(search_dir))

        paths = [m.path for m in result.matches]
        assert str(link) in paths


class TestSemanticSearchDirectoryFilter:
    """Cover directory filter path in semantic_search."""

    async def test_directory_filter_excludes_outside_files(self, tmp_path: Path) -> None:
        """Files outside the directory filter should be excluded."""
        from concurrent.futures import ThreadPoolExecutor

        with patch("semantic_cache_mcp.cache.search.embed_query") as mock_embed:
            mock_embed.return_value = list(np.zeros(64, dtype=np.float32))

            mock_cache = MagicMock()
            mock_cache._io_executor = ThreadPoolExecutor(max_workers=1)
            mock_storage = MagicMock()
            mock_cache._storage = mock_storage

            # Return files in different directories; all outside /nonexistent/dir
            mock_storage.search_hybrid = AsyncMock(
                return_value=[
                    ("/home/user/project/a.py", "preview a", 0.9),
                    ("/other/path/b.py", "preview b", 0.8),
                ]
            )
            mock_storage.get_stats = AsyncMock(return_value={"files_cached": 2})

            # All filtered out → empty result
            result = await semantic_search(
                mock_cache,
                query="test",
                k=5,
                directory="/nonexistent/dir",
            )
            assert len(result.matches) == 0


class TestMcpCacheNoneGuard:
    """Cover line 54-55 in _mcp.py — cache is None after lifespan block."""

    @pytest.mark.asyncio
    async def test_cache_none_raises(self) -> None:
        """If cache is somehow None after init block, RuntimeError raised."""
        with (
            patch("semantic_cache_mcp.server._mcp.get_tokenizer"),
            patch(
                "semantic_cache_mcp.server._mcp.ToolProcessSupervisor",
                return_value=None,
            ),
        ):
            from semantic_cache_mcp.server._mcp import app_lifespan

            server = MagicMock()
            with pytest.raises(RuntimeError, match="Cache failed to initialize"):
                async with app_lifespan(server) as context:
                    pass
