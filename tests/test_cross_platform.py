"""Cross-platform hardening tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from semantic_cache_mcp.cache._helpers import _is_binary_content
from semantic_cache_mcp.cache.store import _get_rss_mb
from semantic_cache_mcp.config import _get_cache_dir

# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------


class TestGetCacheDir:
    def test_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("SEMANTIC_CACHE_DIR", str(tmp_path / "custom"))
        result = _get_cache_dir()
        assert result == tmp_path / "custom"

    def test_linux_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SEMANTIC_CACHE_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        with patch.object(sys, "platform", "linux"):
            result = _get_cache_dir()
        assert result == Path.home() / ".cache" / "semantic-cache-mcp"

    def test_linux_xdg(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("SEMANTIC_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
        with patch.object(sys, "platform", "linux"):
            result = _get_cache_dir()
        assert result == tmp_path / "xdg" / "semantic-cache-mcp"

    def test_macos_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SEMANTIC_CACHE_DIR", raising=False)
        with patch.object(sys, "platform", "darwin"):
            result = _get_cache_dir()
        assert result == Path.home() / "Library" / "Caches" / "semantic-cache-mcp"

    def test_windows_localappdata(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.delenv("SEMANTIC_CACHE_DIR", raising=False)
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
        with patch.object(sys, "platform", "win32"):
            result = _get_cache_dir()
        assert result == tmp_path / "Local" / "semantic-cache-mcp"

    def test_windows_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SEMANTIC_CACHE_DIR", raising=False)
        monkeypatch.delenv("LOCALAPPDATA", raising=False)
        with patch.object(sys, "platform", "win32"):
            result = _get_cache_dir()
        assert result == Path.home() / "AppData" / "Local" / "semantic-cache-mcp"


# ---------------------------------------------------------------------------
# RSS memory
# ---------------------------------------------------------------------------


class TestGetRssMb:
    def test_returns_float_or_none(self) -> None:
        result = _get_rss_mb()
        assert result is None or isinstance(result, float)

    def test_never_raises(self) -> None:
        # Patch sys.platform to something unknown — should return None, not raise
        with patch.object(sys, "platform", "freebsd"):
            result = _get_rss_mb()
        assert result is None

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux-only")
    def test_linux_positive(self) -> None:
        result = _get_rss_mb()
        assert result is not None
        assert result > 0


# ---------------------------------------------------------------------------
# BOM-aware binary detection
# ---------------------------------------------------------------------------


class TestBinaryDetection:
    def test_utf16le_not_binary(self) -> None:
        data = b"\xff\xfe" + "Hello World".encode("utf-16-le")
        assert _is_binary_content(data) is False

    def test_utf16be_not_binary(self) -> None:
        data = b"\xfe\xff" + "Hello World".encode("utf-16-be")
        assert _is_binary_content(data) is False

    def test_utf32le_not_binary(self) -> None:
        data = b"\xff\xfe\x00\x00" + "Hi".encode("utf-32-le")
        assert _is_binary_content(data) is False

    def test_utf32be_not_binary(self) -> None:
        data = b"\x00\x00\xfe\xff" + "Hi".encode("utf-32-be")
        assert _is_binary_content(data) is False

    def test_utf8_bom_not_binary(self) -> None:
        data = b"\xef\xbb\xbf" + b"Hello World"
        assert _is_binary_content(data) is False

    def test_png_still_binary(self) -> None:
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert _is_binary_content(data) is True

    def test_elf_still_binary(self) -> None:
        data = b"\x7fELF" + b"\x00" * 100
        assert _is_binary_content(data) is True

    def test_null_heavy_still_binary(self) -> None:
        data = b"\x00" * 1000
        assert _is_binary_content(data) is True

    def test_plain_text_not_binary(self) -> None:
        data = b"Hello, this is plain ASCII text.\n"
        assert _is_binary_content(data) is False

    def test_empty_not_binary(self) -> None:
        assert _is_binary_content(b"") is False


# ---------------------------------------------------------------------------
# Tokenizer uses config cache dir
# ---------------------------------------------------------------------------


class TestTokenizerCacheDir:
    def test_uses_config_cache_dir(self) -> None:
        from semantic_cache_mcp.config import CACHE_DIR
        from semantic_cache_mcp.core.tokenizer import TOKENIZER_CACHE_DIR

        assert TOKENIZER_CACHE_DIR == CACHE_DIR / "tokenizer"
