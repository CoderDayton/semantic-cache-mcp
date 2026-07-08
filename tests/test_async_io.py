"""Real-file (unmocked) coverage for the async I/O wrappers in utils._async_io."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from semantic_cache_mcp.utils._async_io import (
    aread_bytes,
    aread_text,
    astat,
    aunlink,
    awrite_atomic,
)


class TestAreadBytes:
    async def test_reads_exact_bytes(self, tmp_path: Path) -> None:
        f = tmp_path / "data.bin"
        payload = b"\x00\x01binary\xff"
        f.write_bytes(payload)
        assert await aread_bytes(f) == payload

    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await aread_bytes(tmp_path / "absent.bin")


class TestAreadText:
    async def test_reads_utf8(self, tmp_path: Path) -> None:
        f = tmp_path / "text.txt"
        f.write_text("héllo wörld\n", encoding="utf-8")
        assert await aread_text(f) == "héllo wörld\n"

    async def test_strict_raises_on_invalid_utf8(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.txt"
        f.write_bytes(b"ok \x80\x81 nope\n")
        with pytest.raises(UnicodeDecodeError):
            await aread_text(f)

    async def test_replace_degrades_gracefully(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.txt"
        f.write_bytes(b"ok \x80 end\n")
        text = await aread_text(f, errors="replace")
        assert "ok" in text and "end" in text
        assert "�" in text


class TestAstat:
    async def test_matches_os_stat(self, tmp_path: Path) -> None:
        f = tmp_path / "stat_me.txt"
        f.write_text("12345")
        st = await astat(f)
        assert st.st_size == 5
        assert st.st_mtime == os.stat(f).st_mtime

    async def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await astat(tmp_path / "absent")


class TestAwriteAtomic:
    async def test_creates_new_file(self, tmp_path: Path) -> None:
        f = tmp_path / "new.txt"
        await awrite_atomic(f, "content\n")
        assert f.read_text(encoding="utf-8") == "content\n"

    async def test_overwrites_completely(self, tmp_path: Path) -> None:
        f = tmp_path / "existing.txt"
        f.write_text("old content that is longer\n")
        await awrite_atomic(f, "new\n")
        assert f.read_text(encoding="utf-8") == "new\n"

    async def test_leaves_no_temp_files(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.txt"
        await awrite_atomic(f, "x\n")
        assert [p.name for p in tmp_path.iterdir()] == ["clean.txt"]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
    async def test_preserves_permissions(self, tmp_path: Path) -> None:
        f = tmp_path / "perms.txt"
        f.write_text("old\n")
        os.chmod(f, 0o600)
        await awrite_atomic(f, "new\n")
        assert stat.S_IMODE(os.stat(f).st_mode) == 0o600

    async def test_write_failure_cleans_up_temp(self, tmp_path: Path) -> None:
        target = tmp_path / "no_dir" / "file.txt"
        with pytest.raises(OSError):
            await awrite_atomic(target, "x\n")
        assert list(tmp_path.iterdir()) == []


class TestAunlink:
    async def test_removes_file(self, tmp_path: Path) -> None:
        f = tmp_path / "doomed.txt"
        f.write_text("x")
        await aunlink(f)
        assert not f.exists()

    async def test_missing_raises_by_default(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            await aunlink(tmp_path / "absent")

    async def test_missing_ok_suppresses(self, tmp_path: Path) -> None:
        await aunlink(tmp_path / "absent", missing_ok=True)
