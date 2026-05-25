"""Tests for the `read_image` pass-through tool."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastmcp import Client

from semantic_cache_mcp.cache.read import _sniff_image_mime
from semantic_cache_mcp.server._mcp import mcp

# Smallest valid 1x1 PNG — used as an inline test fixture so we don't
# need to depend on Pillow just to create one.
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c620001000005000101dab2880000000049454e44ae426082"
)

# Magic-byte headers padded to a plausible size. read_image sniffs the
# leading bytes and never decodes pixels, so a structurally valid header is
# enough to exercise format detection. A real BMP carries a DIB header size
# at bytes 14-17 — 'BM' alone is too weak to trust.
_BMP_HEADER = (
    b"BM"
    + (122).to_bytes(4, "little")  # file size
    + b"\x00\x00\x00\x00"  # reserved
    + (54).to_bytes(4, "little")  # pixel data offset
    + (40).to_bytes(4, "little")  # DIB header size = BITMAPINFOHEADER
    + b"\x00" * 104
)
_TIFF_LE_HEADER = b"II*\x00" + b"\x00" * 124
_TIFF_BE_HEADER = b"MM\x00*" + b"\x00" * 124
_JPEG_HEADER = b"\xff\xd8\xff\xe0" + b"\x00" * 124
_GIF_HEADER = b"GIF89a" + b"\x00" * 122
_WEBP_HEADER = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 116


@pytest.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


async def _expect_error(client, args: dict, match: str) -> None:
    from fastmcp.exceptions import ClientError, ToolError

    try:
        await client.call_tool("read_image", args)
    except (ToolError, ClientError) as e:
        assert match in str(e), f"expected {match!r} in {e!s}"
        return
    except Exception as e:
        assert match in str(e), f"unexpected error: {type(e).__name__}: {e}"
        return
    pytest.fail(f"expected error containing {match!r}; tool succeeded")


def test_sniff_rejects_non_bmp_starting_with_bm() -> None:
    """'BM' is only a 2-byte signature; a non-BMP binary that merely begins
    with 0x42 0x4D must not be mis-detected as image/bmp."""
    fake = b"BM" + b"not a bitmap, just arbitrary bytes\x00\x01\x02\x03\x04\x05"
    assert _sniff_image_mime(fake) is None


def test_sniff_accepts_structurally_valid_bmp() -> None:
    """A BMP with a valid DIB header size is detected on content alone."""
    assert _sniff_image_mime(_BMP_HEADER) == "image/bmp"


async def test_read_image_returns_image_content_block(mcp_client, tmp_path: Path) -> None:
    """Happy path: PNG bytes are returned as an MCP image content block."""
    png = tmp_path / "tiny.png"
    png.write_bytes(_TINY_PNG)

    result = await mcp_client.call_tool("read_image", {"path": str(png)})

    # FastMCP's Client unpacks the content list onto result.content.
    blocks = list(result.content)
    image_blocks = [b for b in blocks if getattr(b, "type", None) == "image"]
    assert len(image_blocks) == 1, f"expected one image block, got {blocks!r}"
    img = image_blocks[0]
    assert img.mimeType == "image/png"
    assert base64.b64decode(img.data) == _TINY_PNG

    # Structured metadata sidecar.
    meta = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    assert meta is not None
    assert meta["mime"] == "image/png"
    assert meta["size"] == len(_TINY_PNG)
    assert meta["tool"] == "read_image"


async def test_read_image_rejects_non_image(mcp_client, tmp_path: Path) -> None:
    """Text and arbitrary binaries are refused — `read` owns those."""
    txt = tmp_path / "notes.txt"
    txt.write_text("just some text\n")
    await _expect_error(mcp_client, {"path": str(txt)}, "not a recognized image")


async def test_read_image_rejects_text_with_image_extension(mcp_client, tmp_path: Path) -> None:
    """Extension is not trusted: text saved as `.png` is refused on content."""
    fake = tmp_path / "fake.png"
    fake.write_text("this is plain text, not a PNG\n")
    await _expect_error(mcp_client, {"path": str(fake)}, "not a recognized image")


@pytest.mark.parametrize(
    ("header", "ext", "expected_mime"),
    [
        (_BMP_HEADER, ".bmp", "image/bmp"),
        (_TIFF_LE_HEADER, ".tiff", "image/tiff"),
        (_TIFF_BE_HEADER, ".tiff", "image/tiff"),
        (_JPEG_HEADER, ".jpg", "image/jpeg"),
        (_GIF_HEADER, ".gif", "image/gif"),
        (_WEBP_HEADER, ".webp", "image/webp"),
    ],
)
async def test_read_image_accepts_formats_by_magic(
    mcp_client, tmp_path: Path, header: bytes, ext: str, expected_mime: str
) -> None:
    """BMP/TIFF/JPEG are detected by magic bytes, including BMP/TIFF that
    the magic table previously had no entry for."""
    f = tmp_path / f"img{ext}"
    f.write_bytes(header)
    result = await mcp_client.call_tool("read_image", {"path": str(f)})
    image_blocks = [b for b in result.content if getattr(b, "type", None) == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0].mimeType == expected_mime


async def test_read_image_accepts_image_with_wrong_extension(mcp_client, tmp_path: Path) -> None:
    """A real image with a non-image extension is still accepted on content."""
    f = tmp_path / "screenshot.dat"
    f.write_bytes(_TINY_PNG)
    result = await mcp_client.call_tool("read_image", {"path": str(f)})
    image_blocks = [b for b in result.content if getattr(b, "type", None) == "image"]
    assert len(image_blocks) == 1
    assert image_blocks[0].mimeType == "image/png"


async def test_read_image_missing_file(mcp_client, tmp_path: Path) -> None:
    missing = tmp_path / "nope.png"
    await _expect_error(mcp_client, {"path": str(missing)}, "File not found")


async def test_read_image_oversize_rejected(
    mcp_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """File over the cap is refused before bytes are read into memory."""
    # Shrink the cap to a value smaller than our fixture so the test stays
    # cheap and deterministic.
    from semantic_cache_mcp.server import tools as tools_mod

    monkeypatch.setattr(tools_mod, "_MAX_IMAGE_BYTES", 10)

    png = tmp_path / "big.png"
    png.write_bytes(_TINY_PNG)  # ~70 bytes — well over the 10-byte cap
    await _expect_error(mcp_client, {"path": str(png)}, "image too large")


async def test_read_image_rejects_directory(mcp_client, tmp_path: Path) -> None:
    """A directory path hits the S_ISREG guard, not a confusing I/O error."""
    d = tmp_path / "a_dir"
    d.mkdir()
    await _expect_error(mcp_client, {"path": str(d)}, "Not a regular file")


def test_parse_max_image_bytes_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from semantic_cache_mcp.server import tools as tools_mod

    monkeypatch.delenv("SCMCP_MAX_IMAGE_BYTES", raising=False)
    assert tools_mod._parse_max_image_bytes() == tools_mod._DEFAULT_MAX_IMAGE_BYTES


def test_parse_max_image_bytes_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from semantic_cache_mcp.server import tools as tools_mod

    monkeypatch.setenv("SCMCP_MAX_IMAGE_BYTES", "1048576")
    assert tools_mod._parse_max_image_bytes() == 1048576


def test_parse_max_image_bytes_bad_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-integer override falls back to the default, doesn't crash."""
    from semantic_cache_mcp.server import tools as tools_mod

    monkeypatch.setenv("SCMCP_MAX_IMAGE_BYTES", "not-a-number")
    assert tools_mod._parse_max_image_bytes() == tools_mod._DEFAULT_MAX_IMAGE_BYTES


def test_parse_max_image_bytes_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    """An absurdly small override is clamped up to the 1 KiB floor."""
    from semantic_cache_mcp.server import tools as tools_mod

    monkeypatch.setenv("SCMCP_MAX_IMAGE_BYTES", "1")
    assert tools_mod._parse_max_image_bytes() == 1024


async def test_read_image_rejects_oversized_encoded_payload(
    mcp_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file under the raw cap whose base64 expansion would blow the wire
    cap must be refused before encoding — surfaces as a tool error, not an
    opaque upstream 400.
    """
    from semantic_cache_mcp.server import tools as tools_mod

    png = tmp_path / "img.png"
    # 600 raw bytes encode to 800 base64 bytes (4 * ceil(600/3)). Setting
    # the encoded cap to 700 catches it; the raw cap stays high so this test
    # exercises only the encoded guard.
    png.write_bytes(_TINY_PNG + b"\x00" * (600 - len(_TINY_PNG)))
    monkeypatch.setattr(tools_mod, "_MAX_IMAGE_BYTES", 10_000)
    monkeypatch.setattr(tools_mod, "_MAX_ENCODED_IMAGE_BYTES", 700)

    await _expect_error(mcp_client, {"path": str(png)}, "image too large after base64")


def test_parse_max_encoded_image_bytes_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from semantic_cache_mcp.server import tools as tools_mod

    monkeypatch.delenv("SCMCP_MAX_ENCODED_IMAGE_BYTES", raising=False)
    assert tools_mod._parse_max_encoded_image_bytes() == tools_mod._DEFAULT_MAX_ENCODED_IMAGE_BYTES


def test_parse_max_encoded_image_bytes_bad_value(monkeypatch: pytest.MonkeyPatch) -> None:
    from semantic_cache_mcp.server import tools as tools_mod

    monkeypatch.setenv("SCMCP_MAX_ENCODED_IMAGE_BYTES", "not-a-number")
    assert tools_mod._parse_max_encoded_image_bytes() == tools_mod._DEFAULT_MAX_ENCODED_IMAGE_BYTES


def test_predicted_base64_len_matches_actual() -> None:
    """The predicted length must equal len(base64.b64encode(...)) for every
    residue class mod 3 — otherwise the encoded-size guard is wrong.
    """
    from semantic_cache_mcp.server import tools as tools_mod

    for n in (0, 1, 2, 3, 4, 5, 6, 100, 999, 1000, 1001):
        expected = len(base64.b64encode(b"\x00" * n))
        assert tools_mod._predicted_base64_len(n) == expected, n


async def test_read_image_rejects_oversized_after_read(
    mcp_client, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-read st_size gate races a file that grows between the stat and
    the read. read_image must re-check the bytes it actually read and reject
    an oversized image before base64-encoding it into the response.
    """
    from semantic_cache_mcp.server import tools as tools_mod

    png = tmp_path / "small.png"
    png.write_bytes(_TINY_PNG)  # tiny on disk — passes the pre-read st_size gate

    async def _grown_read(_path, _executor):
        # Simulate the file having grown after the stat: return far more
        # bytes than the cap even though the on-disk file is tiny.
        return _TINY_PNG + b"\x00" * 4096

    monkeypatch.setattr(tools_mod, "_MAX_IMAGE_BYTES", 1024)
    monkeypatch.setattr(tools_mod, "aread_bytes", _grown_read)
    await _expect_error(mcp_client, {"path": str(png)}, "image too large")
