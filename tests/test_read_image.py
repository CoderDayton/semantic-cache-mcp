"""Tests for the `read_image` pass-through tool."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastmcp import Client

from semantic_cache_mcp.server._mcp import mcp

# Smallest valid 1x1 PNG — used as an inline test fixture so we don't
# need to depend on Pillow just to create one.
_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c620001000005000101dab2880000000049454e44ae426082"
)


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
    await _expect_error(mcp_client, {"path": str(txt)}, "not an image")


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
