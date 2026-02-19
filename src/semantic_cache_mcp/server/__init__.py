"""Server package - MCP server entry point."""

from __future__ import annotations

from . import tools  # noqa: F401 — side-effect: registers @mcp.tool() decorators
from ._mcp import mcp
from .tools import _expand_globs  # re-export for test compatibility

__all__ = ["mcp", "main", "_expand_globs"]


def main() -> None:
    """Run the MCP server."""
    mcp.run()
