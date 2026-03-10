"""MCP server entry point."""

from __future__ import annotations

from . import tools  # noqa: F401 — side-effect: registers @mcp.tool() decorators
from ._mcp import mcp
from .tools import _expand_globs  # re-export for test compatibility

__all__ = ["mcp", "main", "_expand_globs"]


def main() -> None:
    _setup_signal_handlers()
    mcp.run()


def _setup_signal_handlers() -> None:
    """Register signal handlers for graceful shutdown.

    SIGPIPE is critical for stdio transport: without it, writing to a closed
    pipe (client disconnect) kills the process before lifespan cleanup runs.
    This is the most likely cause of "1 MCP server failed before closing."
    """
    import signal  # noqa: PLC0415

    # SIGPIPE: ignore so writes to closed pipe raise BrokenPipeError instead
    # of killing the process. asyncio and FastMCP handle BrokenPipeError
    # gracefully, triggering lifespan cleanup.
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
