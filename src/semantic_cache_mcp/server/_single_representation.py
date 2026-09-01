"""Middleware: send each tool result exactly once.

MCP lets a tool answer with a text block and a `structuredContent` object at
the same time, and FastMCP does exactly that for any handler returning a dict.
The two carry identical JSON, so a client that forwards both charges the model
twice for every file this server delivers — measured at 8.8k + 8.9k tokens for
one 584-line file. Clients disagree about which they forward, so the safe
shape is one representation.

The text block is the one kept: every client renders content blocks, and this
server's payloads are already compact JSON. `structuredContent` is restored by
setting `SCMCP_STRUCTURED_CONTENT=1`, which is also forced on whenever output
schemas are published (a declared schema obliges the server to return
structured content).
"""

from __future__ import annotations

import logging

from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.tools import ToolResult

from ..config import STRUCTURED_CONTENT

logger = logging.getLogger(__name__)


class SingleRepresentationMiddleware(Middleware):
    """Drop the duplicated `structuredContent` from outgoing tool results."""

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        result = await call_next(context)

        if STRUCTURED_CONTENT:
            return result

        # A ToolResult subclass carries protocol meaning of its own
        # (`InputRequiredToolResult` is a multi-round-trip ask, not an answer),
        # so only the plain shape is rewritten.
        if type(result) is not ToolResult:
            return result

        if result.structured_content is None:
            return result

        if not result.content:
            # Structured content is the only thing carrying the answer here.
            # Dropping it would return an empty result — a silent data loss
            # far more expensive than the duplication this exists to remove.
            logger.debug(
                "keeping structured content for %s: no content blocks to fall back on",
                getattr(context.message, "name", "?"),
            )
            return result

        return ToolResult(
            content=result.content,
            structured_content=None,
            meta=result.meta,
            is_error=result.is_error,
        )
