"""Response rendering helpers for the MCP server."""

from __future__ import annotations

import json
from typing import Any

from ..config import TOOL_MAX_RESPONSE_TOKENS, TOOL_OUTPUT_MODE
from ..core import count_tokens

_MODE_NORMAL = {"normal", "debug"}
_MODE_DEBUG = "debug"


def _response_mode() -> str:
    """Global response mode from environment-backed config."""
    return TOOL_OUTPUT_MODE


def _response_token_cap() -> int | None:
    """Global response token cap from environment-backed config."""
    return TOOL_MAX_RESPONSE_TOKENS if TOOL_MAX_RESPONSE_TOKENS > 0 else None


def _minimal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build minimal JSON payload when response exceeds token budget."""
    keep_order = (
        "ok",
        "tool",
        "status",
        "path",
        "path1",
        "path2",
        "summary",
        "skipped",
        "files_read",
        "files_skipped",
        "succeeded",
        "failed",
        "message",
        "error",
    )
    minimal: dict[str, Any] = {}
    for key in keep_order:
        if key in payload:
            minimal[key] = payload[key]
    minimal["truncated"] = True
    if "message" not in minimal:
        minimal["message"] = "Response truncated by max_response_tokens"
    return minimal


def _render_response(payload: dict[str, Any], max_response_tokens: int | None) -> str:
    """Render tool response as compact JSON with optional token cap."""
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    if max_response_tokens is not None and max_response_tokens > 0:
        if count_tokens(body) > max_response_tokens:
            body = json.dumps(_minimal_payload(payload), separators=(",", ":"), ensure_ascii=False)
        if count_tokens(body) > max_response_tokens:
            body = json.dumps({"ok": False, "truncated": True}, separators=(",", ":"))
    return body


def _render_error(tool: str, message: str, max_response_tokens: int | None) -> str:
    """Render consistent error responses."""
    payload = {"ok": False, "tool": tool, "error": message}
    return _render_response(payload, max_response_tokens)
