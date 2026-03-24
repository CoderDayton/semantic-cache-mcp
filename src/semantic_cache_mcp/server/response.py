"""Response rendering helpers."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from ..config import TOOL_MAX_RESPONSE_TOKENS, TOOL_OUTPUT_MODE
from ..core import count_tokens

_MODE_NORMAL = {"normal", "debug"}
_MODE_DEBUG = "debug"
_UNSET = object()
_response_mode_override: ContextVar[str | None] = ContextVar("response_mode_override", default=None)
_response_token_cap_override: ContextVar[int | None | object] = ContextVar(
    "response_token_cap_override",
    default=_UNSET,
)


def _response_mode() -> str:
    override = _response_mode_override.get()
    return override if override is not None else TOOL_OUTPUT_MODE


def _response_token_cap() -> int | None:
    override = _response_token_cap_override.get()
    if override is not _UNSET:
        return override  # type: ignore[return-value]
    return TOOL_MAX_RESPONSE_TOKENS if TOOL_MAX_RESPONSE_TOKENS > 0 else None


@contextmanager
def _response_overrides(mode: str, max_response_tokens: int | None):
    mode_token = _response_mode_override.set(mode)
    cap_token = _response_token_cap_override.set(max_response_tokens)
    try:
        yield
    finally:
        _response_mode_override.reset(mode_token)
        _response_token_cap_override.reset(cap_token)


def _minimal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip payload to essential fields when response exceeds token budget."""
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
    """Serialize payload to compact JSON, truncating if it exceeds max_response_tokens."""
    # In compact mode, strip diagnostic wrapper fields from success responses.
    # Errors (ok=False) keep all fields so Claude can identify what failed.
    if TOOL_OUTPUT_MODE == "compact" and payload.get("ok") is True:
        payload.pop("ok", None)
        payload.pop("tool", None)
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    if max_response_tokens is not None and max_response_tokens > 0:
        if count_tokens(body) > max_response_tokens:
            body = json.dumps(_minimal_payload(payload), separators=(",", ":"), ensure_ascii=False)
        if count_tokens(body) > max_response_tokens:
            body = json.dumps({"ok": False, "truncated": True}, separators=(",", ":"))
    return body


def _render_error(tool: str, message: str, max_response_tokens: int | None) -> str:
    payload = {"ok": False, "tool": tool, "error": message}
    return _render_response(payload, max_response_tokens)
