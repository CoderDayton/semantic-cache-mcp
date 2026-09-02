"""Response rendering helpers."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Never

from fastmcp.exceptions import ToolError

from ..config import TOOL_MAX_RESPONSE_TOKENS, TOOL_OUTPUT_MODE
from ..core import count_tokens

_MODE_NORMAL = {"normal", "debug"}
_MODE_DEBUG = "debug"
_UNSET = object()
_NO_CHANGES_DIFF = "// No changes"
# Quotes and comma around one rendered grep line in the JSON array.
_LINE_BUDGET_OVERHEAD = 3
# Chars per token to assume when refitting. Serialized matches are not prose:
# quoted keys, punctuation and line numbers tokenize denser, so budgeting at
# the ~4-chars-per-token prose rate builds a payload back over the cap.
_MIN_CHARS_PER_TOKEN = 2
# Characters reserved for the envelope (counts, flags, root, path) when
# refitting grep matches into a truncated response.
_TRUNCATION_ENVELOPE_CHARS = 200
_SUPPRESSED_DIFF_PREFIX = "[diff suppressed:"
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


def _fit_grep_files(files: list[Any], budget: int, spent: int) -> tuple[list[dict[str, Any]], bool]:
    """Keep as many grep matches as the remaining budget allows.

    Dropping every match was a net token *loss*: the caller learns a count it
    cannot act on and runs the same grep again. Keeping a prefix that fits
    means the common case — an estimate that came in slightly high — still
    answers the question it was asked.
    """
    kept: list[dict[str, Any]] = []
    dropped = False
    for entry in files:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        lines = entry.get("lines")
        if not isinstance(lines, list):
            # Not a grep match list. `grep(output="paths")` legitimately sends
            # a bare `{"path": ...}`, but a `batch_read` entry carries content
            # and status here — reducing it to its path drops the answer, so
            # it counts as a truncation rather than a free fit.
            dropped = dropped or len(entry) > 1
            kept.append({"path": path})
            continue
        fitted: list[str] = []
        for line in lines:
            cost = len(str(line)) + _LINE_BUDGET_OVERHEAD
            if spent + cost > budget:
                dropped = True
                break
            spent += cost
            fitted.append(line)
        if not fitted:
            dropped = dropped or bool(lines)
            break
        kept.append({"path": path, "lines": fitted})
        if dropped:
            break
    return kept, dropped or len(kept) < len(files)


def _minimal_payload(payload: dict[str, Any], budget: int | None = None) -> dict[str, Any]:
    """Strip payload to essential fields when response exceeds token budget.

    The bulky fields go; the ones that say what the answer *was* stay. A
    response cut down to `{"path": ..., "truncated": true}` reads as a failure
    with no result, when the tool in fact counted 400 matches and knows the
    scan was capped — the caller needs those scalars far more than it needs
    the match lines, and they cost a handful of tokens.

    When a ``budget`` is given, grep matches degrade rather than vanish: as
    many lines as fit are kept, and `truncated` says the rest were cut.
    """
    keep_order = (
        "ok",
        "tool",
        "status",
        "path",
        "path1",
        "path2",
        "pattern",
        # Relative `files[].path` entries are meaningless without it.
        "root",
        "summary",
        "skipped",
        "files_read",
        "files_skipped",
        "succeeded",
        "failed",
        "total_matches",
        "files_matched",
        "files_in_response",
        "truncated_matches",
        "truncated_files",
        "complete",
        "limit_reached",
        "files_not_searched",
        "reason",
        "hint",
        "message",
        "error",
    )
    minimal: dict[str, Any] = {}
    for key in keep_order:
        if key in payload:
            minimal[key] = payload[key]

    diff_content = payload.get("diff")
    diff_state = payload.get("diff_state")
    diff_omitted = payload.get("diff_omitted")
    if isinstance(diff_content, str):
        if diff_state in {"unchanged", "suppressed"}:
            minimal["diff"] = diff_content
            minimal["diff_state"] = diff_state
        else:
            minimal["diff_state"] = "omitted"
            minimal["diff_omitted"] = True
    elif diff_omitted:
        minimal["diff_state"] = "omitted"
        minimal["diff_omitted"] = True
    elif diff_state is not None:
        minimal["diff_state"] = diff_state

    files = payload.get("files")
    if budget is not None and isinstance(files, list) and files:
        # Charge at the same conservative chars-per-token rate the grep tool
        # budgets with, so a refit that says it fits actually does.
        char_budget = max(0, budget * _MIN_CHARS_PER_TOKEN - _TRUNCATION_ENVELOPE_CHARS)
        kept, dropped = _fit_grep_files(files, char_budget, 0)
        if kept:
            minimal["files"] = kept
        if not dropped:
            # Everything fitted after the bulky fields went, so this is no
            # longer a truncated answer to the question that was asked.
            minimal.pop("message", None)
            return minimal

    minimal["truncated"] = True
    if "message" not in minimal:
        minimal["message"] = "Response truncated by max_response_tokens"
    return minimal


def _diff_state(diff_content: str | None) -> str | None:
    """Classify a diff payload for clients without making them parse strings."""
    if not diff_content:
        return None
    if diff_content == _NO_CHANGES_DIFF:
        return "unchanged"
    if diff_content.startswith(_SUPPRESSED_DIFF_PREFIX):
        return "suppressed"
    return "full"


def _finalize_payload(payload: dict[str, Any], max_response_tokens: int | None) -> dict[str, Any]:
    """Apply response shaping without serializing, for FastMCP structured results."""
    body = payload.copy()
    if _response_mode() == "compact" and body.get("ok") is True:
        body.pop("ok", None)
        body.pop("tool", None)

    if max_response_tokens is not None and max_response_tokens > 0:
        rendered = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        # tokens <= chars for any BPE tokenizer (no token spans zero chars),
        # so len <= max_response_tokens is a safe fast-pass that skips BPE.
        # Anything larger must be measured: density varies (CJK/emoji can
        # approach 1 token per char), and the token cap is the source of truth.
        if len(rendered) > max_response_tokens and count_tokens(rendered) > max_response_tokens:
            body = _minimal_payload(body, max_response_tokens)
            rendered = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
            if len(rendered) > max_response_tokens and count_tokens(rendered) > max_response_tokens:
                # The refit overshot anyway; fall back to the bare form rather
                # than ship a payload over the cap the caller set.
                body = _minimal_payload(body)
                rendered = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
                if (
                    len(rendered) > max_response_tokens
                    and count_tokens(rendered) > max_response_tokens
                ):
                    body = {"ok": False, "truncated": True}

    return body


def _render_response(payload: dict[str, Any], max_response_tokens: int | None) -> str:
    """Serialize payload to compact JSON, truncating if it exceeds max_response_tokens."""
    body = _finalize_payload(payload, max_response_tokens)
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False)


def _render_error(tool: str, message: str, max_response_tokens: int | None) -> str:
    payload = {"ok": False, "tool": tool, "error": message}
    return _render_response(payload, max_response_tokens)


def _tool_error_message(tool: str, message: str, max_response_tokens: int | None) -> str:
    text = f"{tool}: {message}"
    if (
        max_response_tokens is not None
        and max_response_tokens > 0
        and count_tokens(text) > max_response_tokens
    ):
        text = f"{tool}: error truncated by max_response_tokens"
    return text


def _raise_tool_error(tool: str, message: str, max_response_tokens: int | None) -> Never:
    raise ToolError(_tool_error_message(tool, message, max_response_tokens))
