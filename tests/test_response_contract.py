"""Contract: every key a tool emits is declared in its response model.

The response shape for each tool lives in several places that can silently
drift apart:

  * the hand-assembled, mode-gated ``payload`` dict in ``server/tools``;
  * the Pydantic ``*Response`` model in ``server/_tool_models.py`` that backs
    the published MCP ``output_schema``;
  * the ``_minimal_payload`` truncation allow-list in ``server/response.py``.

The Pydantic models use ``extra="ignore"``, so a payload key that is missing
from the model is *silently dropped* from a schema-aware client's structured
output — no error, just a vanished field. This test makes the model the single
declared contract and fails loudly if a tool emits a key the model does not
declare, across every output mode (which is where mode-gated fields appear).

Scope: top-level payload keys (the main drift surface). Nested sub-models
(e.g. ``ReadResponse.lines``) are checked for presence at the top level, not
field-by-field.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastmcp import Context
from pydantic import BaseModel

from semantic_cache_mcp.cache import SemanticCache, smart_read
from semantic_cache_mcp.server._tool_models import (
    BatchEditResponse,
    BatchReadResponse,
    ClearResponse,
    DeleteResponse,
    EditPreviewResponse,
    EditResponse,
    GlobResponse,
    GrepResponse,
    ReadResponse,
    SearchResponse,
    WriteResponse,
)
from semantic_cache_mcp.server.response import _response_overrides
from semantic_cache_mcp.server.tools import (
    batch_edit,
    batch_read,
    clear,
    delete,
    edit,
    edit_preview,
    glob,
    grep,
    read,
    search,
    write,
)

# Envelope keys live on every payload but are deliberately not typed model
# fields: compact mode strips them, schema-aware clients ignore them.
_ENVELOPE = {"ok", "tool"}

_MODES = ["compact", "normal", "debug"]


def _make_ctx(cache: SemanticCache) -> MagicMock:
    ctx = MagicMock(spec=Context)
    ctx.lifespan_context = {"cache": cache}
    return ctx


# name -> (callable, model, kwargs-from-(dir, file), needs a seeded cache)
_TOOLS: dict[str, tuple[Any, type[BaseModel], Callable[[Path, Path], dict[str, Any]], bool]] = {
    "read": (read, ReadResponse, lambda d, f: {"path": str(f)}, False),
    "clear": (clear, ClearResponse, lambda d, f: {}, False),
    "delete": (delete, DeleteResponse, lambda d, f: {"path": str(f), "dry_run": True}, False),
    "write": (
        write,
        WriteResponse,
        lambda d, f: {"path": str(d / "new.txt"), "content": "hi\n"},
        False,
    ),
    "edit": (
        edit,
        EditResponse,
        lambda d, f: {"path": str(f), "old_string": "world", "new_string": "planet"},
        False,
    ),
    "edit_preview": (
        edit_preview,
        EditPreviewResponse,
        lambda d, f: {"path": str(f), "old_string": "return"},
        False,
    ),
    "batch_edit": (
        batch_edit,
        BatchEditResponse,
        lambda d, f: {"path": str(f), "edits": json.dumps([["world", "planet"]])},
        False,
    ),
    "search": (search, SearchResponse, lambda d, f: {"query": "hello", "show_preview": True}, True),
    "batch_read": (batch_read, BatchReadResponse, lambda d, f: {"paths": str(f)}, False),
    "glob": (glob, GlobResponse, lambda d, f: {"pattern": "*.py", "directory": str(d)}, False),
    "grep": (grep, GrepResponse, lambda d, f: {"pattern": "hello"}, True),
}


@pytest.mark.parametrize("tool_name", list(_TOOLS))
@pytest.mark.parametrize("mode", _MODES)
async def test_payload_keys_are_declared_in_model(
    tool_name: str, mode: str, tmp_path: Path
) -> None:
    fn, model, build_kwargs, needs_seed = _TOOLS[tool_name]

    work = tmp_path / f"{tool_name}_{mode}"
    work.mkdir()
    src = work / "mod.py"
    src.write_text("def hello():\n    return 'world'\n")

    cache = SemanticCache(db_path=work / "cache.db")
    ctx = _make_ctx(cache)
    if needs_seed:
        await smart_read(cache=cache, path=str(src))

    # No token cap so truncation never strips keys; force the output mode so
    # mode-gated fields (normal/debug) are exercised.
    with _response_overrides(mode, None):
        result = await fn(ctx, **build_kwargs(work, src))

    assert isinstance(result, dict), f"{tool_name} did not return a dict payload"

    declared = set(model.model_fields)
    allowed = declared | _ENVELOPE
    leaked = set(result) - allowed
    assert not leaked, (
        f"{tool_name} (mode={mode}) emits {sorted(leaked)} not declared in "
        f"{model.__name__}; schema-aware clients would silently drop these "
        f"fields. Add them to {model.__name__} or stop emitting them."
    )


def test_contract_has_teeth() -> None:
    """A payload key absent from the model must be detected (guard is not vacuous)."""
    declared = set(ReadResponse.model_fields) | _ENVELOPE
    payload = {"ok": True, "tool": "read", "path": "/x", "undeclared_field": 1}
    assert set(payload) - declared == {"undeclared_field"}
