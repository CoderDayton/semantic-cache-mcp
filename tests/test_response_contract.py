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


# Each tool shapes its payload differently per branch — a capped grep, a partial
# batch_edit and a dry-run write emit keys the happy path never does. Covering
# one call per tool leaves those branches unguarded, which is exactly how
# `reason`/`hint` reached `grep`'s payload without reaching its model. One
# scenario per distinct payload shape.
_Setup = Callable[[Path, Path, SemanticCache], Any]

_SCENARIOS: list[tuple[str, Any, type[BaseModel], _Setup]] = [
    ("read/full", read, ReadResponse, lambda d, f, c: {"path": str(f)}),
    (
        "read/ranged",
        read,
        ReadResponse,
        lambda d, f, c: {"path": str(f), "offset": 1, "limit": 1},
    ),
    # A file over `max_size` comes back summarized: `truncated`, `hint`,
    # `total_tokens` and a `partial:`-prefixed `file_hash` instead of a
    # claimable `content_hash`.
    (
        "read/summarized",
        read,
        ReadResponse,
        lambda d, f, c: {"path": str(d / "large.py"), "max_size": 1024},
    ),
    (
        "read/binary",
        read,
        ReadResponse,
        lambda d, f, c: {"path": str(d / "blob.bin")},
    ),
    ("clear/empty", clear, ClearResponse, lambda d, f, c: {}),
    (
        "delete/dry_run",
        delete,
        DeleteResponse,
        lambda d, f, c: {"path": str(f), "dry_run": True},
    ),
    ("delete/real", delete, DeleteResponse, lambda d, f, c: {"path": str(d / "gone.py")}),
    (
        "delete/not_found",
        delete,
        DeleteResponse,
        lambda d, f, c: {"path": str(d / "never.py")},
    ),
    (
        "write/create",
        write,
        WriteResponse,
        lambda d, f, c: {"path": str(d / "new.txt"), "content": "hi\n"},
    ),
    (
        "write/update_show_diff",
        write,
        WriteResponse,
        lambda d, f, c: {
            "path": str(f),
            "content": "def hello():\n    return 'moon'\n",
            "show_diff": True,
        },
    ),
    (
        "write/dry_run",
        write,
        WriteResponse,
        lambda d, f, c: {"path": str(f), "content": "x\n", "dry_run": True},
    ),
    (
        "write/append",
        write,
        WriteResponse,
        lambda d, f, c: {"path": str(f), "content": "# tail\n", "append": True},
    ),
    (
        "edit/basic",
        edit,
        EditResponse,
        lambda d, f, c: {"path": str(f), "old_string": "world", "new_string": "planet"},
    ),
    (
        "edit/dry_run",
        edit,
        EditResponse,
        lambda d, f, c: {
            "path": str(f),
            "old_string": "world",
            "new_string": "planet",
            "dry_run": True,
        },
    ),
    (
        "edit/show_diff",
        edit,
        EditResponse,
        lambda d, f, c: {
            "path": str(f),
            "old_string": "world",
            "new_string": "planet",
            "show_diff": True,
        },
    ),
    (
        "edit_preview/found",
        edit_preview,
        EditPreviewResponse,
        lambda d, f, c: {"path": str(f), "old_string": "return"},
    ),
    (
        "edit_preview/miss",
        edit_preview,
        EditPreviewResponse,
        lambda d, f, c: {"path": str(f), "old_string": "no_such_anchor"},
    ),
    (
        "batch_edit/all_ok",
        batch_edit,
        BatchEditResponse,
        lambda d, f, c: {"path": str(f), "edits": json.dumps([["world", "planet"]])},
    ),
    (
        "batch_edit/partial",
        batch_edit,
        BatchEditResponse,
        lambda d, f, c: {
            "path": str(f),
            "edits": json.dumps([["world", "planet"], ["absent_anchor", "x"]]),
        },
    ),
    (
        "batch_edit/no_changes",
        batch_edit,
        BatchEditResponse,
        lambda d, f, c: {"path": str(f), "edits": json.dumps([["absent_anchor", "x"]])},
    ),
    (
        "search/hit",
        search,
        SearchResponse,
        lambda d, f, c: {"query": "hello", "show_preview": True},
    ),
    ("search/miss", search, SearchResponse, lambda d, f, c: {"query": "zzz_absent_term"}),
    ("batch_read/basic", batch_read, BatchReadResponse, lambda d, f, c: {"paths": str(f)}),
    (
        "batch_read/budget_exhausted",
        batch_read,
        BatchReadResponse,
        lambda d, f, c: {"paths": str(f), "max_total_tokens": 1},
    ),
    (
        "glob/basic",
        glob,
        GlobResponse,
        lambda d, f, c: {"pattern": "*.py", "directory": str(d)},
    ),
    ("grep/hit", grep, GrepResponse, lambda d, f, c: {"pattern": "hello"}),
    (
        "grep/capped",
        grep,
        GrepResponse,
        lambda d, f, c: {"pattern": "e", "max_matches": 1},
    ),
    (
        "grep/miss_under_path",
        grep,
        GrepResponse,
        lambda d, f, c: {"pattern": "zzz_absent", "path": "nowhere/absent.py"},
    ),
    (
        "grep/with_context",
        grep,
        GrepResponse,
        lambda d, f, c: {"pattern": "hello", "context_lines": 2},
    ),
]


@pytest.mark.parametrize("scenario", _SCENARIOS, ids=lambda s: s[0])
@pytest.mark.parametrize("mode", _MODES)
async def test_payload_keys_are_declared_in_model(
    scenario: tuple[str, Any, type[BaseModel], _Setup], mode: str, tmp_path: Path
) -> None:
    name, fn, model, build_kwargs = scenario

    work = tmp_path / f"{name.replace('/', '_')}_{mode}"
    work.mkdir()
    src = work / "mod.py"
    src.write_text("def hello():\n    return 'world'\n")
    (work / "gone.py").write_text("delete me\n")
    (work / "large.py").write_text(
        "\n".join(f"def helper_{i}():\n    return {i}" for i in range(400))
    )
    (work / "blob.bin").write_bytes(b"\x00\x01\x02\xff" * 64)

    cache = SemanticCache(db_path=work / "cache.db")
    ctx = _make_ctx(cache)
    # Seed unconditionally: grep and search read only cached files, and a
    # populated cache is also what exercises the `unchanged`/diff branches.
    await smart_read(cache=cache, path=str(src))

    kwargs = build_kwargs(work, src, cache)

    # No token cap so truncation never strips keys; force the output mode so
    # mode-gated fields (normal/debug) are exercised.
    with _response_overrides(mode, None):
        result = await fn(ctx, **kwargs)

    assert isinstance(result, dict), f"{name} did not return a dict payload"

    declared = set(model.model_fields)
    allowed = declared | _ENVELOPE
    leaked = set(result) - allowed
    assert not leaked, (
        f"{name} (mode={mode}) emits {sorted(leaked)} not declared in "
        f"{model.__name__}; schema-aware clients would silently drop these "
        f"fields. Add them to {model.__name__} or stop emitting them."
    )


async def test_grep_truncation_keys_are_actually_reached(tmp_path: Path) -> None:
    """The capped-grep scenario must really hit the branch it exists to cover.

    Without this, `grep/capped` could stop capping (a default change, a smarter
    prefilter) and quietly go back to guarding nothing.
    """
    work = tmp_path / "capped"
    work.mkdir()
    src = work / "many.py"
    src.write_text("hello\n" * 20)
    cache = SemanticCache(db_path=work / "cache.db")
    await smart_read(cache=cache, path=str(src))

    with _response_overrides("normal", None):
        result = await grep(_make_ctx(cache), pattern="hello", max_matches=2)

    assert result["complete"] is False
    assert result["limit_reached"] == "max_matches"
    assert "hint" in result


def test_contract_has_teeth() -> None:
    """A payload key absent from the model must be detected (guard is not vacuous)."""
    declared = set(ReadResponse.model_fields) | _ENVELOPE
    payload = {"ok": True, "tool": "read", "path": "/x", "undeclared_field": 1}
    assert set(payload) - declared == {"undeclared_field"}
