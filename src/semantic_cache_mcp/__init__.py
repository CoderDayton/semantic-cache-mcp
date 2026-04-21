"""Semantic Cache MCP public package surface.

Keep the package root import-light. Submodule imports such as
``semantic_cache_mcp.utils`` should not eagerly pull in the server, cache, or
embedding stack just to resolve the package namespace.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cache import SemanticCache, smart_read
    from .server import mcp
    from .types import CacheEntry, ReadResult

__version__ = _pkg_version("semantic-cache-mcp")

__all__ = [
    "SemanticCache",
    "CacheEntry",
    "ReadResult",
    "smart_read",
    "mcp",
    "CACHE_DIR",
    "DB_PATH",
    "MAX_CACHE_ENTRIES",
    "MAX_CONTENT_SIZE",
    "SIMILARITY_THRESHOLD",
    "TOOL_OUTPUT_MODE",
    "TOOL_MAX_RESPONSE_TOKENS",
]


def __getattr__(name: str) -> Any:
    if name in {"SemanticCache", "smart_read"}:
        module = import_module(".cache", __name__)
        return getattr(module, name)
    if name in {"CacheEntry", "ReadResult"}:
        module = import_module(".types", __name__)
        return getattr(module, name)
    if name == "mcp":
        module = import_module(".server", __name__)
        return module.mcp
    if name in {
        "CACHE_DIR",
        "DB_PATH",
        "MAX_CACHE_ENTRIES",
        "MAX_CONTENT_SIZE",
        "SIMILARITY_THRESHOLD",
        "TOOL_OUTPUT_MODE",
        "TOOL_MAX_RESPONSE_TOKENS",
    }:
        module = import_module(".config", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
