"""Semantic Cache MCP - Lightweight semantic file caching with 80%+ token reduction."""

from importlib.metadata import version as _pkg_version

from .cache import SemanticCache, smart_read
from .config import (
    CACHE_DIR,
    DB_PATH,
    MAX_CACHE_ENTRIES,
    MAX_CONTENT_SIZE,
    SIMILARITY_THRESHOLD,
    TOOL_MAX_RESPONSE_TOKENS,
    TOOL_OUTPUT_MODE,
)
from .server import mcp
from .types import CacheEntry, ReadResult

__version__ = _pkg_version("semantic-cache-mcp")

__all__ = [
    # Main classes
    "SemanticCache",
    "CacheEntry",
    "ReadResult",
    # Functions
    "smart_read",
    # Server
    "mcp",
    # Configuration
    "CACHE_DIR",
    "DB_PATH",
    "MAX_CACHE_ENTRIES",
    "MAX_CONTENT_SIZE",
    "SIMILARITY_THRESHOLD",
    "TOOL_OUTPUT_MODE",
    "TOOL_MAX_RESPONSE_TOKENS",
]
