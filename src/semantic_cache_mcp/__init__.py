"""Semantic Cache MCP - Lightweight semantic file caching with 80%+ token reduction."""

from .cache import SemanticCache, smart_read
from .config import (
    CACHE_DIR,
    DB_PATH,
    MAX_CACHE_ENTRIES,
    MAX_CONTENT_SIZE,
    SIMILARITY_THRESHOLD,
)
from .server import mcp
from .types import CacheEntry, ReadResult

__version__ = "0.4.0"

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
]
