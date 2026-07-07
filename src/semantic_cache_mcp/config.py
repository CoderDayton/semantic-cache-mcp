"""Configuration constants for semantic-cache-mcp."""

import sys
from os import environ
from pathlib import Path
from typing import Final

from .logger import DEFAULT_LOG_FORMAT, configure_logging, get_log_dir, get_log_file_path


# Paths
def _get_cache_dir() -> Path:
    """Platform-appropriate cache directory.

    Priority: $SEMANTIC_CACHE_DIR > platform default.
    - Linux: $XDG_CACHE_HOME/semantic-cache-mcp or ~/.cache/semantic-cache-mcp
    - macOS: ~/Library/Caches/semantic-cache-mcp
    - Windows: %LOCALAPPDATA%/semantic-cache-mcp
    """
    env_override = environ.get("SEMANTIC_CACHE_DIR")
    if env_override:
        return Path(env_override).expanduser().resolve()

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "semantic-cache-mcp"
    if sys.platform == "win32":
        local = environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "semantic-cache-mcp"
        return Path.home() / "AppData" / "Local" / "semantic-cache-mcp"
    # Linux / other POSIX
    xdg = environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "semantic-cache-mcp"
    return Path.home() / ".cache" / "semantic-cache-mcp"


CACHE_DIR: Final = _get_cache_dir()
DB_PATH: Final = CACHE_DIR / "cache.db"
LOG_DIR: Final = get_log_dir(CACHE_DIR, environ.get("LOG_DIR"))
LOG_FILE_PATH: Final = get_log_file_path(LOG_DIR)


# Logging configuration — explicit stderr handler to prevent stdout pollution
# in stdio MCP transport, plus a dated file handler for post-mortem debugging.
# An unknown LOG_LEVEL falls back to INFO instead of letting setLevel() raise
# during import (same lenient policy as _env_int/_env_float below).
_VALID_LOG_LEVELS: Final = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_raw_log_level = environ.get("LOG_LEVEL", "INFO").upper()
LOG_LEVEL: Final = _raw_log_level if _raw_log_level in _VALID_LOG_LEVELS else "INFO"
LOG_FORMAT: Final = DEFAULT_LOG_FORMAT

configure_logging(LOG_DIR, LOG_FILE_PATH, log_level=LOG_LEVEL, log_format=LOG_FORMAT)


def _env_int(name: str, default: int) -> int:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_mode(name: str, default: str) -> str:
    raw = environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower()


# Cache limits
MAX_CONTENT_SIZE: Final = _env_int("MAX_CONTENT_SIZE", 100_000)  # 100KB default max return size
MAX_CACHE_ENTRIES: Final = _env_int("MAX_CACHE_ENTRIES", 10_000)  # W-TinyLFU eviction limit

# Tool response policy (global, not model-selected per call)
TOOL_OUTPUT_MODE: Final = _env_mode("TOOL_OUTPUT_MODE", "compact")
TOOL_MAX_RESPONSE_TOKENS: Final = _env_int("TOOL_MAX_RESPONSE_TOKENS", 0)
TOOL_TIMEOUT: Final = _env_float("TOOL_TIMEOUT", 30.0)  # seconds before tool call times out

# Chunking
CHUNK_MIN_SIZE: Final = 2048
CHUNK_MAX_SIZE: Final = 65536

# W-TinyLFU eviction
ACCESS_HISTORY_SIZE: Final = 5  # accesses tracked per entry for frequency scoring

# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def _validate_config() -> None:
    """Fail fast on invalid config at import time."""
    errors: list[str] = []

    if CHUNK_MAX_SIZE <= CHUNK_MIN_SIZE:
        errors.append(
            f"CHUNK_MAX_SIZE ({CHUNK_MAX_SIZE}) must be > CHUNK_MIN_SIZE ({CHUNK_MIN_SIZE})"
        )

    if MAX_CACHE_ENTRIES <= 0:
        errors.append(f"MAX_CACHE_ENTRIES ({MAX_CACHE_ENTRIES}) must be > 0")

    if MAX_CONTENT_SIZE <= 0:
        errors.append(f"MAX_CONTENT_SIZE ({MAX_CONTENT_SIZE}) must be > 0")

    if TOOL_OUTPUT_MODE not in {"compact", "normal", "debug"}:
        errors.append(
            f"TOOL_OUTPUT_MODE ({TOOL_OUTPUT_MODE}) must be one of: compact, normal, debug"
        )

    if TOOL_MAX_RESPONSE_TOKENS < 0:
        errors.append(f"TOOL_MAX_RESPONSE_TOKENS ({TOOL_MAX_RESPONSE_TOKENS}) must be >= 0")

    if TOOL_TIMEOUT <= 0:
        errors.append(f"TOOL_TIMEOUT ({TOOL_TIMEOUT}) must be > 0")

    if ACCESS_HISTORY_SIZE <= 0:
        errors.append(f"ACCESS_HISTORY_SIZE ({ACCESS_HISTORY_SIZE}) must be > 0")

    if errors:
        raise ValueError("Configuration errors:\n  " + "\n  ".join(errors))


_validate_config()
