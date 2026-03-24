"""Configuration constants for semantic-cache-mcp."""

import logging
import sys
from os import environ
from pathlib import Path
from typing import Final

# Logging configuration — explicit stderr handler to prevent stdout pollution
# in stdio MCP transport. Default basicConfig would also use stderr, but being
# explicit guards against accidental reconfiguration by third-party libraries.
LOG_LEVEL: Final = environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: Final = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

if not logging.root.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.root.addHandler(_handler)
logging.root.setLevel(LOG_LEVEL)


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

# Crash sentinel — written on startup, removed on clean shutdown.
# If present at next startup → previous run crashed → wipe vecdb to avoid
# heap corruption from corrupted usearch index files.
STARTUP_SENTINEL: Final = CACHE_DIR / ".startup.lock"


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
MAX_CACHE_ENTRIES: Final = _env_int("MAX_CACHE_ENTRIES", 10_000)  # LRU-K eviction limit

# Embedding device: "cpu" (default), "gpu"/"cuda" (NVIDIA GPU), "auto" (detect)
_raw_device = environ.get("EMBEDDING_DEVICE", "cpu").strip().lower()
EMBEDDING_DEVICE: Final = "cuda" if _raw_device == "gpu" else _raw_device

# Embedding model: any FastEmbed-supported model name
EMBEDDING_MODEL: Final = environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip()

# Tool response policy (global, not model-selected per call)
TOOL_OUTPUT_MODE: Final = _env_mode("TOOL_OUTPUT_MODE", "compact")
TOOL_MAX_RESPONSE_TOKENS: Final = _env_int("TOOL_MAX_RESPONSE_TOKENS", 0)
TOOL_TIMEOUT: Final = _env_float("TOOL_TIMEOUT", 30.0)  # seconds before tool call times out

# Similarity
SIMILARITY_THRESHOLD: Final = 0.85  # Semantic similarity threshold
NEAR_DUPLICATE_THRESHOLD: Final = 0.98  # Early termination threshold

# Chunking (Rabin fingerprinting)
CHUNK_MIN_SIZE: Final = 2048
CHUNK_MAX_SIZE: Final = 65536
RH_PRIME: Final = 31
RH_MOD: Final = (1 << 32) - 1
RH_WINDOW: Final = 48
RH_MASK: Final = 0x1FFF  # 13 bits - average chunk size ~8KB
RH_POW_OUT: Final = pow(RH_PRIME, RH_WINDOW - 1, RH_MOD)

# Compression (Brotli)
ENTROPY_SAMPLE_SIZE: Final = 4096
COMPRESSION_QUALITY: Final = {
    "high_entropy": 1,  # Already compressed (entropy > 7.0)
    "medium_entropy": 4,  # Medium compressibility (entropy > 5.5)
    "low_entropy": 6,  # Highly compressible (entropy <= 5.5)
}

# LRU-K
LRU_K: Final = 2  # K-th most recent access for eviction scoring
ACCESS_HISTORY_SIZE: Final = 5  # Number of accesses to track

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

    if EMBEDDING_DEVICE not in {"cpu", "cuda", "auto"}:
        errors.append(f"EMBEDDING_DEVICE ({EMBEDDING_DEVICE}) must be one of: cpu, gpu, cuda, auto")

    if not 0 < SIMILARITY_THRESHOLD < 1:
        errors.append(f"SIMILARITY_THRESHOLD ({SIMILARITY_THRESHOLD}) must be between 0 and 1")

    if not 0 < NEAR_DUPLICATE_THRESHOLD <= 1:
        errors.append(
            f"NEAR_DUPLICATE_THRESHOLD ({NEAR_DUPLICATE_THRESHOLD}) must be between 0 and 1"
        )

    if NEAR_DUPLICATE_THRESHOLD < SIMILARITY_THRESHOLD:
        errors.append(
            f"NEAR_DUPLICATE_THRESHOLD ({NEAR_DUPLICATE_THRESHOLD}) must be >= "
            f"SIMILARITY_THRESHOLD ({SIMILARITY_THRESHOLD})"
        )

    if errors:
        raise ValueError("Configuration errors:\n  " + "\n  ".join(errors))


_validate_config()
