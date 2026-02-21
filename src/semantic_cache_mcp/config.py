"""Configuration constants for semantic-cache-mcp."""

import logging
from os import environ
from pathlib import Path
from typing import Final

# Logging configuration
LOG_LEVEL: Final = environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: Final = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# Paths
CACHE_DIR: Final = Path.home() / ".cache" / "semantic-cache-mcp"
DB_PATH: Final = CACHE_DIR / "cache.db"


def _env_int(name: str, default: int) -> int:
    """Read positive integer env var with fallback."""
    raw = environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_mode(name: str, default: str) -> str:
    """Read normalized response mode env var."""
    raw = environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower()


# Cache limits
MAX_CONTENT_SIZE: Final = _env_int("MAX_CONTENT_SIZE", 100_000)  # 100KB default max return size
MAX_CACHE_ENTRIES: Final = _env_int("MAX_CACHE_ENTRIES", 10_000)  # LRU-K eviction limit

# Embedding device: "cpu" (default, no VRAM usage), "cuda" (GPU), "auto" (detect)
EMBEDDING_DEVICE: Final = environ.get("EMBEDDING_DEVICE", "cpu").strip().lower()

# Tool response policy (global, not model-selected per call)
TOOL_OUTPUT_MODE: Final = _env_mode("TOOL_OUTPUT_MODE", "compact")
TOOL_MAX_RESPONSE_TOKENS: Final = _env_int("TOOL_MAX_RESPONSE_TOKENS", 0)

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
    """Validate configuration constants at module load time."""
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
        errors.append(f"EMBEDDING_DEVICE ({EMBEDDING_DEVICE}) must be one of: cpu, cuda, auto")

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
