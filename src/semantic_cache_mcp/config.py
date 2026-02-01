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

# Cache limits
MAX_CONTENT_SIZE: Final = 100_000  # 100KB default max return size
MAX_CACHE_ENTRIES: Final = 10_000  # LRU-K eviction limit

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
