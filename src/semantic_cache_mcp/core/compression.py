"""Adaptive compression using Brotli."""

from __future__ import annotations

import math
from collections import Counter

import brotli

from ..config import COMPRESSION_QUALITY, ENTROPY_SAMPLE_SIZE


def estimate_entropy(data: bytes) -> float:
    """Estimate Shannon entropy of data (bits per byte).

    Used to select compression quality. High entropy data (>7) is already
    compressed and needs less effort. Low entropy data (<5.5) is highly
    compressible and benefits from higher quality.

    Args:
        data: Raw bytes to analyze

    Returns:
        Entropy in bits per byte (0-8 range)
    """
    if not data:
        return 0.0

    n = len(data)
    counts = Counter(data)  # C-implemented, ~2-3x faster than manual loop
    log2_n = math.log2(n)

    return -sum(
        count * (math.log2(count) - log2_n) / n for count in counts.values()
    )


def compress_adaptive(data: bytes) -> bytes:
    """Compress data with adaptive quality based on entropy.

    Samples the first 4KB to estimate compressibility, then selects
    appropriate Brotli quality level.

    Args:
        data: Raw bytes to compress

    Returns:
        Brotli-compressed bytes
    """
    entropy = estimate_entropy(data[:ENTROPY_SAMPLE_SIZE])

    if entropy > 7.0:
        quality = COMPRESSION_QUALITY["high_entropy"]
    elif entropy > 5.5:
        quality = COMPRESSION_QUALITY["medium_entropy"]
    else:
        quality = COMPRESSION_QUALITY["low_entropy"]

    return brotli.compress(data, quality=quality)


def decompress(data: bytes) -> bytes:
    """Decompress Brotli data.

    Args:
        data: Brotli-compressed bytes

    Returns:
        Decompressed bytes
    """
    return brotli.decompress(data)
