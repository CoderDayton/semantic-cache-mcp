"""Content-defined chunking using Rabin fingerprinting."""

from __future__ import annotations

from typing import Iterator

from ..config import (
    CHUNK_MAX_SIZE,
    CHUNK_MIN_SIZE,
    RH_MASK,
    RH_MOD,
    RH_POW_OUT,
    RH_PRIME,
    RH_WINDOW,
)


def content_defined_chunking(
    content: bytes,
    min_size: int = CHUNK_MIN_SIZE,
    max_size: int = CHUNK_MAX_SIZE,
) -> Iterator[bytes]:
    """Split content into chunks using rolling hash (Rabin fingerprinting).

    Content-defined chunking finds natural boundaries that survive insertions
    and deletions, enabling better deduplication than fixed-size chunking.

    The rolling hash is inlined for performance (~3x faster than method calls).

    Args:
        content: Raw bytes to chunk
        min_size: Minimum chunk size (default: 2KB)
        max_size: Maximum chunk size (default: 64KB)

    Yields:
        Content chunks with natural boundaries
    """
    n = len(content)
    if n == 0:
        return

    # Inline constants for performance
    PRIME, MOD, WINDOW, MASK = RH_PRIME, RH_MOD, RH_WINDOW, RH_MASK
    pow_out = RH_POW_OUT

    h = 0
    buf = [0] * WINDOW  # Circular buffer
    pos = 0
    full = False
    chunk_start = 0

    for i in range(n):
        b = content[i]

        # Inlined rolling hash update
        if full:
            h = (h - buf[pos] * pow_out) % MOD
        h = (h * PRIME + b) % MOD
        buf[pos] = b
        pos += 1
        if pos == WINDOW:
            pos = 0
            full = True

        chunk_size = i - chunk_start + 1
        if chunk_size >= min_size and ((h & MASK) == 0 or chunk_size >= max_size):
            yield content[chunk_start : i + 1]
            chunk_start = i + 1

    # Emit final chunk
    if chunk_start < n:
        yield content[chunk_start:]
