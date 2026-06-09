from __future__ import annotations

from collections.abc import Callable, Iterator

from ._gear import (
    DEFAULT_CONFIG,
    TURBO_CONFIG,
    HyperCDCConfig,
    hierarchical_hypercdc_chunks,
    hypercdc_boundaries,
    hypercdc_boundaries_turbo,
    hypercdc_chunks,
)

try:
    # _parallel_cdc_boundaries re-exported for tests; not public API (absent from __all__).
    from ._simd import (  # noqa: F401
        _parallel_cdc_boundaries,
        hypercdc_simd_boundaries,
        hypercdc_simd_chunks,
    )

    _SIMD_AVAILABLE = True
except ImportError:  # numpy missing — SIMD path unavailable, Gear path still works
    _SIMD_AVAILABLE = False


def get_optimal_chunker(prefer_simd: bool = True) -> Callable[..., Iterator[bytes]]:
    """Return the fastest available chunker.

    numpy is a declared dependency, so the SIMD path is normally available and
    selected by default. This performs a real availability check: ``prefer_simd=False``
    — or a numpy-less environment — falls back to the pure-Python Gear-hash chunker.

    Warning: the SIMD and Gear chunkers produce DIFFERENT boundaries for the same
    input; they are not interchangeable. Route all callers through this function
    (or pick one explicitly and stick with it). Mixing them on the same content
    makes content-defined dedup store two disjoint chunk sets per file.
    """
    if prefer_simd and _SIMD_AVAILABLE:
        return hypercdc_simd_chunks
    return hypercdc_chunks


__all__ = [
    "HyperCDCConfig",
    "DEFAULT_CONFIG",
    "TURBO_CONFIG",
    "hypercdc_chunks",
    "hypercdc_boundaries",
    "hypercdc_boundaries_turbo",
    "hierarchical_hypercdc_chunks",
    "hypercdc_simd_chunks",
    "hypercdc_simd_boundaries",
    "get_optimal_chunker",
]
