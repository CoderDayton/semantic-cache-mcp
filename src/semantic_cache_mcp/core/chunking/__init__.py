"""Chunking sub-package: Gear-hash HyperCDC and SIMD-accelerated CDC."""

from __future__ import annotations

from ._gear import (
    DEFAULT_CONFIG,
    TURBO_CONFIG,
    HyperCDCConfig,
    hierarchical_hypercdc_chunks,
    hypercdc_boundaries,
    hypercdc_boundaries_turbo,
    hypercdc_chunks,
)
from ._simd import (
    _parallel_cdc_boundaries,
    get_optimal_chunker,
    hypercdc_simd_boundaries,
    hypercdc_simd_chunks,
)

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
    "_parallel_cdc_boundaries",
]
