from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

from ...config import EMBEDDING_DEVICE
from ...utils import retry
from ._constants import FASTEMBED_CACHE_DIR, FASTEMBED_MODEL
from ._cuda import _cuda_provider_is_available
from ._registry import _register_custom_model

if TYPE_CHECKING:
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# Singleton instance
_embedding_model: TextEmbedding | None = None
_model_ready: bool = False
_embedding_dim: int = 0
_execution_provider: str = "unknown"


def _get_model() -> TextEmbedding:
    """Lazy-init the embedding model singleton."""
    global _embedding_model, _execution_provider

    if _embedding_model is None:
        from fastembed import TextEmbedding

        logger.info(f"Loading embedding model: {FASTEMBED_MODEL}")
        logger.info(f"Model cache directory: {FASTEMBED_CACHE_DIR}")

        FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        init_kwargs: dict[str, Any] = {
            "model_name": FASTEMBED_MODEL,
            "cache_dir": str(FASTEMBED_CACHE_DIR),
            "lazy_load": False,  # Load immediately for predictable startup
        }

        cuda_available = _cuda_provider_is_available()
        if EMBEDDING_DEVICE == "cuda" and not cuda_available:
            logger.warning(
                "EMBEDDING_DEVICE=gpu/cuda but CUDAExecutionProvider unavailable. "
                "Install GPU support: pip install 'semantic-cache-mcp[gpu]'. "
                "Falling back to CPU."
            )
        use_cuda = cuda_available and EMBEDDING_DEVICE in ("cuda", "auto")
        if use_cuda:
            # Configure CUDA provider to limit VRAM arena growth:
            # - kSameAsRequested: allocate exact size needed (default kNextPowerOfTwo
            #   doubles on each expansion, wasting VRAM)
            # - gpu_mem_limit: 3x ONNX file size (weights + activations + overhead),
            #   defaults to 4GB for built-in models
            # - cudnn_conv_algo_search: HEURISTIC avoids cuDNN workspace bloat
            init_kwargs["providers"] = [
                (
                    "CUDAExecutionProvider",
                    {
                        "arena_extend_strategy": "kSameAsRequested",
                        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB default
                        "cudnn_conv_algo_search": "HEURISTIC",
                    },
                ),
            ]

        def _try_init(kwargs: dict[str, Any]) -> TextEmbedding:
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")
                model = TextEmbedding(**kwargs)

            # fastembed may emit warning and silently fall back to CPU if CUDA init fails.
            # Guard on "providers" in kwargs (not the outer `use_cuda`) so that after
            # a CPU fallback removes providers, we don't raise on a stale closure variable.
            if "providers" in kwargs and any(
                "Attempt to set CUDAExecutionProvider failed" in str(w.message)
                for w in captured_warnings
            ):
                raise RuntimeError("CUDA provider initialization failed at runtime")
            return model

        def _init_or_fallback(kwargs: dict[str, Any]) -> tuple[TextEmbedding, str]:
            """Try CUDA init with retries, fall back to CPU on non-ValueError failures."""
            try:
                if use_cuda:
                    # Only retry RuntimeError (transient CUDA init failure).
                    # TypeError (unsupported providers arg) propagates immediately.
                    model = retry(
                        lambda: _try_init(kwargs),
                        delays=(0.1, 0.2, 0.4),
                        exceptions=(RuntimeError,),
                        label="CUDA embedding init",
                    )
                    return model, "CUDAExecutionProvider"
                return _try_init(kwargs), "CPUExecutionProvider"
            except ValueError:
                raise  # Custom model registration handled by caller
            except Exception as e:
                if "providers" not in kwargs:
                    raise
                logger.warning(f"{e} — falling back to CPU")
                kwargs.pop("providers", None)
                return _try_init(kwargs), "CPUExecutionProvider"

        try:
            _embedding_model, _execution_provider = _init_or_fallback(init_kwargs)
        except ValueError as e:
            if "not supported" not in str(e):
                raise
            # Model not in fastembed's built-in list — auto-register from HuggingFace Hub
            model_info = _register_custom_model(FASTEMBED_MODEL)
            # Recalculate gpu_mem_limit from actual ONNX size: 3x for activations + overhead
            if use_cuda and model_info.get("onnx_size_bytes"):
                # 3x model size + 512MB headroom for attention spikes and kernel workspace
                mem_limit = int(model_info["onnx_size_bytes"]) * 3 + 512 * 1024 * 1024  # type: ignore[arg-type]
                init_kwargs["providers"][0][1]["gpu_mem_limit"] = mem_limit
                logger.info(f"GPU memory limit set to {mem_limit / (1024**3):.1f}GB from ONNX size")
            _embedding_model, _execution_provider = _init_or_fallback(init_kwargs)

        logger.info(f"Embedding execution provider: {_execution_provider}")
        logger.info("Embedding model loaded")

    return _embedding_model


def warmup() -> None:
    """Warmup the embedding model with realistic workloads.

    Runs multiple embeddings of varying length to prime ONNX Runtime's
    CUDA kernel cache, memory allocator, and batch inference path.
    A single-word warmup leaves these cold, causing the first real
    request to pay a ~600ms penalty.
    """
    global _model_ready, _embedding_dim

    if _model_ready:
        return

    logger.info("Warming up embedding model...")
    model = _get_model()

    # Representative texts at different lengths to warm up ONNX kernel cache
    # and memory allocator.
    warmup_texts = [
        "def compute_hash(data: bytes) -> str",
        (
            "The architecture of modern distributed systems relies "
            "heavily on eventual consistency models, where nodes communicate through "
            "asynchronous message passing. Each node maintains its own state replica "
            "and conflicts are resolved through vector clocks or CRDTs."
        ),
        (
            "Database indexing strategies significantly impact query performance. "
            "B-tree indexes provide logarithmic lookup time for range queries while "
            "hash indexes offer constant-time point lookups. Composite indexes can "
            "serve multiple query patterns but increase write amplification. The "
            "query optimizer must consider index selectivity and cardinality."
        ),
    ]

    results = list(model.embed(warmup_texts))
    if results:
        _embedding_dim = len(results[0])

    _model_ready = True
    logger.info(f"Embedding model warmed up (dim={_embedding_dim})")
