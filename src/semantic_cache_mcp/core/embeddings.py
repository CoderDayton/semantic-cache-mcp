"""FastEmbed-based embedding service with local model inference.

Provides fast, local text embeddings using ONNX models without external API calls.
Uses CUDA provider when available and falls back to CPU automatically.
"""

from __future__ import annotations

import array
import logging
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastembed import TextEmbedding

from ..config import CACHE_DIR, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)

# Model configuration — bge-small is 6x smaller than nomic (33M vs 137M params),
# 384-dim vs 768-dim, fast on CPU, and quality is sufficient for file similarity.
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
FASTEMBED_CACHE_DIR = CACHE_DIR / "models"

# Singleton instance
_embedding_model: TextEmbedding | None = None
_model_ready: bool = False
_embedding_dim: int = 0
_execution_provider: str = "unknown"


def _cuda_provider_is_available() -> bool:
    """Return True when ONNX Runtime reports CUDA provider support."""
    try:
        import onnxruntime as ort
    except Exception as e:
        logger.info(f"ONNX Runtime not available for provider check: {e}")
        return False

    try:
        providers = ort.get_available_providers()
    except Exception as e:
        logger.warning(f"Could not query ONNX Runtime providers: {e}")
        return False

    has_cuda = "CUDAExecutionProvider" in providers
    if has_cuda:
        logger.info("CUDAExecutionProvider detected; enabling GPU embeddings")
    else:
        logger.info(f"CUDAExecutionProvider not detected (available: {providers})")
    return has_cuda


def _get_model() -> TextEmbedding:
    """Get or initialize the embedding model singleton."""
    global _embedding_model, _execution_provider

    if _embedding_model is None:
        from fastembed import TextEmbedding

        logger.info(f"Loading embedding model: {FASTEMBED_MODEL}")
        logger.info(f"Model cache directory: {FASTEMBED_CACHE_DIR}")

        # Ensure cache directory exists
        FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        init_kwargs: dict[str, Any] = {
            "model_name": FASTEMBED_MODEL,
            "cache_dir": str(FASTEMBED_CACHE_DIR),
            "lazy_load": False,  # Load immediately for predictable startup
        }

        use_cuda = EMBEDDING_DEVICE == "cuda" or (
            EMBEDDING_DEVICE == "auto" and _cuda_provider_is_available()
        )
        if use_cuda:
            # Configure CUDA provider to limit VRAM arena growth:
            # - kSameAsRequested: allocate exact size needed (default kNextPowerOfTwo
            #   doubles on each expansion, wasting ~3GB for a 137M param model)
            # - gpu_mem_limit: 4GB cap — model needs ~1.5GB weights + ~500MB activations
            #   for max sequence length; headroom for attention matmul spikes
            # - cudnn_conv_algo_search: HEURISTIC avoids cuDNN workspace bloat
            init_kwargs["providers"] = [
                (
                    "CUDAExecutionProvider",
                    {
                        "arena_extend_strategy": "kSameAsRequested",
                        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # 4GB
                        "cudnn_conv_algo_search": "HEURISTIC",
                    },
                ),
            ]

        try:
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")
                _embedding_model = TextEmbedding(**init_kwargs)

            # fastembed may emit warning and silently fall back to CPU if CUDA init fails.
            cuda_warning = any(
                "Attempt to set CUDAExecutionProvider failed" in str(w.message)
                for w in captured_warnings
            )
            if use_cuda and cuda_warning:
                raise RuntimeError("CUDA provider initialization failed at runtime")

            _execution_provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
        except TypeError:
            # Backward compatibility: some fastembed versions may not accept providers.
            if "providers" not in init_kwargs:
                raise
            logger.warning("TextEmbedding does not accept providers arg, falling back")
            init_kwargs.pop("providers", None)
            _embedding_model = TextEmbedding(**init_kwargs)
            _execution_provider = "CPUExecutionProvider"
        except Exception as e:
            # If CUDA init fails (driver/runtime mismatch), retry on CPU providers.
            if "providers" not in init_kwargs:
                raise
            logger.warning(f"CUDA initialization failed ({e}), falling back to CPU provider")
            init_kwargs.pop("providers", None)
            _embedding_model = TextEmbedding(**init_kwargs)
            _execution_provider = "CPUExecutionProvider"

        logger.info(f"Embedding execution provider: {_execution_provider}")
        logger.info("Embedding model loaded successfully")

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


def embed(text: str) -> array.array[float] | None:
    """Generate embedding for text.

    Args:
        text: Text to embed (truncated to 8000 chars)

    Returns:
        Embedding as array.array or None on error
    """
    try:
        model = _get_model()

        truncated = text[:8000]

        embeddings = list(model.embed([truncated]))
        if embeddings:
            return array.array("f", embeddings[0].tolist())
        return None

    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def embed_query(text: str) -> array.array[float] | None:
    """Generate embedding for a search query.

    Uses 'Represent this sentence:' prefix for bge retrieval.

    Args:
        text: Query text to embed

    Returns:
        Embedding as array.array or None on error
    """
    try:
        model = _get_model()

        # Official bge query instruction for retrieval tasks
        prefixed = f"Represent this sentence for searching relevant passages: {text[:8000]}"

        embeddings = list(model.embed([prefixed]))
        if embeddings:
            return array.array("f", embeddings[0].tolist())
        return None

    except Exception as e:
        logger.warning(f"Query embedding failed: {e}")
        return None


def get_embedding_dim() -> int:
    """Return the embedding dimension of the loaded model.

    Falls back to 0 if the model hasn't been warmed up yet.
    """
    return _embedding_dim


def get_model_info() -> dict[str, str | int]:
    """Get information about the loaded model.

    Returns:
        Dict with model name, dimension, provider, and readiness
    """
    return {
        "model": FASTEMBED_MODEL,
        "dim": _embedding_dim,
        "cache_dir": str(FASTEMBED_CACHE_DIR),
        "provider": _execution_provider,
        "ready": _model_ready,
    }
