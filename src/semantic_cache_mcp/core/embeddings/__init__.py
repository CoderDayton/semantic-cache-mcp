from __future__ import annotations

import array
import logging

import numpy as np

from ._constants import FASTEMBED_CACHE_DIR
from ._cuda import _cuda_provider_is_available
from ._model import (
    FASTEMBED_MODEL,
    _embedding_dim,
    _embedding_model,
    _execution_provider,
    _get_model,
    _model_ready,
    warmup,
)
from ._registry import _register_custom_model

logger = logging.getLogger(__name__)


def embed(text: str) -> array.array[float] | None:
    """Generate embedding, truncated to 8000 chars."""
    try:
        model = _get_model()

        truncated = text[:8000]

        embeddings = list(model.embed([truncated]))
        if embeddings:
            result = array.array("f")
            result.frombytes(embeddings[0].astype(np.float32).tobytes())
            return result
        return None

    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def embed_batch(texts: list[str]) -> list[array.array[float] | None]:
    """Embed N texts in one model call.

    Amortizes ONNX Runtime overhead — critical for batch_smart_read where N files
    need embedding on first cache miss. Returns same-length list; None on failure.
    """
    if not texts:
        return []
    try:
        model = _get_model()
        truncated = [t[:8000] for t in texts]
        results = list(model.embed(truncated))
        out: list[array.array[float] | None] = []
        for r in results:
            a = array.array("f")
            a.frombytes(r.astype(np.float32).tobytes())
            out.append(a)
        return out
    except Exception as e:
        logger.warning(f"Batch embedding failed: {e}")
        return [None] * len(texts)


def embed_query(text: str) -> array.array[float] | None:
    """Embed a search query with the bge retrieval prefix."""
    try:
        model = _get_model()

        # Official bge query instruction for retrieval tasks
        prefixed = f"Represent this sentence for searching relevant passages: {text[:8000]}"

        embeddings = list(model.embed([prefixed]))
        if embeddings:
            result = array.array("f")
            result.frombytes(embeddings[0].astype(np.float32).tobytes())
            return result
        return None

    except Exception as e:
        logger.warning(f"Query embedding failed: {e}")
        return None


def get_embedding_dim() -> int:
    """Return embedding dimension; 0 if model not yet warmed up."""
    # Must re-import to read current singleton value
    import semantic_cache_mcp.core.embeddings._model as _m

    return _m._embedding_dim


def get_model_info() -> dict[str, str | int]:
    """Return model name, dim, provider, cache_dir, and readiness flag."""
    import semantic_cache_mcp.core.embeddings._model as _m

    return {
        "model": FASTEMBED_MODEL,
        "dim": _m._embedding_dim,
        "cache_dir": str(FASTEMBED_CACHE_DIR),
        "provider": _m._execution_provider,
        "ready": _m._model_ready,
    }


__all__ = [
    "embed",
    "embed_batch",
    "embed_query",
    "get_embedding_dim",
    "get_model_info",
    "warmup",
    # Private exports needed by tests and internal modules
    "_cuda_provider_is_available",
    "_register_custom_model",
    "_get_model",
    "FASTEMBED_MODEL",
    "FASTEMBED_CACHE_DIR",
    "_embedding_model",
    "_model_ready",
    "_embedding_dim",
    "_execution_provider",
]
