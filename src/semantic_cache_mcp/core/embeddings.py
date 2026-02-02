"""FastEmbed-based embedding service with local model inference.

Provides fast, local text embeddings using ONNX models without external API calls.
"""

from __future__ import annotations

import array
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastembed import TextEmbedding

from ..config import CACHE_DIR

logger = logging.getLogger(__name__)

# Model configuration
FASTEMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
FASTEMBED_CACHE_DIR = CACHE_DIR / "models"

# Singleton instance
_embedding_model: TextEmbedding | None = None
_model_ready: bool = False


def _get_model() -> TextEmbedding:
    """Get or initialize the embedding model singleton."""
    global _embedding_model

    if _embedding_model is None:
        from fastembed import TextEmbedding

        logger.info(f"Loading embedding model: {FASTEMBED_MODEL}")
        logger.info(f"Model cache directory: {FASTEMBED_CACHE_DIR}")

        # Ensure cache directory exists
        FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        _embedding_model = TextEmbedding(
            model_name=FASTEMBED_MODEL,
            cache_dir=str(FASTEMBED_CACHE_DIR),
            lazy_load=False,  # Load immediately for predictable startup
        )
        logger.info("Embedding model loaded successfully")

    return _embedding_model


def warmup() -> None:
    """Warmup the embedding model by running a test embedding.

    Call this at startup to ensure the model is fully loaded and
    cached before handling requests.
    """
    global _model_ready

    if _model_ready:
        return

    logger.info("Warming up embedding model...")
    model = _get_model()

    # Run a test embedding to ensure everything is loaded
    _ = list(model.embed(["warmup"]))

    _model_ready = True
    logger.info("Embedding model warmed up and ready")


def embed(text: str) -> array.array[float] | None:
    """Generate embedding for text.

    Args:
        text: Text to embed (uses first 8192 chars for nomic model)

    Returns:
        Embedding as array.array or None on error
    """
    try:
        model = _get_model()

        # nomic model supports 8192 tokens, but we limit text for efficiency
        # Add search_document prefix as recommended by nomic
        prefixed_text = f"search_document: {text[:8000]}"

        embeddings = list(model.embed([prefixed_text]))
        if embeddings:
            return array.array("f", embeddings[0].tolist())
        return None

    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def embed_query(text: str) -> array.array[float] | None:
    """Generate embedding for a search query.

    Uses search_query prefix for better retrieval performance.

    Args:
        text: Query text to embed

    Returns:
        Embedding as array.array or None on error
    """
    try:
        model = _get_model()

        # Use search_query prefix for queries
        prefixed_text = f"search_query: {text[:8000]}"

        embeddings = list(model.embed([prefixed_text]))
        if embeddings:
            return array.array("f", embeddings[0].tolist())
        return None

    except Exception as e:
        logger.warning(f"Query embedding failed: {e}")
        return None


def get_model_info() -> dict[str, str | int]:
    """Get information about the loaded model.

    Returns:
        Dict with model name, dimension, and cache location
    """
    return {
        "model": FASTEMBED_MODEL,
        "cache_dir": str(FASTEMBED_CACHE_DIR),
        "ready": _model_ready,
    }
