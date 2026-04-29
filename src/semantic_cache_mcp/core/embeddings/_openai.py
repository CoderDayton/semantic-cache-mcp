from __future__ import annotations

import array
import logging
from typing import Any

from ...config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_EMBEDDING_DIMENSIONS,
    OPENAI_EMBEDDING_DIMENSIONS_RAW,
    OPENAI_EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

_client: Any | None = None
_embedding_dim: int = OPENAI_EMBEDDING_DIMENSIONS or 0


def _get_client() -> Any:
    global _client

    if _client is None:
        from openai import OpenAI  # noqa: PLC0415

        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


def get_embedding_dim() -> int:
    return _embedding_dim


def _to_array(values: list[float]) -> array.array[float]:
    global _embedding_dim

    result = array.array("f", values)
    actual_dim = len(result)
    if OPENAI_EMBEDDING_DIMENSIONS is not None and actual_dim != OPENAI_EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"OpenAI embedding dimension mismatch: got {actual_dim}, "
            f"expected {OPENAI_EMBEDDING_DIMENSIONS}"
        )
    if actual_dim > 0:
        _embedding_dim = actual_dim
    return result


def embed_texts(texts: list[str]) -> list[array.array[float] | None]:
    if not texts:
        return []

    params: dict[str, Any] = {
        "model": OPENAI_EMBEDDING_MODEL,
        "input": [text[:8000] for text in texts],
    }
    if OPENAI_EMBEDDING_DIMENSIONS_RAW is not None:
        params["dimensions"] = OPENAI_EMBEDDING_DIMENSIONS

    response = _get_client().embeddings.create(**params)
    out: list[array.array[float] | None] = [None] * len(texts)
    for item in response.data:
        index = int(item.index)
        if 0 <= index < len(out):
            out[index] = _to_array(item.embedding)
    return out
