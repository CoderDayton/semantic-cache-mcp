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


def _get_client() -> Any:
    global _client

    if _client is None:
        from openai import OpenAI  # noqa: PLC0415

        _client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client


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
            out[index] = array.array("f", item.embedding)
    return out
