"""Tests for embedding provider selection and fallback behavior."""

from __future__ import annotations

import sys
import types

import pytest


def _reset_embedding_state() -> None:
    from semantic_cache_mcp.core import embeddings

    embeddings._embedding_model = None
    embeddings._model_ready = False
    embeddings._execution_provider = "unknown"


@pytest.fixture(autouse=True)
def reset_embedding_state_fixture():
    """Ensure global embedding singleton state does not leak between tests."""
    _reset_embedding_state()
    yield
    _reset_embedding_state()


class TestEmbeddingProviders:
    """Validate GPU/CPU provider selection paths."""

    def test_uses_cuda_provider_when_available(self, monkeypatch) -> None:
        """Model init should request CUDA provider when runtime supports it."""
        from semantic_cache_mcp.core import embeddings

        calls: list[dict] = []

        class FakeTextEmbedding:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def embed(self, items):
                return []

        monkeypatch.setattr(embeddings, "_cuda_provider_is_available", lambda: True)
        monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))

        embeddings._get_model()

        assert len(calls) == 1
        # Provider is a tuple with arena config options
        providers = calls[0]["providers"]
        assert len(providers) == 1
        provider_name = providers[0] if isinstance(providers[0], str) else providers[0][0]
        assert provider_name == "CUDAExecutionProvider"
        assert embeddings._execution_provider == "CUDAExecutionProvider"

    def test_falls_back_to_cpu_when_cuda_not_available(self, monkeypatch) -> None:
        """Model init should not set CUDA providers when unavailable."""
        from semantic_cache_mcp.core import embeddings

        calls: list[dict] = []

        class FakeTextEmbedding:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def embed(self, items):
                return []

        monkeypatch.setattr(embeddings, "_cuda_provider_is_available", lambda: False)
        monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))

        embeddings._get_model()

        assert len(calls) == 1
        assert "providers" not in calls[0]
        assert embeddings._execution_provider == "CPUExecutionProvider"

    def test_falls_back_if_providers_argument_unsupported(self, monkeypatch) -> None:
        """If providers arg is unsupported, retry initialization without it."""
        from semantic_cache_mcp.core import embeddings

        calls: list[dict] = []

        class FakeTextEmbedding:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                if "providers" in kwargs:
                    raise TypeError("unexpected providers")

            def embed(self, items):
                return []

        monkeypatch.setattr(embeddings, "_cuda_provider_is_available", lambda: True)
        monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))

        embeddings._get_model()

        assert len(calls) == 2
        assert "providers" in calls[0]
        assert "providers" not in calls[1]
        assert embeddings._execution_provider == "CPUExecutionProvider"
