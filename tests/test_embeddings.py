"""Tests for embedding provider selection and fallback behavior."""

from __future__ import annotations

import array
import sys
import types

import pytest


def _reset_embedding_state() -> None:
    from semantic_cache_mcp.core.embeddings import _model as _emb_model
    from semantic_cache_mcp.core.embeddings import _openai as _openai_model

    _emb_model._embedding_model = None
    _emb_model._model_ready = False
    _emb_model._embedding_dim = 0
    _emb_model._execution_provider = "unknown"
    _openai_model._client = None


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
        from semantic_cache_mcp.core.embeddings import _model as _emb_model

        calls: list[dict] = []

        class FakeTextEmbedding:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def embed(self, items):
                return []

        monkeypatch.setattr(_emb_model, "EMBEDDING_DEVICE", "auto")
        monkeypatch.setattr(_emb_model, "_cuda_provider_is_available", lambda: True)
        monkeypatch.setitem(
            sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        )

        embeddings._get_model()

        assert len(calls) == 1
        # Provider is a tuple with arena config options
        providers = calls[0]["providers"]
        assert len(providers) == 1
        provider_name = providers[0] if isinstance(providers[0], str) else providers[0][0]
        assert provider_name == "CUDAExecutionProvider"
        assert _emb_model._execution_provider == "CUDAExecutionProvider"

    def test_falls_back_to_cpu_when_cuda_not_available(self, monkeypatch) -> None:
        """Model init should not set CUDA providers when unavailable."""
        from semantic_cache_mcp.core import embeddings
        from semantic_cache_mcp.core.embeddings import _model as _emb_model

        calls: list[dict] = []

        class FakeTextEmbedding:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def embed(self, items):
                return []

        monkeypatch.setattr(_emb_model, "_cuda_provider_is_available", lambda: False)
        monkeypatch.setitem(
            sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        )

        embeddings._get_model()

        assert len(calls) == 1
        assert "providers" not in calls[0]
        assert _emb_model._execution_provider == "CPUExecutionProvider"

    def test_falls_back_if_providers_argument_unsupported(self, monkeypatch) -> None:
        """If providers arg is unsupported, retry initialization without it."""
        from semantic_cache_mcp.core import embeddings
        from semantic_cache_mcp.core.embeddings import _model as _emb_model

        calls: list[dict] = []

        class FakeTextEmbedding:
            def __init__(self, **kwargs):
                calls.append(kwargs)
                if "providers" in kwargs:
                    raise TypeError("unexpected providers")

            def embed(self, items):
                return []

        monkeypatch.setattr(_emb_model, "EMBEDDING_DEVICE", "auto")
        monkeypatch.setattr(_emb_model, "_cuda_provider_is_available", lambda: True)
        monkeypatch.setitem(
            sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        )

        embeddings._get_model()

        assert len(calls) == 2
        assert "providers" in calls[0]
        assert "providers" not in calls[1]
        assert _emb_model._execution_provider == "CPUExecutionProvider"


class TestOpenAIEmbeddings:
    """Validate OpenAI-compatible embedding routing."""

    def test_openai_embed_skips_local_model(self, monkeypatch) -> None:
        from semantic_cache_mcp.core import embeddings

        monkeypatch.setattr(embeddings, "OPENAI_EMBEDDINGS_ENABLED", True)
        monkeypatch.setattr(embeddings, "_get_model", lambda: pytest.fail("loaded local model"))
        monkeypatch.setattr(
            embeddings,
            "_openai_embed_texts",
            lambda texts: [array.array("f", [1.0, 2.0, 3.0])],
        )

        result = embeddings.embed("hello")

        assert result == array.array("f", [1.0, 2.0, 3.0])

    def test_openai_query_does_not_apply_bge_prefix(self, monkeypatch) -> None:
        from semantic_cache_mcp.core import embeddings

        calls: list[list[str]] = []

        def fake_embed_texts(texts: list[str]) -> list[array.array[float] | None]:
            calls.append(texts)
            return [array.array("f", [0.5])]

        monkeypatch.setattr(embeddings, "OPENAI_EMBEDDINGS_ENABLED", True)
        monkeypatch.setattr(embeddings, "_openai_embed_texts", fake_embed_texts)

        assert embeddings.embed_query("find cache code") == array.array("f", [0.5])
        assert calls == [["find cache code"]]

    def test_openai_model_info_uses_configured_remote_metadata(self, monkeypatch) -> None:
        from semantic_cache_mcp.core import embeddings

        monkeypatch.setattr(embeddings, "OPENAI_EMBEDDINGS_ENABLED", True)
        monkeypatch.setattr(embeddings, "OPENAI_EMBEDDING_MODEL", "nomic-embed-text")
        monkeypatch.setattr(embeddings, "OPENAI_EMBEDDING_DIMENSIONS", 768)

        assert embeddings.get_embedding_dim() == 768
        assert embeddings.get_model_info() == {
            "model": "nomic-embed-text",
            "dim": 768,
            "cache_dir": "",
            "provider": "OpenAI",
            "ready": True,
        }

    def test_openai_warmup_does_not_load_local_model(self, monkeypatch) -> None:
        from semantic_cache_mcp.core.embeddings import _model as _emb_model

        monkeypatch.setattr(_emb_model, "OPENAI_EMBEDDINGS_ENABLED", True)
        monkeypatch.setattr(_emb_model, "OPENAI_EMBEDDING_DIMENSIONS", 768)
        monkeypatch.setattr(_emb_model, "_get_model", lambda: pytest.fail("loaded local model"))

        _emb_model.warmup()

        assert _emb_model._model_ready is True
        assert _emb_model._embedding_dim == 768
        assert _emb_model._execution_provider == "OpenAI"

    def test_openai_adapter_sends_dimensions_only_when_explicit(self, monkeypatch) -> None:
        from semantic_cache_mcp.core.embeddings import _openai

        calls: list[dict] = []

        class FakeEmbeddings:
            def create(self, **kwargs):
                calls.append(kwargs)
                item = types.SimpleNamespace(index=0, embedding=[0.1, 0.2])
                return types.SimpleNamespace(data=[item])

        monkeypatch.setattr(_openai, "OPENAI_EMBEDDING_DIMENSIONS_RAW", None)
        monkeypatch.setattr(
            _openai, "_get_client", lambda: types.SimpleNamespace(embeddings=FakeEmbeddings())
        )

        assert _openai.embed_texts(["abc"]) == [array.array("f", [0.1, 0.2])]
        assert "dimensions" not in calls[0]


class TestCustomModelRegistration:
    """Validate auto-registration of HuggingFace models not in fastembed defaults."""

    def test_register_custom_model_happy_path(self, monkeypatch, tmp_path) -> None:
        """Should download config, find ONNX, verify SHA256, and call add_custom_model."""
        from unittest.mock import MagicMock

        from semantic_cache_mcp.core import embeddings

        # --- Fake HF Hub responses ---
        config_file = tmp_path / "config.json"
        config_file.write_text('{"hidden_size": 768}')

        pooling_file = tmp_path / "1_Pooling" / "config.json"
        pooling_file.parent.mkdir()
        pooling_file.write_text('{"pooling_mode_cls_token": true}')

        # Fake ONNX file with known SHA256
        onnx_file = tmp_path / "onnx" / "model.onnx"
        onnx_file.parent.mkdir()
        onnx_content = b"fake-onnx-model-data"
        onnx_file.write_bytes(onnx_content)

        import hashlib

        expected_sha = hashlib.sha256(onnx_content).hexdigest()

        def fake_hf_download(repo_id: str, filename: str, **kwargs) -> str:
            if filename == "config.json":
                return str(config_file)
            if filename == "1_Pooling/config.json":
                return str(pooling_file)
            if filename == "onnx/model.onnx":
                return str(onnx_file)
            return str(tmp_path / filename)

        # Fake repo tree entry with LFS sha256
        onnx_entry = types.SimpleNamespace(
            path="onnx/model.onnx",
            lfs=types.SimpleNamespace(sha256=expected_sha, size=len(onnx_content)),
        )
        fake_list_repo_tree = MagicMock(return_value=[onnx_entry])

        # Fake fastembed classes
        mock_add_custom_model = MagicMock()
        FakePoolingType = types.SimpleNamespace(
            MEAN=types.SimpleNamespace(name="MEAN"),
            CLS=types.SimpleNamespace(name="CLS"),
        )
        FakeModelSource = MagicMock()

        fake_fastembed = types.SimpleNamespace(
            TextEmbedding=types.SimpleNamespace(add_custom_model=mock_add_custom_model),
        )
        fake_model_desc = types.SimpleNamespace(
            ModelSource=FakeModelSource,
            PoolingType=FakePoolingType,
        )
        fake_hf_hub = types.SimpleNamespace(
            hf_hub_download=fake_hf_download,
            list_repo_tree=fake_list_repo_tree,
        )

        monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)
        monkeypatch.setitem(sys.modules, "fastembed.common.model_description", fake_model_desc)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)

        result = embeddings._register_custom_model("org/custom-model")

        # Verify add_custom_model was called with correct args
        mock_add_custom_model.assert_called_once()
        kw = mock_add_custom_model.call_args
        assert kw.kwargs["model"] == "org/custom-model"
        assert kw.kwargs["dim"] == 768
        assert kw.kwargs["pooling"].name == "CLS"
        assert kw.kwargs["model_file"] == "onnx/model.onnx"

        # Verify returned model info
        assert result["dim"] == 768
        assert result["onnx_size_bytes"] == len(onnx_content)

    def test_register_custom_model_no_onnx_raises(self, monkeypatch, tmp_path) -> None:
        """Should raise ValueError when model has no ONNX files."""
        from unittest.mock import MagicMock

        from semantic_cache_mcp.core import embeddings

        config_file = tmp_path / "config.json"
        config_file.write_text('{"hidden_size": 384}')

        def fake_hf_download(repo_id: str, filename: str, **kwargs) -> str:
            if filename == "config.json":
                return str(config_file)
            if filename == "1_Pooling/config.json":
                raise FileNotFoundError("no pooling config")
            return str(tmp_path / filename)

        # Empty repo — no ONNX files
        fake_list_repo_tree = MagicMock(
            return_value=[types.SimpleNamespace(path="pytorch_model.bin")]
        )

        fake_fastembed = types.SimpleNamespace(
            TextEmbedding=types.SimpleNamespace(add_custom_model=MagicMock()),
        )
        fake_model_desc = types.SimpleNamespace(
            ModelSource=MagicMock(),
            PoolingType=types.SimpleNamespace(MEAN="MEAN", CLS="CLS"),
        )
        fake_hf_hub = types.SimpleNamespace(
            hf_hub_download=fake_hf_download,
            list_repo_tree=fake_list_repo_tree,
        )

        monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)
        monkeypatch.setitem(sys.modules, "fastembed.common.model_description", fake_model_desc)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)

        with pytest.raises(ValueError, match="no ONNX export"):
            embeddings._register_custom_model("org/no-onnx-model")

    def test_register_custom_model_sha256_mismatch_raises(self, monkeypatch, tmp_path) -> None:
        """Should raise ValueError when ONNX file hash doesn't match LFS metadata."""
        from unittest.mock import MagicMock

        from semantic_cache_mcp.core import embeddings

        config_file = tmp_path / "config.json"
        config_file.write_text('{"hidden_size": 384}')

        onnx_file = tmp_path / "onnx" / "model.onnx"
        onnx_file.parent.mkdir()
        onnx_file.write_bytes(b"actual-content")

        def fake_hf_download(repo_id: str, filename: str, **kwargs) -> str:
            if filename == "config.json":
                return str(config_file)
            if filename == "1_Pooling/config.json":
                raise FileNotFoundError
            if filename == "onnx/model.onnx":
                return str(onnx_file)
            return str(tmp_path / filename)

        onnx_entry = types.SimpleNamespace(
            path="onnx/model.onnx",
            lfs=types.SimpleNamespace(
                sha256="0000000000000000000000000000000000000000000000000000000000000000"
            ),
        )
        fake_list_repo_tree = MagicMock(return_value=[onnx_entry])

        fake_fastembed = types.SimpleNamespace(
            TextEmbedding=types.SimpleNamespace(add_custom_model=MagicMock()),
        )
        fake_model_desc = types.SimpleNamespace(
            ModelSource=MagicMock(),
            PoolingType=types.SimpleNamespace(MEAN="MEAN", CLS="CLS"),
        )
        fake_hf_hub = types.SimpleNamespace(
            hf_hub_download=fake_hf_download,
            list_repo_tree=fake_list_repo_tree,
        )

        monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)
        monkeypatch.setitem(sys.modules, "fastembed.common.model_description", fake_model_desc)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)

        with pytest.raises(ValueError, match="hash mismatch"):
            embeddings._register_custom_model("org/tampered-model")

    def test_register_custom_model_network_error_raises(self, monkeypatch, tmp_path) -> None:
        """Should raise ValueError with helpful message when HF Hub is unreachable."""
        from unittest.mock import MagicMock

        from semantic_cache_mcp.core import embeddings

        def fake_hf_download(repo_id: str, filename: str, **kwargs) -> str:
            raise ConnectionError("Network unreachable")

        fake_fastembed = types.SimpleNamespace(
            TextEmbedding=types.SimpleNamespace(add_custom_model=MagicMock()),
        )
        fake_model_desc = types.SimpleNamespace(
            ModelSource=MagicMock(),
            PoolingType=types.SimpleNamespace(MEAN="MEAN", CLS="CLS"),
        )
        fake_hf_hub = types.SimpleNamespace(
            hf_hub_download=fake_hf_download,
            list_repo_tree=MagicMock(),
        )

        monkeypatch.setitem(sys.modules, "fastembed", fake_fastembed)
        monkeypatch.setitem(sys.modules, "fastembed.common.model_description", fake_model_desc)
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_hub)

        with pytest.raises(ValueError, match="Could not download config"):
            embeddings._register_custom_model("org/unreachable-model")
