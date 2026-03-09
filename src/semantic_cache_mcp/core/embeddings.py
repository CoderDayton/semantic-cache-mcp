"""FastEmbed-based embedding service with local model inference.

Provides fast, local text embeddings using ONNX models without external API calls.
Uses CUDA provider when available and falls back to CPU automatically.
"""

from __future__ import annotations

import array
import contextlib
import hashlib
import json
import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from fastembed import TextEmbedding

from ..config import CACHE_DIR, EMBEDDING_DEVICE, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Model configuration — default bge-small is 6x smaller than nomic (33M vs 137M params),
# 384-dim vs 768-dim, fast on CPU, and quality is sufficient for file similarity.
# Configurable via EMBEDDING_MODEL env var for users who want different models.
FASTEMBED_MODEL = EMBEDDING_MODEL
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


def _register_custom_model(model_name: str) -> dict[str, int | str | None]:
    """Auto-register an unsupported HuggingFace model with fastembed.

    Downloads config.json and pooling config from HF Hub to infer
    embedding dim, pooling type, and ONNX model path, then registers
    via TextEmbedding.add_custom_model().

    Returns model info dict with dim, model_file, and onnx_size_bytes.
    """
    from fastembed import TextEmbedding
    from fastembed.common.model_description import ModelSource, PoolingType
    from huggingface_hub import hf_hub_download, list_repo_tree

    logger.info(f"Model {model_name} not in fastembed defaults, registering from HuggingFace Hub")

    # 1. Get embedding dimension from config.json
    try:
        config_path = hf_hub_download(model_name, "config.json", cache_dir=str(FASTEMBED_CACHE_DIR))
    except Exception as exc:
        raise ValueError(
            f"Could not download config for {model_name} from HuggingFace Hub. "
            f"Check your network or use a built-in model. ({exc})"
        ) from exc
    with open(config_path) as f:
        config = json.load(f)
    dim = config.get("hidden_size", 768)

    # 2. Infer pooling type from sentence-transformers config
    pooling = PoolingType.MEAN
    try:
        pooling_path = hf_hub_download(
            model_name, "1_Pooling/config.json", cache_dir=str(FASTEMBED_CACHE_DIR)
        )
        with open(pooling_path) as f:
            pooling_config = json.load(f)
        if pooling_config.get("pooling_mode_cls_token"):
            pooling = PoolingType.CLS
    except Exception:  # noqa: BLE001 — pooling config is optional
        pass

    # 3. Find the ONNX model file and its expected SHA256 from repo metadata
    model_file = "onnx/model.onnx"
    expected_sha256: str | None = None
    onnx_size_bytes: int | None = None
    try:
        entries = list(list_repo_tree(model_name, recursive=True))
        onnx_entries = sorted(
            [e for e in entries if hasattr(e, "path") and e.path.endswith(".onnx")],
            key=lambda e: e.path,
        )
        if not onnx_entries:
            raise ValueError(
                f"Model {model_name} has no ONNX export on HuggingFace. "
                f"Use a model with ONNX support (look for the 'onnx' tag)."
            )
        # Prefer the base model.onnx over quantized/optimized variants
        preferred = [e for e in onnx_entries if e.path.endswith("/model.onnx")]
        chosen = preferred[0] if preferred else onnx_entries[0]
        model_file = chosen.path
        # LFS entries carry a sha256; non-LFS carry blob_id (git SHA-1)
        if hasattr(chosen, "lfs") and chosen.lfs is not None:
            expected_sha256 = chosen.lfs.sha256
            if hasattr(chosen.lfs, "size"):
                onnx_size_bytes = chosen.lfs.size
    except ValueError:
        raise
    except Exception as exc:
        logger.warning(f"Could not list repo files for {model_name}: {exc}")

    # 4. Pre-download the ONNX model file + tokenizer so fastembed finds them
    #    in its HF cache snapshot directory at init time.
    required_files = [
        model_file,
        model_file + "_data",  # external weights for large models (e.g. model.onnx_data)
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ]
    for fname in required_files:
        with contextlib.suppress(Exception):
            hf_hub_download(model_name, fname, cache_dir=str(FASTEMBED_CACHE_DIR))

    # 5. Verify ONNX model integrity against HF-reported SHA256
    if expected_sha256:
        onnx_path = hf_hub_download(
            model_name, model_file, cache_dir=str(FASTEMBED_CACHE_DIR), local_files_only=True
        )
        sha256 = hashlib.sha256()
        with open(onnx_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"ONNX model hash mismatch for {model_name}: "
                f"expected {expected_sha256}, got {actual}"
            )

    logger.info(f"Custom model: dim={dim}, pooling={pooling.name}, onnx={model_file}")

    TextEmbedding.add_custom_model(
        model=model_name,
        pooling=pooling,
        normalization=True,
        sources=ModelSource(hf=model_name),
        dim=dim,
        model_file=model_file,
    )

    return {"dim": dim, "model_file": model_file, "onnx_size_bytes": onnx_size_bytes}


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
            cuda_warning = any(
                "Attempt to set CUDAExecutionProvider failed" in str(w.message)
                for w in captured_warnings
            )
            if use_cuda and cuda_warning:
                raise RuntimeError("CUDA provider initialization failed at runtime")
            return model

        try:
            _embedding_model = _try_init(init_kwargs)
            _execution_provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
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
            try:
                _embedding_model = _try_init(init_kwargs)
                _execution_provider = (
                    "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
                )
            except Exception as cuda_err:
                if "providers" not in init_kwargs:
                    raise
                logger.warning(
                    f"CUDA init failed for custom model ({cuda_err}), falling back to CPU"
                )
                init_kwargs.pop("providers", None)
                _embedding_model = _try_init(init_kwargs)
                _execution_provider = "CPUExecutionProvider"
        except TypeError:
            # Backward compatibility: some fastembed versions may not accept providers.
            if "providers" not in init_kwargs:
                raise
            logger.warning("TextEmbedding does not accept providers arg, falling back")
            init_kwargs.pop("providers", None)
            _embedding_model = _try_init(init_kwargs)
            _execution_provider = "CPUExecutionProvider"
        except Exception as e:
            # If CUDA init fails (driver/runtime mismatch), retry on CPU providers.
            if "providers" not in init_kwargs:
                raise
            logger.warning(f"CUDA initialization failed ({e}), falling back to CPU provider")
            init_kwargs.pop("providers", None)
            _embedding_model = _try_init(init_kwargs)
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
            result = array.array("f")
            result.frombytes(embeddings[0].astype(np.float32).tobytes())
            return result
        return None

    except Exception as e:
        logger.warning(f"Embedding failed: {e}")
        return None


def embed_batch(texts: list[str]) -> list[array.array[float] | None]:
    """Generate embeddings for multiple texts in a single model call.

    Amortizes ONNX Runtime overhead across N texts — critical for
    batch_smart_read where N files need embedding on first cache miss.

    Args:
        texts: Texts to embed (each truncated to 8000 chars)

    Returns:
        List of embeddings (same length as input, None entries on failure)
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
            result = array.array("f")
            result.frombytes(embeddings[0].astype(np.float32).tobytes())
            return result
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
