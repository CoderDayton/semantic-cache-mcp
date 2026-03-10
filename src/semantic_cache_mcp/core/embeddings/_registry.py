"""Auto-registration of custom HuggingFace embedding models with fastembed."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging

from ._constants import FASTEMBED_CACHE_DIR as _FASTEMBED_CACHE_DIR

logger = logging.getLogger(__name__)


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
        config_path = hf_hub_download(
            model_name, "config.json", cache_dir=str(_FASTEMBED_CACHE_DIR)
        )
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
            model_name, "1_Pooling/config.json", cache_dir=str(_FASTEMBED_CACHE_DIR)
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
            hf_hub_download(model_name, fname, cache_dir=str(_FASTEMBED_CACHE_DIR))

    # 5. Verify ONNX model integrity against HF-reported SHA256
    if expected_sha256:
        onnx_path = hf_hub_download(
            model_name, model_file, cache_dir=str(_FASTEMBED_CACHE_DIR), local_files_only=True
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
