from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _cuda_provider_is_available() -> bool:
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
