"""FastMCP instance and application lifespan."""

from __future__ import annotations

import contextlib
import logging
import sys

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan

from ..cache import SemanticCache
from ..core.embeddings import get_model_info, warmup
from ..core.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


@lifespan
async def app_lifespan(server: FastMCP):
    """Initialize cache and embedding model on startup."""
    logger.info("Semantic cache MCP server starting...")

    cache: SemanticCache | None = None

    # Redirect stdout → stderr during initialization to prevent third-party
    # libraries (fastembed, onnxruntime) from printing to stdout and corrupting
    # the stdio MCP transport. The lifespan runs BEFORE stdio_server() captures
    # sys.stdout.buffer, so we must restore before yielding.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            logger.info("Initializing tokenizer...")
            get_tokenizer()

            logger.info("Initializing embedding model...")
            warmup()

            model_info = get_model_info()
            if not model_info.get("ready", False):
                logger.error(
                    "Embedding model failed to initialize. "
                    "Semantic similarity features will be disabled. "
                    "Check network connectivity and disk space."
                )
            else:
                logger.info(f"Embedding model ready: {model_info['model']}")

            cache = SemanticCache()
            logger.info("Semantic cache MCP server started")
        except Exception:
            logger.exception("Failed to initialize semantic cache")
            raise

    if cache is None:
        raise RuntimeError("Cache failed to initialize")

    try:
        yield {"cache": cache}
    finally:
        cache.metrics.persist()
        logger.info("Semantic cache MCP server stopped")


mcp = FastMCP("semantic-cache-mcp", lifespan=app_lifespan)
