"""FastMCP instance and application lifespan."""

from __future__ import annotations

import logging

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

    # Warmup tokenizer (loads 200K vocab from disk, ~600ms one-time cost)
    logger.info("Initializing tokenizer...")
    get_tokenizer()

    # Warmup embedding model (downloads if needed, loads into memory)
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

    # Initialize cache
    cache = SemanticCache()
    logger.info("Semantic cache MCP server started")

    try:
        yield {"cache": cache}
    finally:
        logger.info("Semantic cache MCP server stopped")


mcp = FastMCP("semantic-cache-mcp", lifespan=app_lifespan)
