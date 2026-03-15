"""FastMCP instance and application lifespan."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan

from ..cache import SemanticCache
from ..config import DB_PATH
from ..core.embeddings import get_model_info, warmup
from ..core.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)


def _migrate_v2_to_v3() -> None:
    """Remove legacy v0.2.0 SQLite cache on first v0.3.0 startup.

    v0.3.0 switched from SQLiteStorage (cache.db with chunks/files/lsh_index tables)
    to VectorStorage (vecdb.db). The old database is incompatible and just wastes disk.
    """
    if not DB_PATH.exists():
        return
    try:
        import sqlite3

        conn = sqlite3.connect(str(DB_PATH))
        try:
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            conn.close()
        # Only delete if it has the old v0.2.0 schema — don't touch unrelated DBs
        if {"chunks", "files", "lsh_index"} <= tables:
            DB_PATH.unlink()
            # Also remove WAL/SHM if present
            for suffix in ("-wal", "-shm"):
                wal = DB_PATH.with_name(DB_PATH.name + suffix)
                if wal.exists():
                    wal.unlink()
            logger.info(
                "Migrated from v0.2.0: removed legacy cache.db "
                "(cache will rebuild automatically as files are read)"
            )
    except Exception:
        logger.debug("Could not check legacy cache.db for migration", exc_info=True)


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

            # Initialize VectorStorage (usearch) before onnxruntime (fastembed).
            # Loading onnxruntime first and then usearch causes heap corruption
            # (free(): corrupted unsorted chunks) on Linux due to allocator conflicts.
            logger.info("Initializing cache storage...")
            cache = SemanticCache()
            _migrate_v2_to_v3()

            logger.info("Initializing embedding model...")
            await asyncio.to_thread(warmup)

            model_info = get_model_info()
            # Detect embedding model change and rebuild index if needed
            if model_info.get("ready") and cache is not None:
                cache._storage.clear_if_model_changed(
                    str(model_info["model"]), int(model_info["dim"])
                )
            if not model_info.get("ready", False):
                logger.error(
                    "Embedding model failed to initialize. "
                    "Semantic similarity features will be disabled. "
                    "Check network connectivity and disk space."
                )
            else:
                logger.info(f"Embedding model ready: {model_info['model']}")
            logger.info("Semantic cache MCP server started")
        except Exception:
            logger.exception("Failed to initialize semantic cache")
            raise

    if cache is None:
        raise RuntimeError("Cache failed to initialize")

    # Install graceful shutdown handlers so SIGTERM/SIGINT trigger the
    # lifespan cleanup path instead of raising KeyboardInterrupt mid-operation.
    loop = asyncio.get_running_loop()
    shutdown_received = False

    def _graceful_shutdown(sig: int) -> None:
        nonlocal shutdown_received
        sig_name = signal.Signals(sig).name
        if not shutdown_received:
            shutdown_received = True
            logger.info(f"Received {sig_name} — initiating graceful shutdown")
            cache.request_shutdown()
            # Cancel all tasks so their finally blocks run (including lifespan
            # cleanup → async_close). Shielded writes continue to completion.
            for task in asyncio.all_tasks(loop):
                task.cancel()
        else:
            logger.warning(f"Received {sig_name} again — forcing exit")
            import os  # noqa: PLC0415

            os._exit(128 + sig)

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError, OSError):
            loop.add_signal_handler(sig, _graceful_shutdown, sig)

    try:
        yield {"cache": cache}
    finally:
        # Remove signal handlers before cleanup to avoid re-entrance
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, OSError):
                loop.remove_signal_handler(sig)
        await cache.async_close()
        # Flush streams before exit — prevents lost log output when running
        # as a subprocess (stdio transport) or in containers.
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                if not stream.closed:
                    stream.flush()
        logger.info("Semantic cache MCP server stopped")


mcp = FastMCP("semantic-cache-mcp", lifespan=app_lifespan)
