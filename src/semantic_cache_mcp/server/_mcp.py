"""FastMCP instance and application lifespan."""

from __future__ import annotations

import contextlib
import logging
import sys

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan

from ..config import DB_PATH
from ..core.tokenizer import get_tokenizer
from ._tool_worker import ToolProcessSupervisor

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

    cache: ToolProcessSupervisor | None = None

    # Redirect stdout → stderr during initialization to prevent third-party
    # libraries (fastembed, onnxruntime) from printing to stdout and corrupting
    # the stdio MCP transport. The lifespan runs BEFORE stdio_server() captures
    # sys.stdout.buffer, so we must restore before yielding.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            logger.info("Initializing tokenizer...")
            get_tokenizer()

            _migrate_v2_to_v3()
            logger.info("Starting tool worker...")
            cache = ToolProcessSupervisor()
            if cache is None:
                raise RuntimeError("Cache failed to initialize")
            await cache.start()
            logger.info("Semantic cache MCP server started")
        except Exception:
            logger.exception("Failed to initialize semantic cache")
            raise

    if cache is None:
        raise RuntimeError("Cache failed to initialize")

    try:
        yield {"cache": cache}
    finally:
        await cache.async_close()
        # Flush streams before exit — prevents lost log output when running
        # as a subprocess (stdio transport) or in containers.
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                if not stream.closed:
                    stream.flush()
        logger.info("Semantic cache MCP server stopped")


mcp = FastMCP("semantic-cache-mcp", lifespan=app_lifespan)
