"""FastMCP instance and application lifespan."""

from __future__ import annotations

import contextlib
import logging
import sys

from fastmcp import FastMCP
from fastmcp.server.lifespan import lifespan

from ..config import DB_PATH
from ..core.tokenizer import get_tokenizer
from ._param_hints import ParamHintsMiddleware
from ._tool_worker import ToolProcessSupervisor

logger = logging.getLogger(__name__)


def _migrate_v2_to_v3() -> None:
    """Remove legacy v0.2.0 SQLite cache on first v0.3.0 startup.

    v0.3.0 switched from SQLiteStorage (cache.db with chunks/files/lsh_index tables)
    to ContentStorage (docstore.db). The old database is incompatible and just wastes disk.
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
    """Initialize cache on startup."""
    logger.info("Semantic cache MCP server starting...")

    # Redirect stdout → stderr during initialization to prevent third-party
    # libraries from printing to stdout and corrupting the stdio MCP
    # transport. The lifespan runs BEFORE stdio_server() captures
    # sys.stdout.buffer, so we must restore before yielding.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            logger.info("Initializing tokenizer...")
            get_tokenizer()

            _migrate_v2_to_v3()
            logger.info("Starting tool worker...")
            cache = ToolProcessSupervisor()
            await cache.start()
            logger.info("Semantic cache MCP server started")
        except Exception:
            logger.exception("Failed to initialize semantic cache")
            raise

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


# Client-visible server instructions. The hash-echo discipline is what turns
# the cache from storage into savings, and it spans every file tool, so it is
# stated once here rather than repeated across thirteen tool descriptions.
INSTRUCTIONS = """\
Every file operation runs through one cache, and every response that delivers a \
file carries its `content_hash`.

Keep those hashes and pass them back: `known_hash` on `read`, a `known_hashes` \
entry on `batch_read`. A matching hash is the only evidence the server has that \
the content is still in your context — a warm cache proves the *server* holds \
the file, never that you do — so it answers `unchanged` instead of re-sending \
bytes you already have. Without a hash it always sends the file in full.

That makes forgetting cheap to recover from: after a context compaction, or \
any point where you no longer hold a file's text, read it again without the \
hash and you get the whole thing back.

Pass `known_hash` when you edit or append too. Editing a file is not the same \
as having read it — an anchor can come from `grep` — so an edit only returns a \
claimable `content_hash` to a caller that showed it held the text being \
changed. With it, you never need a read after an edit to learn what the file \
now contains.

A hash reported as `file_hash` (prefixed `partial:`) came from a partial or \
summarized read. It identifies the file across reads but is not proof you hold \
it, and is never accepted as `known_hash`.

A ranged read (`offset`/`limit`) instead returns a `coverage_token` recording \
the lines it sent you. Keep it per file and pass it back as `known_hash` on \
your next ranged read: a window you already hold answers `unchanged`, a new \
window widens the coverage, and once the windows add up to the whole file you \
get a claimable `content_hash`. It vouches only for the lines actually \
delivered, so pass it back only while you still hold them.
"""

mcp = FastMCP("semantic-cache-mcp", instructions=INSTRUCTIONS, lifespan=app_lifespan)
mcp.add_middleware(ParamHintsMiddleware())
