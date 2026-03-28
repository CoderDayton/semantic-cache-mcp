"""Logging bootstrap for semantic-cache-mcp."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(process)d - %(threadName)s - %(name)s - %(levelname)s - %(message)s"
)
_STDERR_HANDLER_NAME = "semantic-cache-mcp.stderr"
_FILE_HANDLER_NAME = "semantic-cache-mcp.file"


def log_marker(logger: logging.Logger, marker: str, **fields: object) -> None:
    """Emit a grep-friendly execution marker at INFO level."""
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value).replace("\n", "\\n")
        parts.append(f"{key}={text!r}" if (" " in text or "=" in text) else f"{key}={text}")

    if parts:
        logger.info("[%s] %s", marker, " ".join(parts))
    else:
        logger.info("[%s]", marker)


def get_log_dir(cache_dir: Path, override: str | None = None) -> Path:
    """Return the directory used for file logs."""
    if override:
        return Path(override).expanduser().resolve()
    return (cache_dir / "logs").resolve()


def get_log_file_path(log_dir: Path, *, now: datetime | None = None) -> Path:
    """Return the dated log file path for the current day."""
    if now is None:
        now = datetime.now()
    return log_dir / f"semantic-cache-mcp-{now.strftime('%Y-%m-%d')}.log"


def configure_logging(
    log_dir: Path,
    log_file_path: Path,
    *,
    log_level: str,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> None:
    """Install stderr and daily file handlers without duplicating them on reload."""
    root = logging.getLogger()
    root.setLevel(log_level.upper())

    stderr_handler = next(
        (h for h in root.handlers if getattr(h, "name", "") == _STDERR_HANDLER_NAME),
        None,
    )
    if stderr_handler is None:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.name = _STDERR_HANDLER_NAME
        root.addHandler(stderr_handler)
    stderr_handler.setLevel(logging.NOTSET)
    stderr_handler.setFormatter(logging.Formatter(log_format))

    file_handler = next(
        (h for h in root.handlers if getattr(h, "name", "") == _FILE_HANDLER_NAME),
        None,
    )
    if file_handler is not None and Path(file_handler.baseFilename) != log_file_path:
        root.removeHandler(file_handler)
        file_handler.close()
        file_handler = None

    if file_handler is None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8", delay=True)
        file_handler.name = _FILE_HANDLER_NAME
        root.addHandler(file_handler)

    file_handler.setLevel(logging.NOTSET)
    file_handler.setFormatter(logging.Formatter(log_format))
