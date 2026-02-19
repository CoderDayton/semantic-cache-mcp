"""Private helpers and constants shared across the cache package."""

from __future__ import annotations

import bisect
import logging
import shutil
import subprocess  # nosec B404 - used for formatter execution with hardcoded commands
from pathlib import Path

from ..core import count_tokens

logger = logging.getLogger(__name__)

# Size limits for DoS protection
MAX_WRITE_SIZE = 10 * 1024 * 1024  # 10MB max content size for write
MAX_EDIT_SIZE = 10 * 1024 * 1024  # 10MB max file size for edit
MAX_MATCHES = 10000  # Max occurrences for replace_all
MAX_RETURN_DIFF_TOKENS = 8000  # Hard cap for emitted diff payloads
MAX_DIFF_TO_FULL_RATIO = 0.9  # Suppress diff payloads near full-content size

# Auto-format configuration: extension -> (command, args...)
# Command must accept file path as final argument
FORMATTERS: dict[str, tuple[str, ...]] = {
    ".py": ("ruff", "format"),
    ".pyi": ("ruff", "format"),
    ".js": ("prettier", "--write"),
    ".ts": ("prettier", "--write"),
    ".tsx": ("prettier", "--write"),
    ".jsx": ("prettier", "--write"),
    ".json": ("prettier", "--write"),
    ".css": ("prettier", "--write"),
    ".scss": ("prettier", "--write"),
    ".md": ("prettier", "--write"),
    ".yaml": ("prettier", "--write"),
    ".yml": ("prettier", "--write"),
    ".go": ("gofmt", "-w"),
    ".rs": ("rustfmt",),
}


def _format_file(path: Path) -> bool:
    """Format file in-place using appropriate formatter.

    Args:
        path: Path to file to format

    Returns:
        True if formatted successfully, False if formatter not found or failed
    """
    formatter = FORMATTERS.get(path.suffix.lower())
    if not formatter:
        return False

    cmd_name = formatter[0]
    # Check if formatter is installed
    if not shutil.which(cmd_name):
        logger.debug(f"Formatter not found: {cmd_name}")
        return False

    try:
        cmd = [*formatter, str(path)]
        result = subprocess.run(  # nosec B603 - commands from hardcoded FORMATTERS dict
            cmd,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            logger.debug(f"Formatted {path} with {cmd_name}")
            return True
        else:
            logger.warning(f"Formatter {cmd_name} failed: {result.stderr.decode()[:200]}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Formatter {cmd_name} timed out on {path}")
        return False
    except OSError as e:
        logger.warning(f"Failed to run {cmd_name}: {e}")
        return False


def _is_binary_content(data: bytes) -> bool:
    """Check if content is binary using multiple heuristics.

    Checks:
    1. Null bytes (most reliable indicator)
    2. Common binary file magic numbers
    3. High ratio of non-printable characters
    """
    if not data:
        return False

    sample = data[:8192]

    # 1. Null bytes - definitive binary indicator
    if b"\x00" in sample:
        return True

    # 2. Check for binary file magic numbers
    binary_signatures = (
        b"\x89PNG",  # PNG
        b"\xff\xd8\xff",  # JPEG
        b"GIF8",  # GIF
        b"PK\x03\x04",  # ZIP/DOCX/XLSX
        b"\x1f\x8b",  # GZIP
        b"BZ",  # BZIP2
        b"\x7fELF",  # ELF executable
        b"MZ",  # Windows executable
        b"%PDF",  # PDF (text header but binary content)
        b"\xd0\xcf\x11\xe0",  # MS Office OLE
    )
    for sig in binary_signatures:
        if sample.startswith(sig):
            return True

    # 3. High ratio of non-printable characters (>30% is suspicious)
    # Exclude common whitespace: tab(9), newline(10), carriage return(13)
    non_printable = sum(1 for b in sample if b < 32 and b not in (9, 10, 13))
    return len(sample) > 0 and non_printable / len(sample) > 0.3


def _find_match_line_numbers(content: str, search_string: str) -> list[int]:
    """Find line numbers where search_string appears.

    Returns 1-based line numbers for each occurrence.
    Uses binary search for O(M log N) complexity where M=matches, N=lines.
    """
    lines = content.splitlines(keepends=True)
    line_numbers: list[int] = []

    if not lines:
        return line_numbers

    # Build cumulative character positions for binary search
    char_pos = 0
    line_starts: list[int] = []
    for line in lines:
        line_starts.append(char_pos)
        char_pos += len(line)

    # Find all occurrences with O(log N) line lookup
    start = 0
    while True:
        idx = content.find(search_string, start)
        if idx == -1:
            break

        # Binary search for line number - O(log N) per match
        # bisect_right returns insertion point; -1 gives us the line containing idx
        line_num = bisect.bisect_right(line_starts, idx)
        line_numbers.append(line_num)  # Already 1-based due to bisect_right behavior
        start = idx + 1

    return line_numbers


def _choose_min_token_content(options: dict[str, str]) -> tuple[str, str, int]:
    """Pick the response payload with the smallest token count."""
    best_kind = ""
    best_content = ""
    best_tokens: int | None = None

    for kind, content in options.items():
        tokens = count_tokens(content)
        if best_tokens is None or tokens < best_tokens:
            best_kind = kind
            best_content = content
            best_tokens = tokens

    return best_kind, best_content, best_tokens or 0


def _suppress_large_diff(diff_content: str | None, full_tokens: int) -> str | None:
    """Suppress large diff payloads to optimize returned token count.

    Returns the diff unchanged for small files, a summary string for large
    diffs that exceed the token cap, or None for empty input.
    """
    if not diff_content:
        return None

    diff_tokens = count_tokens(diff_content)
    full_tokens = max(full_tokens, 1)

    # Preserve diffs for small files where readability is more useful than suppression.
    if full_tokens <= 200:
        return diff_content

    should_suppress = (
        diff_tokens > MAX_RETURN_DIFF_TOKENS
        or diff_tokens >= int(full_tokens * MAX_DIFF_TO_FULL_RATIO)
    )
    if should_suppress:
        # Count added/removed lines and hunks from unified diff
        added = 0
        removed = 0
        hunks = 0
        for line in diff_content.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
            elif line.startswith("@@"):
                hunks += 1
        return (
            f"[diff suppressed: {diff_tokens} tokens > cap] "
            f"+{added} -{removed} lines across {hunks} hunks"
        )

    return diff_content
