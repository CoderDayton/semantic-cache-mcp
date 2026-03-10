"""Delta compression, semantic truncation, and diff utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff

# ---------------------------------------------------------------------------
# Delta compression
# ---------------------------------------------------------------------------


@dataclass
class DiffDelta:
    """Compact diff for files that change minimally (10-100x smaller than storing both texts)."""

    old_hash: str  # Hash of original text
    new_hash: str  # Hash of updated text
    insertions: list[tuple[int, str]]  # (line_number, content)
    deletions: list[tuple[int, str]]  # (line_number, content)
    modifications: list[tuple[int, str, str]]  # (line_number, old, new)
    size_bytes: int  # Compressed size in bytes


def compute_delta(old: str, new: str) -> DiffDelta:
    from ..hashing import hash_content  # noqa: PLC0415

    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    # Truncate to 16 chars for compact storage — still collision-free at this scale
    old_hash = hash_content(old)[:16]
    new_hash = hash_content(new)[:16]

    sm = SequenceMatcher(None, old_lines, new_lines)
    opcodes = sm.get_opcodes()

    insertions = []
    deletions = []
    modifications = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "replace":
            # Could be modification or deletion+insertion
            if i2 - i1 == j2 - j1:
                # Line-by-line modification
                for i, j in zip(range(i1, i2), range(j1, j2), strict=True):
                    modifications.append((i, old_lines[i].rstrip(), new_lines[j].rstrip()))
            else:
                # Treat as deletion + insertion
                for i in range(i1, i2):
                    deletions.append((i, old_lines[i]))
                for j in range(j1, j2):
                    insertions.append((j, new_lines[j]))
        elif tag == "delete":
            for i in range(i1, i2):
                deletions.append((i, old_lines[i]))
        elif tag == "insert":
            for j in range(j1, j2):
                insertions.append((j, new_lines[j]))

    # Estimate compressed size
    size_bytes = len(old_hash) + len(new_hash) + sum(len(s) for _, s in insertions + deletions)
    for _, old_s, new_s in modifications:
        size_bytes += len(old_s) + len(new_s)

    return DiffDelta(
        old_hash=old_hash,
        new_hash=new_hash,
        insertions=insertions,
        deletions=deletions,
        modifications=modifications,
        size_bytes=size_bytes,
    )


# ---------------------------------------------------------------------------
# Semantic-aware truncation
# ---------------------------------------------------------------------------

# Regex patterns for semantic boundaries (Python/TypeScript/Go)
_SEMANTIC_PATTERNS = {
    "python": [
        r"^\s*class\s+\w+",
        r"^\s*def\s+\w+",
        r"^\s*@\w+",
        r"^\s*if\s+__name__",
    ],
    "typescript": [
        r"^\s*(export\s+)?(class|interface|type|function)",
        r"^\s*(export\s+)?const\s+\w+",
        r"^\s*async\s+function",
    ],
    "go": [
        r"^\s*type\s+\w+",
        r"^\s*func\s+(\(.*?\)\s+)?\w+",
        r"^\s*func\s+init",
    ],
}


def _detect_language(content: str) -> str:
    if "import " in content and "def " in content:
        return "python"
    elif "interface" in content or "export" in content:
        return "typescript"
    elif "func " in content and "package " in content:
        return "go"
    return "generic"


def _find_semantic_boundaries(lines: list[str], language: str) -> list[int]:
    patterns = _SEMANTIC_PATTERNS.get(language, [])
    if not patterns:
        return []

    boundaries = [0]  # Always include start
    combined_pattern = "|".join(f"({p})" for p in patterns)
    regex = re.compile(combined_pattern)

    for i, line in enumerate(lines):
        if regex.match(line):
            boundaries.append(i)

    boundaries.append(len(lines))  # Always include end
    return sorted(set(boundaries))


def truncate_semantic(
    content: str,
    max_size: int,
    keep_top: int = 10,
    keep_bottom: int = 5,
) -> str:
    """Truncate at semantic boundaries (class/function defs) rather than arbitrary line numbers."""
    if len(content) <= max_size:
        return content

    lines = content.splitlines(keepends=True)
    n_lines = len(lines)

    if n_lines <= keep_top + keep_bottom:
        return content[: max_size - 20] + "\n// [TRUNCATED]"

    language = _detect_language(content)
    boundaries = _find_semantic_boundaries(lines, language)

    # Find best cut point in top section
    top_cutoff = keep_top
    for b in boundaries:
        if b <= keep_top:
            top_cutoff = b
        else:
            break

    # Find best cut point in bottom section
    bottom_start = n_lines - keep_bottom
    bottom_cutoff = bottom_start
    for b in reversed(boundaries):
        if b >= bottom_start:
            bottom_cutoff = b
        else:
            break

    top_content = "".join(lines[:top_cutoff])
    bottom_content = "".join(lines[bottom_cutoff:]) if bottom_cutoff < n_lines else ""
    truncation_msg = (
        f"\n\n// ... [{n_lines - top_cutoff - (n_lines - bottom_cutoff)} lines truncated] ...\n\n"
    )

    total = len(top_content) + len(truncation_msg) + len(bottom_content)
    if total > max_size:
        return content[: max_size - 20] + "\n// [TRUNCATED]"

    return f"{top_content}{truncation_msg}{bottom_content}"


def truncate_smart(
    content: str,
    max_size: int,
    keep_top: int = 80,
    keep_bottom: int = 40,
    use_semantic: bool = True,
) -> str:
    if use_semantic:
        return truncate_semantic(content, max_size, keep_top, keep_bottom)

    if len(content) <= max_size:
        return content

    lines = content.splitlines(keepends=True)
    n_lines = len(lines)

    if n_lines <= keep_top + keep_bottom:
        return content[: max_size - 20] + "\n// [TRUNCATED]"

    top_content = "".join(lines[:keep_top])
    bottom_content = "".join(lines[-keep_bottom:]) if keep_bottom > 0 else ""
    truncation_msg = f"\n\n// ... [{n_lines - keep_top - keep_bottom} lines truncated] ...\n\n"

    total = len(top_content) + len(truncation_msg) + len(bottom_content)
    if total > max_size:
        return content[: max_size - 20] + "\n// [TRUNCATED]"

    return f"{top_content}{truncation_msg}{bottom_content}"


def generate_diff(old: str, new: str, context_lines: int = 3) -> str:
    """Return unified diff string, or '// No changes' if identical."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)

    diff = unified_diff(old_lines, new_lines, fromfile="old", tofile="new", n=context_lines)

    result = "".join(diff)
    return result if result else "// No changes"


# ---------------------------------------------------------------------------
# Utility: diff statistics
# ---------------------------------------------------------------------------


def diff_stats(old: str, new: str) -> dict:
    delta = compute_delta(old, new)

    return {
        "insertions": len(delta.insertions),
        "deletions": len(delta.deletions),
        "modifications": len(delta.modifications),
        "delta_size_bytes": delta.size_bytes,
        "original_size": len(old.encode()),
        "compression_ratio": delta.size_bytes / len(old.encode()) if old else 0,
    }
