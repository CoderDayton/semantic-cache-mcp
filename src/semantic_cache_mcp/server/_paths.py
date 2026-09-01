"""Name a shared directory once instead of repeating it on every path.

Every path this server returns is fully resolved, so a response listing files
pays for the same absolute prefix once per entry — ~25 tokens against ~12 for
the repo-relative form. `glob` alone returns up to 1000 matches, which is
~13k tokens of identical prefix.

The saving is only allowed to be a saving. A shortened path has to rebuild
into the original by joining it back onto the reported `root`, so the prefix
is computed component-wise (`/a/bc` and `/a/bd` share `/a`, never `/a/b`) and
a path that does not live under the root is left absolute.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

# Characters the `root` field itself costs: the key, the quotes, the comma.
# The prefix has to save more than this across the paths it shortens, or
# reporting it is a net loss.
_ROOT_FIELD_OVERHEAD = 10

# One path has nothing to share a prefix with — `root` plus a relative path is
# strictly longer than the absolute path it replaces.
_MIN_PATHS = 2


def shared_root(paths: Sequence[str], client_root: str | None = None) -> str | None:
    """The directory prefix worth reporting once, or ``None`` if there isn't one.

    ``client_root`` is preferred when every path lives under it, so responses
    from one session keep naming the same root instead of drifting with
    whichever files happened to match.
    """
    absolute = [p for p in paths if os.path.isabs(p)]
    if len(absolute) < _MIN_PATHS or len(absolute) != len(paths):
        # A mix of absolute and relative paths has no common ground worth
        # claiming, and `commonpath` refuses it outright.
        return None

    if client_root and all(_is_under(p, client_root) for p in absolute):
        candidate = client_root
    else:
        try:
            candidate = os.path.commonpath([os.path.dirname(p) for p in absolute])
        except ValueError:
            # Different drives on Windows, or an empty list.
            return None

    # The prefix is written once and removed from every path, so it saves
    # roughly `len(candidate)` per path beyond the first. Report it only when
    # that beats what the field itself costs.
    under = sum(1 for p in absolute if _is_under(p, candidate))
    if under < _MIN_PATHS or len(candidate) * (under - 1) <= _ROOT_FIELD_OVERHEAD:
        return None
    return candidate


def _is_under(path: str, root: str) -> bool:
    """True when *path* is inside *root*, comparing whole components.

    A plain ``startswith`` would read ``/a/bc/x`` as living under ``/a/b``.
    """
    if path == root:
        return False
    return path.startswith(root.rstrip(os.sep) + os.sep)


def relativize(path: str, root: str | None) -> str:
    """Strip *root* from *path*, or return *path* unchanged if it is elsewhere."""
    if not root or not _is_under(path, root):
        return path
    return path[len(root.rstrip(os.sep)) + 1 :]


__all__ = ["relativize", "shared_root"]
