"""Generic retry utility with exponential backoff.

Kept deliberately minimal — no external deps, no async variant (not needed yet).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


def retry[T](
    fn: Callable[[], T],
    *,
    delays: tuple[float, ...] = (0.1, 0.2, 0.4),
    exceptions: tuple[type[Exception], ...] = (Exception,),
    label: str = "operation",
) -> T:
    """Call fn up to len(delays)+1 times with exponential backoff.

    Exceptions not in *exceptions* propagate immediately without retrying.

    Raises:
        The last exception from fn after all retries are exhausted.
    """
    last_err: Exception = RuntimeError(f"{label}: no attempts made")
    attempts = len(delays) + 1

    for attempt in range(attempts):
        try:
            return fn()
        except exceptions as exc:
            last_err = exc
            remaining = attempts - attempt - 1
            if remaining == 0:
                break
            delay = delays[attempt]
            logger.warning(
                f"{label}: attempt {attempt + 1}/{attempts} failed ({exc}); retrying in {delay}s..."
            )
            time.sleep(delay)

    logger.warning(f"{label}: all {attempts} attempts failed")
    raise last_err
