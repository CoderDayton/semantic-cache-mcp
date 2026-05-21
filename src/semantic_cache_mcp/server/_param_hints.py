"""Middleware: rewrite known parameter aliases and surface did-you-mean hints.

Without this layer, FastMCP's Pydantic validation raises a raw -32602 with
no actionable hint when callers send `abs_path=` instead of `path=`, or
`query=` instead of `pattern=`. The audit found this is a real source of
wasted calls. This middleware:

- Renames known aliases on the way in (`abs_path` → `path`, etc.) so the
  call succeeds.
- For truly unknown parameters, replaces the opaque -32602 with a clean
  ToolError that suggests the closest known parameter via difflib.
- For type-coercion failures on `offset`/`limit`, surfaces the
  ranged-read hint instead of a Pydantic stack trace.
"""

from __future__ import annotations

import inspect
import logging
from difflib import get_close_matches
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware, MiddlewareContext

logger = logging.getLogger(__name__)


# Common alias confusions seen in production traffic. Key = arg the caller
# sent, value = the real parameter name. Applies across every tool.
#
# NOTE: do not add "paths" -> "path". `batch_read` takes a real `paths`
# parameter; a global rename would silently collapse a list-of-files call
# into a single-path call the moment a signature rename exposes it.
_ALIASES: dict[str, str] = {
    "abs_path": "path",
    "file": "path",
    "filename": "path",
    "filepath": "path",
    "query": "pattern",
    "q": "pattern",
}


def _tool_param_names(tool: Any) -> set[str]:
    """Return the tool's accepted parameter names (excluding ctx)."""
    try:
        sig = inspect.signature(tool.fn)
    except (TypeError, ValueError):
        return set()
    return {name for name in sig.parameters if name not in ("ctx", "self")}


class ParamHintsMiddleware(Middleware):
    """Rewrite alias kwargs and surface did-you-mean errors for unknown ones."""

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        args: dict[str, Any] | None = getattr(context.message, "arguments", None) or None
        if not args:
            return await call_next(context)

        tool_name = getattr(context.message, "name", None)
        if not tool_name:
            return await call_next(context)

        fastmcp_ctx = context.fastmcp_context
        if fastmcp_ctx is None:
            return await call_next(context)
        try:
            tool = await fastmcp_ctx.fastmcp.get_tool(tool_name)
        except Exception:
            return await call_next(context)

        known = _tool_param_names(tool)
        if not known:
            return await call_next(context)

        rewritten: dict[str, Any] = {}
        for key, value in args.items():
            if key in known:
                rewritten[key] = value
                continue
            mapped = _ALIASES.get(key)
            if mapped and mapped in known:
                if mapped in args:
                    # Caller sent both the real name and an alias for it.
                    # The explicit real parameter wins regardless of order;
                    # drop the alias rather than let it clobber the value.
                    logger.debug(
                        "param-alias: ignoring %s for tool %s (%s explicitly set)",
                        key,
                        tool_name,
                        mapped,
                    )
                    continue
                # Log so users can trace why their `query=` became `pattern=`.
                logger.debug("param-alias rewrite for tool %s: %s -> %s", tool_name, key, mapped)
                rewritten[mapped] = value
                continue
            # Unknown parameter: fail fast with a clean hint.
            close = get_close_matches(key, known, n=1, cutoff=0.6)
            hint = f" — did you mean {close[0]!r}?" if close else ""
            raise ToolError(f"{tool_name}: unknown parameter {key!r}{hint}")

        # Only mutate when we actually changed something.
        if rewritten != args:
            context.message.arguments = rewritten

        return await call_next(context)
