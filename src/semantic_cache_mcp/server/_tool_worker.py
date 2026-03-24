"""Supervised subprocess for unsafe tool execution."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

from fastmcp.exceptions import ToolError

from ..cache import SemanticCache
from ..core.embeddings import get_model_info, warmup
from ..core.tokenizer import get_tokenizer
from .response import _response_overrides

logger = logging.getLogger(__name__)

_WORKER_TIMEOUT_SENTINEL = 24 * 60 * 60.0


@dataclass(slots=True)
class _WorkerContext:
    lifespan_context: dict[str, Any]


_PROTOCOL_ERROR = "Worker protocol error"


class ToolProcessSupervisor:
    """Owns a single tool worker subprocess and restarts it on timeout/failure."""

    _is_tool_process_supervisor = True

    __slots__ = (
        "_ctx",
        "_process",
        "_conn",
        "_lock",
        "_worker_target",
        "_startup_timeout",
        "_shutdown_timeout",
    )

    def __init__(
        self,
        *,
        worker_target: Any | None = None,
        startup_timeout: float = 120.0,
        shutdown_timeout: float = 8.0,
    ) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._process: BaseProcess | None = None
        self._conn: Connection | None = None
        self._lock = asyncio.Lock()
        self._worker_target = worker_target or _tool_worker_main
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout

    def _is_running(self) -> bool:
        return self._process is not None and self._process.is_alive()

    async def start(self) -> None:
        async with self._lock:
            if self._is_running():
                return
            await asyncio.to_thread(self._start_blocking)

    async def call_tool(
        self,
        tool: str,
        kwargs: dict[str, Any],
        *,
        output_mode: str,
        max_response_tokens: int | None,
        timeout: float,
    ) -> Any:
        async with self._lock:
            if not self._is_running():
                await asyncio.to_thread(self._start_blocking)

            request = {
                "op": "call_tool",
                "tool": tool,
                "kwargs": kwargs,
                "output_mode": output_mode,
                "max_response_tokens": max_response_tokens,
            }
            pending = asyncio.create_task(asyncio.to_thread(self._request_blocking, request))
            try:
                response = await asyncio.wait_for(pending, timeout=timeout)
            except TimeoutError:
                logger.warning(f"{tool} timed out after {timeout}s in worker process")
                await asyncio.to_thread(self._restart_blocking, f"{tool} timeout")
                raise
            except Exception:
                await asyncio.to_thread(self._restart_blocking, f"{tool} worker failure")
                raise

            try:
                return _worker_result(response)
            except RuntimeError:
                await asyncio.to_thread(self._restart_blocking, f"{tool} protocol failure")
                raise

    async def async_close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._close_blocking)

    def _start_blocking(self) -> None:
        self._close_blocking()

        parent_conn, child_conn = self._ctx.Pipe()
        process = self._ctx.Process(
            target=self._worker_target,
            args=(child_conn,),
            name="semantic-cache-tool-worker",
            daemon=True,
        )
        process.start()
        child_conn.close()

        self._process = process
        self._conn = parent_conn

        if not parent_conn.poll(self._startup_timeout):
            self._close_blocking(force=True)
            raise RuntimeError(f"Tool worker startup timed out after {self._startup_timeout}s")

        try:
            response = parent_conn.recv()
        except EOFError as exc:
            self._close_blocking()
            raise RuntimeError("Tool worker exited before signaling readiness") from exc

        if response.get("op") != "ready":
            error = response.get("error", "Tool worker failed to start")
            detail = response.get("traceback")
            self._close_blocking()
            if detail:
                raise RuntimeError(f"{error}\n{detail}")
            raise RuntimeError(error)

    def _request_blocking(self, request: dict[str, Any]) -> dict[str, Any]:
        conn = self._conn
        process = self._process
        if conn is None or process is None or not process.is_alive():
            raise RuntimeError("Tool worker is not running")

        conn.send(request)
        try:
            return conn.recv()
        except EOFError as exc:
            raise RuntimeError("Tool worker exited while handling a request") from exc

    def _restart_blocking(self, reason: str) -> None:
        logger.warning(f"Restarting tool worker: {reason}")
        self._close_blocking(force=True)
        self._start_blocking()

    def _close_blocking(self, *, force: bool = False) -> None:
        conn = self._conn
        process = self._process
        self._conn = None
        self._process = None

        if conn is not None:
            try:
                if not force:
                    conn.send({"op": "shutdown"})
            except Exception:
                pass

        if process is not None:
            process.join(timeout=0 if force else self._shutdown_timeout)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)

        if conn is not None:
            conn.close()


def _worker_result(response: object) -> Any:
    """Decode a worker reply at a single validated protocol boundary."""
    if not isinstance(response, dict):
        raise RuntimeError(f"{_PROTOCOL_ERROR}: expected dict response")

    op = response.get("op")
    if op == "result":
        if "result" not in response:
            raise RuntimeError(f"{_PROTOCOL_ERROR}: result response missing 'result'")
        return response["result"]
    if op == "tool_error":
        error = response.get("error")
        if not isinstance(error, str) or not error:
            raise RuntimeError(f"{_PROTOCOL_ERROR}: tool_error response missing 'error'")
        raise ToolError(error)

    error = response.get("error")
    if isinstance(error, str) and error:
        raise RuntimeError(error)
    raise RuntimeError(f"{_PROTOCOL_ERROR}: unsupported op {op!r}")


async def _send_worker_message(conn: Connection, payload: dict[str, Any]) -> None:
    await asyncio.to_thread(conn.send, payload)


def _tool_worker_main(conn: Connection) -> None:
    asyncio.run(_tool_worker_main_async(conn))


async def _tool_worker_main_async(conn: Connection) -> None:
    cache: SemanticCache | None = None
    try:
        import semantic_cache_mcp.server.tools as tools_mod

        tools_mod._TOOL_TIMEOUT = _WORKER_TIMEOUT_SENTINEL

        get_tokenizer()
        cache = SemanticCache()
        warmup()

        model_info = get_model_info()
        if model_info.get("ready") and cache is not None:
            cache._storage.clear_if_model_changed(str(model_info["model"]), int(model_info["dim"]))

        await _send_worker_message(conn, {"op": "ready"})

        while True:
            try:
                request = await asyncio.to_thread(conn.recv)
            except EOFError:
                break

            if request.get("op") == "shutdown":
                break

            if request.get("op") != "call_tool":
                await _send_worker_message(
                    conn,
                    {"op": "error", "error": f"Unsupported worker op: {request.get('op')}"},
                )
                continue

            try:
                result = await _dispatch_tool_request(
                    cache=cache,
                    tool=str(request["tool"]),
                    kwargs=dict(request["kwargs"]),
                    output_mode=str(request["output_mode"]),
                    max_response_tokens=request.get("max_response_tokens"),
                )
            except ToolError as exc:
                await _send_worker_message(conn, {"op": "tool_error", "error": str(exc)})
                continue
            except Exception as exc:
                await _send_worker_message(
                    conn,
                    {
                        "op": "error",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                continue

            await _send_worker_message(conn, {"op": "result", "result": result})
    except Exception as exc:
        try:
            conn.send({"op": "error", "error": str(exc), "traceback": traceback.format_exc()})
        except Exception:
            logger.exception("Failed to report worker startup error")
    finally:
        if cache is not None:
            await cache.async_close()
        conn.close()


async def _dispatch_tool_request(
    *,
    cache: SemanticCache,
    tool: str,
    kwargs: dict[str, Any],
    output_mode: str,
    max_response_tokens: int | None,
) -> Any:
    import semantic_cache_mcp.server.tools as tools_mod

    fn = getattr(tools_mod, tool, None)
    if fn is None:
        raise RuntimeError(f"Unknown tool: {tool}")

    ctx = _WorkerContext(lifespan_context={"cache": cache})
    with _response_overrides(output_mode, max_response_tokens):
        return await fn(ctx, **kwargs)
