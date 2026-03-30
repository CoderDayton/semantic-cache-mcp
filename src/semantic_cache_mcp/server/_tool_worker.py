"""Supervised subprocess for unsafe tool execution."""

from __future__ import annotations

import asyncio
import itertools
import logging
import multiprocessing
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Any

from fastmcp.exceptions import ToolError

from ..cache import SemanticCache
from ..core.embeddings import get_embedding_dim, get_model_info
from ..core.tokenizer import get_tokenizer
from ..logger import log_marker
from .response import _response_overrides

logger = logging.getLogger(__name__)

_WORKER_TIMEOUT_SENTINEL = 24 * 60 * 60.0


@dataclass(slots=True)
class _WorkerContext:
    lifespan_context: dict[str, Any]


_PROTOCOL_ERROR = "Worker protocol error"
_REQUEST_SEQUENCE = itertools.count(1)


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

            request_id = next(_REQUEST_SEQUENCE)
            request = {
                "op": "call_tool",
                "request_id": request_id,
                "tool": tool,
                "kwargs": kwargs,
                "output_mode": output_mode,
                "max_response_tokens": max_response_tokens,
            }
            started = time.perf_counter()
            log_marker(
                logger,
                "tool.call.begin",
                request_id=request_id,
                tool=tool,
                timeout_s=timeout,
            )
            pending = asyncio.create_task(asyncio.to_thread(self._request_blocking, request))
            try:
                response = await asyncio.wait_for(pending, timeout=timeout)
            except TimeoutError:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                log_marker(
                    logger,
                    "tool.call.timeout",
                    request_id=request_id,
                    tool=tool,
                    timeout_s=timeout,
                    elapsed_ms=elapsed_ms,
                )
                logger.warning(f"{tool} timed out after {timeout}s in worker process")
                # Drop the wedged worker immediately so the timeout stays bounded.
                # Starting a replacement worker can trigger tokenizer/model init and
                # GPU warmup, which would otherwise make a "30s" timeout take far
                # longer before the caller sees the error.
                await asyncio.to_thread(self._invalidate_worker_blocking, f"{tool} timeout")
                raise
            except Exception as exc:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                log_marker(
                    logger,
                    "tool.call.error",
                    request_id=request_id,
                    tool=tool,
                    error=type(exc).__name__,
                    elapsed_ms=elapsed_ms,
                )
                await asyncio.to_thread(
                    self._invalidate_worker_blocking,
                    f"{tool} worker failure",
                )
                raise

            try:
                result = _worker_result(response)
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                log_marker(
                    logger,
                    "tool.call.end",
                    request_id=request_id,
                    tool=tool,
                    elapsed_ms=elapsed_ms,
                )
                return result
            except RuntimeError as exc:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
                log_marker(
                    logger,
                    "tool.call.protocol_error",
                    request_id=request_id,
                    tool=tool,
                    error=type(exc).__name__,
                    elapsed_ms=elapsed_ms,
                )
                await asyncio.to_thread(
                    self._invalidate_worker_blocking,
                    f"{tool} protocol failure",
                )
                raise

    async def async_close(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._close_blocking)

    def _start_blocking(self) -> None:
        started = time.perf_counter()
        log_marker(logger, "worker.start.begin", startup_timeout_s=self._startup_timeout)
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
            log_marker(
                logger,
                "worker.start.timeout",
                startup_timeout_s=self._startup_timeout,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
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
            log_marker(
                logger,
                "worker.start.error",
                error=error,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            self._close_blocking()
            if detail:
                raise RuntimeError(f"{error}\n{detail}")
            raise RuntimeError(error)
        log_marker(
            logger,
            "worker.start.ready",
            pid=process.pid,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )

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

    def _invalidate_worker_blocking(self, reason: str) -> None:
        """Discard a bad worker. The next call will start a fresh one lazily."""
        logger.warning(f"Discarding tool worker: {reason}")
        self._close_blocking(force=True)

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

        init_started = time.perf_counter()
        log_marker(logger, "worker.init.begin")
        stage_started = time.perf_counter()
        log_marker(logger, "worker.init.tokenizer.begin")
        get_tokenizer()
        log_marker(
            logger,
            "worker.init.tokenizer.end",
            elapsed_ms=round((time.perf_counter() - stage_started) * 1000, 1),
        )
        stage_started = time.perf_counter()
        log_marker(logger, "worker.init.cache.begin")
        cache = SemanticCache()
        log_marker(
            logger,
            "worker.init.cache.end",
            elapsed_ms=round((time.perf_counter() - stage_started) * 1000, 1),
        )
        stage_started = time.perf_counter()
        log_marker(logger, "worker.init.embedding_meta.begin")
        embedding_dim = get_embedding_dim()
        log_marker(
            logger,
            "worker.init.embedding_meta.end",
            dim=embedding_dim,
            elapsed_ms=round((time.perf_counter() - stage_started) * 1000, 1),
        )
        model_info = get_model_info()
        if cache is not None and embedding_dim > 0:
            cache._storage.clear_if_model_changed(str(model_info["model"]), embedding_dim)

        log_marker(
            logger,
            "worker.init.end",
            ready=model_info.get("ready"),
            provider=model_info.get("provider"),
            elapsed_ms=round((time.perf_counter() - init_started) * 1000, 1),
        )
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
                request_id = request.get("request_id")
                tool = str(request["tool"])
                started = time.perf_counter()
                log_marker(logger, "worker.exec.begin", request_id=request_id, tool=tool)
                result = await _dispatch_tool_request(
                    cache=cache,
                    tool=tool,
                    kwargs=dict(request["kwargs"]),
                    output_mode=str(request["output_mode"]),
                    max_response_tokens=request.get("max_response_tokens"),
                )
            except ToolError as exc:
                log_marker(
                    logger,
                    "worker.exec.tool_error",
                    request_id=request.get("request_id"),
                    tool=request.get("tool"),
                    error=type(exc).__name__,
                )
                await _send_worker_message(conn, {"op": "tool_error", "error": str(exc)})
                continue
            except Exception as exc:
                log_marker(
                    logger,
                    "worker.exec.error",
                    request_id=request.get("request_id"),
                    tool=request.get("tool"),
                    error=type(exc).__name__,
                )
                await _send_worker_message(
                    conn,
                    {
                        "op": "error",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                continue

            log_marker(
                logger,
                "worker.exec.end",
                request_id=request_id,
                tool=tool,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
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
