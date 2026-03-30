"""Helpers for live supervisor probes that require multiprocessing spawn."""

from __future__ import annotations

import os
import time
from multiprocessing.connection import Connection
from pathlib import Path


def restart_probe_worker(conn: Connection) -> None:
    """First request hangs, restarted worker returns a synthetic read payload."""
    state_path = Path(os.environ["SEMANTIC_CACHE_WORKER_STATE"])
    spawn_count = int(state_path.read_text()) if state_path.exists() else 0
    state_path.write_text(str(spawn_count + 1))

    conn.send({"op": "ready"})
    request = conn.recv()
    if request.get("op") == "shutdown":
        conn.close()
        return

    if spawn_count == 0:
        time.sleep(60)
        return

    conn.send(
        {
            "op": "result",
            "result": '{"ok":true,"tool":"read","path":"/tmp/probe.txt","content":"recovered"}',
        }
    )
    while True:
        request = conn.recv()
        if request.get("op") == "shutdown":
            conn.close()
            return
        conn.send(
            {
                "op": "result",
                "result": '{"ok":true,"tool":"read","path":"/tmp/probe.txt","content":"recovered"}',
            }
        )
