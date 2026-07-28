#!/usr/bin/env python
"""Drive the real MCP server through a realistic agentic coding session.

This is not a unit test and not a benchmark. It spawns the actual server over
stdio — the same binary a client launches, forwarding to the same worker
subprocess — and walks it through the work an agent really does: orient in an
unfamiliar tree, locate a symbol, confirm an anchor, edit it, verify the edit,
absorb a change someone else made, page through a file too big to hold, and
clean up after itself.

Every tool is exercised because the workflow needs it, not because a checklist
says so. The point is the seam between them: a hash minted by `batch_read` is
spent on `read`, the hash `edit` hands back is spent on the next `read`, and a
`coverage_token` from one window is spent on the next. Those hand-offs are the
whole product, and they only break in sequence.

The session runs twice against two isolated caches — once carrying every hash
forward, once discarding them, which is what an agent that ignores the protocol
does — and reports the difference in tokens actually delivered.

Usage:
    uv run python scripts/agentic_workflow.py [--keep] [--verbose]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastmcp.client import Client
from fastmcp.client.transports.stdio import StdioTransport

from semantic_cache_mcp.config import CACHE_DIR

REPO_ROOT = Path(__file__).resolve().parents[1]

# Large enough to trip the summarizer on a whole-file read, so the windowed
# read phase has something real to page through.
BIG_MODULE_ROWS = 4_000

# Windows the agent pages through in the large-file phase.
FIRST_WINDOW = (1, 120)
SECOND_WINDOW = (121, 240)
SUB_WINDOW = (40, 60)


# --------------------------------------------------------------------------
# result plumbing
# --------------------------------------------------------------------------


def unwrap(result: Any) -> dict[str, Any]:
    """Pull the payload out of an MCP tool result.

    Structured content first: it is the tool's declared output schema. Some
    tools additionally render a human-readable text block (`stats` returns
    markdown), so parsing the text alone silently yields nothing for them.
    """
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict) and structured:
        return structured
    for block in getattr(result, "content", ()) or ():
        text = getattr(block, "text", None)
        if text is None:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}
    return {}


@dataclass
class Ledger:
    """What the session actually cost, and what it would have cost naively."""

    delivered_tokens: int = 0
    unchanged_hits: int = 0
    diffs_received: int = 0
    full_bodies: int = 0
    # Paging through a file too large to hold costs both runs the same: the
    # windows are new content either way. Kept separate so it cannot flatter
    # or dilute the figure for the working set, which is where the hash
    # discipline actually decides anything.
    paging_chars: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class Session:
    """One scripted agent run against its own private cache."""

    client: Client
    carry_hashes: bool
    ledger: Ledger = field(default_factory=Ledger)
    checks_passed: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    verbose: bool = False

    async def call(self, tool: str, **args: Any) -> dict[str, Any]:
        result = await self.client.call_tool(tool, args, raise_on_error=False)
        payload = unwrap(result)
        if self.verbose:
            trimmed = {k: v for k, v in payload.items() if k not in ("content", "files")}
            print(f"      · {tool}({', '.join(args)}) -> {json.dumps(trimmed)[:160]}")
        return payload

    def hash_for(self, held: str | None) -> dict[str, Any]:
        """Spend a held hash — or withhold it, for the naive baseline."""
        if held and self.carry_hashes:
            return {"known_hash": held}
        return {}

    def account(self, payload: dict[str, Any], *, paging: bool = False) -> None:
        """Record what this response actually delivered."""
        if payload.get("unchanged") is True:
            self.ledger.unchanged_hits += 1
            return
        body = payload.get("content")
        if body is None:
            return
        if payload.get("is_diff"):
            self.ledger.diffs_received += 1
        else:
            self.ledger.full_bodies += 1
        # Chars, not tokens: the server's own counter is the authority for
        # totals, and this only needs to rank two runs against each other.
        if paging:
            self.ledger.paging_chars += len(body)
        else:
            self.ledger.delivered_tokens += len(body)

    def check(self, name: str, cond: bool, detail: str = "") -> bool:
        if cond:
            self.checks_passed += 1
            print(f"      ok   {name}")
        else:
            self.failures.append((name, detail))
            print(f"      FAIL {name}\n           {detail[:300]}")
        return cond


# --------------------------------------------------------------------------
# the project the agent is asked to work on
# --------------------------------------------------------------------------


def build_project(root: Path) -> dict[str, Path]:
    """A small service with a real call graph, so the workflow has somewhere to go."""
    src = root / "svc"
    src.mkdir(parents=True)

    files = {
        "client.py": '''"""HTTP client for the upstream billing service."""

from __future__ import annotations

import time


DEFAULT_TIMEOUT = 30.0


class BillingClient:
    """Talks to the billing API."""

    def __init__(self, base_url: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url
        self.timeout = timeout

    def request(self, method: str, path: str) -> dict:
        """Issue a single request and return the decoded body."""
        started = time.monotonic()
        response = self._send(method, path)
        self._record_latency(time.monotonic() - started)
        return response

    def _send(self, method: str, path: str) -> dict:
        raise NotImplementedError

    def _record_latency(self, elapsed: float) -> None:
        pass
''',
        "invoices.py": '''"""Invoice operations built on the billing client."""

from __future__ import annotations

from .client import BillingClient


def fetch_invoice(client: BillingClient, invoice_id: str) -> dict:
    """Fetch one invoice by id."""
    return client.request("GET", f"/invoices/{invoice_id}")


def void_invoice(client: BillingClient, invoice_id: str) -> dict:
    """Void an invoice."""
    return client.request("POST", f"/invoices/{invoice_id}/void")


def list_invoices(client: BillingClient) -> dict:
    """List every invoice."""
    return client.request("GET", "/invoices")
''',
        "accounts.py": '''"""Account lookups."""

from __future__ import annotations

from .client import BillingClient


def get_account(client: BillingClient, account_id: str) -> dict:
    """Fetch one account."""
    return client.request("GET", f"/accounts/{account_id}")
''',
        # Deliberately substantial: below a floor the server re-sends a window
        # rather than diffing it, because the @@-header overhead outweighs the
        # saving. A toy file would never exercise the diff path at all.
        "config.py": '''"""Service configuration.

Every tunable the billing client reads at start-up. Values here are the
defaults; deployment tooling overrides them per environment.
"""

from __future__ import annotations

BASE_URL = "https://billing.internal"
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.5

'''
        + "\n".join(
            f"# {name} tuning\n{name.upper()}_ENABLED = True\n"
            f"{name.upper()}_TIMEOUT_SECONDS = {i + 1}.0\n"
            f"{name.upper()}_MAX_CONCURRENCY = {i + 2}\n"
            for i, name in enumerate(
                (
                    "invoices",
                    "accounts",
                    "payments",
                    "refunds",
                    "disputes",
                    "subscriptions",
                    "webhooks",
                    "exports",
                    "audits",
                    "ledger",
                    "taxes",
                    "coupons",
                    "credits",
                    "payouts",
                    "reports",
                )
            )
        )
        + "\n",
        "README.md": """# billing service

Thin wrapper over the upstream billing API. `BillingClient.request` is the
single choke point every operation goes through.
""",
    }

    written = {}
    for name, body in files.items():
        p = src / name
        p.write_text(body)
        written[name] = p

    # A module too large to hand back whole — the windowed-read phase needs one.
    big = src / "generated_schema.py"
    big.write_text(
        '"""Generated — do not edit by hand."""\n\n'
        + "\n".join(
            f"FIELD_{i} = {{'name': 'field_{i}', 'kind': 'string', 'index': {i}}}"
            for i in range(BIG_MODULE_ROWS)
        )
        + "\n"
    )
    written["generated_schema.py"] = big
    return written


# --------------------------------------------------------------------------
# the workflow
# --------------------------------------------------------------------------


async def run_workflow(s: Session, project: Path, files: dict[str, Path]) -> None:
    """One agent, one task: put retry handling behind the client's choke point."""
    src = project / "svc"
    held: dict[str, str] = {}

    def remember(path: Path, payload: dict[str, Any]) -> None:
        h = payload.get("content_hash")
        if h:
            held[str(path)] = h

    # -- 1. orient ---------------------------------------------------------
    print("\n  [1] orient in an unfamiliar tree")

    gl = await s.call("glob", pattern="*.py", directory=str(src))
    names = {Path(m["path"]).name for m in gl.get("matches", [])}
    s.check("glob discovers the modules", {"client.py", "invoices.py"} <= names, str(names))
    s.check(
        "and reports nothing is cached yet",
        gl.get("cached_count", 0) == 0,
        json.dumps({k: v for k, v in gl.items() if k != "matches"}),
    )

    seed = [str(files[n]) for n in ("client.py", "invoices.py", "accounts.py", "config.py")]
    br = await s.call("batch_read", paths=json.dumps(seed))
    for entry in br.get("files", []):
        h = entry.get("content_hash")
        if h:
            held[entry["path"]] = h
        s.ledger.delivered_tokens += len(entry.get("content", ""))
        s.ledger.full_bodies += 1
    s.check(
        "batch_read seeds the working set",
        len(br.get("files", [])) == 4,
        json.dumps(br.get("summary")),
    )
    s.check("and every file comes back with a hash", len(held) == 4, str(list(held)))

    # -- 2. locate the work ------------------------------------------------
    print("\n  [2] locate the choke point")

    se = await s.call("search", query="request timeout latency client", k=3, show_preview=True)
    top = (se.get("matches") or [{}])[0].get("path", "")
    s.check(
        "search ranks the client first",
        top.endswith("client.py"),
        json.dumps([m.get("path") for m in se.get("matches", [])]),
    )

    gr = await s.call("grep", pattern=r"client\.request\(", path=str(src))
    call_sites = {Path(f["path"]).name for f in gr.get("files", [])}
    s.check(
        "grep finds every call site through the directory filter",
        call_sites == {"invoices.py", "accounts.py"},
        json.dumps({k: v for k, v in gr.items() if k != "files"}) + f" sites={call_sites}",
    )
    s.check("and counts them all", gr.get("total_matches") == 4, str(gr.get("total_matches")))

    # -- 3. read what we already hold --------------------------------------
    print("\n  [3] re-read the target we already hold")

    client_py = str(files["client.py"])
    rd = await s.call("read", path=client_py, **s.hash_for(held.get(client_py)))
    s.account(rd)
    if s.carry_hashes:
        s.check(
            "the held hash buys 'unchanged' instead of the bytes",
            rd.get("unchanged") is True,
            json.dumps(rd)[:200],
        )
    else:
        s.check("[naive] no hash means the whole file again", "content" in rd, json.dumps(rd)[:150])

    # -- 4. confirm the anchor before touching it --------------------------
    print("\n  [4] confirm the anchor is unique before editing")

    anchor = "        response = self._send(method, path)"
    pv = await s.call("edit_preview", path=client_py, old_string=anchor)
    s.check(
        "edit_preview finds exactly one anchor", pv.get("match_count") == 1, json.dumps(pv)[:250]
    )
    s.check("and pins its line number", bool(pv.get("line_numbers")), json.dumps(pv)[:250])

    ambiguous = await s.call("edit_preview", path=client_py, old_string="        return response")
    s.check(
        "a probe that would be ambiguous is visible up front",
        ambiguous.get("match_count", 0) >= 1,
        json.dumps(ambiguous)[:250],
    )

    # -- 5. make the change ------------------------------------------------
    print("\n  [5] edit the choke point")

    ed = await s.call(
        "edit",
        path=client_py,
        old_string=anchor,
        new_string="        response = self._send_with_retries(method, path)",
        **s.hash_for(held.get(client_py)),
    )
    s.check("the edit applies", ed.get("replaced") == 1, json.dumps(ed)[:250])
    if s.carry_hashes:
        s.check(
            "proving possession earns a claimable hash back",
            bool(ed.get("content_hash")),
            json.dumps(ed)[:250],
        )
        remember(files["client.py"], ed)
    else:
        s.check(
            "[naive] editing without proof earns only file_hash",
            bool(ed.get("file_hash")) and not ed.get("content_hash"),
            json.dumps(ed)[:250],
        )

    # -- 6. verify without re-reading --------------------------------------
    print("\n  [6] verify the edit without paying to re-read")

    vr = await s.call("read", path=client_py, **s.hash_for(held.get(client_py)))
    s.account(vr)
    if s.carry_hashes:
        s.check(
            "the hash the edit returned redeems for 'unchanged'",
            vr.get("unchanged") is True,
            json.dumps(vr)[:200],
        )
        s.ledger.notes.append("post-edit verification cost 0 content tokens")
    else:
        s.check("[naive] the file is delivered again", "content" in vr, json.dumps(vr)[:150])

    # -- 7. multi-site refactor -------------------------------------------
    print("\n  [7] update the call sites in one atomic batch")

    inv = str(files["invoices.py"])
    batch = await s.call(
        "batch_edit",
        path=inv,
        edits=json.dumps(
            [
                [
                    'return client.request("GET", f"/invoices/{invoice_id}")',
                    'return client.request("GET", f"/invoices/{invoice_id}", retry=True)',
                ],
                [
                    'return client.request("POST", f"/invoices/{invoice_id}/void")',
                    'return client.request("POST", f"/invoices/{invoice_id}/void", retry=False)',
                ],
                [
                    'return client.request("GET", "/invoices")',
                    'return client.request("GET", "/invoices", retry=True)',
                ],
            ]
        ),
        **s.hash_for(held.get(inv)),
    )
    s.check(
        "all three call sites land",
        batch.get("succeeded") == 3 and batch.get("status") == "edited",
        json.dumps(batch)[:300],
    )
    if s.carry_hashes:
        remember(files["invoices.py"], batch)

    bad = await s.call(
        "batch_edit",
        path=inv,
        edits=json.dumps([["client.request(", "client.call("]]),
        **s.hash_for(held.get(inv)),
    )
    s.check(
        "an ambiguous anchor is refused rather than guessed at",
        bad.get("status") in {"partial", "no_changes"} and bool(bad.get("failures")),
        json.dumps(bad)[:300],
    )

    # -- 8. absorb someone else's change -----------------------------------
    print("\n  [8] absorb a change made outside the session")

    cfg = str(files["config.py"])
    original = files["config.py"].read_text()
    files["config.py"].write_text(
        original.replace("RETRY_ATTEMPTS = 3", "RETRY_ATTEMPTS = 5\nRETRY_JITTER = 0.1")
        + "\n# touched by the deploy tooling\n"
    )
    ch = await s.call("read", path=cfg, **s.hash_for(held.get(cfg)))
    s.account(ch)
    if s.carry_hashes:
        s.check(
            "a moved file comes back as a diff, not a re-send",
            ch.get("is_diff") is True,
            json.dumps({k: v for k, v in ch.items() if k != "content"})[:250],
        )
        s.check(
            "and the diff carries the change",
            "RETRY_JITTER" in ch.get("content", ""),
            ch.get("content", "")[:200],
        )
        remember(files["config.py"], ch)
    else:
        s.check(
            "[naive] the changed file is re-sent whole",
            "content" in ch and not ch.get("is_diff"),
            json.dumps(ch)[:150],
        )

    # -- 9. page through a file too big to hold ----------------------------
    print("\n  [9] page through a file too big to hand back whole")

    big = str(files["generated_schema.py"])
    whole = await s.call("read", path=big)
    s.account(whole, paging=True)
    s.check(
        "a whole-file read of it is summarized",
        whole.get("truncated") is True,
        json.dumps({k: v for k, v in whole.items() if k != "content"})[:250],
    )
    s.check(
        "and a summary is never claimable",
        str(whole.get("file_hash", "")).startswith("partial:") and not whole.get("content_hash"),
        json.dumps({k: v for k, v in whole.items() if k != "content"})[:250],
    )

    w1 = await s.call("read", path=big, offset=FIRST_WINDOW[0], limit=FIRST_WINDOW[1])
    s.account(w1, paging=True)
    token = w1.get("coverage_token")
    s.check(
        "a ranged read earns a coverage token",
        bool(token),
        json.dumps({k: v for k, v in w1.items() if k != "content"})[:250],
    )

    again = await s.call(
        "read", path=big, offset=FIRST_WINDOW[0], limit=FIRST_WINDOW[1], **s.hash_for(token)
    )
    s.account(again, paging=True)
    if s.carry_hashes:
        s.check(
            "re-reading a window you hold costs nothing",
            again.get("unchanged") is True,
            json.dumps(again)[:200],
        )
        sub = await s.call(
            "read",
            path=big,
            offset=SUB_WINDOW[0],
            limit=SUB_WINDOW[1] - SUB_WINDOW[0],
            **s.hash_for(token),
        )
        s.account(sub, paging=True)
        s.check(
            "a sub-window of a held window costs nothing either",
            sub.get("unchanged") is True,
            json.dumps(sub)[:200],
        )
    else:
        s.check(
            "[naive] the same window is delivered again",
            "content" in again,
            json.dumps(again)[:150],
        )

    w2 = await s.call(
        "read",
        path=big,
        offset=SECOND_WINDOW[0],
        limit=SECOND_WINDOW[1] - SECOND_WINDOW[0],
        **s.hash_for(token),
    )
    s.account(w2, paging=True)
    s.check(
        "a new window is delivered",
        "content" in w2,
        json.dumps({k: v for k, v in w2.items() if k != "content"})[:200],
    )
    if s.carry_hashes:
        s.check(
            "and folds into a widened token",
            bool(w2.get("coverage_token")) and w2.get("coverage_token") != token,
            json.dumps({k: v for k, v in w2.items() if k != "content"})[:250],
        )

    # -- 10. add a file, then extend it ------------------------------------
    print("\n  [10] add a module and extend it")

    retry_py = src / "retry.py"
    retry_module = (
        '"""Retry helpers."""\n\n'
        "from __future__ import annotations\n\n\n"
        "def backoff(attempt: int, base: float) -> float:\n"
        '    """Exponential backoff for one attempt."""\n'
        "    return base * (2**attempt)\n"
    )
    wr = await s.call("write", path=str(retry_py), content=retry_module)
    s.check("write creates the module", wr.get("status") == "created", json.dumps(wr)[:200])
    s.check("a full write is always claimable", bool(wr.get("content_hash")), json.dumps(wr)[:200])
    remember(retry_py, wr)

    retry_tail = (
        "\n\ndef should_retry(status: int) -> bool:\n"
        '    """Retry on transient upstream failures."""\n'
        "    return status in {429, 502, 503, 504}\n"
    )
    ap = await s.call(
        "write",
        path=str(retry_py),
        content=retry_tail,
        append=True,
        **s.hash_for(held.get(str(retry_py))),
    )
    if s.carry_hashes:
        s.check(
            "an append backed by proof stays claimable",
            bool(ap.get("content_hash")),
            json.dumps(ap)[:250],
        )
        remember(retry_py, ap)
    else:
        s.check(
            "[naive] an append with nothing to vouch for is not claimable",
            bool(ap.get("file_hash")) and not ap.get("content_hash"),
            json.dumps(ap)[:250],
        )

    # -- 11. confirm the cache tracked every mutation ----------------------
    print("\n  [11] confirm the cache followed the edits")

    post = await s.call("grep", pattern="retry=True", path=str(src))
    s.check(
        "grep sees text written this session",
        post.get("total_matches") == 2,
        json.dumps({k: v for k, v in post.items() if k != "files"})[:250],
    )

    gone = await s.call("grep", pattern=r"self\._send\(method, path\)", path=str(src))
    s.check(
        "and no longer sees what was edited away",
        gone.get("total_matches") == 0,
        json.dumps({k: v for k, v in gone.items() if k != "files"})[:250],
    )

    findable = await s.call("search", query="backoff exponential attempt", k=3)
    s.check(
        "search finds the module added this session",
        any(m.get("path", "").endswith("retry.py") for m in findable.get("matches", [])),
        json.dumps([m.get("path") for m in findable.get("matches", [])])[:250],
    )

    # -- 12. an image, because agents read screenshots ---------------------
    print("\n  [12] look at an image")

    logo = REPO_ROOT / "assets" / "logo-128.png"
    if logo.exists():
        im = await s.call("read_image", path=str(logo))
        s.check(
            "read_image returns a typed image",
            im.get("mime") == "image/png" and im.get("size", 0) > 0,
            json.dumps(im)[:200],
        )
    else:
        s.ledger.notes.append("read_image skipped: assets/logo-128.png absent")

    # -- 13. clean up after the session ------------------------------------
    print("\n  [13] clean up")

    scratch = src / "scratch_notes.md"
    await s.call("write", path=str(scratch), content="# scratch\n\ntemporary working notes\n")
    dry = await s.call("delete", path=str(scratch), dry_run=True)
    s.check(
        "delete previews before removing",
        dry.get("status") == "would_delete" and scratch.exists(),
        json.dumps(dry)[:200],
    )

    rm = await s.call("delete", path=str(scratch))
    s.check(
        "delete removes the file",
        rm.get("status") == "deleted" and not scratch.exists(),
        json.dumps(rm)[:200],
    )

    evicted = await s.call("grep", pattern="temporary working notes", path=str(src))
    s.check(
        "and evicts it from the cache",
        evicted.get("total_matches") == 0,
        json.dumps({k: v for k, v in evicted.items() if k != "files"})[:250],
    )

    # -- 14. what the session cost ----------------------------------------
    print("\n  [14] account for the session")

    st = await s.call("stats")
    sess = st.get("session", {})
    s.check("stats reports the session", sess.get("tool_calls") is not None, json.dumps(sess)[:250])
    s.ledger.notes.append(
        f"server-side: tokens_saved={sess.get('tokens_saved')} "
        f"hit_rate={sess.get('hit_rate_pct')}% diffs_served={sess.get('diffs_served')}"
    )

    cl = await s.call("clear")
    s.check(
        "clear empties the cache it owns",
        cl.get("status") == "cleared" and cl.get("count", 0) > 0,
        json.dumps(cl)[:200],
    )
    reseed = await s.call("read", path=client_py)
    s.account(reseed)
    s.check("and the next read re-seeds from disk", "content" in reseed, json.dumps(reseed)[:150])


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------


def prepare_cache(cache_dir: Path) -> bool:
    """Give the server its own cache, reusing the real tokenizer if present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_src = CACHE_DIR / "tokenizer"
    if not (tokenizer_src / "o200k_base.tiktoken").exists():
        return False
    dst = cache_dir / "tokenizer"
    if not dst.exists():
        try:
            dst.symlink_to(tokenizer_src, target_is_directory=True)
        except OSError:
            shutil.copytree(tokenizer_src, dst)
    return True


async def run_one(root: Path, *, carry_hashes: bool, verbose: bool) -> Session:
    label = "hash-disciplined" if carry_hashes else "naive (hashes discarded)"
    print(f"\n{'=' * 72}\n  RUN: {label}\n{'=' * 72}")

    project = root / ("disciplined" if carry_hashes else "naive")
    files = build_project(project)

    cache_dir = root / f"cache-{'d' if carry_hashes else 'n'}"
    prepare_cache(cache_dir)

    env = os.environ.copy()
    env["SEMANTIC_CACHE_DIR"] = str(cache_dir)
    env["LOG_LEVEL"] = "ERROR"
    env["TOOL_TIMEOUT"] = "120"

    transport = StdioTransport(
        command="uv",
        args=["run", "semantic-cache-mcp"],
        env=env,
        cwd=str(REPO_ROOT),
        keep_alive=False,
        log_file=root / f"server-{'d' if carry_hashes else 'n'}.log",
    )

    async with Client(transport, timeout=180, init_timeout=120) as client:
        tools = {t.name for t in await client.list_tools()}
        print(f"\n  server exposes {len(tools)} tools: {', '.join(sorted(tools))}")
        s = Session(client=client, carry_hashes=carry_hashes, verbose=verbose)
        s.check(
            "every tool this workflow uses is exposed",
            tools
            >= {
                "read",
                "batch_read",
                "write",
                "edit",
                "batch_edit",
                "edit_preview",
                "grep",
                "glob",
                "search",
                "stats",
                "delete",
                "clear",
                "read_image",
            },
            str(sorted(tools)),
        )
        await run_workflow(s, project, files)
    return s


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="keep the scratch tree")
    ap.add_argument("--verbose", action="store_true", help="echo every tool response")
    args = ap.parse_args()

    root = Path(tempfile.mkdtemp(prefix="scmcp-agentic-"))
    if not prepare_cache(root / "probe"):
        print("The tokenizer is not cached; the server would download it on start.")
        print("Run any tool once against the real server first, then retry.")
        return 2

    try:
        disciplined = await run_one(root, carry_hashes=True, verbose=args.verbose)
        naive = await run_one(root, carry_hashes=False, verbose=args.verbose)
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"\nscratch kept at {root}")

    print(f"\n{'=' * 72}\n  LEDGER\n{'=' * 72}")
    d, n = disciplined.ledger, naive.ledger
    print(f"  {'':32s} {'disciplined':>14s} {'naive':>14s}")
    print(
        f"  {'working-set chars delivered':32s}"
        f" {d.delivered_tokens:>14,d} {n.delivered_tokens:>14,d}"
    )
    print(f"  {'large-file paging chars':32s} {d.paging_chars:>14,d} {n.paging_chars:>14,d}")
    print(f"  {'full bodies sent':32s} {d.full_bodies:>14,d} {n.full_bodies:>14,d}")
    print(f"  {'diffs instead of bodies':32s} {d.diffs_received:>14,d} {n.diffs_received:>14,d}")
    print(f"  {'answered unchanged':32s} {d.unchanged_hits:>14,d} {n.unchanged_hits:>14,d}")
    if n.delivered_tokens:
        saved = 1 - (d.delivered_tokens / n.delivered_tokens)
        print(f"\n  on the working set, carrying hashes cut delivered content by {saved:.1%}")
    if n.delivered_tokens + n.paging_chars:
        overall = 1 - (
            (d.delivered_tokens + d.paging_chars) / (n.delivered_tokens + n.paging_chars)
        )
        print(f"  across the whole session, including paging, {overall:.1%}")
    for note in d.notes:
        print(f"  · {note}")

    print(f"\n{'=' * 72}")
    failures = disciplined.failures + naive.failures
    total = disciplined.checks_passed + naive.checks_passed
    print(f"  {total} checks passed, {len(failures)} failed")
    for name, detail in failures:
        print(f"    FAIL {name}: {detail[:200]}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
