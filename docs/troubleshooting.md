# Troubleshooting

## Tokenizer Issues

**"Failed to download tokenizer: ..."**
- **Cause:** Failed to download `o200k_base.tiktoken` from OpenAI; the error after the colon says why (timeout, HTTP error, over the 32 MB ceiling)
- **Verify:** Check internet connectivity and that `openaipublic.blob.core.windows.net` is reachable
- **Effect:** Token counts use a word-and-character heuristic instead of exact o200k counts, so reported figures are approximate. The server does not retry again in that process.
- **Fix:** The server retries on next startup. To force a retry now, delete `~/.cache/semantic-cache-mcp/tokenizer/` and restart.

**"Hash verification failed, re-downloading"**
- **Cause:** Corrupted or incomplete download
- **Fix:** Delete `~/.cache/semantic-cache-mcp/tokenizer/` and restart. A fresh download will be verified automatically.

---

## Cache Issues

**"Database is locked"**
- **Cause:** More than one server process writing the same database for longer than SQLite's 5-second busy timeout. By default every client shares one daemon, so this happens with `SCMCP_SHARED_SERVER=0`, or when clients with different configuration (different `PATH`, `LOG_LEVEL`, any `SCMCP_` variable) get separate daemons that point at the same `SEMANTIC_CACHE_DIR`.
- **Fix:** Give the clients identical environments, or point them at different `SEMANTIC_CACHE_DIR`s. List the daemons with `pgrep -af "semantic_cache_mcp.server serve"`.

**Cache not reducing tokens**
- **Cause:** Files haven't been read yet (cold cache), or `known_hash` isn't being echoed back. The server answers `unchanged` or a diff only to a caller that sends the `content_hash` of the version it holds.
- **Fix:** The first `read` of any file populates the cache. Pass the returned `content_hash` as `known_hash` on later reads. Use `stats` to verify files are being cached.

**All files reporting "unchanged" after model context compression**
- **Cause (fixed in 0.5.2):** `batch_read` used to answer `unchanged` for any file the *server* had cached. The cache is on disk and outlives your context window, so after a compaction or a `/clear` you were told you already had files you had never seen.
- **Now:** `unchanged` is only ever produced for a file whose `content_hash` you echoed back — `known_hash` on `read`, or a `known_hashes` entry on `batch_read`. Anything you cannot vouch for is sent in full, so simply omitting the hashes after a compaction is always safe.
- **Note:** a partial read (a line range, or a summary of a large file) reports its hash as `file_hash`, prefixed `partial:`. It identifies the file but is not proof you hold it and will not be honoured as a `known_hash`.
- **Since 0.5.3:** a ranged read additionally returns a `coverage_token` recording the lines it did deliver, and that *is* honoured as a `known_hash` — for those lines only. Re-reading a window you hold answers `unchanged`, a new window widens the token, and full coverage upgrades to a claimable `content_hash`. Like every other claim it is yours to make honestly: pass it back only while you still hold the lines it names, and after a compaction just omit it.

**Stale content returned**
- **Cause:** File was modified outside normal flow (e.g., by another process) and the mtime wasn't updated
- **Fix:** Use `clear` to reset the cache. To delete `~/.cache/semantic-cache-mcp/docstore.db` instead, stop the shared daemon first (see [Shared Server](#shared-server)), then restart your client.
- **Fixed in 0.5.2:** two paths that produced this on their own. A write landing between a cold read's `aread_bytes` and its `stat` was cached as pre-write content with a post-write mtime, so the entry looked fresh forever and the next edit wrote it back over the newer file — the stat is now taken first. And a file rewritten without changing its mtime (`cp -p`, `tar -x`, `touch -d`) is now detected by comparing content hashes on the full-read path, not mtimes.

**`search` returns no results / stale results**
- **Cause:** Only cached files are searched, and since 0.6.0 `search` without `directory` and `grep` with a relative or no `path` cover only the client's current project. New or unread files aren't in the cache yet.
- **Fix:** Seed the cache with `read`, `batch_read` or `warm` first. To search another project, pass an absolute `directory` or `path`.
- **Note (0.5.2+):** query terms are joined with `OR`, so one word your corpus doesn't contain no longer empties the results — it just stops contributing to the ranking. Before 0.5.2 a query like `"password hashing session token"` returned nothing unless a single file held all four words. An empty result now means no cached file matched *any* term.

**`grep` fails with "invalid regex pattern" or "pattern too long"**
- **Cause (0.5.2+):** the pattern didn't compile, or it exceeds the 1,000-character ReDoS cap. Earlier versions logged a warning and returned no matches, which is indistinguishable from a genuine miss.
- **Fix:** correct the regex, or pass `fixed_string=true` to match the text literally (`grep pattern="foo(bar" fixed_string=true`).

**`grep` rejects a pattern that looks valid to you**
- **Cause (0.5.3+):** the pattern has a repeatable group wrapping an unbounded quantifier — `(a+)+`, `(\w*)*`, `(x+){2,}` — which can backtrack exponentially. Left to run, `(a+)$` against 40 characters exceeded two minutes and could not be interrupted, so it is refused before compilation instead.
- **Fix:** the error names a safe equivalent; `(a+)+` almost always wants to be `a+`. If you meant the text literally, pass `fixed_string=true`.
- **Note:** the check is on shape, so it can refuse a pattern that would have been fine in practice. It never accepts one it should have refused silently — a rejection is always an error, never an empty result.

**Repeated `search` queries return instantly (< 1 ms)**
- **Cause (0.4.6+):** `SemanticCache` keeps an in-session 32-entry LRU of search results, keyed on `(query, k, directory)`. Identical queries skip the BM25 round-trip entirely.
- **When this is wrong:** the LRU is invalidated on every cache mutation (`put`, `clear`, `delete_path`, `update_mtime`), so callers never see results that predate a write. If you suspect staleness, run `clear` to flush state.

---

## Server Hangs

**Server freezes during read/write/search operations**
- **Cause (pre-0.3.4):** SQLite catalog scans and subprocess formatter calls ran synchronously on the asyncio event loop, blocking all other operations for the duration.
- **Fix:** Upgrade to 0.3.4+. All blocking calls now run in thread pools. Storage I/O uses a dedicated single-thread executor to prevent pool starvation.

**Tool call returns "timed out after Ns"**
- **Cause:** The call exceeded `TOOL_TIMEOUT` (default 30 s). The executor is reset, so the next call works without a restart.
- **Fix:** Raise `TOOL_TIMEOUT` for slow disks or very large files.

**Server dies with a closed pipe when the client disconnects**
- **Cause:** On older builds, a write to the closed pipe killed the process before storage cleanup.
- **Fix:** Upgrade. `SIGPIPE` is now ignored so the write raises `BrokenPipeError` and the normal shutdown path runs. The tool worker is given 8 s to stop before it is forced.

---

## Performance Issues

**High memory usage**
- **Cause:** A large cache holds file text and metadata in SQLite, plus the in-memory eviction index. There is no model held in memory.
- **Options:**
  - Use `clear` to evict cached entries and reduce DB size
  - Reduce `MAX_CACHE_ENTRIES` to lower the number of cached entries

**`docstore.db` is far larger than the files it caches**
- **Cause (fixed in 0.5.3):** two independent accumulations, neither visible in `stats`. FTS5 records a deletion as an index entry rather than removing one, so a store that evicts files keeps the vocabulary of every file it ever held; and SQLite retains freed pages instead of returning them, pinning the file at its high-water mark. A real store holding 244 files and 4.2 MB of text had grown to 153 MB.
- **Now:** shutdown merges the index and hands the freed pages back, and stores are opened in incremental auto-vacuum mode. That same store settles at 17.5 MB.
- **Upgrading:** a store created before 0.5.3 is rewritten once on its first open (~50 ms for 153 MB) to enable the mode. Nothing is lost — the schema and every document are untouched, and the log line reads `enabled incremental auto-vacuum`.
- **Note:** the merge only runs on a clean shutdown. A store whose server is repeatedly `SIGKILL`ed never gets one, and will drift upward again. The shared daemon's idle exit and `SIGTERM` are clean shutdowns.

**`docstore.db-wal` is far larger than the database**
- **Cause (fixed in 0.7.0):** the WAL grew without bound while another process held a read snapshot. One live store had a 126 MB WAL next to a 22 MB database.
- **Now:** the WAL is capped at 16 MB and truncated on close. A reader mid-transaction can delay the truncate by up to 1 s, after which the WAL is left as it is until the next close.

**Glob timeout**
- **Cause:** Very broad pattern (e.g., `**/*.py` on a large monorepo) exceeds the 5-second timeout
- **Fix:** Narrow the pattern or add a `directory` argument to limit scope. The timeout is a safety guard, and results up to the timeout are still returned.

---

## Shared Server

The `semantic-cache-mcp` command relays to one per-user daemon (`python -m semantic_cache_mcp.server serve`), starting it when none is running. Relay warnings go to the client's stderr, prefixed `semantic-cache-mcp:`. The daemon itself runs detached with no stderr, so its messages are only in the log file (`logs/` under the cache dir). Design: [architecture.md](architecture.md#process-model).

**`shared server unavailable (...); serving in-process`**
- **Cause:** The relay could not use the daemon endpoint. The usual one is an unsafe runtime directory: `semantic-cache-mcp-<uid>/` under `$XDG_RUNTIME_DIR` (or the temp dir) exists but is not a directory you own with mode `0700`, and the message reads `... must be a directory owned by this user, mode 0700`.
- **Effect:** This client gets its own server instead of the shared one. Nothing is lost; memory use is higher.
- **Fix:** `chmod 700` the directory if you own it, or remove it if it is not yours to keep. Set `XDG_RUNTIME_DIR` to a directory you own.

**`shared server did not start; serving in-process` / `shared server never answered; serving in-process`**
- **Cause:** The daemon exited before listening, did not listen within 60 s, or never answered the handshake within 10 s on two attempts. `shared server dropped the connection (...)` precedes the second case.
- **Fix:** Read the newest file in `logs/` for the daemon's error. Set `SCMCP_SHARED_SERVER=0` to skip the daemon entirely.

**Opt out of sharing**
- Set `SCMCP_SHARED_SERVER=0` (also `false`, `no`, `off`) in the client's `env`. Each client then runs its own server, as before the shared daemon.

**Stop a daemon**
- It exits by itself after `SCMCP_IDLE_TIMEOUT` seconds (default 600) with no client connected.
- To stop it now on Linux/macOS: `pkill -TERM -f "semantic_cache_mcp.server serve"`. It stops accepting, removes its endpoint, and lets open connections finish. On Windows, end the `python` process running `semantic_cache_mcp.server serve`.
- Stop it before deleting `docstore.db`, or the daemon keeps the old file open.

**Old behavior after an upgrade**
- A daemon is keyed by version and install path, so a new build starts a new daemon rather than reusing the old one. The old daemon exits when idle. If you see old behavior, check for a second daemon with `pgrep -af "semantic_cache_mcp.server serve"`.

---

## Cache Locations

| Path                                         | Contents                     |
|----------------------------------------------|------------------------------|
| `~/.cache/semantic-cache-mcp/docstore.db`    | Primary store (raw text and metadata, FTS5 index) |
| `~/.cache/semantic-cache-mcp/metrics.db`     | Session metrics (token savings, tool calls, lifetime stats) |
| `~/.cache/semantic-cache-mcp/cache.db`       | Legacy SQLite from pre-0.3.0 (deleted at startup if it has the old schema; safe to delete) |
| `~/.cache/semantic-cache-mcp/tokenizer/`     | o200k_base BPE tokenizer file |
| `~/.cache/semantic-cache-mcp/logs/`          | Daily log files, `semantic-cache-mcp-YYYY-MM-DD.log` (`LOG_DIR` overrides) |
| `$XDG_RUNTIME_DIR/semantic-cache-mcp-<uid>/` | Shared daemon sockets, port files and locks (system temp dir if unset; Windows: `%LOCALAPPDATA%\semantic-cache-mcp\run\`) |

Paths assume Linux without `SEMANTIC_CACHE_DIR` or `XDG_CACHE_HOME`; see [env_variables.md](env_variables.md) for the others.

---

## Debug Logging

Enable verbose logging to diagnose issues:

```bash
export LOG_LEVEL=DEBUG
semantic-cache-mcp
```

| Level     | What is logged                                                     |
|-----------|--------------------------------------------------------------------|
| `INFO`    | Server start, file cache and eviction events                       |
| `DEBUG`   | Cache hits and misses, chunk storage, SQL timing                   |
| `WARNING` | Hash verification, tokenizer fallback                              |
| `ERROR`   | Unhandled exceptions, startup failures                             |

Output goes to stderr and to a daily file under `logs/` in the cache dir. The shared daemon has no stderr, so read the file. Changing `LOG_LEVEL` gives the client a new daemon.

---

## Common Log Messages

| Message                              | Meaning                        | Action                     |
|--------------------------------------|--------------------------------|----------------------------|
| `Loading o200k_base tokenizer from ...` (DEBUG) | Tokenizer loading   | Wait for completion        |
| `Cache hit: /path` (DEBUG)           | File found unchanged in cache  | Working correctly          |
| `Cached file: /path (N tokens)` (DEBUG) | File stored in cache        | Working correctly          |
| `Cache eviction: removed N files ...` | W-TinyLFU cleanup triggered   | Normal, no action needed   |
| `Hash verification failed, re-downloading` | Corrupted tokenizer file | Nothing; it re-downloads. If it recurs, delete the tokenizer dir, restart |
| `Failed to download tokenizer: ...`  | Download failed                | Check internet; token counts approximate |
| `shared server unavailable (...); serving in-process` | Daemon endpoint unusable | See [Shared Server](#shared-server) |
| `Invalid <NAME>=...; using ...`      | Malformed environment variable | Fix the value; the default is in use |

---

## Getting Help

1. **Enable debug logging:** `LOG_LEVEL=DEBUG semantic-cache-mcp`
2. **Check the cache:** `stats` tool shows file count, token totals, and savings percentage
3. **Reset state:** `clear` tool resets all cache entries; deleting `~/.cache/semantic-cache-mcp/` does a full reset
4. **Report issues:** [GitHub Issues](https://github.com/CoderDayton/semantic-cache-mcp/issues)

---

[← Back to README](../README.md)
