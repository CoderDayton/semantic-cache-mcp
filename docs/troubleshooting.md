# Troubleshooting

## Tokenizer Issues

**"Tokenizer not loaded, using heuristic fallback"**
- **Cause:** Failed to download `o200k_base.tiktoken` from OpenAI
- **Verify:** Check internet connectivity and that `openaipublic.blob.core.windows.net` is reachable
- **Effect:** Token counts use a heuristic (`len(text) / 4`), which is functional but approximate
- **Fix:** The server retries on next startup. To force a retry now, delete `~/.cache/semantic-cache-mcp/tokenizer/` and restart.

**"Hash verification failed, re-downloading"**
- **Cause:** Corrupted or incomplete download
- **Fix:** Delete `~/.cache/semantic-cache-mcp/tokenizer/` and restart. A fresh download will be verified automatically.

---

## Cache Issues

**"Database is locked"**
- **Cause:** Multiple MCP server instances accessing the same database
- **Fix:** Ensure only one instance of `semantic-cache-mcp` is running. Check with `pgrep semantic-cache-mcp`.

**Cache not reducing tokens**
- **Cause:** Files haven't been read yet (cold cache), or `diff_mode=False` is set
- **Fix:** The first `read` of any file populates the cache. Subsequent reads return diffs. Use `stats` to verify files are being cached.

**All files reporting "unchanged" after model context compression**
- **Cause (fixed in 0.5.2):** `batch_read` used to answer `unchanged` for any file the *server* had cached. The cache is on disk and outlives your context window, so after a compaction or a `/clear` you were told you already had files you had never seen.
- **Now:** `unchanged` is only ever produced for a file whose `content_hash` you echoed back — `known_hash` on `read`, or a `known_hashes` entry on `batch_read`. Anything you cannot vouch for is sent in full, so simply omitting the hashes after a compaction is always safe.
- **Note:** a partial read (a line range, or a summary of a large file) reports its hash as `file_hash`, prefixed `partial:`. It identifies the file but is not proof you hold it and will not be honoured as a `known_hash`.

**Stale content returned**
- **Cause:** File was modified outside normal flow (e.g., by another process) and the mtime wasn't updated
- **Fix:** Use `clear` to reset the cache, or delete `~/.cache/semantic-cache-mcp/docstore.db` and restart
- **Fixed in 0.5.2:** two paths that produced this on their own. A write landing between a cold read's `aread_bytes` and its `stat` was cached as pre-write content with a post-write mtime, so the entry looked fresh forever and the next edit wrote it back over the newer file — the stat is now taken first. And a file rewritten without changing its mtime (`cp -p`, `tar -x`, `touch -d`) is now detected by comparing content hashes on the full-read path, not mtimes.

**`search` returns no results / stale results**
- **Cause:** Only cached files are searched. New or unread files aren't in the cache yet.
- **Fix:** Seed the cache with `read` or `batch_read` first.
- **Note (0.5.2+):** query terms are joined with `OR`, so one word your corpus doesn't contain no longer empties the results — it just stops contributing to the ranking. Before 0.5.2 a query like `"password hashing session token"` returned nothing unless a single file held all four words. An empty result now means no cached file matched *any* term.

**`grep` fails with "invalid regex pattern" or "pattern too long"**
- **Cause (0.5.2+):** the pattern didn't compile, or it exceeds the 1,000-character ReDoS cap. Earlier versions logged a warning and returned no matches, which is indistinguishable from a genuine miss.
- **Fix:** correct the regex, or pass `fixed_string=true` to match the text literally (`grep pattern="foo(bar" fixed_string=true`).

**Repeated `search` queries return instantly (< 1 ms)**
- **Cause (0.4.6+):** `SemanticCache` keeps an in-session 32-entry LRU of search results, keyed on `(query, k, directory)`. Identical queries skip the BM25 round-trip entirely.
- **When this is wrong:** the LRU is invalidated on every cache mutation (`put`, `clear`, `delete_path`, `update_mtime`), so callers never see results that predate a write. If you suspect staleness, run `clear` to flush state.

---

## Server Hangs

**Server freezes during read/write/search operations**
- **Cause (pre-0.3.4):** SQLite catalog scans and subprocess formatter calls ran synchronously on the asyncio event loop, blocking all other operations for the duration.
- **Fix:** Upgrade to 0.3.4+. All blocking calls now run in thread pools. Storage I/O uses a dedicated single-thread executor to prevent pool starvation.

**Server hangs on shutdown (SIGTERM/SIGINT)**
- **Cause (pre-0.3.4):** No signal handlers; SIGTERM killed the process before storage cleanup.
- **Fix:** 0.3.4+ installs graceful shutdown handlers. First signal drains in-flight operations (8s timeout) and closes cleanly. Second signal forces `os._exit()`.

---

## Performance Issues

**High memory usage**
- **Cause:** A large cache holds file text and metadata in SQLite, plus the in-memory eviction index. There is no model held in memory.
- **Options:**
  - Use `clear` to evict cached entries and reduce DB size
  - Reduce `MAX_CACHE_ENTRIES` to lower the number of cached entries

**Glob timeout**
- **Cause:** Very broad pattern (e.g., `**/*.py` on a large monorepo) exceeds the 5-second timeout
- **Fix:** Narrow the pattern or add a `directory` argument to limit scope. The timeout is a safety guard, and results up to the timeout are still returned.

---

## Cache Locations

| Path                                         | Contents                     |
|----------------------------------------------|------------------------------|
| `~/.cache/semantic-cache-mcp/docstore.db`    | Primary store (raw text and metadata, FTS5 index) |
| `~/.cache/semantic-cache-mcp/metrics.db`     | Session metrics (token savings, tool calls, lifetime stats) |
| `~/.cache/semantic-cache-mcp/cache.db`       | Legacy SQLite from pre-0.3.0 (only inspected at startup for migration; safe to delete) |
| `~/.cache/semantic-cache-mcp/tokenizer/`     | o200k_base BPE tokenizer file |

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

---

## Common Log Messages

| Message                              | Meaning                        | Action                     |
|--------------------------------------|--------------------------------|----------------------------|
| `Loading o200k_base tokenizer`       | Tokenizer downloading/loading  | Wait for completion        |
| `Cache hit: /path`                   | File found unchanged in cache  | Working correctly          |
| `Cached file: /path (N tokens)`      | File stored in cache           | Working correctly          |
| `Cache eviction: removed N entries`  | W-TinyLFU cleanup triggered    | Normal, no action needed   |
| `Hash verification failed`           | Corrupted download             | Delete tokenizer dir, restart |
| `Tokenizer not loaded, using heuristic fallback` | Download failed   | Check internet; token counts approximate |

---

## Getting Help

1. **Enable debug logging:** `LOG_LEVEL=DEBUG semantic-cache-mcp`
2. **Check the cache:** `stats` tool shows file count, token totals, and savings percentage
3. **Reset state:** `clear` tool resets all cache entries; deleting `~/.cache/semantic-cache-mcp/` does a full reset
4. **Report issues:** [GitHub Issues](https://github.com/CoderDayton/semantic-cache-mcp/issues)

---

[← Back to README](../README.md)
