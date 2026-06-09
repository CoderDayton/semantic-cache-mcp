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
- **Cause:** The server only answers `unchanged` when the same session already received the file, or when you pass a `known_hash` that still matches. After context compression you no longer hold the text.
- **Fix:** Read the file again without `known_hash` and you get full content back. If you only need part of it, use `read` with `offset`/`limit`.

**Stale content returned**
- **Cause:** File was modified outside normal flow (e.g., by another process) and the mtime wasn't updated
- **Fix:** Use `clear` to reset the cache, or delete `~/.cache/semantic-cache-mcp/docstore.db` and restart

**`search` returns no results / stale results**
- **Cause:** Only cached files are searched. New or unread files aren't in the cache yet.
- **Fix:** Seed the cache with `read` or `batch_read` first.

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
