# Troubleshooting

## Embedding Model Issues

**"Embedding model not loaded"**
- **Cause:** First startup triggers a one-time download (~500MB)
- **Fix:** Wait for download to complete. Progress is logged at INFO level. Subsequent starts are fast.
- **Model location:** `~/.cache/semantic-cache-mcp/models/`

**Slow first request after restart**
- **Cause:** ONNX Runtime warms up the model on the first inference
- **Fix:** The server performs a warmup pass at startup. If latency is still high on first use, check available memory — the model needs ~500MB resident.

**"Embedding failed" in logs**
- **Cause:** Model not yet loaded, or ONNX Runtime error
- **Fix:** Check `LOG_LEVEL=DEBUG` output. Verify `~/.cache/semantic-cache-mcp/models/` exists and is not empty.

---

## Tokenizer Issues

**"Tokenizer not loaded, using heuristic fallback"**
- **Cause:** Failed to download `o200k_base.tiktoken` from OpenAI
- **Verify:** Check internet connectivity and that `openaipublic.blob.core.windows.net` is reachable
- **Effect:** Token counts use a heuristic (`len(text) / 4`) — functional but approximate
- **Fix:** The server retries on next startup. To force a retry now, delete `~/.cache/semantic-cache-mcp/tokenizer/` and restart.

**"Hash verification failed, re-downloading"**
- **Cause:** Corrupted or incomplete download
- **Fix:** Delete `~/.cache/semantic-cache-mcp/tokenizer/` and restart — a fresh download will be verified automatically

---

## Cache Issues

**"Database is locked"**
- **Cause:** Multiple MCP server instances accessing the same database
- **Fix:** Ensure only one instance of `semantic-cache-mcp` is running. Check with `pgrep semantic-cache-mcp`.

**Cache not reducing tokens**
- **Cause:** Files haven't been read yet (cold cache), or `diff_mode=False` is set
- **Fix:** The first `read` of any file populates the cache. Subsequent reads return diffs. Use `stats` to verify files are being cached.

**All files reporting "unchanged" after model context compression**
- **Cause:** The cache tracks filesystem state (mtime), not LLM context state. After context compression, the LLM no longer has the file content, but the cache still reports the file as unchanged.
- **Fix:** Call `batch_read` or `read` with `diff_mode=False` to force full content return, bypassing the cache hit path.
  ```
  batch_read paths="src/**/*.py" diff_mode=false
  ```

**Stale content returned**
- **Cause:** File was modified outside normal flow (e.g., by another process) and the mtime wasn't updated
- **Fix:** Use `clear` to reset the cache, or delete `~/.cache/semantic-cache-mcp/cache.db` and restart

**`search` or `similar` returns no results**
- **Cause:** These tools only search cached files. If the cache is empty or the relevant files haven't been read, nothing will match.
- **Fix:** Seed the cache first with `read` or `batch_read`, then call `search` or `similar`.

---

## Performance Issues

**Slow first startup**
- **Cause:** Embedding model download (~500MB) on first use, followed by ONNX initialization
- **Expected:** Normal on first use. Model is cached in `~/.cache/semantic-cache-mcp/models/`.

**High memory usage**
- **Cause:** Embedding model holds ~500MB resident + SQLite page cache (up to 64MB)
- **Options:**
  - Use `clear` to evict cached entries and reduce DB size
  - Reduce `MAX_CACHE_ENTRIES` to lower the number of cached embeddings

**Glob timeout**
- **Cause:** Very broad pattern (e.g., `**/*.py` on a large monorepo) exceeds the 5-second timeout
- **Fix:** Narrow the pattern or add a `directory` argument to limit scope. The timeout is a safety guard — results up to the timeout are still returned.

---

## Configuration Reference

| Variable                    | Default    | Description                                          |
|-----------------------------|------------|------------------------------------------------------|
| `LOG_LEVEL`                 | `INFO`     | Verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`       |
| `TOOL_OUTPUT_MODE`          | `compact`  | Response detail: `compact`, `normal`, `debug`        |
| `TOOL_MAX_RESPONSE_TOKENS`  | `0`        | Global response cap (0 = disabled)                   |
| `MAX_CONTENT_SIZE`          | `100000`   | Max bytes returned by `read`                         |
| `MAX_CACHE_ENTRIES`         | `10000`    | Max entries before LRU-K eviction                    |

---

## Cache Locations

| Path                                         | Contents                     |
|----------------------------------------------|------------------------------|
| `~/.cache/semantic-cache-mcp/cache.db`       | SQLite database (chunks + file metadata) |
| `~/.cache/semantic-cache-mcp/tokenizer/`     | o200k_base BPE tokenizer file |
| `~/.cache/semantic-cache-mcp/models/`        | FastEmbed ONNX model (~500MB) |

---

## Debug Logging

Enable verbose logging to diagnose issues:

```bash
export LOG_LEVEL=DEBUG
semantic-cache-mcp
```

| Level     | What is logged                                                     |
|-----------|--------------------------------------------------------------------|
| `INFO`    | Server start, model loading, file cache/eviction events            |
| `DEBUG`   | Cache hits/misses, chunk storage, embedding generation, SQL timing |
| `WARNING` | Embedding failures, hash verification, tokenizer fallback          |
| `ERROR`   | Unhandled exceptions, startup failures                             |

---

## Common Log Messages

| Message                              | Meaning                        | Action                     |
|--------------------------------------|--------------------------------|----------------------------|
| `Loading embedding model`            | Model initializing             | Wait for completion        |
| `Embedding model ready`              | Ready to process               | None needed                |
| `Loading o200k_base tokenizer`       | Tokenizer downloading/loading  | Wait for completion        |
| `Cache hit: /path`                   | File found unchanged in cache  | Working correctly          |
| `Cached file: /path (N tokens)`      | File stored in cache           | Working correctly          |
| `Embedding failed`                   | Model inference error          | Check DEBUG logs           |
| `Cache eviction: removed N entries`  | LRU-K cleanup triggered        | Normal — no action needed  |
| `Hash verification failed`           | Corrupted download             | Delete tokenizer dir, restart |
| `Tokenizer not loaded, using heuristic fallback` | Download failed   | Check internet; token counts approximate |

---

## Getting Help

1. **Enable debug logging:** `LOG_LEVEL=DEBUG semantic-cache-mcp`
2. **Check the cache:** `stats` tool shows file count, token totals, and compression ratio
3. **Reset state:** `clear` tool resets all cache entries; deleting `~/.cache/semantic-cache-mcp/` does a full reset
4. **Report issues:** [GitHub Issues](https://github.com/CoderDayton/semantic-cache-mcp/issues)

---

[← Back to README](../README.md)
