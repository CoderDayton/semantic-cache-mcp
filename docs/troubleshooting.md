# Troubleshooting

## Embedding Model Issues

**"Embedding model not loaded"**
- **Cause:** First startup downloads the model (~500MB)
- **Fix:** Wait for download to complete; subsequent starts are fast
- **Location:** Model cached in `~/.cache/semantic-cache-mcp/models/`

**Slow first embedding**
- **Cause:** Model initialization and ONNX warmup
- **Fix:** Server performs warmup at startup; first request should be fast

---

## Tokenizer Issues

**"Tokenizer not loaded, using heuristic fallback"**
- **Cause:** Failed to download o200k_base.tiktoken from OpenAI
- **Fix:** Check internet connection, verify firewall allows access to `openaipublic.blob.core.windows.net`
- **Workaround:** Token counts will be approximate but functional

**"Hash verification failed"**
- **Cause:** Corrupted download or file tampering
- **Fix:** Delete `~/.cache/semantic-cache-mcp/tokenizer/` and restart

---

## Cache Issues

**"Database is locked"**
- **Cause:** Multiple processes accessing cache simultaneously
- **Fix:** Ensure only one MCP server instance is running

**Cache not reducing tokens**
- **Cause:** Files are new (not yet cached) or `diff_mode=false`
- **Fix:** Read same files again to see token reduction

**Stale cache data**
- **Cause:** File was modified outside normal flow
- **Fix:** Use `clear` tool or delete `~/.cache/semantic-cache-mcp/cache.db`

---

## Performance Issues

**Slow first startup**
- **Cause:** Model download (~500MB) and initialization
- **Fix:** Normal on first use; model is cached for subsequent starts

**High memory usage**
- **Cause:** Embedding model (~500MB) plus cached entries
- **Fix:** Use `clear` tool to reset cache, or reduce `MAX_CACHE_ENTRIES` in config

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |

---

## Cache Locations

| Path | Purpose |
|------|---------|
| `~/.cache/semantic-cache-mcp/cache.db` | SQLite database |
| `~/.cache/semantic-cache-mcp/tokenizer/` | Tokenizer files |
| `~/.cache/semantic-cache-mcp/models/` | FastEmbed model files |

---

## Debug Logging

Enable verbose logging to diagnose issues:

```bash
export LOG_LEVEL=DEBUG
semantic-cache-mcp
```

Log messages include:
- `INFO`: Server start, model loading, file caching, eviction events
- `DEBUG`: Cache hits, chunk storage, embedding generation
- `WARNING`: Embedding failures, hash verification issues

---

## Common Log Messages

| Message | Meaning | Action |
|---------|---------|--------|
| `Loading embedding model` | Model initializing | Wait for completion |
| `Embedding model ready` | Ready to process | None needed |
| `Loading o200k_base tokenizer` | Tokenizer initializing | Normal, wait for completion |
| `Cache hit: /path` | File found in cache | Working as expected |
| `Cached file: /path (N tokens, M chunks)` | File stored | Working as expected |
| `Embedding failed` | Model error | Check logs for details |
| `Cache eviction: removed N entries` | LRU-K cleanup | Normal maintenance |

---

## Getting Help

1. **Check logs:** Set `LOG_LEVEL=DEBUG` for verbose output
2. **Verify installation:** Run `semantic-cache-mcp --help`
3. **Clear cache:** Delete `~/.cache/semantic-cache-mcp/` to reset
4. **Report issues:** [GitHub Issues](https://github.com/your-repo/semantic-cache-mcp/issues)

---

[← Back to README](../README.md)
