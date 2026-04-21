# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.3] - 2026-04-20

### Added

- **Explicit verbosity toggles** — `write`, `edit`, and `batch_edit` now accept `show_diff`, and `search` now accepts `show_preview`, so large payloads are opt-in when they materially affect the next decision.

### Changed

- **Lean default mutation responses** — Clean deterministic `write`, `edit`, and `batch_edit` results no longer return full diffs by default. They now expose machine-readable `diff_state` metadata and reserve full diffs for partial applies, debug mode, or explicit requests.
- **Lean unchanged reads** — Unchanged `read` responses now return `unchanged=true` without replaying cached file content by default.
- **Compressed batch/search/glob/stats payloads** — `batch_read` now returns `unchanged_count` by default instead of full unchanged path lists, skipped-file guidance moved to a summary hint, `search` omits previews by default, `glob` omits per-match `tokens`/`mtime` outside debug, and `stats` text output is shorter while preserving structured data.

### Fixed

- **Diff contract stability** — Truncated responses now preserve diff metadata instead of silently dropping it.
- **Diff state accuracy** — Unchanged writes no longer misreport `diff_omitted=true`, and diff-bearing tools now distinguish `full`, `unchanged`, and `omitted` states consistently.

## [0.4.2] - 2026-04-10

### Changed

- **simplevecdb 2.5.0** — Bumped minimum dependency to pick up the new
  `delete_collection`, `store_embeddings`, and pagination APIs along with
  fixes to delete ordering, FTS retries, and connection health probes.
- **`store_embeddings=True`** — VectorStorage now opts into SQLite-side
  embedding storage. simplevecdb 2.5.0 changed the default to `False` to save
  ~2× storage; without opting in, `get_embeddings_by_ids` would return `None`
  and break embedding-aware similarity reuse in `SemanticCache.get()`.
- **Atomic collection reset** — `clear()` and `clear_if_model_changed()` now
  call `delete_collection()`, which drops the SQLite tables, FTS index, and
  usearch file in one call, replacing the previous per-id loop and manual
  file unlinks. The new helper `_reset_collection_sync()` handles the
  startup-path (no event loop) variant.
- **Sync VectorDB + manual async wrapper** — Replaced `AsyncVectorDB` with a
  direct sync `VectorDB` plus a manually-built `AsyncVectorCollection`
  wrapper. `AsyncVectorDB.collection()` does not expose `store_embeddings`
  in 2.5.0 (no kwargs forwarding, no setter), so we need the sync collection
  factory anyway. Going through the public sync `VectorDB` deletes every
  remaining `simplevecdb` private-attribute access from the project: no more
  `_db._db`, `_db._executor`, or `_collection._collection` reach-throughs.
  A new `VectorStorage.rebind_executor()` method gives `SemanticCache.reset_executor`
  a public seam to swap the IO executor after a hung worker.

## [0.4.1] - 2026-04-02

### Changed

- **Automatic cache behavior** — Removed `diff_mode` parameter from `read` and `batch_read`. The server now automatically detects whether a file is new, unchanged, or modified and returns the optimal response (full content, `"unchanged":true` marker, or unified diff). No configuration needed.

### Fixed

- **Embedding dimension mismatch guard** — `_resolve_embedding` validates vector dimensions before passing to usearch, raising `ValueError` instead of segfaulting on model change mid-session.
- **Runtime dimension check** — `clear_if_model_changed` now verifies the live index dimension matches the model, catching stale indexes even when the sidecar metadata is missing.
- **Save race condition** — `save()` skips if `close()` is already running on the daemon thread, preventing concurrent usearch saves that caused heap corruption.
- **Oversized file truncation** — Files producing >500 CDC chunks now fall back to single-doc storage instead of silently truncating content.
- **ReDoS mitigation** — Grep rejects regex patterns longer than 1,000 characters.
- **Stats crash on missing DB** — `get_stats()` handles deleted database files gracefully.

## [0.4.0] - 2026-03-30

### Added

- **`delete` tool** — Added a narrow cache-aware delete operation for one file or one symlink path, with `dry_run` support and immediate cache eviction.
- **Path-filtered `grep`** — Exact cached-content search can now be scoped to one file, suffix, or glob path filter to reduce noise and token spend.

### Changed

- **LLM tool routing prompts** — Rewrote tool docstrings and README guidance so models choose the right cache-first tool more reliably and recover cleanly from empty or unchanged results.
- **Relative path resolution** — Tool paths now resolve against the client project root instead of the server process cwd.
- **FastMCP 3.1 alignment** — Normalized tool outputs and remote dispatch behavior to match current FastMCP response handling.

### Fixed

- **Tool hangs under concurrent access** — Blocking file I/O, SQLite catalog work, and all ONNX inference paths are isolated from the event loop and serialized safely, eliminating the GPU-spin / no-response hang class under load.
- **Timeout recovery** — Added a supervised tool worker that drops and restarts wedged executors after tool timeouts or worker protocol failures without stretching the caller's timeout budget.
- **Embedding dimension detection** — Removed the hardcoded 384-dimension fallback so non-default embedding models no longer corrupt vector storage shape.
- **Stats consistency** — Internal stats counters now stay coherent across clears, rewrites, and cache refreshes.

### Performance

- **Cache hit ratio** — `read` and `batch_read` now block `diff_mode=false` for unchanged cached full-file reads so callers reuse the cached version instead of forcing redundant disk I/O.
- **Embedding reuse** — Small edits reuse cached embeddings when possible, and `similar` avoids recomputing source embeddings for fresh cached files.
- **Freshness checks** — `diff` now uses the same mtime-plus-content-hash freshness logic as read/write paths, avoiding cache misses on touch-only changes.
- **Adaptive refresh timeout** — Cache refreshes now choose a timeout based on remaining work, reducing unnecessary executor resets after slow but healthy write/edit refreshes.
- **Lower startup churn** — Removed the embedding keepalive task and unnecessary cache rewrites during worker initialization.

## [0.3.4] - 2026-03-15

### Fixed

- **Event loop blocking** — ONNX embedding inference, SQLite catalog operations, and subprocess formatter calls were running synchronously on the asyncio event loop, causing the server to hang under load. All blocking calls now run via `asyncio.to_thread()`.
- **Graceful shutdown** — SIGTERM/SIGINT handlers cancel all tasks so lifespan cleanup runs. Write/edit operations are shielded from `CancelledError` via `asyncio.shield()` to prevent file corruption. `async_close()` drains in-flight operations (8s timeout) before closing storage.
- **Use-after-close crashes** — All VectorStorage async methods now guard against closed state, returning safe defaults instead of crashing during shutdown.
- **Embedding dimension mismatch** — `_resolve_embedding` now queries the actual model dimension instead of hardcoding 384, preventing `Vector dimension 384 != index dimension N` errors with non-default models (e.g. `Snowflake/snowflake-arctic-embed-m-v2.0`).
- **`_format_file` blocking** — Replaced `subprocess.run()` with `asyncio.create_subprocess_exec()` so auto-formatting no longer freezes the server.
- **`_expand_globs` unbounded** — Added 5-second deadline to prevent recursive `**` glob patterns from blocking indefinitely.
- **Connection pool timeout** — Reduced SQLite pool wait from 10s to 5s to surface exhaustion faster.

### Performance

- **Dedicated embedding executor** — ONNX calls use a single-thread `ThreadPoolExecutor` so concurrent embeddings don't starve the default thread pool (used by storage I/O).
- **Parallel cache lookups** — `batch_smart_read` gathers all `cache.get()` calls via `asyncio.gather()` instead of N serial awaits, and reuses results in the pre-scan loop (eliminates ~N redundant SQLite queries per batch).
- **No double-fetch on diff path** — `smart_read` saves the cache entry before the sentinel-null and restores it for diff generation (eliminates 1 SQLite query per changed-file read).
- **Embedding reuse** — `find_similar_files` reuses `cached.embedding` when available instead of calling ONNX (saves 20–100ms per cached file).

## [0.3.3] - 2026-03-10

### Fixed

- **Eviction miscounting** — LRU-K eviction counted documents instead of files, under-evicting at cache capacity.
- **Semantic boundary snapping** — Zero-distance sentinel allowed worse candidates to overwrite perfect matches.
- **`HierarchicalHasher.finalize_content`** — Always returned empty chunk list due to clearing before copy.
- **SQLite connection leak** — Migration helper leaked connection on query exception.
- **Duplicate log handlers** — Module re-import added redundant stderr handlers.
- **Batch edit crash** — Non-UTF-8 files caused unhandled `UnicodeDecodeError`.
- **Shutdown hang** — Graceful shutdown could block indefinitely on client disconnect.
- **Input validation** — Hardened storage layer against missing/malformed inputs.
- **`close()` blocking** — Cache close could hang when background save was stuck.

### Changed

- Stripped padding, repetition, and template prose across all `.py` and `.md` (net −1,350 lines).

### Removed

- Dead code: `_myers_diff`, `_unified_diff_fast`, `generate_diff_streaming`, `invert_diff`, `apply_delta`, `_fit_content_to_max_size`, `save_session`, `_zero_embedding`, stale singleton re-exports.

### Performance

- `estimate_min_tokens` returns cached token counts instead of re-reading full files.
- `find_similar_files` no longer double-computes embeddings for uncached files.
- `grep` skips fetching context lines in compact mode.

## [0.3.2] - 2026-03-08

### Added

- **Custom embedding model support** — Set `EMBEDDING_MODEL` to any HuggingFace model with an ONNX export. Models not in fastembed's built-in list are automatically downloaded and registered from HuggingFace Hub on first startup.
- **SHA256 verification** — Downloaded ONNX model files are verified against HuggingFace-reported hashes to prevent tampering.
- **Clear error messages** — Specific errors for models without ONNX exports and for network failures when downloading custom models.

## [0.3.1] - 2026-03-08

### Changed

- **Removed explicit `onnxruntime` dependency** — `fastembed` now owns the ONNX Runtime dependency. Users with `fastembed-gpu` get `onnxruntime-gpu` automatically instead of being forced to CPU.

### Added

- **`[gpu]` optional extra** — Install with `semantic-cache-mcp[gpu]` to get NVIDIA GPU acceleration via `fastembed-gpu`.
- **`gpu` alias for `EMBEDDING_DEVICE`** — `EMBEDDING_DEVICE=gpu` now accepted as an alias for `cuda`.
- **Startup warning on missing CUDA** — When `EMBEDDING_DEVICE=gpu/cuda` but `CUDAExecutionProvider` is unavailable, logs a warning with install instructions before falling back to CPU.

## [0.3.0] - 2026-03-08 — Storage Rewrite

Complete storage backend rewrite from compressed chunks (SQLiteStorage) to raw text + vector embeddings (VectorStorage via simplevecdb). Simpler data path, better search, same caching semantics.

### Changed

- **Storage backend: SQLiteStorage → VectorStorage** — Files stored as plain text with HNSW embedding vectors. Eliminates compression/decompression overhead.
- **Small files** (< 8KB) stored as a single document; large files split via HyperCDC into content-defined chunks, each with its own embedding.
- **Thread safety** — `threading.RLock` on all public VectorStorage methods for safe concurrent access.
- **Dependencies** — Replaced `fastembed-gpu` (broken Rust rewrite) with `fastembed`. Removed `onnxruntime-gpu` (fastembed handles provider selection).
- **Stats tool** — Now returns token savings, hit/miss ratio, DB size, and session uptime in a flat JSON structure.
- **Search scores** — Normalized to 0–1 range (best result = 1.0) instead of raw RRF scores.

### Added

- **Content hash freshness** — BLAKE3 hash comparison when mtime changes but content is identical (touch, git checkout, editor re-save). Returns "unchanged" instead of re-reading. Applied across all 7 freshness check locations.
- **Truncation hints** — `read`/`batch_read` responses include `hint` with offset to continue reading.
- **Configurable embedding model** — `EMBEDDING_MODEL` env var (default: `BAAI/bge-small-en-v1.5`).
- **`grep` tool** — Regex/literal pattern search across cached files with line numbers and context.
- **`docs/env_variables.md`** — Full reference for all configurable env vars.
- **Auto-migration** — Detects and removes legacy v0.2.0 `cache.db` on first startup.

### Fixed

- **Stale cache** — `touch`, `git checkout`, editor re-saves no longer invalidate cache when content is identical.
- **`find_similar_files` returning 0 results** — Always computes embedding via `cache.get_embedding()` instead of relying on VectorStorage.get().
- **`stats` key mismatch** — Fixed `total_files` → `files_cached` in 3 locations.

### Removed

- Compressed chunk storage (ZSTD/LZ4/Brotli layer)
- File locking (`filelock`) — replaced by in-process `threading.RLock`
- Dead code: `_backtrack()` in `_diff.py`

## [0.2.0] - 2026-03-02

### Added

- **Cross-process file locking** — `filelock` serializes database access across concurrent MCP instances (e.g. Cursor + Claude Desktop sharing the same cache). Lock timeout produces a clear `RuntimeError` instead of cryptic SQLite crashes.
- **Atomic file writes** — All `write`/`edit`/`batch_edit` operations use temp-file + rename to prevent data loss on crash or signal interruption.
- **Thread-safe connection pool** — `threading.Lock` around pool counter prevents connection overflow under concurrent access.
- **Thread-safe tokenizer init** — Double-checked locking prevents duplicate downloads when multiple threads call `get_tokenizer()` simultaneously. Download is now atomic (temp file + rename).
- **Thread-safe ZSTD compressor cache** — Double-checked locking on lazy compressor/decompressor initialization.

### Fixed

- **Directory filter bypass** — `search(directory=...)` used `startswith()` which matched `/project_evil` when filtering for `/project`. Now uses `Path.is_relative_to()`.
- **Special files passed to formatter** — `_format_file` now rejects char devices, pipes, and `/proc` entries via `stat.S_ISREG` before spawning subprocess.
- **Startup crash on init failure** — `UnboundLocalError` when `SemanticCache()` or `warmup()` raised during lifespan. `cache` is now initialized to `None` with proper guards.
- **Negative offset/limit silently wrapping** — `read` tool now validates `offset >= 1` and `limit >= 1`; `max_size` clamped to prevent unbounded reads.
- **`executescript` breaking transactions** — `clear()` used `executescript` which auto-commits, defeating the connection pool's transaction management. Replaced with separate `execute()` calls.
- **O(N) eviction loading all metadata** — Eviction now uses `ORDER BY json_extract(...) LIMIT ?` in SQL instead of loading all rows + JSON parsing in Python.
- **LRU cache memory bloat** — Content hash cache now bypasses `@lru_cache` for files > 64KB, bounding worst-case retention to ~128MB instead of ~20GB.
- **`k=0` / `k<0` passing search guards** — `min(k, MAX)` now wrapped with `max(1, ...)` for both `search` and `similar`.
- **`compare_files` crash on missing/binary files** — Now validates file existence and catches `UnicodeDecodeError` with clean error messages.
- **`assert` used for control flow** — Three `assert` statements in `write.py` replaced with `TypeError` raises (assertions are stripped by `-O`).
- **Symlink traversal in glob** — `glob_with_cache_status` now skips symlinks that resolve outside the base directory.
- **`SEMANTIC_CACHE_DIR` env var not resolved** — Now calls `.expanduser().resolve()` on the override path.
- **Operator precedence ambiguity** — Added explicit parentheses in `_summarize.py` for `or`/`and` expression.
- **Redundant `ORDER BY` in `find_similar`** — Removed wasted sort; similarity search already ranks results.
- **Double chunking pass in `put()`** — Removed chunk counting loop that duplicated work done by storage layer.

### Performance

- **Vectorized hamming distance** — `hamming_distance_batch` now uses `np.unpackbits` on uint8 view of XOR results instead of Python-level popcount loops. Scalar `hamming_distance` uses Kernighan's bit-counting algorithm.
- **Vectorized SimHash bit packing** — `compute_simhash` replaces Python loop with `np.uint64` power-of-two dot product. `compute_simhash_batch` uses pre-allocated matrix instead of `np.vstack`.
- **O(N) top-K selection** — `np.argpartition` replaces `np.argsort` in similarity ranking (2 call sites), reducing top-K from O(N log N) to O(N).
- **O(N) pruning threshold** — `np.partition` replaces `np.percentile` for dimension pruning cutoff in cosine similarity.
- **Native binary quantization** — `np.packbits`/`np.unpackbits` replaces Python bit-manipulation loops in `quantize_binary`/`dequantize_binary`.
- **Buffer protocol blob deserialization** — Single `b"".join()` + `np.frombuffer` reshape replaces per-row `struct.unpack` loop in batch cosine similarity.
- **Pre-allocated matrices** — `np.empty` + fill replaces `np.vstack` with list comprehension in 3 hot paths (LSH batch, cosine batch ×2).

### Changed

- **Stdout redirect uses `contextlib.redirect_stdout`** — Replaces manual `sys.stdout` swap for thread-safety and re-entrancy.
- **Explicit stderr logging handler** — `logging.StreamHandler(sys.stderr)` instead of `basicConfig()` to guard against third-party reconfiguration.
- Type annotations tightened: `dict[str, Any]` → `dict[str, bool | int]` in `get_hash_stats`, `-> list` → `-> list[float]` in `cosine_similarity_batch`, `params: list` → `params: list[str]` in `find_similar`.
- README: added `uvx` vs `uv tool install` explanation, cross-platform cache paths, `SEMANTIC_CACHE_DIR` env var.

## [0.1.1] - 2026-02-21

### Fixed

- **macOS/Windows installation** — `fastembed-gpu` and `onnxruntime-gpu` (Linux-only wheels) replaced with platform-conditional dependencies. CPU variants install on macOS/Windows; GPU variants remain on Linux.
- **Cross-platform cache directory** — respects `$SEMANTIC_CACHE_DIR` env override, then uses platform-appropriate defaults: `$XDG_CACHE_HOME` on Linux, `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows.
- **Cross-platform RSS memory stats** — `/proc/self/status` replaced with platform-aware helper: `resource.getrusage` on macOS, `K32GetProcessMemoryInfo` on Windows, graceful `None` on unsupported platforms.
- **UTF-16/32 files falsely detected as binary** — BOM-aware check (UTF-32 LE/BE, UTF-16 LE/BE, UTF-8 BOM) now runs before the null-byte heuristic.
- **Inline binary checks consolidated** — `read.py` now uses the shared `_is_binary_content()` helper instead of duplicating null-byte checks.

### Changed

- Installation docs updated to use `uvx` instead of `uv tool install`.
- CI: action versions bumped (checkout v6, setup-uv v7, codecov v5, upload-artifact v6), macOS added to test matrix.

## [0.1.0] - 2026-02-21

### Added

- Initial release
- Session metrics: per-session and lifetime tracking of tokens saved, cache hits/misses, files read/written/edited, diffs served, and tool call counts. Persisted to SQLite on shutdown and aggregated across sessions via the `stats` tool.
- 11 MCP tools: `read`, `write`, `edit`, `batch_edit`, `search`, `similar`, `glob`, `batch_read`, `diff`, `stats`, `clear`
- Smart file reading with diff-mode — unchanged files cost ~5 tokens, modified files return unified diffs (80–95% savings)
- Semantic similarity search via local ONNX embeddings (BAAI/bge-small-en-v1.5, no API keys)
- Persistent LSH index for O(1) similarity lookups; serialized to SQLite, survives restarts
- Batch embedding — all new/changed files in a `batch_read` are embedded in a single model call
- Line-range editing for `edit` and `batch_edit` — scoped find/replace and direct line replacement
- int8 quantized embedding storage (388 bytes/vector, 22x smaller than float32)
- SIMD-parallel content-defined chunking (~70–95 MB/s), BLAKE3 hashing, ZSTD compression
- LRU-K eviction with 10,000-entry default; DoS limits on write size, match count, and glob scope
- `diff_mode=false` on `batch_read` for full content recovery after LLM context compression
- `append=true` on `write` for chunked large file writes
- `cached_only=true` on `glob` to filter to already-cached files

[Unreleased]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.3.3...HEAD
[0.3.3]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/CoderDayton/semantic-cache-mcp/releases/tag/v0.1.0
