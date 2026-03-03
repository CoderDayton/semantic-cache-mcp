# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/CoderDayton/semantic-cache-mcp/releases/tag/v0.1.0
