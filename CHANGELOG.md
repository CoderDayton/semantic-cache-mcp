# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `diff_mode` parameter on `batch_read` tool — set `false` after LLM context compression to force full content delivery instead of diff-only responses
- `append=True` mode on `write` tool for chunked large file writes; appends content to existing file rather than overwriting
- `cached_only=True` filter on `glob` tool to return only files already present in cache
- LSH acceleration for `search` and `similar` tools — O(1) candidate retrieval for caches ≥ 100 files, falls back to exhaustive O(N) scan for smaller caches
- `@overload` decorators on `LSHIndex.query` for precise mypy type narrowing (`Literal[True]` → `list[tuple[int, float]]`, `Literal[False]` → `list[int]`)
- `nosec` annotations for bandit false positives (B310 `urlretrieve`, B608 parameterized `IN`-clause queries)
- Line-range editing for `edit` and `batch_edit` tools — two new modes: scoped find/replace (searches only within `start_line`/`end_line`) and direct line replacement (`old_string=None`, replaces range wholesale). Fully backward-compatible; existing callers unchanged. 28 new tests covering all modes, validation, and edge cases.

### Changed

- Restructured `cache.py` (1581 lines) into `cache/` package: `cache/__init__.py`, `cache/store.py`, `cache/read.py`, `cache/write.py`, `cache/search.py`, `cache/_helpers.py`
- Restructured `server.py` (947 lines) into `server/` package: `server/__init__.py`, `server/_mcp.py`, `server/response.py`, `server/tools.py`
- Restructured `core/` into focused sub-packages: `core/chunking/` (Gear hash + SIMD parallel CDC), `core/similarity/` (cosine + LSH + int8/binary/ternary quantization), `core/text/` (diff generation + semantic summarization)
- Applied `_suppress_large_diff` to `diff` tool output — large diffs now auto-summarized within token budget
- Rewrote all 11 tool docstrings for clarity, discoverability, and accurate parameter documentation

### Fixed

- `lefthook` bandit hook updated to use `python -m bandit` — resolves binary spawn permission issue on some systems

## [1.0.0] - 2026-02-03

### Added

- SIMD-accelerated Parallel CDC chunking with 5–7x speedup (`core/chunking_simd.py`)
- Semantic summarization based on TCRA-LLM research (arXiv:2310.15556)
- LSH approximate similarity search for fast nearest-neighbor lookups
- Binary and ternary quantization for extreme compression (up to 100x)
- Comprehensive CDC benchmarking framework comparing 5 algorithms
- `get_optimal_chunker()` for automatic SIMD/serial selection
- Claude Code hooks system with install script for automatic token savings
- Security documentation (`docs/security.md`)
- Pre-commit hooks with lefthook (ruff, mypy, bandit, tests)
- CI/CD workflows (GitHub Actions)
- Dependabot configuration for automated updates
- Config validation at module load time
- Binary file detection with clear error messages
- Path validation (is_file check, symlink logging)

### Changed

- Integrated SIMD chunking into production cache (`smart_read` uses parallel CDC)
- Replaced simple truncation with semantic summarization for large files
- First segment (docstrings, imports) always preserved in summarization
- Embedding conversion wrapper for numpy compatibility
- int8 quantized embeddings enabled by default (22x storage reduction)
- Pre-quantized binary storage format for embeddings
- Improved error handling with specific exception types
- Better logging for fallback paths (tokenizer, compression, hashing)
- Updated binary file error from `UnicodeDecodeError` to `ValueError`

### Fixed

- Size limit enforcement in semantic summarization (dynamic marker length)
- First segment preservation bug (was being skipped if < 3 lines)
- Type annotations for embedding functions (`EmbeddingVector` → `NDArray` conversion)
- `UnicodeDecodeError` handling in chunk retrieval
- Embedding model initialization validation on startup
- Type annotations (removed `Any`, fixed `type: ignore`)
- Explicit UTF-8 encoding for file reads
- Thread pool and connection pool cleanup

### Security

- Added path traversal validation
- Bandit security scanning in CI

## [0.3.0] - 2026-01-28

### Added

- SQL optimizations with connection pooling and WAL mode
- O(N log M) BPE tokenizer with priority queue optimization
- HyperCDC chunking (2.7x faster than Rabin)
- BLAKE3 hashing with BLAKE2b fallback (3.8x faster)
- ZSTD adaptive compression with LZ4/Brotli fallbacks

### Changed

- Improved MCP tool descriptions and metadata
- Enhanced documentation for performance optimizations

## [0.2.0] - 2026-01-25

### Added

- Semantic similarity search using local FastEmbed
- LRU-K cache eviction strategy
- Content-addressable storage with deduplication
- Diff-based responses for changed files

### Changed

- Upgraded to FastMCP 3.0

## [0.1.0] - 2026-01-20

### Added

- Initial release
- Basic file caching with mtime tracking
- Token counting with o200k_base tokenizer
- MCP server implementation

[Unreleased]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/CoderDayton/semantic-cache-mcp/releases/tag/v0.1.0
