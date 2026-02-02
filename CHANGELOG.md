# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-02-01

### Added
- Claude Code hooks system with install script for automatic token savings
- Security documentation (`docs/security.md`)
- Pre-commit hooks with lefthook (ruff, mypy, bandit, tests)
- CI/CD workflows (GitHub Actions)
- Dependabot configuration for automated updates
- Config validation at module load time
- Binary file detection with clear error messages
- Path validation (is_file check, symlink logging)

### Changed
- int8 quantized embeddings enabled by default (22x storage reduction)
- Pre-quantized binary storage format for embeddings
- Improved error handling with specific exception types
- Better logging for fallback paths (tokenizer, compression, hashing)
- Updated binary file error from UnicodeDecodeError to ValueError

### Fixed
- UnicodeDecodeError handling in chunk retrieval
- Embedding model initialization validation on startup
- Type annotations (removed Any, fixed type:ignore)
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
