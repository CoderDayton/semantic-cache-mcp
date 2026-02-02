# Contributing to Semantic Cache MCP

Thanks for your interest in contributing! This document covers everything you need to get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/CoderDayton/semantic-cache-mcp.git
cd semantic-cache-mcp

# Install dependencies (requires uv)
uv sync

# Run tests
uv run pytest

# Run linting and type checking
uv run ruff check src/
uv run mypy src/
```

## Code Standards

### Python Version

- **Python 3.12+** required
- Use modern syntax: `match` statements, PEP 695 type params, `|` union types

### Type Hints

- All functions must have type hints
- No `Any` without a comment explaining why
- Use `from __future__ import annotations` at top of files
- Import-only types go in `TYPE_CHECKING` blocks

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import CacheEntry
```

### Code Style

- **Line length:** 100 characters (configured in ruff)
- **Imports:** Absolute imports, sorted by ruff
- **Docstrings:** Google style, focus on "why" not "what"
- **Classes:** Use `__slots__` for memory efficiency

### Performance

This is a performance-critical project. When modifying hot paths:

- Profile before optimizing (`python -m cProfile`)
- Document complexity with O(N) notation
- Cache aggressively (`@lru_cache` for pure functions)
- Batch operations where possible

## Commit Conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |

### Examples

```
feat: Add semantic similarity caching for related files
fix: Handle empty files in chunking algorithm
perf: Enable int8 quantization for 22x embedding storage reduction
docs: Update installation instructions for uv
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feat/your-feature`
3. **Make changes** following the code standards above
4. **Run tests:** `uv run pytest`
5. **Run linting:** `uv run ruff check src/`
6. **Commit** using conventional commits
7. **Push** and open a PR

### PR Checklist

- [ ] Tests pass (`uv run pytest`)
- [ ] No lint errors (`uv run ruff check src/`)
- [ ] Type hints added for new code
- [ ] Docstrings for public functions
- [ ] Performance impact considered for hot paths

## Project Structure

```
src/semantic_cache_mcp/
├── core/           # Pure algorithms (stateless, no I/O)
│   ├── chunking.py     # HyperCDC content-defined chunking
│   ├── compression.py  # ZSTD/LZ4/Brotli adaptive compression
│   ├── embeddings.py   # FastEmbed local embeddings
│   ├── hashing.py      # BLAKE3/BLAKE2b hashing
│   ├── similarity.py   # int8 quantized cosine similarity
│   ├── text.py         # Diff generation, truncation
│   └── tokenizer.py    # BPE token counting
├── storage/        # Persistence layer
│   └── sqlite.py       # SQLite with connection pooling
├── cache.py        # Orchestration facade
├── server.py       # MCP interface (FastMCP)
├── config.py       # Configuration constants
└── types.py        # Data models
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_core.py

# Run specific test
uv run pytest tests/test_core.py::TestSimilarity -v
```

### Writing Tests

- Test files: `tests/test_*.py`
- Use `pytest` and `pytest-asyncio`
- Mock I/O for fast unit tests
- Property tests for: parsing, hashing, compression

## Questions?

Open an issue or start a discussion on GitHub.
