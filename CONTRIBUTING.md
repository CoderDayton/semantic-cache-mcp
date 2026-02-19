# Contributing to Semantic Cache MCP

Thanks for contributing. This document covers setup, standards, and workflow.

## Development Setup

```bash
# Clone
git clone https://github.com/CoderDayton/semantic-cache-mcp.git
cd semantic-cache-mcp

# Install dependencies
uv sync

# Install pre-commit hooks (lefthook)
uv run lefthook install

# Verify everything works
uv run python -m pytest
```

## Development Commands

```bash
uv run ruff check src/ tests/        # Lint
uv run ruff format src/ tests/       # Format
uv run python -m mypy src/           # Type check
uv run python -m pytest              # Tests
uv run python -m pytest --cov        # Tests + coverage
uv run python -m bandit -r src/      # Security scan
```

The pre-commit hooks (lefthook) run ruff-check, ruff-format, mypy, and bandit automatically on every commit. Tests run on push.

## Package Structure

```
src/semantic_cache_mcp/
├── cache/          # Orchestration: store.py, read.py, write.py, search.py
├── server/         # MCP interface: tools.py, response.py, _mcp.py
├── core/
│   ├── chunking/   # _gear.py (HyperCDC), _simd.py (parallel CDC)
│   ├── similarity/ # _cosine.py, _lsh.py, _quantization.py
│   └── text/       # _diff.py, _summarize.py
├── storage/        # sqlite.py (SQLiteStorage)
├── config.py       # Environment-variable configuration
└── types.py        # All shared data models
```

**Architecture rules:**
- `core/` must remain **stateless and I/O-free** — no file reads, no DB access, no subprocess calls
- New algorithms belong in the appropriate sub-package (`chunking/`, `similarity/`, `text/`)
- New MCP tools go in `server/tools.py` only; business logic stays in `cache/`
- `storage/` is a swappable backend — `core/` must never import from it

## Code Standards

### Python Version

Python 3.12+ required. Use modern syntax:
- `match` statements over long `if/elif` chains
- `X | Y` union types (not `Optional[X]` or `Union[X, Y]`)
- `type` alias statement (PEP 695) for complex type aliases

### Type Hints

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import CacheEntry
```

- All functions must have type hints, including return types
- No `Any` without an inline comment and a TODO explaining why
- `@overload` for functions with conditional return types (see `LSHIndex.query`)
- `TYPE_CHECKING` blocks for import-only types to avoid circular deps

### Code Style

- **Line length:** 100 characters (configured in ruff)
- **Imports:** absolute, sorted by ruff (isort compatible)
- **Docstrings:** Google style; explain *why*, not *what*
- **Classes:** use `__slots__` for memory efficiency on data-heavy classes
- **Complexity:** max cyclomatic complexity 10 (ruff enforced)

### Performance

This is a performance-critical project. When touching hot paths (chunking, hashing, similarity, tokenization, embeddings):

- Profile before and after with `cProfile`: `python -m cProfile -o out.prof script.py`
- Document algorithmic complexity with O(N) notation in comments
- Use `@lru_cache` for pure functions called repeatedly
- Batch operations over loops where possible
- Add benchmarks to the PR description

### Security

- All inputs validated before I/O
- No `# type: ignore` — fix the underlying type issue
- No bare `except:` — catch specific exception types
- Size limits enforced for write/edit (10MB max)
- Parameterized queries only — no string interpolation for SQL data

## Writing Tests

```python
# tests/test_*.py — pytest discovers all files matching this pattern

import pytest
from semantic_cache_mcp.cache import smart_read, SemanticCache

@pytest.fixture
def cache(tmp_path):
    from semantic_cache_mcp.storage.sqlite import SQLiteStorage
    storage = SQLiteStorage(db_path=tmp_path / "test.db")
    return SemanticCache(storage=storage)

def test_read_unchanged_returns_message(cache, tmp_path):
    path = tmp_path / "file.py"
    path.write_text("content")
    smart_read(cache, str(path))          # prime cache
    result = smart_read(cache, str(path)) # second read
    assert "unchanged" in result.content
    assert result.tokens_saved > 0
```

- `asyncio_mode = "auto"` — all async tests work without `@pytest.mark.asyncio`
- Mock disk I/O for unit tests; use `tmp_path` fixture for integration tests
- Property-based tests for: hashing (determinism), compression (round-trip), similarity (range), tokenization (byte-level correctness)

## Commit Conventions

Enforced by the `commit-msg` lefthook:

```
<type>(<optional scope>): <description>
```

| Type       | When to use                                      |
|------------|--------------------------------------------------|
| `feat`     | New feature or tool                              |
| `fix`      | Bug fix                                          |
| `perf`     | Performance improvement                          |
| `refactor` | Code change with no behavior change              |
| `docs`     | Documentation only                               |
| `test`     | Adding or updating tests                         |
| `build`    | Build system, deps, CI                           |
| `chore`    | Maintenance (config, tooling)                    |

**Examples:**

```
feat: add cached_only filter to glob tool
fix: propagate diff_mode through batch_smart_read
perf: accelerate similarity search with LSH for caches ≥100 files
refactor: restructure cache.py into cache/ sub-package
docs: update architecture.md with sub-package structure
test: add property tests for int8 quantization round-trip
```

## Pull Request Process

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b feat/your-feature
   ```

2. **Implement** following the code standards above

3. **Run the full pipeline** before opening a PR:
   ```bash
   uv run python -m pytest --cov
   uv run ruff check src/ tests/
   uv run python -m mypy src/
   uv run python -m bandit -r src/
   ```

4. **Commit** using conventional commits

5. **Open the PR** with:
   - Clear description of what changed and *why*
   - Benchmark results if touching a hot path
   - Any migration notes if the public API changed

### PR Checklist

- [ ] Tests pass (`uv run python -m pytest`)
- [ ] No lint errors (`uv run ruff check src/`)
- [ ] No type errors (`uv run python -m mypy src/`)
- [ ] No security issues (`uv run python -m bandit -r src/`)
- [ ] Docstrings updated for any changed public functions
- [ ] `core/` changes are still I/O-free (grep for `open(`, `sqlite3`, `subprocess`)
- [ ] Performance impact considered; benchmarks included if touching hot paths

## Questions

Open an issue or start a discussion on [GitHub](https://github.com/CoderDayton/semantic-cache-mcp).

---

[← Back to README](../README.md)
