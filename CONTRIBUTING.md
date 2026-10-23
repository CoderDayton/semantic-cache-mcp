# Contributing to Semantic Cache MCP

## Development Setup

```bash
# Clone
git clone https://github.com/CoderDayton/semantic-cache-mcp.git
cd semantic-cache-mcp

# Install dependencies plus the dev tools (pytest, ruff, mypy, bandit).
# Plain `uv sync` omits the `dev` extra and removes them.
uv sync --extra dev

# Install git hooks. lefthook is a standalone binary, not a Python dependency:
# `npm install -g @evilmartians/lefthook` or `brew install lefthook`
lefthook install

# Verify everything works
uv run pytest -q
```

Runtime dependencies are only `blake3`, `fastmcp>=4.0.0,<5` and `numpy`. Do not add one without a strong reason.

## Development Commands

```bash
uv run ruff check src/ tests/                  # Lint
uv run ruff format src/ tests/                 # Format
uv run mypy src/                               # Type check
uv run pytest -q                               # Tests (add a path or -k <expr> for a subset)
uv run bandit -c pyproject.toml -r src/        # Security scan
uv run python benchmarks/benchmark_performance.py  # Benchmarks
```

Git hooks (lefthook):

| Hook | Runs |
|------|------|
| `pre-commit` | ruff-check (`--fix`), ruff-format, bandit |
| `commit-msg` | Conventional Commits check |
| `pre-push` | `pytest tests/ -x -q`, mypy |

CI runs the same commands (`ruff format --check` instead of `ruff format`).

## Architecture

See `docs/architecture.md` for the full package layout.

**Rules:**
- `core/` must remain **stateless and I/O-free** — no file reads, no DB access, no subprocess calls. The one exception is the tokenizer's one-time BPE download, primed at server startup; do not add more
- New algorithms belong in the appropriate sub-package (`chunking/`, `text/`, `hashing/`, `tokenizer/`)
- New MCP tools go in `server/tools/` (`__init__.py`) only; business logic stays in `cache/`. Callers go through the `SemanticCache` facade, never straight to `storage/`
- `core/` must never import from `storage/`
- Storage is one SQLite connection serialized through a single IO thread; keep it that way
- Search is BM25 keyword ranking over SQLite FTS5. There are no embeddings or vectors

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
- `@overload` for functions with conditional return types
- `TYPE_CHECKING` blocks for import-only types to avoid circular deps

### Code Style

- **Line length:** 100 characters (configured in ruff)
- **Imports:** absolute, sorted by ruff (isort compatible)
- **Docstrings:** Google style; explain *why*, not *what*
- **Classes:** use `__slots__` for memory efficiency on data-heavy classes
- **Lint rules:** `E F I N W UP B C4 SIM`, configured in `pyproject.toml` only; no CLI `--ignore` lists

### Performance

This is a performance-critical project. When touching hot paths (chunking, hashing, tokenization, diffing):

- Profile before and after with `cProfile`: `uv run python -m cProfile -o out.prof benchmarks/benchmark_performance.py` (see `docs/performance.md#profiling` — profiling the server itself only sees the forwarding process)
- Document algorithmic complexity with O(N) notation in comments
- Use `@lru_cache` for pure functions called repeatedly
- Batch operations over loops where possible
- Add benchmarks to the PR description

### Security

- All inputs validated before I/O
- Avoid `# type: ignore` — fix the underlying type issue. Where unavoidable, use a specific code (`warn_unused_ignores` is on)
- No bare `except:` — catch specific exception types
- Size limits enforced for write/edit (10MB max)
- Parameterized queries only — no string interpolation for SQL data

## Writing Tests

```python
# tests/test_*.py — pytest discovers all files matching this pattern
# `semantic_cache` is a fixture in tests/conftest.py (SemanticCache on a temp DB)

from semantic_cache_mcp.cache import SemanticCache, smart_read

async def test_second_read_comes_from_cache(semantic_cache: SemanticCache, tmp_path):
    path = tmp_path / "file.py"
    path.write_text("content")
    first = await smart_read(semantic_cache, str(path))   # primes the cache
    second = await smart_read(semantic_cache, str(path))
    assert first.from_cache is False
    assert second.from_cache is True
```

- `asyncio_mode = "auto"` — all async tests work without `@pytest.mark.asyncio`
- Shared fixtures live in `tests/conftest.py`; tests are flat files under `tests/`
- Use the `tmp_path` / `temp_dir` fixtures for anything that touches disk
- Cover determinism for hashing, boundary stability for chunking, and byte-level correctness for tokenization

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
| `style`    | Formatting only                                  |
| `build`    | Build system, deps                               |
| `ci`       | CI configuration                                 |
| `chore`    | Maintenance (config, tooling)                    |
| `revert`   | Revert a previous commit                         |

Keep the subject short and imperative. No AI attribution trailers.

**Examples:**

```
feat: add cached_only filter to glob tool
fix: propagate diff_mode through batch_smart_read
perf: speed up BM25 ranking for large cached corpora
refactor: restructure cache.py into cache/ sub-package
docs: update architecture.md with sub-package structure
test: add property tests for chunk boundary stability
```

## Pull Request Process

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b feat/your-feature
   ```

2. **Run the full pipeline** before opening a PR:
   ```bash
   uv run pytest tests/
   uv run ruff check src/ tests/
   uv run ruff format --check src/ tests/
   uv run mypy src/
   uv run bandit -c pyproject.toml -r src/
   ```

3. **Commit** using conventional commits

4. **Open the PR** with:
   - Clear description of what changed and *why*
   - Benchmark results if touching a hot path
   - Any migration notes if the public API changed

### PR Checklist

- [ ] Tests pass (`uv run pytest tests/`)
- [ ] No lint or format errors (`uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/`)
- [ ] No type errors (`uv run mypy src/`)
- [ ] No security issues (`uv run bandit -c pyproject.toml -r src/`)
- [ ] Docstrings updated for any changed public functions
- [ ] `core/` changes are still I/O-free (grep for `open(`, `sqlite3`, `subprocess`)
- [ ] Performance impact considered; benchmarks included if touching hot paths

[← Back to README](README.md)
