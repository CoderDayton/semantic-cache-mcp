# Contributing

We welcome contributions! This project follows enterprise-level code quality standards.

## Development Setup

```bash
# Clone repository
git clone <repository-url>
cd semantic-cache-mcp

# Install with dev dependencies
uv sync

# Run tests
uv run pytest

# Format code
uv run ruff format src/

# Lint code
uv run ruff check src/

# Type check
uv run mypy src/
```

---

## Code Quality Standards

### Type Hints

Strict typing with `mypy --strict`:

```python
def process_file(path: str, content: bytes) -> CacheEntry:
    ...
```

### Formatting

Ruff with 100-character line length:

```bash
uv run ruff format src/
```

### Linting

Ruff rules: E, F, I, N, W, UP, B, C4, SIM

```bash
uv run ruff check src/ --fix
```

---

## Architecture Guidelines

### Single Responsibility

Each module has one clear purpose:

| Module | Responsibility |
|--------|----------------|
| `chunking.py` | Content-defined chunking only |
| `compression.py` | Adaptive compression only |
| `hashing.py` | Content hashing only |

### Facade Pattern

`SemanticCache` coordinates all components:

```python
class SemanticCache:
    def __init__(self):
        self._storage = SQLiteStorage(...)
        # All coordination happens here
```

### Performance First

- Profile hot paths before optimizing
- Minimize allocations in loops
- Use generators for large data
- Batch database operations

---

## Commit Guidelines

```bash
# Format
<type>: <description>

Co-Authored-By: Your Name <email@example.com>
```

### Types

| Type | Use |
|------|-----|
| `feat` | New feature |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code restructuring |
| `docs` | Documentation |
| `test` | Tests |
| `chore` | Maintenance |

### Examples

```
feat: Add semantic similarity matching for cached files

fix: Handle database lock on concurrent access

perf: Inline rolling hash calculation for 2x speedup
```

---

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes with tests
4. Run quality checks: `uv run ruff format src/ && uv run ruff check src/ && uv run mypy src/`
5. Commit with conventional format
6. Push and create PR

---

## Testing

### Unit Tests

```python
def test_chunking_deterministic():
    """Same input produces same chunks."""
    content = b"test content here"
    chunks1 = list(hypercdc_chunks(content))
    chunks2 = list(hypercdc_chunks(content))
    assert chunks1 == chunks2
```

### Integration Tests

```python
def test_cache_round_trip():
    """Store and retrieve file content."""
    cache = SemanticCache(client=None)
    cache.put("/test/file.py", "content", 1234.0)
    entry = cache.get("/test/file.py")
    assert cache.get_content(entry) == "content"
```

---

## Documentation

- Update README for user-facing changes
- Add docstrings to public functions
- Include type hints in signatures

---

[← Back to README](../README.md)
