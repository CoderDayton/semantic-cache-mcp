"""Pytest fixtures for semantic-cache-mcp tests."""

from __future__ import annotations

import array
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from semantic_cache_mcp.cache import SemanticCache
from semantic_cache_mcp.storage.sqlite import SQLiteStorage
from semantic_cache_mcp.types import EmbeddingVector


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_cache(temp_dir: Path) -> Generator[SQLiteStorage, None, None]:
    """Create a temporary SQLite cache for testing."""
    db_path = temp_dir / "test_cache.db"
    storage = SQLiteStorage(db_path)
    yield storage


@pytest.fixture
def semantic_cache(temp_dir: Path) -> Generator[SemanticCache, None, None]:
    """Create a SemanticCache for testing."""
    db_path = temp_dir / "semantic_cache.db"
    cache = SemanticCache(db_path=db_path)
    yield cache


@pytest.fixture
def semantic_cache_no_embeddings(temp_dir: Path) -> Generator[SemanticCache, None, None]:
    """Create a SemanticCache with embeddings disabled via mock."""
    db_path = temp_dir / "semantic_cache_no_emb.db"

    # Mock the embed function to return None (simulating no embeddings)
    with patch("semantic_cache_mcp.cache.embed", return_value=None):
        cache = SemanticCache(db_path=db_path)
        yield cache


@pytest.fixture
def semantic_cache_with_embeddings(temp_dir: Path, mock_embeddings: EmbeddingVector) -> Generator[SemanticCache, None, None]:
    """Create a SemanticCache with mocked embeddings."""
    db_path = temp_dir / "semantic_cache_emb.db"

    # Mock the embed function to return consistent embeddings
    with patch("semantic_cache_mcp.cache.embed", return_value=mock_embeddings):
        cache = SemanticCache(db_path=db_path)
        yield cache


@pytest.fixture
def sample_files(temp_dir: Path) -> dict[str, Path]:
    """Create sample test files."""
    files = {}

    # Simple text file
    simple = temp_dir / "simple.txt"
    simple.write_text("Hello, World!\nThis is a test file.\n")
    files["simple"] = simple

    # Python file
    python_file = temp_dir / "sample.py"
    python_file.write_text(
        '''"""Sample Python module."""

def hello(name: str) -> str:
    """Return greeting."""
    return f"Hello, {name}!"

class Calculator:
    """Simple calculator."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def subtract(self, a: int, b: int) -> int:
        """Subtract b from a."""
        return a - b
'''
    )
    files["python"] = python_file

    # Empty file
    empty = temp_dir / "empty.txt"
    empty.write_text("")
    files["empty"] = empty

    # Unicode file
    unicode_file = temp_dir / "unicode.txt"
    unicode_file.write_text("Hello World\n")
    files["unicode"] = unicode_file

    # Large file (>100KB)
    large = temp_dir / "large.txt"
    large.write_text("x" * 150_000)
    files["large"] = large

    # JSON file
    json_file = temp_dir / "data.json"
    json_file.write_text('{"name": "test", "value": 42, "items": [1, 2, 3]}')
    files["json"] = json_file

    # Modified file (for diff testing)
    original = temp_dir / "original.txt"
    original.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")
    files["original"] = original

    return files


@pytest.fixture
def mock_embeddings() -> EmbeddingVector:
    """Create a mock embedding vector (normalized, 768 dimensions for nomic)."""
    import math

    raw = [0.1] * 768
    magnitude = math.sqrt(sum(x * x for x in raw))
    normalized = [x / magnitude for x in raw]
    return array.array("f", normalized)


@pytest.fixture
def mock_embeddings_similar() -> EmbeddingVector:
    """Create a mock embedding vector similar to mock_embeddings."""
    import math

    # Slightly different from mock_embeddings but still similar
    raw = [0.1] * 768
    raw[0] = 0.11  # Small difference
    raw[1] = 0.09
    magnitude = math.sqrt(sum(x * x for x in raw))
    normalized = [x / magnitude for x in raw]
    return array.array("f", normalized)


@pytest.fixture
def mock_embeddings_different() -> EmbeddingVector:
    """Create a mock embedding vector different from mock_embeddings."""
    import math

    # Very different values
    raw = [-0.1] * 384 + [0.2] * 384
    magnitude = math.sqrt(sum(x * x for x in raw))
    normalized = [x / magnitude for x in raw]
    return array.array("f", normalized)


@pytest.fixture
def corrupted_cache_db(temp_dir: Path) -> Path:
    """Create a corrupted SQLite database."""
    db_path = temp_dir / "corrupted.db"
    db_path.write_bytes(b"This is not a valid SQLite database!")
    return db_path


@pytest.fixture
def binary_file(temp_dir: Path) -> Path:
    """Create a binary file (not valid UTF-8)."""
    binary_path = temp_dir / "binary.bin"
    binary_path.write_bytes(b"\x00\x01\x02\xff\xfe\x80\x81\x90")
    return binary_path
