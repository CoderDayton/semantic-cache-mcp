"""SemanticCache — orchestration facade over VectorStorage, SQLite metrics, and embeddings."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, cast

from ..config import CACHE_DIR
from ..core import count_tokens
from ..storage import SQLiteStorage, VectorStorage
from ..storage.vector import VECDB_PATH
from ..types import CacheEntry, EmbeddingVector
from .metrics import SessionMetrics

logger = logging.getLogger(__name__)


def _get_rss_mb() -> float | None:
    """Cross-platform resident set size in MB. Returns None on failure."""
    try:
        if sys.platform == "linux":
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return round(int(line.split()[1]) / 1024, 1)
            return None
        if sys.platform == "darwin":
            import resource  # noqa: PLC0415

            # macOS ru_maxrss is in bytes
            return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1)
        if sys.platform == "win32":
            import ctypes  # noqa: PLC0415
            import ctypes.wintypes  # noqa: PLC0415

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = ProcessMemoryCounters()
            pmc.cb = ctypes.sizeof(pmc)
            handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[union-attr]
            if ctypes.windll.kernel32.K32GetProcessMemoryInfo(  # type: ignore[union-attr]
                handle, ctypes.byref(pmc), pmc.cb
            ):
                return round(pmc.WorkingSetSize / (1024 * 1024), 1)
            return None
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# File-type semantic labels — prepended before embedding so the model gets
# intent-rich context instead of raw syntactic noise.
# ---------------------------------------------------------------------------

# Exact basename → label  (checked first, highest priority)
_STEM_LABELS: dict[str, str] = {
    "package.json": (
        "npm package manifest listing project dependencies scripts and version constraints"
    ),
    "package-lock.json": "npm lockfile pinning exact dependency versions",
    "tsconfig.json": "TypeScript compiler configuration",
    "jsconfig.json": "JavaScript project configuration",
    ".eslintrc.json": "ESLint linter configuration",
    ".prettierrc": "Prettier formatter configuration",
    ".prettierrc.json": "Prettier formatter configuration",
    "biome.json": "Biome linter and formatter configuration",
    "composer.json": "PHP Composer package manifest with dependencies",
    "Cargo.toml": "Rust Cargo package manifest with dependencies and build settings",
    "Cargo.lock": "Rust Cargo lockfile pinning exact dependency versions",
    "go.mod": "Go module definition with dependencies",
    "go.sum": "Go module checksum lockfile",
    "pyproject.toml": "Python project manifest with dependencies and build configuration",
    "setup.py": "Python package setup script with dependencies",
    "setup.cfg": "Python package setup configuration with dependencies",
    "requirements.txt": "Python pip requirements listing package dependencies",
    "Pipfile": "Python Pipenv package dependencies",
    "Pipfile.lock": "Python Pipenv lockfile pinning exact dependency versions",
    "Gemfile": "Ruby Bundler gem dependencies",
    "Gemfile.lock": "Ruby Bundler lockfile pinning exact gem versions",
    "pom.xml": "Maven Java project configuration with dependencies",
    "build.gradle": "Gradle Java build configuration with dependencies",
    "build.gradle.kts": "Gradle Kotlin build configuration with dependencies",
    "Makefile": "Make build automation targets and commands",
    "CMakeLists.txt": "CMake build system configuration",
    "Dockerfile": "Docker container image build instructions",
    "docker-compose.yml": "Docker Compose container orchestration defining services and deployment",
    "docker-compose.yaml": (
        "Docker Compose container orchestration defining services and deployment"
    ),
    "docker-compose.prod.yml": "Docker Compose production deployment orchestration",
    "docker-compose.dev.yml": "Docker Compose development environment orchestration",
    "compose.yml": "Docker Compose container orchestration defining services and deployment",
    "compose.yaml": "Docker Compose container orchestration defining services and deployment",
    ".dockerignore": "Docker build context ignore patterns",
    ".gitignore": "Git ignore patterns for untracked files",
    ".gitattributes": "Git attributes configuration",
    ".env": "Environment variables configuration",
    ".env.example": "Environment variables template",
    ".env.local": "Local environment variables override",
    "README.md": "Project README documentation",
    "CONTRIBUTING.md": "Contribution guidelines for project contributors",
    "CHANGELOG.md": "Project changelog and version history",
    "LICENSE": "Software license terms",
    "LICENSE.md": "Software license terms",
    "CLAUDE.md": "Claude Code project instructions and conventions",
    "jest.config.js": "Jest testing framework configuration",
    "jest.config.ts": "Jest testing framework configuration",
    "vitest.config.ts": "Vitest testing framework configuration",
    "webpack.config.js": "Webpack module bundler configuration",
    "vite.config.ts": "Vite build tool and dev server configuration",
    "next.config.js": "Next.js framework configuration",
    "next.config.mjs": "Next.js framework configuration",
    "next.config.ts": "Next.js framework configuration",
    "nuxt.config.ts": "Nuxt.js framework configuration",
    "tailwind.config.js": "Tailwind CSS utility framework configuration",
    "tailwind.config.ts": "Tailwind CSS utility framework configuration",
    "postcss.config.js": "PostCSS CSS processing configuration",
    ".babelrc": "Babel JavaScript transpiler configuration",
    "babel.config.js": "Babel JavaScript transpiler configuration",
    "nginx.conf": "Nginx web server configuration",
    "Procfile": "Process manager deployment commands",
    "Vagrantfile": "Vagrant virtual machine configuration",
    "Taskfile.yml": "Task runner automation commands",
    "Justfile": "Just command runner recipes",
    "flake.nix": "Nix flake package and environment definition",
    "shell.nix": "Nix shell development environment",
    "renovate.json": "Renovate automated dependency update configuration",
    "dependabot.yml": "Dependabot automated dependency update configuration",
    "fly.toml": "Fly.io deployment configuration",
    "vercel.json": "Vercel deployment configuration",
    "netlify.toml": "Netlify deployment configuration",
    "railway.json": "Railway deployment configuration",
    "render.yaml": "Render deployment configuration",
    "app.yaml": "Google App Engine deployment configuration",
    "serverless.yml": "Serverless Framework deployment configuration",
    "terraform.tf": "Terraform infrastructure as code",
    "main.tf": "Terraform infrastructure as code main configuration",
    "variables.tf": "Terraform infrastructure variable definitions",
    "outputs.tf": "Terraform infrastructure output definitions",
    "Pulumi.yaml": "Pulumi infrastructure as code configuration",
    "ansible.cfg": "Ansible automation configuration",
    "playbook.yml": "Ansible automation playbook",
    "inventory.yml": "Ansible host inventory",
    "tox.ini": "Python tox testing automation configuration",
    "pytest.ini": "Pytest testing configuration",
    "mypy.ini": "Mypy Python type checker configuration",
    ".flake8": "Flake8 Python linter configuration",
    "ruff.toml": "Ruff Python linter and formatter configuration",
    ".rubocop.yml": "RuboCop Ruby linter configuration",
    "Rakefile": "Ruby Rake build automation tasks",
    "Brewfile": "Homebrew package dependencies",
    "CODEOWNERS": "GitHub code ownership and review assignment rules",
}

# Extension → label  (fallback when basename not in _STEM_LABELS)
_EXT_LABELS: dict[str, str] = {
    ".json": "JSON data file",
    ".jsonc": "JSON with comments configuration file",
    ".json5": "JSON5 configuration file",
    ".yaml": "YAML configuration file",
    ".yml": "YAML configuration file",
    ".toml": "TOML configuration file",
    ".ini": "INI configuration file",
    ".cfg": "configuration file",
    ".conf": "configuration file",
    ".xml": "XML document",
    ".csv": "CSV tabular data file",
    ".tsv": "TSV tabular data file",
    ".sql": "SQL database queries and schema definitions",
    ".graphql": "GraphQL schema and query definitions",
    ".gql": "GraphQL schema and query definitions",
    ".proto": "Protocol Buffers schema definition",
    ".prisma": "Prisma database schema with models and relations",
    ".py": "Python source code",
    ".pyi": "Python type stub definitions",
    ".js": "JavaScript source code",
    ".jsx": "React JSX component",
    ".ts": "TypeScript source code",
    ".tsx": "React TypeScript component",
    ".rs": "Rust source code",
    ".go": "Go source code",
    ".java": "Java source code",
    ".kt": "Kotlin source code",
    ".scala": "Scala source code",
    ".rb": "Ruby source code",
    ".php": "PHP source code",
    ".c": "C source code",
    ".h": "C/C++ header file",
    ".cpp": "C++ source code",
    ".hpp": "C++ header file",
    ".cs": "C# source code",
    ".swift": "Swift source code",
    ".m": "Objective-C source code",
    ".r": "R statistical computing source code",
    ".R": "R statistical computing source code",
    ".jl": "Julia source code",
    ".lua": "Lua source code",
    ".ex": "Elixir source code",
    ".exs": "Elixir script",
    ".erl": "Erlang source code",
    ".hs": "Haskell source code",
    ".ml": "OCaml source code",
    ".clj": "Clojure source code",
    ".dart": "Dart source code",
    ".zig": "Zig source code",
    ".v": "V source code",
    ".nim": "Nim source code",
    ".sh": "Shell script",
    ".bash": "Bash shell script",
    ".zsh": "Zsh shell script",
    ".fish": "Fish shell script",
    ".ps1": "PowerShell script",
    ".bat": "Windows batch script",
    ".md": "Markdown documentation",
    ".mdx": "MDX documentation with embedded components",
    ".rst": "reStructuredText documentation",
    ".txt": "plain text file",
    ".tex": "LaTeX document",
    ".adoc": "AsciiDoc documentation",
    ".html": "HTML web page",
    ".htm": "HTML web page",
    ".css": "CSS stylesheet",
    ".scss": "SCSS stylesheet",
    ".sass": "Sass stylesheet",
    ".less": "Less stylesheet",
    ".svg": "SVG vector graphic",
    ".tf": "Terraform infrastructure as code",
    ".hcl": "HashiCorp configuration language",
    ".nix": "Nix expression",
    ".dhall": "Dhall configuration",
    ".lock": "dependency lockfile",
    ".env": "environment variables file",
    ".editorconfig": "editor configuration",
}


def _file_label(path: str) -> str:
    """Derive a semantic label for a file path.

    Checks the exact basename first (for well-known files like Makefile,
    docker-compose.yml, package.json), then falls back to extension-based
    labels, and finally to just the filename.
    """
    p = Path(path)
    name = p.name

    # 1. Exact basename match (highest specificity)
    label = _STEM_LABELS.get(name)
    if label:
        return f"{name} ({label})"

    # 2. Extension-based fallback
    suffix = p.suffix.lower()
    label = _EXT_LABELS.get(suffix)
    if label:
        return f"{name} ({label})"

    # 3. Bare filename (still better than nothing)
    return name


class SemanticCache:
    """Facade over VectorStorage (simplevecdb/HNSW), SQLite metrics, and FastEmbed embeddings."""

    __slots__ = ("_storage", "_metrics_storage", "_metrics", "_closed")

    def __init__(self, db_path: Path = VECDB_PATH) -> None:
        self._storage = VectorStorage(db_path)
        # Keep SQLiteStorage only for session metrics persistence
        metrics_db = CACHE_DIR / "metrics.db"
        self._metrics_storage = SQLiteStorage(metrics_db)
        self._metrics = SessionMetrics(self._metrics_storage._pool)
        self._closed = False

    @property
    def metrics(self) -> SessionMetrics:
        """Current session metrics accumulator."""
        return self._metrics

    def close(self) -> None:
        """Persist metrics and close all storage backends.

        Called from the lifespan finally block. Idempotent — safe to call
        multiple times. Must not raise — a failed close should not mask
        the original shutdown reason.
        """
        if self._closed:
            return
        self._closed = True

        try:
            self._metrics.persist()
        except Exception as e:
            logger.warning(f"Failed to persist metrics on close: {e}")

        try:
            self._storage.close()
        except Exception as e:
            logger.warning(f"Failed to close VectorStorage: {e}")

        try:
            self._metrics_storage._pool.close_all()
        except Exception as e:
            logger.warning(f"Failed to close metrics pool: {e}")

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    def get_embedding(self, text: str, path: str = "") -> EmbeddingVector | None:
        """Embed text using FastEmbed. When path is given, prepends a file-type label
        (e.g. "Python source code: ...") so the model gets intent-rich context instead
        of raw syntactic noise — improves retrieval for JSON, YAML, and config files.
        """
        try:
            # Late import from package so patch("semantic_cache_mcp.cache.embed") in tests works.
            # By call time the cache package is fully initialized in sys.modules.
            from . import embed as _embed  # noqa: PLC0415

            if path:
                text = f"{_file_label(path)}: {text}"

            result = _embed(text)
            if result:
                logger.debug(f"Embedding generated for {text[:50]}...")
            return result
        except Exception as e:
            logger.warning(f"Failed to get embedding: {e}")
            return None

    # -------------------------------------------------------------------------
    # Delegated operations
    # -------------------------------------------------------------------------

    async def get(self, path: str) -> CacheEntry | None:
        entry = await self._storage.get(path)
        if entry:
            logger.debug(f"Cache hit: {path}")
        return entry

    async def put(
        self,
        path: str,
        content: str,
        mtime: float,
        embedding: EmbeddingVector | None = None,
    ) -> None:
        tokens = count_tokens(content)
        await self._storage.put(path, content, mtime, embedding)
        logger.info(f"Cached file: {path} ({tokens} tokens)")

    async def get_content(self, entry: CacheEntry) -> str:
        return await self._storage.get_content(entry)

    async def record_access(self, path: str) -> None:
        await self._storage.record_access(path)

    async def update_mtime(self, path: str, new_mtime: float) -> None:
        """Update cached mtime without re-storing content or re-embedding."""
        await self._storage.update_mtime(path, new_mtime)

    async def find_similar(
        self, embedding: EmbeddingVector, exclude_path: str | None = None
    ) -> str | None:
        return await self._storage.find_similar(embedding, exclude_path)

    async def get_stats(self) -> dict[str, Any]:
        """Cache statistics: occupancy, process memory, session, and lifetime metrics."""
        stats: dict[str, Any] = {**await self._storage.get_stats()}

        # Add process memory stats
        rss = _get_rss_mb()
        if rss is not None:
            stats["process_rss_mb"] = rss

        # Add merge cache stats
        from ..core.tokenizer import _tokenizer  # noqa: PLC0415

        if _tokenizer is not None:
            stats["merge_cache_entries"] = len(_tokenizer._merge_cache)
            stats["merge_cache_maxsize"] = _tokenizer._merge_cache_maxsize

        # Add embedding model readiness
        from ..core.embeddings import _execution_provider, _model_ready  # noqa: PLC0415

        stats["embedding_ready"] = _model_ready
        stats["embedding_provider"] = _execution_provider

        # Session metrics
        stats["session"] = self._metrics.snapshot()

        # Lifetime metrics (aggregated from all completed sessions)
        try:
            stats["lifetime"] = self._metrics_storage.get_lifetime_stats()
        except Exception as e:
            logger.warning(f"Failed to load lifetime stats: {e}")

        return stats

    def get_embeddings_batch(
        self, path_content_pairs: list[tuple[str, str]]
    ) -> list[EmbeddingVector | None]:
        """Batch-embed files in one model call. Prepends file-type labels like get_embedding."""
        from . import embed_batch as _embed_batch  # noqa: PLC0415

        texts = [
            (f"{_file_label(path)}: {content}" if path else content)[:8000]
            for path, content in path_content_pairs
        ]
        return cast(list[EmbeddingVector | None], _embed_batch(texts))

    async def clear(self) -> int:
        return await self._storage.clear()
