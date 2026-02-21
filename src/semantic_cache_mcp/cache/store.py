"""SemanticCache class - high-level cache interface with semantic similarity support."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import DB_PATH
from ..core import count_tokens, get_optimal_chunker
from ..storage import SQLiteStorage
from ..types import CacheEntry, EmbeddingVector

logger = logging.getLogger(__name__)

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
    """High-level cache interface with semantic similarity support.

    This facade coordinates:
    - Storage backend (SQLite with content-addressable chunks)
    - Local embedding generation (FastEmbed)
    - Caching strategies (diff, truncate, semantic match)
    """

    __slots__ = ("_storage",)

    def __init__(self, db_path: Path = DB_PATH) -> None:
        """Initialize cache.

        Args:
            db_path: Path to SQLite database
        """
        self._storage = SQLiteStorage(db_path)

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    def get_embedding(self, text: str, path: str = "") -> EmbeddingVector | None:
        """Get embedding vector for text using local FastEmbed model.

        When *path* is provided a semantic file-type label is prepended so the
        model receives rich context (e.g. "npm package manifest …: {content}").
        This dramatically improves retrieval for structured formats like JSON,
        YAML, and well-known config files whose syntactic noise otherwise
        dilutes the semantic signal.

        Args:
            text: Text to embed
            path: Optional file path — used to derive a semantic label prefix

        Returns:
            Embedding as array.array or None if unavailable
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

    def get(self, path: str) -> CacheEntry | None:
        """Get cached entry for path."""
        entry = self._storage.get(path)
        if entry:
            logger.debug(f"Cache hit: {path}")
        return entry

    def put(
        self,
        path: str,
        content: str,
        mtime: float,
        embedding: EmbeddingVector | None = None,
    ) -> None:
        """Store file in cache."""
        tokens = count_tokens(content)
        content_bytes = content.encode()

        # Use optimal chunker (SIMD if available, otherwise Gear hash)
        chunker = get_optimal_chunker(prefer_simd=True)
        chunks = sum(1 for _ in chunker(content_bytes))

        self._storage.put(path, content, mtime, embedding)
        logger.info(f"Cached file: {path} ({tokens} tokens, {chunks} chunks)")

    def get_content(self, entry: CacheEntry) -> str:
        """Get full content from cache entry."""
        return self._storage.get_content(entry)

    def record_access(self, path: str) -> None:
        """Record access for LRU-K tracking."""
        self._storage.record_access(path)

    def find_similar(
        self, embedding: EmbeddingVector, exclude_path: str | None = None
    ) -> str | None:
        """Find semantically similar cached file."""
        return self._storage.find_similar(embedding, exclude_path)

    def get_stats(self) -> dict[str, int | float | str | bool]:
        """Get cache statistics including memory usage."""
        stats: dict[str, int | float | str | bool] = {**self._storage.get_stats()}

        # Add process memory stats
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        stats["process_rss_mb"] = round(int(line.split()[1]) / 1024, 1)
                        break
        except OSError:
            pass

        # Add merge cache stats
        from ..core.tokenizer import _tokenizer

        if _tokenizer is not None:
            stats["merge_cache_entries"] = len(_tokenizer._merge_cache)
            stats["merge_cache_maxsize"] = _tokenizer._merge_cache_maxsize

        # Add embedding model readiness
        from ..core.embeddings import _execution_provider, _model_ready

        stats["embedding_ready"] = _model_ready
        stats["embedding_provider"] = _execution_provider

        return stats

    def clear(self) -> int:
        """Clear all cache entries."""
        return self._storage.clear()
