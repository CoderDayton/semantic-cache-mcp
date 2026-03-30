"""SemanticCache — orchestration facade over VectorStorage, SQLite metrics, and embeddings."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from concurrent.futures import Executor
from pathlib import Path
from typing import Any, cast

from ..config import CACHE_DIR, TOOL_TIMEOUT
from ..core import count_tokens
from ..logger import log_marker
from ..storage import SQLiteStorage, VectorStorage
from ..storage.vector import VECDB_PATH
from ..types import CacheEntry, EmbeddingVector
from ..utils import DetachedExecutor
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

    __slots__ = (
        "_storage",
        "_metrics_storage",
        "_metrics",
        "_closed",
        "_shutting_down",
        "_inflight",
        "_drained",
        "_io_executor",
        "_stale_paths",
    )

    # Grace period for in-flight operations to finish during shutdown.
    _DRAIN_TIMEOUT: float = 8.0

    def __init__(self, db_path: Path = VECDB_PATH) -> None:
        # Single-thread executor shared by ALL blocking operations:
        # file I/O, ONNX embedding inference, and vecdb index saves.
        # MUST be single-threaded — ONNX Runtime and usearch use
        # incompatible allocators that segfault under concurrent access
        # from different threads.
        # Passed to VectorStorage → AsyncVectorDB so simplevecdb's own
        # operations (add_texts, similarity_search, etc.) also serialize
        # on this thread.
        self._io_executor: Executor = DetachedExecutor(thread_name_prefix="semantic-cache-io")
        self._storage = VectorStorage(db_path, executor=self._io_executor)
        metrics_db = CACHE_DIR / "metrics.db"
        self._metrics_storage = SQLiteStorage(metrics_db)
        self._metrics = SessionMetrics(self._metrics_storage._pool)
        self._closed = False
        self._shutting_down = False
        self._inflight = 0
        self._drained: asyncio.Event = asyncio.Event()
        self._drained.set()  # starts drained (no inflight ops)
        self._stale_paths: set[str] = set()

    def reset_executor(self) -> None:
        """Replace the IO executor with a fresh one after a timeout/hang.

        Abandons the stuck thread (it will be GC'd when its task completes or
        the process exits) and creates a new single-threaded executor. All
        references on VectorStorage and simplevecdb are updated atomically.
        """
        logger.warning("Resetting IO executor — previous thread may be stuck")
        old = self._io_executor
        # Don't wait for the old executor — the current call may be wedged in
        # a blocking C extension or a kernel I/O wait and cannot be cancelled.
        old.shutdown(wait=False, cancel_futures=True)

        new_executor: Executor = DetachedExecutor(thread_name_prefix="semantic-cache-io")
        self._io_executor = new_executor
        self._storage._io_executor = new_executor
        self._storage._db._executor = new_executor
        self._storage._collection._executor = new_executor
        logger.debug("IO executor replaced with fresh instance")

    @property
    def metrics(self) -> SessionMetrics:
        """Current session metrics accumulator."""
        return self._metrics

    def request_shutdown(self) -> None:
        """Signal that shutdown has been requested. New operations will be rejected."""
        self._shutting_down = True

    def begin_operation(self) -> bool:
        """Mark the start of an in-flight operation.

        Returns False if shutdown is in progress (caller should bail out).
        No lock needed: asyncio is cooperative and there is no await between
        the guard check and the counter increment.
        """
        if self._shutting_down:
            return False
        self._inflight += 1
        self._drained.clear()
        return True

    def end_operation(self) -> None:
        """Mark the end of an in-flight operation."""
        self._inflight = max(0, self._inflight - 1)
        if self._inflight == 0:
            self._drained.set()

    async def async_close(self) -> None:
        """Graceful shutdown: drain in-flight ops, persist metrics, close backends.

        Called from the lifespan finally block. Waits up to _DRAIN_TIMEOUT
        seconds for in-flight operations to finish before forcing close.
        Idempotent — safe to call multiple times.
        """
        if self._closed:
            return
        self._shutting_down = True

        # Wait for in-flight operations to drain.
        # Catch CancelledError so cleanup proceeds even if our task is cancelled
        # during asyncio.run()'s shutdown (after loop.stop from signal handler).
        if self._inflight > 0:
            logger.debug(f"Waiting for {self._inflight} in-flight operation(s) to finish...")
            try:
                await asyncio.wait_for(self._drained.wait(), timeout=self._DRAIN_TIMEOUT)
                logger.debug("All in-flight operations drained")
            except TimeoutError:
                logger.warning(
                    f"Drain timeout ({self._DRAIN_TIMEOUT}s) expired with "
                    f"{self._inflight} operation(s) still running — forcing close"
                )
            except asyncio.CancelledError:
                logger.warning(
                    f"Drain interrupted by cancellation with "
                    f"{self._inflight} operation(s) still running — forcing close"
                )

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

        self._io_executor.shutdown(wait=False)

        # Remove crash sentinel — signals clean shutdown.
        VectorStorage._remove_sentinel()

    def close(self) -> None:
        """Synchronous close fallback (no drain wait).

        Prefer async_close() in async contexts. This exists for atexit
        and signal handler safety.
        """
        if self._closed:
            return
        self._shutting_down = True
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

        self._io_executor.shutdown(wait=False)

        # Remove crash sentinel — signals clean shutdown.
        VectorStorage._remove_sentinel()

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------

    async def get_embedding(self, text: str, path: str = "") -> EmbeddingVector | None:
        """Embed text using FastEmbed. When path is given, prepends a file-type label
        (e.g. "Python source code: ...") so the model gets intent-rich context instead
        of raw syntactic noise — improves retrieval for JSON, YAML, and config files.

        Runs ONNX inference in a thread executor to avoid blocking the event loop.
        """
        try:
            # Late import from package so patch("semantic_cache_mcp.cache.embed") in tests works.
            # By call time the cache package is fully initialized in sys.modules.
            from . import embed as _embed  # noqa: PLC0415

            if path:
                text = f"{_file_label(path)}: {text}"

            loop = asyncio.get_running_loop()
            started = time.perf_counter()
            log_marker(
                logger,
                "embed.single.begin",
                path=path or None,
                chars=min(len(text), 8000),
            )
            result = await loop.run_in_executor(self._io_executor, _embed, text)
            log_marker(
                logger,
                "embed.single.end",
                path=path or None,
                ok=result is not None,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            if result:
                logger.debug(f"Embedding generated for {text[:50]}...")
            return result
        except Exception as e:
            log_marker(logger, "embed.single.fail", path=path or None, error=type(e).__name__)
            logger.warning(f"Failed to get embedding: {e}")
            return None

    # -------------------------------------------------------------------------
    # Delegated operations
    # -------------------------------------------------------------------------

    async def get(self, path: str) -> CacheEntry | None:
        if path in self._stale_paths:
            logger.debug(f"Treating stale cache entry as miss: {path}")
            return None
        entry = await self._storage.get(path)
        if entry:
            logger.debug(f"Cache hit: {path}")
        return entry

    def mark_stale(self, path: str) -> None:
        self._stale_paths.add(path)

    def clear_stale(self, path: str) -> None:
        self._stale_paths.discard(path)

    def is_stale(self, path: str) -> bool:
        return path in self._stale_paths

    def _compute_refresh_timeout(self, *, has_embedding: bool) -> float:
        """Choose a timeout based on the work still left in refresh_path()."""
        if has_embedding:
            return min(max(1.0, TOOL_TIMEOUT * 0.1), 2.0)

        import semantic_cache_mcp.core.embeddings._model as _emb_model  # noqa: PLC0415

        if _emb_model._model_ready:
            return min(max(2.0, TOOL_TIMEOUT * 0.2), 6.0)

        return min(max(5.0, TOOL_TIMEOUT * 0.5), 15.0)

    async def put(
        self,
        path: str,
        content: str,
        mtime: float,
        embedding: EmbeddingVector | None = None,
    ) -> None:
        tokens = count_tokens(content)
        started = time.perf_counter()
        log_marker(
            logger,
            "cache.put.begin",
            path=path,
            tokens=tokens,
            has_embedding=embedding is not None,
        )
        await self._storage.put(path, content, mtime, embedding)
        log_marker(
            logger,
            "cache.put.end",
            path=path,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        logger.debug(f"Cached file: {path} ({tokens} tokens)")

    async def refresh_path(
        self,
        path: str,
        content: str,
        mtime: float,
        embedding: EmbeddingVector | None = None,
        *,
        embedding_path: str | None = None,
        timeout: float | None = None,
    ) -> bool:
        refresh_timeout = (
            self._compute_refresh_timeout(has_embedding=embedding is not None)
            if timeout is None
            else timeout
        )
        started = time.perf_counter()
        log_marker(
            logger,
            "cache.refresh.begin",
            path=path,
            timeout_s=refresh_timeout,
            has_embedding=embedding is not None,
        )

        async def _refresh() -> None:
            actual_embedding = embedding
            if actual_embedding is None:
                actual_embedding = await self.get_embedding(content, embedding_path or path)
            await self.put(path, content, mtime, actual_embedding)

        try:
            await asyncio.wait_for(_refresh(), timeout=refresh_timeout)
        except TimeoutError:
            self.mark_stale(path)
            log_marker(
                logger,
                "cache.refresh.timeout",
                path=path,
                timeout_s=refresh_timeout,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            logger.warning(f"Cache refresh timed out for {path}; marking stale and resetting IO")
            self.reset_executor()
            return False
        except Exception as e:
            self.mark_stale(path)
            log_marker(
                logger,
                "cache.refresh.fail",
                path=path,
                error=type(e).__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            logger.warning(f"Cache refresh failed for {path}: {e}")
            return False

        self.clear_stale(path)
        log_marker(
            logger,
            "cache.refresh.end",
            path=path,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return True

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
        import semantic_cache_mcp.core.embeddings._model as _emb_model  # noqa: PLC0415

        stats["embedding_ready"] = _emb_model._model_ready
        stats["embedding_provider"] = _emb_model._execution_provider

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
        started = time.perf_counter()
        log_marker(logger, "embed.batch.begin", count=len(texts))
        try:
            result = cast(list[EmbeddingVector | None], _embed_batch(texts))
            log_marker(
                logger,
                "embed.batch.end",
                count=len(texts),
                ok=sum(1 for item in result if item is not None),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return result
        except Exception as exc:
            log_marker(
                logger,
                "embed.batch.fail",
                count=len(texts),
                error=type(exc).__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            raise

    async def clear(self) -> int:
        return await self._storage.clear()

    async def delete_path(self, path: str) -> int:
        """Delete one cached path and clear any stale marker for it."""
        removed = await self._storage.delete_path(path)
        self.clear_stale(path)
        return removed
