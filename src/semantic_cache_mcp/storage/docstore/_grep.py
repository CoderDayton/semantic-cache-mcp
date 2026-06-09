"""Grep capability for :class:`ContentStorage`.

Exact, ripgrep-style pattern matching over cached file content, including the
sound BM25 prefilter that narrows the candidate set without sacrificing
completeness.

Split out of the ``ContentStorage`` god-module: each function takes the storage
instance explicitly (``store``) instead of ``self``, so the whole grep
subsystem lives in one place. ``ContentStorage`` keeps thin delegating methods
for the symbols its callers and tests depend on (``grep``,
``_grep_required_tokens``, ``_grep_sound_candidates``).
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from . import _META_CHUNK_INDEX, _META_IS_PARENT, _META_PATH

if TYPE_CHECKING:
    from . import ContentStorage
    from ._docstore import DocStore

logger = logging.getLogger(__name__)

# Upper bounds for grep parameters — prevent excessive memory/CPU usage.
GREP_MAX_CONTEXT_LINES = 20
GREP_MAX_MATCHES = 10_000
GREP_MAX_FILES = 500

# Prefilter tuning.
#
# A literal token must be at least this long to drive the prefilter — shorter
# tokens expand to too large a slice of the FTS5 vocabulary to be selective.
GREP_TOKEN_MIN_LEN = 4
# Cap on the vocabulary terms a single token may expand to. Past this the
# OR-query is unwieldy and the token is too common to prefilter usefully, so
# the caller falls back to a full scan.
GREP_VOCAB_TERM_CAP = 256
# Cap on BM25 rows fetched. If the match hits this cap the result may be
# truncated and can no longer be trusted as complete — full scan instead.
GREP_PREFILTER_FETCH_CAP = 1000
# Regex metacharacters that can make an extracted token non-mandatory:
# alternation, zero-allowing quantifiers, and character classes — a run like
# [abcd] yields the token "abcd" though the class matches only one character.
# Their presence in a regex pattern disqualifies the token-AND prefilter;
# fixed-string patterns are immune because there the characters are literal.
GREP_UNSAFE_REGEX_CHARS = frozenset("|?*{[")


async def grep(
    store: ContentStorage,
    pattern: str,
    *,
    path: str | None = None,
    fixed_string: bool = False,
    case_sensitive: bool = True,
    context_lines: int = 0,
    max_matches: int = 100,
    max_files: int = 50,
) -> list[dict]:
    """Exact pattern matching across cached files — like ripgrep on the cache.

    Unlike search, returns line numbers and context, not ranked scores.
    """
    if store._closed:
        return []

    # Clamp inputs to prevent excessive memory/CPU usage
    context_lines = max(0, min(context_lines, GREP_MAX_CONTEXT_LINES))
    max_matches = max(1, min(max_matches, GREP_MAX_MATCHES))
    max_files = max(1, min(max_files, GREP_MAX_FILES))

    flags = 0 if case_sensitive else re.IGNORECASE
    if fixed_string:
        compiled = re.compile(re.escape(pattern), flags)
    else:
        # Cap pattern length to mitigate ReDoS from pathological regexes.
        if len(pattern) > 1000:
            logger.warning(f"Regex pattern too long ({len(pattern)} chars), rejecting")
            return []
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            logger.warning(f"Invalid regex pattern: {e}")
            return []

    # Sound BM25 prefilter. Each required literal token is expanded to
    # the FTS5 vocabulary terms that contain it as a substring, then
    # BM25-matched. The candidate set stays complete even though FTS5's
    # unicode61 tokenizer keeps compound identifiers whole —
    # grep("function") still finds a file whose only hit is inside
    # "functionHelper". A None result means the prefilter cannot be
    # trusted for this pattern (complex regex, vocabulary unavailable,
    # or too broad); the caller then does a full scan, always correct.
    candidates = await sound_candidates(store, pattern, fixed_string=fixed_string, path_filter=path)
    files: dict[str, list[tuple[int, str]]] = {}
    if candidates is not None:
        # Vocabulary expansion makes the candidate set exact: empty means
        # nothing matches, non-empty is complete — no full scan needed.
        if not candidates:
            return []
        files = await load_files(store, candidates)
    else:
        all_docs = await store._collection.get_documents()
        for _doc_id, text, meta in all_docs:
            if meta.get(_META_IS_PARENT, False):
                continue  # Parent docs have empty content
            doc_path = meta.get(_META_PATH, "")
            if not doc_path or not path_matches(doc_path, path_filter=path):
                continue
            chunk_idx = meta.get(_META_CHUNK_INDEX, 0)
            files.setdefault(doc_path, []).append((chunk_idx, text))

    # Search each file's content
    results: list[dict] = []
    total_matches = 0

    for doc_path, chunks in files.items():
        if total_matches >= max_matches or len(results) >= max_files:
            break

        # Reconstruct file content from sorted chunks
        chunks.sort(key=lambda c: c[0])
        content = "".join(text for _, text in chunks)
        lines = content.splitlines()

        file_matches: list[dict] = []
        for i, line in enumerate(lines):
            if total_matches >= max_matches:
                break
            if compiled.search(line):
                match_info: dict[str, object] = {
                    "line_number": i + 1,
                    "line": line,
                }
                if context_lines > 0:
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    match_info["before"] = lines[start:i]
                    match_info["after"] = lines[i + 1 : end]
                file_matches.append(match_info)
                total_matches += 1

        if file_matches:
            results.append({"path": doc_path, "matches": file_matches})

    return results


def required_tokens(pattern: str, *, fixed_string: bool) -> list[str] | None:
    """Literal alphanumeric tokens that must appear in every match.

    Returns the tokens of length >= ``GREP_TOKEN_MIN_LEN``, or ``None`` when
    the pattern cannot be soundly prefiltered: a regex carrying alternation, a
    zero-allowing quantifier, or a character class (whose extracted tokens are
    not guaranteed substrings of every match), or a pattern with no token long
    enough to be selective. ``None`` routes the caller to a full scan, which is
    always correct.
    """
    if not fixed_string and any(c in GREP_UNSAFE_REGEX_CHARS for c in pattern):
        return None
    tokens = [t for t in re.findall(r"[A-Za-z0-9]+", pattern) if len(t) >= GREP_TOKEN_MIN_LEN]
    return tokens or None


def vocab_expand(
    store: DocStore,
    tokens: list[str],
    term_cap: int,
) -> list[list[str]] | None:
    """Expand each token to the FTS5 vocabulary terms containing it.

    Runs on the IO executor (blocking sqlite calls), under the store lock so the
    raw-connection access never races other DocStore operations. Returns one
    term list per token — an empty inner list means the token is absent from the
    index entirely. Returns ``None`` when the vocabulary cannot be queried or a
    token expands past ``term_cap`` (too broad to prefilter soundly). Uses an
    ``fts5vocab`` table in the connection-local ``temp`` schema, so it never
    touches the persistent doc-store schema.
    """
    with store._lock:
        conn = store.conn
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%fts5%' LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        fts_table = row[0]
        # Validate before interpolating — table names cannot be bound params.
        if not re.fullmatch(r"[A-Za-z0-9_]+", fts_table):
            return None

        vocab = "scmcp_grep_vocab"
        conn.execute(f"DROP TABLE IF EXISTS temp.{vocab}")
        # The 'main' schema argument is required: fts5vocab otherwise looks
        # for the FTS table in the vocab table's own schema (temp), where it
        # does not exist.
        conn.execute(
            f"CREATE VIRTUAL TABLE temp.{vocab} USING fts5vocab('main', '{fts_table}', 'row')"
        )
        try:
            per_token: list[list[str]] = []
            for token in tokens:
                rows = conn.execute(
                    f"SELECT DISTINCT term FROM temp.{vocab} "
                    f"WHERE term LIKE '%' || ? || '%' LIMIT ?",
                    (token.lower(), term_cap + 1),
                ).fetchall()
                if len(rows) > term_cap:
                    return None  # token too broad for a bounded MATCH query
                per_token.append([r[0] for r in rows])
            return per_token
        finally:
            conn.execute(f"DROP TABLE IF EXISTS temp.{vocab}")


async def sound_candidates(
    store: ContentStorage,
    pattern: str,
    *,
    fixed_string: bool,
    path_filter: str | None,
) -> list[str] | None:
    """Exact candidate paths for grep, or ``None`` to force a full scan.

    Expands each required token to the FTS5 vocabulary terms that contain it as
    a substring, then runs one BM25 MATCH over those terms. The candidate set
    is complete: every document whose line contains the token also contains it
    inside some indexed term, so the OR-of-terms / AND-of-tokens MATCH cannot
    miss it — this is what the raw whole-token MATCH got wrong for compound
    identifiers.

    Returns ``None`` on any condition that would break completeness — an
    unsupported pattern, a vocabulary error, an over-broad token, or a
    possibly-truncated BM25 result — so the caller falls back to a full scan.
    """
    tokens = required_tokens(pattern, fixed_string=fixed_string)
    if tokens is None:
        return None

    loop = asyncio.get_running_loop()
    try:
        per_token_terms = await loop.run_in_executor(
            store._io_executor,
            vocab_expand,
            store._sync_collection,
            tokens,
            GREP_VOCAB_TERM_CAP,
        )
    except Exception as exc:
        logger.debug(f"grep vocab expansion failed: {exc}; falling back to scan")
        return None
    if per_token_terms is None:
        return None
    if any(not terms for terms in per_token_terms):
        return []  # a required token appears in no indexed term

    # AND across tokens, OR across each token's vocabulary expansion.
    match_query = " AND ".join(
        "(" + " OR ".join(f'"{term}"' for term in terms) + ")" for terms in per_token_terms
    )
    try:
        results = await store._collection.keyword_search(match_query, k=GREP_PREFILTER_FETCH_CAP)
    except Exception as exc:
        logger.debug(f"grep BM25 prefilter failed: {exc}; falling back to scan")
        return None
    if len(results) >= GREP_PREFILTER_FETCH_CAP:
        return None  # possibly truncated — completeness no longer assured

    seen: set[str] = set()
    candidates: list[str] = []
    for doc, _score in results:
        meta = doc.metadata
        if meta.get(_META_IS_PARENT, False):
            continue
        doc_path = meta.get(_META_PATH, "")
        if not doc_path or doc_path in seen:
            continue
        if not path_matches(doc_path, path_filter=path_filter):
            continue
        seen.add(doc_path)
        candidates.append(doc_path)
    return candidates


async def load_files(
    store: ContentStorage,
    paths: list[str],
) -> dict[str, list[tuple[int, str]]]:
    """Load chunk text for a set of paths in a single batched lookup.

    Uses the doc store's list-value filter, which compiles to
    ``json_extract(metadata, '$.path') IN (?, ?, ...)`` — one round trip
    through the executor instead of N. Falls back to per-path lookups only if
    the batch query itself errors.
    """
    files: dict[str, list[tuple[int, str]]] = {}
    if not paths:
        return files

    try:
        docs = await store._collection.get_documents(
            filter_dict={_META_PATH: list(paths)},
        )
    except Exception as e:
        logger.debug(f"Batched grep lookup failed ({len(paths)} paths): {e}")
        for path in paths:
            results = await store._find_docs_by_path(path)
            for _doc_id, meta, text in results:
                if meta.get(_META_IS_PARENT, False):
                    continue
                chunk_idx = meta.get(_META_CHUNK_INDEX, 0)
                files.setdefault(path, []).append((chunk_idx, text))
        return files

    for _doc_id, text, meta in docs:
        if meta.get(_META_IS_PARENT, False):
            continue
        doc_path = meta.get(_META_PATH)
        if doc_path is None:
            continue
        chunk_idx = meta.get(_META_CHUNK_INDEX, 0)
        files.setdefault(doc_path, []).append((chunk_idx, text))
    return files


def path_matches(path: str, *, path_filter: str | None) -> bool:
    """Match exact paths, relative suffixes, basenames, and glob filters."""
    if not path_filter:
        return True

    normalized_path = path.replace("\\", "/")
    normalized_filter = path_filter.replace("\\", "/")
    has_glob = any(ch in normalized_filter for ch in "*?[")

    if has_glob:
        return any(
            fnmatch.fnmatchcase(normalized_path, candidate)
            for candidate in (
                normalized_filter,
                f"*/{normalized_filter}",
            )
        )

    return (
        normalized_path == normalized_filter
        or normalized_path.endswith(f"/{normalized_filter}")
        or Path(normalized_path).name == normalized_filter
    )
