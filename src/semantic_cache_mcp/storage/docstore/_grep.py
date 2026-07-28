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
# Longest regex accepted. Caps the ReDoS surface: a pathological pattern needs
# room to express itself, and no legitimate grep needs this much.
GREP_MAX_PATTERN_LEN = 1000

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

# Quantifiers that let their target repeat two or more times. Applied to a group
# that already contains an unbounded quantifier, these are what make a pattern
# exponential: `(a+)+` explores every way to partition the same run of a's.
_UNBOUNDED_QUANTIFIERS = frozenset("*+")
_REPEAT_STARTERS = frozenset("*+{")


class GrepResults(list[dict]):
    """Grep hits, plus whether a cap stopped the scan before it finished.

    Subclasses ``list`` so every existing caller keeps indexing and iterating
    the results unchanged, while the tool layer can tell the caller its answer
    is partial. Without that, a scan that stops at ``max_matches`` reports the
    capped count as though it were the total — the one thing a search must
    never claim about work it did not do. ``files_not_searched`` counts the
    cached files the scan never opened.
    """

    __slots__ = ("truncated_matches", "truncated_files", "files_not_searched")

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.truncated_matches: bool = False
        self.truncated_files: bool = False
        self.files_not_searched: int = 0

    @property
    def complete(self) -> bool:
        """True when every candidate file was scanned to the end."""
        return not (self.truncated_matches or self.truncated_files)


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
) -> GrepResults:
    """Exact pattern matching across cached files — like ripgrep on the cache.

    Unlike search, returns line numbers and context, not ranked scores. The
    result carries whether ``max_matches``/``max_files`` cut the scan short, so
    a capped count is never mistaken for a complete one.
    """
    if store._closed:
        return GrepResults()

    # Clamp inputs to prevent excessive memory/CPU usage
    context_lines = max(0, min(context_lines, GREP_MAX_CONTEXT_LINES))
    max_matches = max(1, min(max_matches, GREP_MAX_MATCHES))
    max_files = max(1, min(max_files, GREP_MAX_FILES))

    flags = 0 if case_sensitive else re.IGNORECASE
    if fixed_string:
        compiled = re.compile(re.escape(pattern), flags)
    else:
        # A pattern the caller cannot use is an error, not an empty result set.
        # Returning [] here made a typo'd regex indistinguishable from a genuine
        # miss — the caller reads "no matches" and moves on believing it.
        if len(pattern) > GREP_MAX_PATTERN_LEN:
            raise ValueError(
                f"grep pattern too long ({len(pattern)} chars, limit {GREP_MAX_PATTERN_LEN})"
            )
        if has_nested_quantifier(pattern):
            raise ValueError(
                f"unsafe regex pattern {pattern!r}: a repeated group contains an "
                "unbounded quantifier (e.g. `(a+)+`), which can take exponential "
                "time to match and cannot be interrupted once started. Drop the "
                "inner or outer repeat — one of them is almost always redundant — "
                "or pass fixed_string=True to match the text literally."
            )
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(
                f"invalid regex pattern {pattern!r}: {e}. "
                f"Pass fixed_string=True to match it literally."
            ) from e

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
            return GrepResults()
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

    return scan_files(
        files,
        compiled,
        context_lines=context_lines,
        max_matches=max_matches,
        max_files=max_files,
    )


def scan_files(
    files: dict[str, list[tuple[int, str]]],
    compiled: re.Pattern[str],
    *,
    context_lines: int,
    max_matches: int,
    max_files: int,
) -> GrepResults:
    """Run *compiled* over already-loaded file chunks. Synchronous, no I/O.

    Every early exit records why, because a capped scan has not seen the rest
    of the corpus and must say so.
    """
    results = GrepResults()
    total_matches = 0
    pending = list(files.items())

    for index, (doc_path, chunks) in enumerate(pending):
        if total_matches >= max_matches:
            results.truncated_matches = True
            results.files_not_searched = len(pending) - index
            break
        if len(results) >= max_files:
            results.truncated_files = True
            results.files_not_searched = len(pending) - index
            break

        # Reconstruct file content from sorted chunks
        chunks.sort(key=lambda c: c[0])
        content = "".join(text for _, text in chunks)
        lines = content.splitlines()

        file_matches: list[dict] = []
        for i, line in enumerate(lines):
            if total_matches >= max_matches:
                # Stopped part-way through this file: the lines after this one
                # were never tested, so the count so far is a floor.
                results.truncated_matches = True
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


def _repeat_allows_many(pattern: str, index: int) -> tuple[bool, int]:
    """Can the quantifier at *index* repeat its target two or more times?

    Returns ``(allows_many, next_index)``. ``?`` and ``{0,1}`` cannot repeat, so
    they cannot compound an inner quantifier and are never flagged.
    """
    char = pattern[index]
    if char in _UNBOUNDED_QUANTIFIERS:
        return True, index + 1
    if char != "{":
        return False, index + 1
    close = pattern.find("}", index)
    if close == -1:
        return False, index + 1  # a literal brace, not a quantifier
    parts = pattern[index + 1 : close].split(",")
    try:
        if len(parts) == 1:
            return int(parts[0]) >= 2, close + 1
        low = int(parts[0]) if parts[0] else 0
        if not parts[1]:
            return True, close + 1  # {n,} is unbounded
        return max(low, int(parts[1])) >= 2, close + 1
    except ValueError:
        return False, close + 1  # not a numeric repeat; treat as a literal


def _skip_character_class(pattern: str, index: int) -> int:
    """Return the index just past the ``[...]`` class starting at *index*."""
    length = len(pattern)
    index += 1
    if index < length and pattern[index] == "^":
        index += 1
    if index < length and pattern[index] == "]":
        index += 1  # a leading ']' is a literal member
    while index < length and pattern[index] != "]":
        index += 2 if pattern[index] == "\\" else 1
    return index + 1


def has_nested_quantifier(pattern: str) -> bool:
    r"""True when a repeatable group encloses an unbounded quantifier.

    This is the shape behind catastrophic backtracking — ``(a+)+``, ``(\w*)*``,
    ``([0-9]{2,})+`` — where the engine explores exponentially many ways to
    split one run of input between the inner and the outer repeat. Detecting it
    statically is the only defense available here: ``re`` offers no match
    budget, and it holds the GIL while matching, so no timeout, worker thread,
    or cancellation can interrupt a match already under way.

    Deliberately narrow. A repeated group with no inner quantifier (``(abc)+``,
    ``(foo|bar)*``) is left alone — refusing ordinary patterns would cost far
    more than the rare exponential one does.
    """
    # One frame per group nesting level, recording whether an unbounded
    # quantifier has been seen inside that group.
    stack: list[bool] = [False]
    index = 0
    length = len(pattern)
    while index < length:
        char = pattern[index]
        if char == "\\":
            index += 2
            continue
        if char == "[":
            index = _skip_character_class(pattern, index)
            continue
        if char == "(":
            stack.append(False)
            index += 1
            continue
        if char == ")":
            inner_unbounded = stack.pop() if len(stack) > 1 else False
            index += 1
            if index < length and pattern[index] in _REPEAT_STARTERS:
                allows_many, index = _repeat_allows_many(pattern, index)
                if allows_many and inner_unbounded:
                    return True
                if allows_many:
                    stack[-1] = True
            # An inner unbounded quantifier still counts toward the enclosing
            # group even when this group itself is not repeated.
            if inner_unbounded:
                stack[-1] = True
            continue
        if char in _REPEAT_STARTERS:
            allows_many, next_index = _repeat_allows_many(pattern, index)
            if allows_many:
                stack[-1] = True
            index = next_index
            continue
        index += 1
    return False


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
