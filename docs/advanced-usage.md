# Advanced Usage

## Programmatic API

Every `smart_*` function, `semantic_search`, `compare_files`, `batch_smart_read`,
`glob_with_cache_status`, and the data methods of `SemanticCache` are coroutines.
The snippets below show the calls only; run them inside `asyncio.run(main())`.

The functions here work on paths and mtimes. The possession rules (`unchanged`
and diffs only for a caller that echoes a matching `known_hash`, `partial:`
file hashes, `coverage_token`, line ranges) are enforced by the MCP tool layer,
which picks `diff_mode`/`force_full` per call. Calling `smart_read` directly with
`diff_mode=True` answers from the cache by mtime, so only do that when your
caller really holds the previous content.

```python
from semantic_cache_mcp.cache import (
    SemanticCache,
    smart_read,
    smart_write,
    smart_edit,
    smart_batch_edit,
    semantic_search,
    compare_files,
    batch_smart_read,
    glob_with_cache_status,
)

# Initialize cache
cache = SemanticCache()

# Smart read: unchanged marker or diff when the file is already cached
result = await smart_read(cache, path="/path/to/file.py", diff_mode=True)
print(f"Tokens saved: {result.tokens_saved}")
print(f"From cache:   {result.from_cache}")
print(f"Is diff:      {result.is_diff}")

# Caller holds nothing: force full content
result = await smart_read(cache, path="/path/to/file.py", diff_mode=False, force_full=True)

# Diff against text the caller proved it holds (e.g. an older stored version);
# full content when the diff would not be smaller
held = await cache.held_version("/path/to/file.py", claim="<content_hash>")
result = await smart_read(
    cache, path="/path/to/file.py",
    base_content=held, diff_mode=False, force_full=True, refresh_cache=False,
)

# A file over max_size comes back as a map, not its text: the outline, or the
# first and last 20 lines under `// L<a>-<b>` anchors. Pass map_oversized=False
# to get the whole file regardless of size.
result = await smart_read(cache, path="/path/to/big.py", max_size=100_000)
if result.truncated:
    print(f"Map of {result.total_lines} lines")

# Line ranges (offset/limit) are a tool-layer feature of `read`; the function
# above has no such parameters. Slice result.content yourself.

# Smart write, returns diff on overwrite, creates if new
write_result = await smart_write(
    cache=cache,
    path="/path/to/file.py",
    content="new file content",
    create_parents=True,
    dry_run=False,       # set True to preview without writing
    auto_format=True,    # run ruff/prettier/gofmt after write
)
print(f"Created:      {write_result.created}")
print(f"Tokens saved: {write_result.tokens_saved}")
print(f"Hash:         {write_result.content_hash}")

# Append mode: write large files in chunks
await smart_write(cache, path="/tmp/out.py", content="# chunk 1\n")
await smart_write(cache, path="/tmp/out.py", content="# chunk 2\n", append=True)
await smart_write(cache, path="/tmp/out.py", content="# chunk 3\n", append=True, auto_format=True)

# smart_edit Mode A: full-file find/replace
edit_result = await smart_edit(
    cache=cache,
    path="/path/to/file.py",
    old_string="old_value",
    new_string="new_value",
    replace_all=False,
    dry_run=True,        # validate before committing
)
print(f"Matches found:    {edit_result.matches_found}")
print(f"Lines modified:   {edit_result.line_numbers}")
print(f"Tokens saved:     {edit_result.tokens_saved}")

# smart_edit Mode B: scoped find/replace within a line range
# Useful when read() gave you line numbers, so a shorter old_string suffices
edit_result = await smart_edit(
    cache=cache,
    path="/path/to/file.py",
    old_string="pass",       # matched only within lines 42 to 42
    new_string="return x",
    start_line=42,
    end_line=42,
)
print(f"Lines modified: {edit_result.line_numbers}")  # absolute line numbers

# smart_edit Mode C: direct line replacement (maximum token savings)
# No old_string needed; replaces the entire range unconditionally
edit_result = await smart_edit(
    cache=cache,
    path="/path/to/file.py",
    old_string=None,
    new_string="    return result\n",
    start_line=80,
    end_line=83,
)

# smart_batch_edit, 2+ independent edits; accepts 2-tuples (Mode A) or 4-tuples (Modes B/C)
multi_result = await smart_batch_edit(
    cache=cache,
    path="/path/to/file.py",
    edits=[
        ("old1", "new1", None, None),    # Mode A: full-file replace
        ("pass", "return x", 42, 42),    # Mode B: scoped replace
        (None, "    return result\n", 80, 83),  # Mode C: line replace
    ],
)
print(f"Succeeded: {multi_result.succeeded}")
print(f"Failed:    {multi_result.failed}")
# Each edit is validated independently, so some can succeed while others fail.
# Three shapes are rejected rather than silently resolved:
#   - two edits whose line ranges overlap
#   - a find/replace whose anchor falls inside another edit's line range
#   - an old_string matching more than once (batch has no replace_all; scope it
#     with start_line/end_line instead)
# Every reported success is verified against the text it produced, so an edit
# whose anchor was consumed by an earlier one in the batch fails with that
# reason instead of being counted as applied.

# 2-tuples (old, new) are also accepted (Mode A only)
multi_result = await smart_batch_edit(
    cache=cache,
    path="/path/to/file.py",
    edits=[("old1", "new1"), ("old2", "new2")],
)

# Keyword search (BM25) over cached files
# Must seed the cache first with smart_read / batch_smart_read
# Query terms are quoted (never parsed as FTS5 operators) and joined with OR,
# so a word the corpus lacks narrows the ranking instead of emptying the results.
# directory=None searches the whole cache; the MCP `search` tool anchors it to the
# client root instead.
search_result = await semantic_search(
    cache=cache,
    query="authentication logic",
    k=5,
    directory="/src",
)
for match in search_result.matches:
    print(f"{match.path}: {match.similarity:.2f}")

# Compare two files (returns a unified diff)
# Not exposed as an MCP tool — the agent-facing surface is read/write/edit/grep.
# It is supported programmatic API and is not going away.
diff_result = await compare_files(
    cache=cache,
    path1="/path/to/old.py",
    path2="/path/to/new.py",
    context_lines=2,
)
print(diff_result.diff_content)

# Batch read. Paths are literal; the MCP `batch_read` tool expands globs before
# calling this, so list matches yourself first (glob_with_cache_status below).
batch_result = await batch_smart_read(
    cache=cache,
    paths=["/src/a.py", "/src/b.py"],
    max_total_tokens=30000,
    priority=["/src/a.py"],   # a.py read first, but still bound by the budget
    diff_mode=True,           # False: never diff or answer unchanged
    # Possession proof: only files whose hash you echo here can come back as
    # `unchanged` or as a diff. Everything else is sent in full, so omitting
    # this after a compaction is always safe.
    known_hashes={"/src/a.py": "8f3c..."},
)
print(f"Files read:    {batch_result.files_read}")
print(f"Tokens saved:  {batch_result.tokens_saved}")
print(f"Unchanged:     {batch_result.unchanged_paths}")

# Skipped files include est_tokens for budget planning
for f in batch_result.files:
    if f.status == "skipped" and f.est_tokens:
        print(f"  {f.path}: ~{f.est_tokens} tokens (read it alone or raise max_total_tokens)")

# Glob with cache status
glob_result = await glob_with_cache_status(
    cache=cache,
    pattern="**/*auth*.py",
    directory="/src",
    cached_only=False,   # set True to see only already-cached files
)
print(f"Total:  {glob_result.total_matches}")
print(f"Cached: {glob_result.cached_count}")
```

---

## Custom Storage Backend

```python
from pathlib import Path
from semantic_cache_mcp.cache import SemanticCache

# Use a custom database path (default: docstore.db in the cache directory)
cache = SemanticCache(db_path=Path("/tmp/my-cache.db"))
# ... use it, then release the database
cache.close()
```

---

## Monitoring Cache Performance

```python
stats = await cache.get_stats()
print(f"Files cached:       {stats['files_cached']}")
print(f"Total tokens:       {stats['total_tokens_cached']}")
print(f"DB size (MB):       {stats['db_size_mb']}")

# Session metrics (current session)
session = stats["session"]
print(f"Tokens saved:       {session['tokens_saved']}")
print(f"Cache hits/misses:  {session['cache_hits']}/{session['cache_misses']}")
print(f"Tool calls:         {session['tool_calls']}")

# Lifetime metrics (aggregated across all completed sessions)
lifetime = stats["lifetime"]
print(f"Total sessions:     {lifetime['total_sessions']}")
print(f"Lifetime saved:     {lifetime['tokens_saved']}")
```

---

## Advanced Algorithms

### Large-File Map

```python
from semantic_cache_mcp.core import large_file_map

# The outline (`<line>: <signature>`), or the first and last 20 lines under
# `// L<a>-<b>` anchors when the file has no definitions.
file_map = large_file_map(large_file_content, filename="module.py")
```

### Token Counting

```python
from semantic_cache_mcp.core import count_tokens
from semantic_cache_mcp.core.tokenizer import tokens_exceed

n = count_tokens(text)            # exact o200k count (matches tiktoken)
big = tokens_exceed(text, 4000)   # stops counting at the limit
```

Counts are exact up to 256K characters; longer text is sampled from 16 windows.

---

## Direct Cache Operations

> **Search-cache invalidation:** `SemanticCache` keeps an in-session LRU of
> recent search results (32 entries, keyed on `query`/`k`/`directory`). All
> mutations (`put`, `clear`, `delete_path`, `update_mtime`) automatically
> call `_bump_search_cache()` to invalidate it, so callers never see a
> result that predates their last write. If you bypass the public API and
> mutate underlying storage directly (rare), call `cache._bump_search_cache()`
> manually.

```python
# Store a file manually
await cache.put(
    path="/path/to/file.py",
    content="file content here",
    mtime=1234567890.0,
)

# Retrieve a cached entry
entry = await cache.get("/path/to/file.py")
if entry:
    content = await cache.get_content(entry)

# A past version the caller proves it holds, by its content_hash (or None).
# The store keeps the last 4 versions per path (2,000 in total).
old_text = await cache.held_version("/path/to/file.py", "<content_hash>")

# Clear all entries
cleared = await cache.clear()
print(f"Cleared {cleared} entries")
```

---

## Return Types Reference

### ReadResult

| Field              | Type          | Description                                    |
|--------------------|---------------|------------------------------------------------|
| `content`          | `str`         | File content, diff, or "unchanged" message     |
| `from_cache`       | `bool`        | Whether result came from cache                 |
| `is_diff`          | `bool`        | Whether content is a unified diff              |
| `tokens_original`  | `int`         | Original token count                           |
| `tokens_returned`  | `int`         | Tokens in the returned content                 |
| `tokens_saved`     | `int`         | Tokens saved vs returning full content         |
| `truncated`        | `bool`        | Whether content is a map of an oversized file  |
| `total_lines`      | `int \| None` | Line count of the whole file; set only when `truncated` (a map) |
| `is_binary`        | `bool`        | Binary file: `content` is empty, `size`/`mime` describe it |
| `size`             | `int \| None` | Byte size; binary files only                   |
| `mime`             | `str \| None` | Guessed MIME type; binary files only           |
| `compression_ratio`| `float`       | Size ratio (returned / original)               |
| `content_hash`     | `str \| None` | BLAKE3 digest of the content this result reflects; `None` for binary fallbacks. The tool layer surfaces it as `content_hash` for a full read, and as `partial:`-prefixed `file_hash` for a ranged or mapped one. The tool layer also attaches a signed `coverage_token` to a ranged read; it names the delivered lines and *is* redeemable as `known_hash` for those lines |

### WriteResult

| Field          | Type          | Description                              |
|----------------|---------------|------------------------------------------|
| `path`         | `str`         | Resolved absolute path                   |
| `bytes_written`| `int`         | Bytes written to disk                    |
| `tokens_written`| `int`        | Token count of written content           |
| `created`      | `bool`        | `True` if new file, `False` if overwrite |
| `diff_content` | `str \| None` | Unified diff from previous content       |
| `diff_stats`   | `dict \| None`| Insertions, deletions, modifications     |
| `tokens_saved` | `int`         | Tokens saved by returning diff           |
| `content_hash` | `str`         | BLAKE3 hex digest                        |
| `previous_hash`| `str \| None` | Digest of the content the write started from; `None` when the file was created. The tool layer matches it against `known_hash` to decide whether an append is claimable |
| `from_cache`   | `bool`        | Whether old content came from cache      |
| `dry_run`      | `bool`        | `True` when nothing was written          |

### EditResult

| Field              | Type       | Description                          |
|--------------------|------------|--------------------------------------|
| `path`             | `str`      | Resolved absolute path               |
| `matches_found`    | `int`      | Total occurrences of `old_string`    |
| `replacements_made`| `int`      | Number of replacements performed     |
| `line_numbers`     | `list[int]`| Lines where replacements occurred    |
| `diff_content`     | `str`      | Unified diff of changes              |
| `diff_stats`       | `dict`     | Insertions, deletions, modifications |
| `tokens_saved`     | `int`      | Tokens saved by cached read          |
| `content_hash`     | `str`      | BLAKE3 hex digest of new content     |
| `previous_hash`    | `str \| None` | Digest of the pre-edit content     |
| `from_cache`       | `bool`     | Whether content came from cache      |
| `dry_run`          | `bool`     | `True` when nothing was written      |

### BatchEditResult

| Field          | Type                    | Description                          |
|----------------|-------------------------|--------------------------------------|
| `path`         | `str`                   | Resolved absolute path               |
| `succeeded`    | `int`                   | Number of successful edits           |
| `failed`       | `int`                   | Number of failed edits               |
| `outcomes`     | `list[SingleEditOutcome]` | Per-edit success/error detail      |
| `diff_content` | `str`                   | Combined unified diff                |
| `diff_stats`   | `dict`                  | Insertions, deletions, modifications |
| `tokens_saved` | `int`                   | Tokens saved by cached read          |
| `content_hash` | `str`                   | BLAKE3 hex digest of final content   |
| `previous_hash`| `str \| None`           | Digest of the pre-edit content       |
| `from_cache`   | `bool`                  | Whether content came from cache      |
| `dry_run`      | `bool`                  | `True` when nothing was written      |

### SearchResult

| Field          | Type                | Description                              |
|----------------|---------------------|------------------------------------------|
| `query`        | `str`               | The query as given                       |
| `matches`      | `list[SearchMatch]` | Ranked matches (see below)               |
| `files_searched` | `int`             | Cached files searched (equals `cached_files`) |
| `cached_files` | `int`               | Files in the cache                       |

`SearchMatch`: `path` (`str`), `similarity` (`float`, BM25 score normalized so the top match is 1.0), `tokens` (`int`), `preview` (`str`).

### BatchReadResult / FileReadSummary

| Field            | Type             | Description                                      |
|------------------|------------------|--------------------------------------------------|
| `files`          | `list[FileReadSummary]` | Per-file status                         |
| `contents`       | `dict[str, str]` | Path → content (excludes unchanged/skipped)      |
| `files_read`     | `int`            | Files successfully read                          |
| `files_skipped`  | `int`            | Files skipped due to budget                      |
| `total_tokens`   | `int`            | Total tokens returned                            |
| `tokens_saved`   | `int`            | Tokens saved by caching                          |
| `unchanged_paths`| `list[str]`      | Files the caller proved it holds (via `known_hashes`) that have not changed |

| Field       | Type        | Description                                                           |
|-------------|-------------|-----------------------------------------------------------------------|
| `path`      | `str`       | File path                                                             |
| `tokens`    | `int`       | Tokens returned (0 for skipped; the marker's size for unchanged)     |
| `status`    | `str`       | `"full"`, `"diff"`, `"truncated"`, `"skipped"`, or `"unchanged"`    |
| `from_cache`| `bool`      | Whether content came from cache                                       |
| `est_tokens`| `int \| None` | Estimated tokens for skipped files (for budget planning)           |
| `content_hash`| `str \| None` | Hash of the delivered content — pass it back in `known_hashes` next time. `None` when what was sent is not the file itself (skipped, or a map of a large file) |

### DiffResult

| Field          | Type             | Description                         |
|----------------|------------------|-------------------------------------|
| `path1`        | `str`            | First file path                     |
| `path2`        | `str`            | Second file path                    |
| `diff_content` | `str`            | Unified diff                        |
| `diff_stats`   | `dict`           | Insertions, deletions, modifications|
| `tokens_saved` | `int`            | Tokens saved by cache               |
| `from_cache`   | `tuple[bool, bool]` | Cache status for each file       |

### GlobResult

| Field                 | Type            | Description                           |
|-----------------------|-----------------|---------------------------------------|
| `pattern`             | `str`           | Glob pattern used                     |
| `directory`           | `str`           | Base directory                        |
| `matches`             | `list[GlobMatch]` | Files with cache status             |
| `total_matches`       | `int`           | Total files matched                   |
| `cached_count`        | `int`           | Files already in cache                |
| `total_cached_tokens` | `int`           | Total tokens of cached files          |

`GlobMatch`: `path` (`str`), `cached` (`bool`), `tokens` (`int | None`), `mtime` (`float`).

---

[← Back to README](../README.md)
