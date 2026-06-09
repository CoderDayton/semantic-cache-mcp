# Advanced Usage

## Programmatic API

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

# Smart read, diffs by default
result = smart_read(cache, path="/path/to/file.py", diff_mode=True)
print(f"Tokens saved: {result.tokens_saved}")
print(f"From cache:   {result.from_cache}")
print(f"Is diff:      {result.is_diff}")

# After context compression: force full content
result = smart_read(cache, path="/path/to/file.py", diff_mode=False)

# Read a specific line range (no full file load)
result = smart_read(cache, path="/path/to/file.py", offset=80, limit=40)

# Smart write, returns diff on overwrite, creates if new
write_result = smart_write(
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
smart_write(cache, path="/tmp/out.py", content="# chunk 1\n")
smart_write(cache, path="/tmp/out.py", content="# chunk 2\n", append=True)
smart_write(cache, path="/tmp/out.py", content="# chunk 3\n", append=True, auto_format=True)

# smart_edit Mode A: full-file find/replace (existing behavior)
edit_result = smart_edit(
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
edit_result = smart_edit(
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
edit_result = smart_edit(
    cache=cache,
    path="/path/to/file.py",
    old_string=None,
    new_string="    return result\n",
    start_line=80,
    end_line=83,
)

# smart_batch_edit, 2+ independent edits; accepts 2-tuples (Mode A) or 4-tuples (Modes B/C)
multi_result = smart_batch_edit(
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

# 2-tuples (old, new) are also accepted for backward compatibility
multi_result = smart_batch_edit(
    cache=cache,
    path="/path/to/file.py",
    edits=[("old1", "new1"), ("old2", "new2")],
)

# Keyword search (BM25) over cached files
# Must seed the cache first with smart_read / batch_smart_read
search_result = semantic_search(
    cache=cache,
    query="authentication logic",
    k=5,
    directory="/src",
)
for match in search_result.matches:
    print(f"{match.path}: {match.similarity:.2f}")

# Compare two files (returns a unified diff)
diff_result = compare_files(
    cache=cache,
    path1="/path/to/old.py",
    path2="/path/to/new.py",
    context_lines=2,
)
print(diff_result.diff_content)

# Batch read with glob expansion
batch_result = batch_smart_read(
    cache=cache,
    paths=["/src/a.py", "/src/b.py", "/src/*.py"],  # glob patterns supported
    max_total_tokens=30000,
    priority=["/src/a.py"],   # a.py read first regardless of size
    diff_mode=True,           # set False after context compression
)
print(f"Files read:    {batch_result.files_read}")
print(f"Tokens saved:  {batch_result.tokens_saved}")
print(f"Unchanged:     {batch_result.unchanged_paths}")

# Skipped files include est_tokens for budget planning
for f in batch_result.files:
    if f.status == "skipped" and f.est_tokens:
        print(f"  {f.path}: ~{f.est_tokens} tokens (use smart_read with offset/limit)")

# Glob with cache status
glob_result = glob_with_cache_status(
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
from semantic_cache_mcp.storage.docstore import ContentStorage
from semantic_cache_mcp.cache import SemanticCache

# Use a custom database path
cache = SemanticCache(db_path=Path("/tmp/my-cache.db"))
```

---

## Monitoring Cache Performance

```python
stats = cache.get_stats()
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

### Semantic Summarization

```python
from semantic_cache_mcp.core import summarize_semantic

# Summarization scores segments on its own; no embedding function is needed.
summary = summarize_semantic(
    content=large_file_content,
    max_size=10000,
)
```

### Delta Compression

```python
from semantic_cache_mcp.core import compute_delta, apply_delta

delta = compute_delta(old_content, new_content)
# For a 15KB file with 5 changed lines: ~245 bytes (98% compression)

recovered = apply_delta(old_content, delta)
assert recovered == new_content
```

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
cache.put(
    path="/path/to/file.py",
    content="file content here",
    mtime=1234567890.0,
)

# Retrieve a cached entry
entry = cache.get("/path/to/file.py")
if entry:
    content = cache.get_content(entry)

# Clear all entries
cleared = cache.clear()
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
| `truncated`        | `bool`        | Whether content was truncated/summarized       |
| `compression_ratio`| `float`       | Size ratio (returned / original)               |
| `semantic_match`   | `str \| None` | Unused; always `None` since the embedding layer was removed |

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
| `from_cache`   | `bool`        | Whether old content came from cache      |

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
| `from_cache`       | `bool`     | Whether content came from cache      |

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
| `from_cache`   | `bool`                  | Whether content came from cache      |

### SearchResult

| Field          | Type                | Description                              |
|----------------|---------------------|------------------------------------------|
| `matches`      | `list[SearchMatch]` | Ranked matches with path + similarity    |
| `cached_files` | `int`               | Total files searched                     |

### BatchReadResult / FileReadSummary

| Field            | Type             | Description                                      |
|------------------|------------------|--------------------------------------------------|
| `files`          | `list[FileReadSummary]` | Per-file status                         |
| `contents`       | `dict[str, str]` | Path → content (excludes unchanged/skipped)      |
| `files_read`     | `int`            | Files successfully read                          |
| `files_skipped`  | `int`            | Files skipped due to budget                      |
| `total_tokens`   | `int`            | Total tokens returned                            |
| `tokens_saved`   | `int`            | Tokens saved by caching                          |
| `unchanged_paths`| `list[str]`      | Files unchanged since last read                  |

| Field       | Type        | Description                                                           |
|-------------|-------------|-----------------------------------------------------------------------|
| `path`      | `str`       | File path                                                             |
| `tokens`    | `int`       | Tokens returned (0 for skipped/unchanged)                            |
| `status`    | `str`       | `"full"`, `"diff"`, `"truncated"`, `"skipped"`, or `"unchanged"`    |
| `from_cache`| `bool`      | Whether content came from cache                                       |
| `est_tokens`| `int \| None` | Estimated tokens for skipped files (for budget planning)           |

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

---

[← Back to README](../README.md)
