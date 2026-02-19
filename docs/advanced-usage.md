# Advanced Usage

## Programmatic API

### Timing Policy (Token-Efficient)

- Use `read` for single-file iteration and verification.
- Use `batch_smart_read` for 2+ files instead of repeated `smart_read`. Use `priority` to control read order.
- Keep `diff_mode=True` while iterating; use `diff_mode=False` only when you need full uncached content.
- Use `smart_edit` for one targeted replacement; use `smart_multi_edit` for 2+ edits in one file.
- Seed cache before `semantic_search` and `find_similar_files`.
- Start search/similar with lower `k` (3-5), then increase only if recall is insufficient.
- Use `compare_files` only for explicit two-file comparisons.
- Use `glob_with_cache_status` to shortlist, then `batch_smart_read` for content retrieval.

```python
from semantic_cache_mcp.cache import (
    SemanticCache,
    smart_read,
    smart_write,
    smart_append,
    smart_edit,
    smart_multi_edit,
    semantic_search,
    compare_files,
    batch_smart_read,
    find_similar_files,
    glob_with_cache_status,
)

# Initialize cache (embeddings handled automatically)
cache = SemanticCache()

# Smart read with caching
result = smart_read(
    cache=cache,
    path="/path/to/file.py",
    max_size=50000,
    diff_mode=True,
)

print(f"Tokens saved: {result.tokens_saved}")
print(f"From cache: {result.from_cache}")
print(f"Is diff: {result.is_diff}")

# Smart write with cache integration
write_result = smart_write(
    cache=cache,
    path="/path/to/file.py",
    content="new file content",
    create_parents=True,
    dry_run=False,
)

print(f"Created: {write_result.created}")
print(f"Tokens saved: {write_result.tokens_saved}")
print(f"Hash: {write_result.content_hash}")

# Chunked write for large files
smart_write(cache=cache, path="/path/to/large.py", content="<chunk1>")
smart_append(cache=cache, path="/path/to/large.py", content="<chunk2>")
smart_append(cache=cache, path="/path/to/large.py", content="<chunk3>", auto_format=True)

# Smart edit (find/replace) using cached reads
edit_result = smart_edit(
    cache=cache,
    path="/path/to/file.py",
    old_string="old_value",
    new_string="new_value",
    replace_all=False,
    dry_run=True,  # validate before committing
)

print(f"Matches found: {edit_result.matches_found}")
print(f"Lines modified: {edit_result.line_numbers}")
print(f"Tokens saved: {edit_result.tokens_saved}")

# Multi-edit (batch find/replace)
multi_result = smart_multi_edit(
    cache=cache,
    path="/path/to/file.py",
    edits=[("old1", "new1"), ("old2", "new2")],
    dry_run=False,  # prefer this for 2+ replacements in same file
)

print(f"Succeeded: {multi_result.succeeded}")
print(f"Failed: {multi_result.failed}")

# Semantic search across cached files
search_result = semantic_search(
    cache=cache,
    query="authentication logic",
    k=5,  # start small, increase only if needed
    directory="/src",
)

for match in search_result.matches:
    print(f"{match.path}: {match.similarity:.2f}")

# Compare two files
diff_result = compare_files(
    cache=cache,
    path1="/path/to/old.py",
    path2="/path/to/new.py",
    context_lines=2,
)

print(f"Similarity: {diff_result.similarity:.2f}")
print(diff_result.diff_content)

# Batch read multiple files
batch_result = batch_smart_read(
    cache=cache,
    paths=["/src/a.py", "/src/b.py", "/src/c.py"],
    max_total_tokens=30000,  # start tighter, increase if necessary
    priority=["/src/a.py"],  # read a.py first regardless of size
)

print(f"Files read: {batch_result.files_read}")
print(f"Tokens saved: {batch_result.tokens_saved}")
print(f"Unchanged: {batch_result.unchanged_paths}")

# Skipped files include est_tokens for budget planning
for f in batch_result.files:
    if f.status == "skipped" and f.est_tokens:
        print(f"  {f.path}: ~{f.est_tokens} tokens (use read with offset/limit)")

# Find similar files
similar_result = find_similar_files(
    cache=cache,
    path="/src/auth.py",
    k=5,  # start 3-5, then expand
)

for f in similar_result.similar_files:
    print(f"{f.path}: {f.similarity:.2f}")

# Glob with cache status
glob_result = glob_with_cache_status(
    cache=cache,
    pattern="**/*auth*.py",  # prefer specific patterns
    directory="/src",
)

print(f"Total: {glob_result.total_matches}")
print(f"Cached: {glob_result.cached_count}")
```

---

## Custom Storage Backend

```python
from semantic_cache_mcp.storage import SQLiteStorage
from semantic_cache_mcp import SemanticCache
from pathlib import Path

# Use custom database location
custom_storage = SQLiteStorage(db_path=Path("/tmp/my-cache.db"))
cache = SemanticCache()
cache._storage = custom_storage
```

---

## Embedding Model Management

```python
from semantic_cache_mcp.core.embeddings import (
    configure,
    warmup,
    embed,
    embed_query,
    get_model_info,
)

# Configure model directory (optional, defaults to ~/.cache/semantic-cache-mcp/models)
# Call before warmup() when using embeddings outside the MCP server
configure(cache_dir="/path/to/custom/models")

# Warmup model (called automatically at server start)
warmup()

# Get model info
info = get_model_info()
print(f"Model: {info['model']}")
print(f"Ready: {info['ready']}")

# Generate embeddings
doc_embedding = embed("This is a document to embed")
query_embedding = embed_query("search query here")
```

---

## Monitoring Cache Performance

```python
# Get detailed statistics
stats = cache.get_stats()

print(f"Files: {stats['files_cached']}")
print(f"Tokens: {stats['total_tokens_cached']}")
print(f"Compression ratio: {stats['compression_ratio']:.1%}")
print(f"Deduplication ratio: {stats['dedup_ratio']:.2f}x")
```

---

## Environment Configuration

```bash
# Debug logging
export LOG_LEVEL=DEBUG

# Start server
semantic-cache-mcp
```

---

## ReadResult Object

The `smart_read` function returns a `ReadResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `str` | The file content, diff, or message |
| `from_cache` | `bool` | Whether result came from cache |
| `is_diff` | `bool` | Whether content is a diff |
| `tokens_original` | `int` | Original token count |
| `tokens_returned` | `int` | Returned token count |
| `tokens_saved` | `int` | Tokens saved |
| `truncated` | `bool` | Whether content was truncated |
| `compression_ratio` | `float` | Size ratio (returned/original) |
| `semantic_match` | `str \| None` | Path of similar file if matched |

---

## WriteResult Object

The `smart_write` function returns a `WriteResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `path` | `str` | Resolved absolute path to written file |
| `bytes_written` | `int` | Number of bytes written |
| `tokens_written` | `int` | Token count of written content |
| `created` | `bool` | True if new file, False if overwrite |
| `diff_content` | `str \| None` | Unified diff from old content |
| `diff_stats` | `dict \| None` | Insertions, deletions, modifications |
| `tokens_saved` | `int` | Tokens saved by returning diff |
| `content_hash` | `str` | BLAKE3 hash for verification |
| `from_cache` | `bool` | Whether old content came from cache |

---

## AppendResult Object

The `smart_append` function returns an `AppendResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `path` | `str` | Resolved absolute path to the file |
| `bytes_appended` | `int` | Number of bytes appended |
| `total_bytes` | `int` | Total file size after append |
| `tokens_appended` | `int` | Estimated token count of appended content |
| `content_hash` | `str` | BLAKE3 hash of appended content chunk |
| `created` | `bool` | True if file was created, False if appended |
| `cache_invalidated` | `bool` | True if a stale cache entry was removed |

---

## EditResult Object

The `smart_edit` function returns an `EditResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `path` | `str` | Resolved absolute path to edited file |
| `matches_found` | `int` | Total occurrences of old_string |
| `replacements_made` | `int` | Number of replacements performed |
| `line_numbers` | `list[int]` | Lines where replacements occurred |
| `diff_content` | `str` | Unified diff of changes |
| `diff_stats` | `dict` | Insertions, deletions, modifications |
| `tokens_saved` | `int` | Tokens saved by cached read |
| `content_hash` | `str` | BLAKE3 hash of new content |
| `from_cache` | `bool` | Whether content came from cache |

---

## MultiEditResult Object

The `smart_multi_edit` function returns a `MultiEditResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `path` | `str` | Resolved absolute path to edited file |
| `succeeded` | `int` | Number of successful edits |
| `failed` | `int` | Number of failed edits |
| `outcomes` | `list[SingleEditOutcome]` | Per-edit results with success/error |
| `diff_content` | `str` | Combined unified diff |
| `diff_stats` | `dict` | Insertions, deletions, modifications |
| `tokens_saved` | `int` | Tokens saved by cached read |
| `content_hash` | `str` | BLAKE3 hash of final content |
| `from_cache` | `bool` | Whether content came from cache |

---

## SearchResult Object

The `semantic_search` function returns a `SearchResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `matches` | `list[SearchMatch]` | Ranked matches with path, similarity, preview |
| `cached_files` | `int` | Total files in cache that were searched |

---

## DiffResult Object

The `compare_files` function returns a `DiffResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `path1` | `str` | First file path |
| `path2` | `str` | Second file path |
| `diff_content` | `str` | Unified diff |
| `diff_stats` | `dict` | Insertions, deletions, modifications |
| `similarity` | `float` | Semantic similarity (0.0-1.0) |
| `tokens_saved` | `int` | Tokens saved by cache |
| `from_cache` | `tuple[bool, bool]` | Cache status for each file |

---

## BatchReadResult Object

The `batch_smart_read` function returns a `BatchReadResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `files` | `list[FileReadSummary]` | Per-file status and token counts |
| `contents` | `dict[str, str]` | Path to content mapping (excludes unchanged files) |
| `files_read` | `int` | Number of files successfully read |
| `files_skipped` | `int` | Files skipped (budget exceeded) |
| `total_tokens` | `int` | Total tokens returned |
| `tokens_saved` | `int` | Tokens saved by caching |
| `unchanged_paths` | `list[str]` | Files that haven't changed since last read |

### FileReadSummary Fields

| Field | Type | Description |
|-------|------|-------------|
| `path` | `str` | File path |
| `tokens` | `int` | Tokens returned (0 for skipped) |
| `status` | `str` | `"full"`, `"diff"`, `"truncated"`, `"skipped"`, or `"unchanged"` |
| `from_cache` | `bool` | Whether content came from cache |
| `est_tokens` | `int \| None` | Estimated tokens for skipped files (for budget planning) |

---

## SimilarFilesResult Object

The `find_similar_files` function returns a `SimilarFilesResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `source_path` | `str` | Source file path |
| `source_tokens` | `int` | Token count of source file |
| `similar_files` | `list[SimilarFile]` | Ranked similar files |
| `files_searched` | `int` | Total cached files searched |

---

## GlobResult Object

The `glob_with_cache_status` function returns a `GlobResult` with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `pattern` | `str` | Glob pattern used |
| `directory` | `str` | Base directory |
| `matches` | `list[GlobMatch]` | Files with cache status |
| `total_matches` | `int` | Total files matched |
| `cached_count` | `int` | Files in cache |
| `total_cached_tokens` | `int` | Total tokens of cached files |

---

## Caching Strategies

### 1. File Unchanged (99% reduction)

When `mtime` matches cached entry:
```
// File unchanged: /path/to/file.py (1234 tokens cached)
```

### 2. File Changed - Diff (80-95% reduction)

When file modified since cache:
```diff
// Diff for /path/to/file.py (changed since cache):
--- cached
+++ current
@@ -10,7 +10,7 @@
 def foo():
-    return "old"
+    return "new"
```

### 3. Semantic Match (70-90% reduction)

When similar file found in cache:
```diff
// Similar to cached: /path/to/similar.py
// Diff from similar file:
--- similar
+++ current
@@ -1,5 +1,5 @@
...
```

### 4. Semantic Summarization (50-80% reduction)

For large files, semantic summarization preserves:
- First segment (docstrings, imports, headers)
- Important functions/classes based on scoring
- High-density code segments
- Diverse content (avoids redundancy)

**Scoring algorithm:**
- Position: U-shaped curve (start/end prioritized)
- Density: Unique tokens, syntax characters
- Diversity: Penalize similarity to already-selected segments

---

## Advanced Algorithms

### SIMD Chunking

```python
from semantic_cache_mcp.core import get_optimal_chunker

# Automatically select best chunker (SIMD if available)
chunker = get_optimal_chunker(prefer_simd=True)
chunks = list(chunker(content.encode()))

# Force serial HyperCDC
from semantic_cache_mcp.core import hypercdc_chunks
chunks = list(hypercdc_chunks(content.encode()))

# Force SIMD parallel (may fall back if unavailable)
from semantic_cache_mcp.core.chunking_simd import hypercdc_simd_chunks
chunks = list(hypercdc_simd_chunks(content.encode()))
```

### Semantic Summarization

```python
from semantic_cache_mcp.core import summarize_semantic

# Summarize with embedding support
def embed_fn(text: str):
    emb = cache.get_embedding(text)
    return np.asarray(emb, dtype=np.float32) if emb else None

summary = summarize_semantic(
    content=large_file_content,
    max_size=10000,
    embed_fn=embed_fn
)
```

### LSH Approximate Search

```python
from semantic_cache_mcp.core.lsh import LSHIndex

# Build LSH index for fast approximate search
index = LSHIndex(dim=768, num_tables=10, num_bits=16)
index.build(embeddings)

# Query (100x faster than exact search on 10K+ vectors)
nearest = index.query(query_embedding, k=5)
```

### Extreme Quantization

```python
from semantic_cache_mcp.core.quantization import (
    quantize_binary,
    quantize_ternary,
    binary_similarity
)

# Binary quantization (32x compression)
binary = quantize_binary(embedding)  # 1 bit per dimension

# Ternary quantization (16x compression, better accuracy)
ternary = quantize_ternary(embedding)  # 2 bits per dimension

# Fast binary similarity
similarity = binary_similarity(binary1, binary2)
```

---

## Disabling Features

```python
# Disable diff mode (always return full content)
result = smart_read(cache, path, diff_mode=False)

# Limit content size
result = smart_read(cache, path, max_size=50000)
```

---

## Direct Cache Operations

```python
# Store a file manually
cache.put(
    path="/path/to/file.py",
    content="file content here",
    mtime=1234567890.0,
    embedding=cache.get_embedding("file content here"),
    tokens=42,  # pass pre-computed token count to avoid redundant tokenization
)

# Get cached entry
entry = cache.get("/path/to/file.py")
if entry:
    content = cache.get_content(entry)

# Find similar files
embedding = cache.get_embedding("search text")
similar_path = cache.find_similar(embedding, exclude_path="/current/file.py")

# Clear cache
cleared = cache.clear()
print(f"Cleared {cleared} entries")
```

---

[← Back to README](../README.md)
