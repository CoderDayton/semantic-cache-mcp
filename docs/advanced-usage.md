# Advanced Usage

## Programmatic API

```python
from semantic_cache_mcp import SemanticCache, smart_read

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
    warmup,
    embed,
    embed_query,
    get_model_info,
)

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

### 4. Truncation (50-80% reduction)

For large files, smart truncation preserves:
- Function/class boundaries
- Important structure
- Beginning and end of file

---

## Disabling Features

```python
# Disable diff mode (always return full content)
result = smart_read(cache, path, diff_mode=False)

# Force full content even if cached
result = smart_read(cache, path, force_full=True)

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
    embedding=cache.get_embedding("file content here")
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
