# Advanced Usage

## Programmatic API

```python
from semantic_cache_mcp.cache import SemanticCache, smart_read, smart_write, smart_edit

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

# Smart edit (find/replace) using cached reads
edit_result = smart_edit(
    cache=cache,
    path="/path/to/file.py",
    old_string="old_value",
    new_string="new_value",
    replace_all=False,
    dry_run=False,
)

print(f"Matches found: {edit_result.matches_found}")
print(f"Lines modified: {edit_result.line_numbers}")
print(f"Tokens saved: {edit_result.tokens_saved}")
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
