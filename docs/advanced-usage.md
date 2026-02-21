# Advanced Usage

## Programmatic API

### Timing Policy (Token-Efficient)

- Use `smart_read` for single-file iteration; `batch_smart_read` for 2+ files
- Keep `diff_mode=True` while iterating; set `diff_mode=False` only after context compression to recover full content
- Use `smart_edit` for one targeted replacement; `smart_batch_edit` for 2+ edits in one file
- Seed cache before `semantic_search` or `find_similar_files` — they only search cached files
- Start `k` at 3–5 for search/similar; increase only if recall is insufficient
- Use `glob_with_cache_status` to shortlist candidates, then `batch_smart_read` to retrieve content

```python
from semantic_cache_mcp.cache import (
    SemanticCache,
    smart_read,
    smart_write,
    smart_edit,
    smart_batch_edit,
    semantic_search,
    find_similar_files,
    compare_files,
    batch_smart_read,
    glob_with_cache_status,
)
from semantic_cache_mcp.core.embeddings import embed_batch

# Initialize cache
cache = SemanticCache()

# Smart read — diffs by default
result = smart_read(cache, path="/path/to/file.py", diff_mode=True)
print(f"Tokens saved: {result.tokens_saved}")
print(f"From cache:   {result.from_cache}")
print(f"Is diff:      {result.is_diff}")

# After context compression: force full content
result = smart_read(cache, path="/path/to/file.py", diff_mode=False)

# Read a specific line range (no full file load)
result = smart_read(cache, path="/path/to/file.py", offset=80, limit=40)

# Smart write — returns diff on overwrite, creates if new
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

# Append mode — write large files in chunks
smart_write(cache, path="/tmp/out.py", content="# chunk 1\n")
smart_write(cache, path="/tmp/out.py", content="# chunk 2\n", append=True)
smart_write(cache, path="/tmp/out.py", content="# chunk 3\n", append=True, auto_format=True)

# smart_edit — Mode A: full-file find/replace (existing behavior)
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

# smart_edit — Mode B: scoped find/replace within a line range
# Useful when read() gave you line numbers — shorter old_string suffices
edit_result = smart_edit(
    cache=cache,
    path="/path/to/file.py",
    old_string="pass",       # matched only within lines 42–42
    new_string="return x",
    start_line=42,
    end_line=42,
)
print(f"Lines modified: {edit_result.line_numbers}")  # absolute line numbers

# smart_edit — Mode C: direct line replacement (maximum token savings)
# No old_string needed — replaces the entire range unconditionally
edit_result = smart_edit(
    cache=cache,
    path="/path/to/file.py",
    old_string=None,
    new_string="    return result\n",
    start_line=80,
    end_line=83,
)

# smart_batch_edit — 2+ independent edits; accepts 2-tuples (Mode A) or 4-tuples (Modes B/C)
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

# Semantic search — embedding-based, NOT keyword
# Must seed cache first with smart_read / batch_smart_read
search_result = semantic_search(
    cache=cache,
    query="authentication logic",
    k=5,
    directory="/src",
)
for match in search_result.matches:
    print(f"{match.path}: {match.similarity:.2f}")

# Find similar files — must seed cache first
similar_result = find_similar_files(cache=cache, path="/src/auth.py", k=5)
for f in similar_result.similar_files:
    print(f"{f.path}: {f.similarity:.2f}")

# Compare two files
diff_result = compare_files(
    cache=cache,
    path1="/path/to/old.py",
    path2="/path/to/new.py",
    context_lines=2,
)
print(f"Similarity: {diff_result.similarity:.2f}")
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
from semantic_cache_mcp.storage.sqlite import SQLiteStorage
from semantic_cache_mcp.cache import SemanticCache

# Use a custom database path
storage = SQLiteStorage(db_path=Path("/tmp/my-cache.db"))
cache = SemanticCache(storage=storage)
```

---

## Embedding Model Management

```python
from semantic_cache_mcp.core.embeddings import configure, warmup, embed, embed_query, get_model_info

# Optional: set custom model directory before warmup
# Defaults to ~/.cache/semantic-cache-mcp/models when omitted
configure(cache_dir="/path/to/custom/models")

# Warmup model — called automatically at server start
warmup()

# Get model status
info = get_model_info()
print(f"Model: {info['model']}")
print(f"Ready: {info['ready']}")

# Generate embeddings
doc_vec   = embed("This is a document.")
query_vec = embed_query("search query here")
```

---

## Monitoring Cache Performance

```python
stats = cache.get_stats()
print(f"Files cached:       {stats['files_cached']}")
print(f"Total tokens:       {stats['total_tokens_cached']}")
print(f"Compression ratio:  {stats['compression_ratio']:.1%}")
print(f"Deduplication:      {stats['dedup_ratio']:.2f}×")

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

### SIMD Chunking

```python
from semantic_cache_mcp.core import get_optimal_chunker, hypercdc_chunks, hypercdc_simd_chunks

# Auto-select: SIMD if available, serial HyperCDC otherwise
chunker = get_optimal_chunker(prefer_simd=True)
chunks  = list(chunker(content.encode()))

# Force serial HyperCDC
chunks = list(hypercdc_chunks(content.encode()))

# Force SIMD (falls back to serial if unavailable)
chunks = list(hypercdc_simd_chunks(content.encode()))
```

### Semantic Summarization

```python
import numpy as np
from semantic_cache_mcp.core import summarize_semantic

def embed_fn(text: str) -> np.ndarray | None:
    vec = cache.get_embedding(text)
    return np.asarray(vec, dtype=np.float32) if vec else None

summary = summarize_semantic(
    content=large_file_content,
    max_size=10000,
    embed_fn=embed_fn,
)
```

### Batch Embedding

When embedding many files at once, amortize ONNX Runtime inference overhead with a single model call:

```python
from semantic_cache_mcp.core.embeddings import embed_batch

texts = [
    "python: def authenticate(user, password): ...",
    "typescript: async function fetchUser(id: string): ...",
    "rust: pub fn parse_config(path: &Path) -> Result<Config, Error>",
]

# Single ONNX Runtime call for N texts (much faster than N separate embed() calls)
vectors = embed_batch(texts)  # list[array.array[float] | None]
for text, vec in zip(texts, vectors):
    if vec is not None:
        print(f"Embedded {len(text)} chars → {len(vec)}-dim vector")
```

`batch_smart_read` calls this automatically for all new/changed files before the main read loop — you typically don't need to call it directly.

### LSH Approximate Search

The LSH index is built on-demand and **persisted to SQLite** automatically — no manual management required. After the first build, subsequent queries load instantly from the database.

```python
from semantic_cache_mcp.core import LSHIndex, LSHConfig

config = LSHConfig(num_bits=64, num_tables=4, band_size=8)
index  = LSHIndex(config=config)

# Add vectors
for item_id, vec in enumerate(embeddings):
    index.add(item_id, vec, store_embedding=True)

# Query — return_distances=True gives list[tuple[int, float]]
results = index.query(query_vec, k=5, return_distances=True)
for item_id, similarity in results:
    print(f"  id={item_id}  sim={similarity:.3f}")

# return_distances=False gives list[int]
item_ids = index.query(query_vec, k=10)
```

### Extreme Quantization

```python
from semantic_cache_mcp.core import (
    quantize_binary,
    quantize_ternary,
    quantize_hybrid,
    hamming_similarity_binary,
    evaluate_quantization_accuracy,
)

# Binary (32× compression, ~82% accuracy)
binary = quantize_binary(embedding)

# Ternary (16× compression, ~88% accuracy)
ternary = quantize_ternary(embedding)

# Hybrid: int8 base + binary index for two-stage search
hybrid = quantize_hybrid(embedding)

# Fast binary similarity (Hamming-based)
sim = hamming_similarity_binary(binary1, binary2)

# Evaluate accuracy vs exact cosine on a sample
report = evaluate_quantization_accuracy(embeddings, method="int8")
print(f"Mean error: {report['mean_error']:.4f}")
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

```python
# Store a file manually (pass pre-computed tokens to avoid redundant tokenization)
cache.put(
    path="/path/to/file.py",
    content="file content here",
    mtime=1234567890.0,
    embedding=cache.get_embedding("file content here"),
    tokens=42,
)

# Retrieve cached entry
entry = cache.get("/path/to/file.py")
if entry:
    content = cache.get_content(entry)

# Find semantically similar cached file
embedding = cache.get_embedding("search text")
similar_path = cache.find_similar(embedding, exclude_path="/current/file.py")

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
| `semantic_match`   | `str \| None` | Path of similar file if matched                |

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

### SearchResult / SimilarFilesResult

| Field          | Type                | Description                              |
|----------------|---------------------|------------------------------------------|
| `matches`      | `list[SearchMatch]` | Ranked matches with path + similarity    |
| `cached_files` | `int`               | Total files searched                     |

| Field           | Type               | Description                           |
|-----------------|--------------------|---------------------------------------|
| `source_path`   | `str`              | Source file                           |
| `source_tokens` | `int`              | Token count of source file            |
| `similar_files` | `list[SimilarFile]`| Ranked similar files                  |
| `files_searched`| `int`              | Total cached files searched           |

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
| `similarity`   | `float`          | Semantic similarity (0.0–1.0)       |
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

## Caching Strategy Reference

| Strategy             | Token Savings | Trigger Condition                              |
|----------------------|--------------|------------------------------------------------|
| Unchanged (mtime)    | ~99%         | File mtime matches cached entry                |
| Diff (changed)       | 80–95%       | File modified since last cache                 |
| Semantic match       | 70–90%       | Similar file found in cache                    |
| Summarized (large)   | 50–80%       | File exceeds `MAX_CONTENT_SIZE` limit          |
| Full (new/cold)      | 0%           | Not in cache; stored for future savings        |

---

[← Back to README](../README.md)
