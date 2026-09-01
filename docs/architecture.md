# Architecture

## Package Structure

```
src/semantic_cache_mcp/
├── config.py               # Constants and environment-variable configuration
├── types.py                # All shared data models (ReadResult, WriteResult, etc.)
├── cache/                  # Orchestration facade, coordinates all components
│   ├── __init__.py         # Public API re-exports
│   ├── store.py            # SemanticCache class: storage and metrics coordination
│   ├── read.py             # smart_read, batch_smart_read
│   ├── write.py            # smart_write, smart_edit, smart_batch_edit
│   ├── search.py           # semantic_search, glob_with_cache_status, compare_files
│   ├── metrics.py          # SessionMetrics: per-session and lifetime metric tracking
│   └── _helpers.py         # Internal utilities: _suppress_large_diff, formatter dispatch
├── server/                 # MCP interface, thin translation layer only
│   ├── __init__.py
│   ├── _mcp.py             # FastMCP app instance, lifespan, startup
│   ├── response.py         # Response formatting, TOOL_OUTPUT_MODE handling
│   └── tools/              # All 14 MCP tool definitions + _shielded_write helper
├── core/                   # Pure algorithms, stateless, zero I/O
│   ├── __init__.py         # Flat re-exports from all sub-packages
│   ├── chunking/           # Content-defined chunking (used for large file splitting)
│   │   ├── __init__.py
│   │   ├── _gear.py        # Serial HyperCDC (Gear hash rolling window)
│   │   └── _simd.py        # SIMD-accelerated parallel CDC
│   ├── hashing/            # BLAKE3/BLAKE2b content hashing and keyed MACs
│   ├── text/               # Diff generation and semantic summarization
│   │   ├── __init__.py
│   │   ├── _diff.py        # generate_diff, diff_with_stats, compute_delta, diff_stats
│   │   └── _summarize.py   # summarize_semantic (TCRA-LLM based)
│   └���─ tokenizer/          # BPE token counting (o200k_base)
└── storage/                # Persistence layer
    ├── __init__.py
    ├── docstore/           # ContentStorage: vendored SQLite + FTS5 keyword store
    └── sqlite.py           # SQLiteStorage: session metrics persistence only
```

## Design Principles

- **Separation of concerns.** `core/` is stateless pure algorithms, `storage/` is persistence only, `cache/` orchestrates, and `server/` translates between MCP and Python
- **Dependency injection.** Storage and config are passed explicitly, with no hidden globals
- **Facade pattern.** `cache/` exposes a clean API, and callers never touch `storage/` directly
- **Performance first in hot paths.** Hashing, chunking, and tokenization are optimized, and everything else favors clarity

---

## Storage (`storage/docstore/`)

### ContentStorage (vendored SQLite + FTS5)

The storage backend is a small SQLite store with FTS5, vendored into the package as `DocStore`. It holds text and metadata only:

- **FTS5 full-text search.** BM25 keyword ranking powers `search` and `grep`
- **Raw text storage.** File contents are stored as plain text in `page_content`, with no compression
- **Metadata filtering.** Path and chunk lookups go through JSON metadata columns
- **Text-free projections.** `get_metadata`, `get_document_ids`, `distinct_paths`, `file_stats` and `count_files` read metadata without selecting the text column, so a caller inspecting one field does not re-materialize the cached corpus
- **Shutdown compaction.** Closing merges the FTS5 index (discarding the delete markers a deletion leaves behind) and returns the freed pages to the filesystem. Stores are opened in incremental auto-vacuum mode; one created before 0.5.3 is rewritten once to enable it. See [performance.md](performance.md#cache-footprint-on-disk)

### Document Model

Files are stored as `Document` rows:

```
Small file (< 8KB):
  └── Single document: page_content=full_text

Large file (≥ 8KB):
  ├── Parent document: page_content="", is_parent=True
  └���─ Child documents (per CDC chunk):
      ├── page_content=chunk_text, chunk_index=0
      ├── page_content=chunk_text, chunk_index=1
      └── ...
```

Large files are split via HyperCDC (content-defined chunking) into multiple child documents. The parent holds file-level metadata; children hold raw text for content retrieval, search, and grep.

### Metadata

Each document carries metadata for cache management:

| Key | Type | Description |
|-----|------|-------------|
| `path` | `str` | Absolute file path |
| `content_hash` | `str` | BLAKE3 hex digest of full file content. Stored in full; delivered to callers as its first 16 hex characters, which is what a possession claim is checked against (see `core/hashing/_wire.py`). A claim is accepted at exactly that length or the full digest, never a shorter prefix. |
| `mtime` | `float` | File modification time |
| `tokens` | `int` | Token count (BPE o200k_base) |
| `chunk_index` | `int` | Chunk ordering (-1 for parent) |
| `total_chunks` | `int` | Number of chunks (1 for small files) |
| `access_history` | `JSON` | Recent access timestamps, used by W-TinyLFU eviction |
| `is_parent` | `bool` | Parent document marker (large files only) |
| `preview` | `str` | First ~200 chars of file content, pre-stored at index time so search results don't re-slice chunked `page_content` at query time |

### W-TinyLFU Eviction

When `MAX_CACHE_ENTRIES` is exceeded, eviction uses W-TinyLFU, the policy Caffeine uses, which scores entries by both frequency and recency:

- Frequency comes from a small 4-bit Count-Min sketch that ages over time, so a file read many times is kept even when it was not the most recent
- Recency keeps a freshly read file from being dropped before it has a chance to prove useful
- The in-memory index bootstraps from each entry's `access_history` metadata on first need, so it survives a restart without a separate table
- This keeps a large one-time read (for example a wide grep seed) from pushing out the files you actually work on

### Session Metrics (`storage/sqlite.py`)

Separate SQLite database for token savings, cache hits/misses, and tool call counts. Not used for file content.

---

## Core Algorithms

### Chunking (`core/chunking/`)

#### Serial HyperCDC (`_gear.py`)

Content-defined chunking using a Gear hash rolling window:

- Pre-computed 256-entry gear table for O(1) byte lookups
- Rolling hash: `h = ((h << 1) + gear[byte]) & MASK_64`
- Boundary when `(h & mask) == 0` (normalized chunking)
- Skip-min: no boundary checks in first 2KB per chunk
- ~8KB average chunk size

Key property: similar files produce identical chunks even when bytes shift position, enabling efficient re-chunking on file changes.

#### SIMD Parallel CDC (`_simd.py`)

Faster chunking via CPU-core-level parallelism:

1. Divide content into N segments (one per available core)
2. Each worker finds CDC boundaries in its segment independently
3. Merge and de-duplicate overlapping boundaries at segment edges

`get_optimal_chunker(prefer_simd=True)` auto-selects; falls back gracefully to serial HyperCDC.

---

### Hashing (`core/hashing/`)

BLAKE3 primary, BLAKE2b fallback. LRU-cached to avoid re-hashing identical data;
the caches are keyed on the buffer they hashed, so they are sized by retained
bytes rather than by entry count.

- **Content hash freshness.** Detects mtime changes with identical content (touch, git checkout)
- **Change detection.** Cached versus current content hash
- **Chunk identity.** `hash_chunk` gives each CDC chunk a stable id, which is what lets a re-write reinsert only the chunks that actually changed
- **Possession proofs.** `keyed_hash` signs the `coverage_token` a ranged read hands back, under a key generated per process

---

### Tokenizer (`core/tokenizer.py`)

GPT-4o compatible (o200k_base) BPE tokenizer.

- **O(N log M)** priority-queue merge vs O(N²) naive
- Merge results memoized to skip repeated BPE sequences
- Single-byte tokens skip BPE entirely (fast path)
- Files > 50KB use sampling for O(1) estimate

---

### Text (`core/text/`)

#### `_diff.py`: Diff and Delta Compression

- Unified diffs via Python `difflib` with diff statistics (insertions, deletions, modifications)
- `diff_with_stats()`: computes the unified diff and its statistics from a single line-matcher pass, shared by the write/edit/batch-edit and compare paths that need both (the output matches calling `generate_diff` and `diff_stats` separately)
- Adaptive context width: files under 100 lines use 2 lines of diff context instead of 3, where the extra line is a large share of a small payload; larger files keep 3
- Delta compression: store only changed lines (10 to 100x smaller for small edits to large files)
- `_suppress_large_diff()` (in `cache/_helpers.py`): caps diff output at a token budget to prevent context overflow. A middle tier keeps the per-hunk `@@` headers (which regions changed and by how much) when a diff is over budget but has few hunks, so the caller can fetch specifics with a ranged `read`; beyond a hunk cap it falls back to a bare count summary

#### `_summarize.py`: Semantic Summarization

Based on TCRA-LLM (arXiv:2310.15556). Preserves structural integrity when files exceed the size budget:

**Algorithm:**
1. Split file at semantic boundaries (function/class definitions, paragraphs)
2. Score each segment:
   - **Position score**: U-shaped curve, highest at the start and end, lowest in the middle
   - **Density score**: unique token ratio + syntax character density + non-whitespace ratio
3. Greedily select highest-scoring segments that fit the budget
4. Always preserve the first segment (docstrings, imports, module header)
5. Reassemble selected segments in original order; `# ... [N lines omitted] ...` markers
   are emitted only when `SummarizationConfig.include_markers=True` (default `False`
   since 0.4.6, because markers added no LLM-visible value and consumed token budget)

**Result:** 50 to 80% token savings on large files versus simple truncation, while preserving the code skeleton and intent.

---

## Threading Model & Graceful Shutdown

### Thread Pools

The server runs a single asyncio event loop. Blocking operations are offloaded to thread pools:

| Executor | Workers | Used for |
|----------|---------|----------|
| **IO executor** | 1 | All `ContentStorage` reads and writes. Single-threaded so the one SQLite connection is never touched concurrently |
| **Default executor** | N (OS-dependent) | `summarize_semantic()` and other CPU-bound work |
| **Async subprocess** | n/a | `_format_file()` (ruff, prettier, etc.) |

Storage operations run on a dedicated single-thread executor so the single SQLite connection is only ever used from one thread, which keeps writes serialized and safe.

### Graceful Shutdown

On SIGTERM/SIGINT:

1. `cache.request_shutdown()` sets the `_shutting_down` flag, and new `begin_operation()` calls return `False`
2. The signal handler cancels all asyncio tasks, so `CancelledError` propagates and runs `finally` blocks
3. Write and edit tool handlers use `asyncio.shield()` via `_shielded_write()`, so the inner task completes even if the outer handler is cancelled
4. Lifespan `finally` calls `async_close()`:
   - Waits up to 8 seconds for in-flight operations to drain (`_drained` event)
   - Catches `CancelledError` during drain so close always proceeds
   - Persists session metrics, then closes ContentStorage, then the SQLite pool, then the IO executor
5. All `ContentStorage` async methods guard `_closed` and return safe defaults instead of crashing
6. Second signal forces `os._exit()` for hard termination

---

## Data Flow

### Read

```
Client ──→ smart_read(path, diff_mode=True)
                │
        1. astat(path)          ◀─ cheap stat() only
        2. cache.get(path)
                │
        ┌───────┴────────────────┐
        │                        │
        ▼                        ▼
  cached + mtime match    cached but mtime drifted, OR not cached
  ── FAST PATH ──         ── SLOW PATH ──
  return "unchanged"      aread_bytes(path) ──→ hash + decode
  no aread_bytes call             │
  (99% savings,           ┌───────┴────────────┐
   ~1 ms latency)         ▼                    ▼
                  hash matches cached    hash changed
                  update_mtime           generate diff
                  return "unchanged"     refresh_path + return diff
                  (99% savings)          (80-95% savings)
                                                │
                                        not cached at all
                                          read full
                                          + return full
```

**Why the fast path matters:** the unchanged case is the most common in interactive
sessions (the LLM re-reads files it already has). Skipping `aread_bytes`, `count_tokens`,
and the hash compute keeps single-file unchanged reads to **~1 ms** (vs ~2 ms when
the disk read was unconditional in pre-0.4.6 builds).

**The possession gate (0.5.2).** The diagram above is `smart_read`, which answers
from cache state alone. That answer is a claim about the *server*: the store is on
disk and outlives the process, the client session, and the caller's context window,
so it can never establish that the caller still holds a file. The tool layer
therefore re-gates it on evidence the caller supplies — `known_hash` on `read`, a
`known_hashes` entry on `batch_read`. Only a matching hash turns the fast path into
an `unchanged` reply; without one the caller gets the file, cheaply, from cache.
The same rule governs diffs, which are just as unusable without the base content.

Mutations answer to the same rule from the other direction: a tool hands back a
claimable `content_hash` only when the caller could derive the result it just
asked for. A full `write` qualifies on its own — the caller supplied every byte.
`edit`, `batch_edit`, and `write append` need `known_hash` to match the
operation's `previous_hash`, because an anchor can come from `grep` and editing a
file is not the same as having read it. `auto_format` never qualifies: the
formatter's output is not what the caller asked for. Everything else is reported
as `file_hash`.

Ranged reads refine the same rule rather than bending it. Delivering one window
is no proof of the file, but it is proof of the window, so a ranged read returns
a `coverage_token` naming the lines it sent — signed with a per-process keyed
hash, so a token the server did not mint verifies as nothing and the bytes go
out. Coverage accumulates in the caller's token, never in server state: tracking
it here would miss a compaction between two windows and certify possession of
bytes the caller had already dropped, which is the failure this whole design
exists to prevent. Once the accumulated windows account for every line, the
caller has been shown the file and earns a real `content_hash`.

This is why there is no compaction detection anywhere in the server, and why
adding one would not help: MCP surfaces no compaction signal, and a client's
`/clear` or auto-compaction does not re-initialize the session. Rather than infer
what the caller has lost, the server declines to assume it has anything.

After context compression, simply omit the hashes — or use `diff_mode=False` on
the Python API — to force full content.

### Batch Read

```
Client ──→ batch_smart_read(paths, diff_mode=True, known_hashes={...})
                │
         1. Gather all cache.get() in parallel (asyncio.gather)
                │
         2. Pre-fetch stat results in parallel (asyncio.gather)
                │
         3. Per file: does the caller's claimed hash match the entry?
                │
         ┌──────┴─────────────────────┐
         ▼                            ▼
   claim matches               no claim / stale claim
   smart_read(diff_mode)       smart_read(diff_mode=False,
   → unchanged or diff           force_full=True,
                                 refresh_cache=False)
                               → literal bytes, and the index
                                 is left alone when already fresh
         └──────┬─────────────────────┘
                │
         4. Return BatchReadResult with per-file status + content_hash
```

### Write / Edit

```
Client ──→ smart_write(path, content)
                │
      read existing content (cache → disk)
                │
         apply new content
                │
        ┌───────┴──────────┐
        ▼                  ▼
   write to disk      update cache
        │                  │
        │             store in ContentStorage
        │
   return diff (not full content)
```

### Search (BM25)

```
query ──→ semantic_search(cache, query, k, directory)
              │
       ┌──────┴──────────────────────────┐
       ▼                                 ▼
 in-session result cache hit?      MISS, BM25 retrieve
 (LRU keyed on q,k,dir)                  │
   YES → return immediately       BM25 keyword search (FTS5 full-text)
   (< 0.01 ms, 2,000×+ faster            │
    than a cold search)            deduplicate by path
                                         │
                                  store in result LRU
                                         │
                                  top-k results
                                  (path, preview, score)
```

The in-session result LRU lives on `SemanticCache._search_cache` (32-entry
`OrderedDict`). It is invalidated on every cache mutation: `put`, `clear`,
`delete_path`, and `update_mtime` all call `_bump_search_cache()`, which
clears the LRU. So callers never see a result that predates a write.

**Query sanitisation (`_sanitize_fts_query`).** A free-text query is split on
whitespace and each run of word characters is quoted, so nothing the caller
types ever reaches the FTS5 parser as an operator; a token that splits into
several word runs (`in-flight`) becomes an adjacency phrase (`"in flight"`).
The quoted terms are then joined with `OR`, not FTS5's implicit `AND`. Under
`AND` a single word the corpus happens not to contain empties the entire result
set — which is exactly what a natural-language query invites, since no one file
carries every word of `"password hashing session token"`. `OR` keeps every
partial match alive and leaves the ordering to BM25, which already ranks a file
matching four terms above one matching a single term.

### Grep

`grep` runs the same corpus through a sound BM25 prefilter (required literal
tokens expanded to the FTS5 vocabulary terms containing them) and then a
compiled `re` pattern over the candidates. A pattern that cannot be used is an
error, not an empty answer: an invalid regex, or one over `GREP_MAX_PATTERN_LEN`
(1,000 characters, which bounds the ReDoS surface), raises `ValueError` and the
tool layer surfaces it. Returning `[]` made a typo'd pattern indistinguishable
from a genuine miss, which the caller then believed. `fixed_string=true` escapes
the pattern instead, for anything meant to match literally.

---

[← Back to README](../README.md)
