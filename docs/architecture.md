# Architecture

## Package Structure

```
src/semantic_cache_mcp/
├── launcher.py             # Console entry point: relay to the shared daemon, else serve in-process
├── config.py               # Constants and environment-variable configuration
├── types.py                # All shared data models (ReadResult, WriteResult, etc.)
├── logger.py               # Logging bootstrap
├── utils/                  # Async file I/O (atomic write), retry, DetachedExecutor (abandonable IO thread)
├── cache/                  # Orchestration facade, coordinates all components
│   ├── __init__.py         # Public API re-exports
│   ├── store.py            # SemanticCache class: storage and metrics coordination
│   ├── read.py             # smart_read, batch_smart_read
│   ├── write.py            # smart_write, smart_edit, smart_batch_edit
│   ├── search.py           # semantic_search, glob_with_cache_status, compare_files
│   ├── metrics.py          # SessionMetrics: per-session and lifetime metric tracking
│   └── _helpers.py         # Internal utilities: _suppress_large_diff, formatter dispatch
├── server/                 # MCP interface, thin translation layer only
│   ├── __init__.py         # In-process entry point (main)
│   ├── __main__.py         # `python -m semantic_cache_mcp.server`: same entry as the console command
│   ├── _daemon.py          # Shared daemon: one lifespan, one session per connection
│   ├── _mcp.py             # FastMCP app instance, lifespan, tool worker startup
│   ├── _tool_worker.py     # ToolProcessSupervisor: tool-worker subprocess that owns the SemanticCache
│   ├── _coverage.py        # Signed coverage_token for ranged reads
│   ├── _paths.py           # Shared-root path shortening for responses
│   ├── _param_hints.py     # Middleware: parameter aliases, did-you-mean hints
│   ├── _single_representation.py  # Middleware: send each result once
│   ├── _tool_models.py     # Tool response models (output schemas)
│   ├── response.py         # Response formatting, TOOL_OUTPUT_MODE handling
│   └── tools/              # All 14 MCP tool definitions, _serialized lock, _shielded_write helper
├── core/                   # Pure algorithms; no I/O except the tokenizer's one-time BPE download
│   ├── __init__.py         # Flat re-exports from all sub-packages
│   ├── chunking/           # Content-defined chunking (used for large file splitting)
│   │   ├── __init__.py
│   │   ├── _gear.py        # Pure-Python HyperCDC (Gear hash rolling window)
│   │   └── _simd.py        # NumPy-vectorized CDC (the default chunker)
│   ├── hashing/            # BLAKE3/BLAKE2b content hashing and keyed MACs
│   │   ├── _blake.py       # hash_content, hash_chunk, keyed_hash
│   │   └── _wire.py        # Wire-form content hash and hash_matches
│   ├── text/               # Diff generation and structural outline
│   │   ├── __init__.py
│   │   ├── _diff.py        # generate_diff, diff_with_stats, rebase_diff_hunks
│   │   └── _outline.py     # extract_outline, render_outline, large_file_map
│   └── tokenizer/          # BPE token counting (o200k_base)
│       └── _bpe.py         # BPETokenizer, count_tokens, tokens_exceed
└── storage/                # Persistence layer
    ├── __init__.py
    ├── docstore/           # ContentStorage: vendored SQLite + FTS5 keyword store
    │   ├── __init__.py     # ContentStorage: put/get, chunking, versions, eviction
    │   ├── _docstore.py    # DocStore (SQLite schema, FTS5, versions, gram tables) + AsyncDocStore
    │   ├── _tinylfu.py     # TinyLFUIndex: in-memory eviction index
    │   ├── _search.py      # BM25 keyword search
    │   ├── _grep.py        # grep: candidate selection and regex scan
    │   └── _grams.py       # Required-literal extraction and trigram queries for grep
    └── sqlite.py           # SQLiteStorage: session metrics persistence only
```

## Design Principles

- **Separation of concerns.** `core/` is stateless pure algorithms, `storage/` is persistence only, `cache/` orchestrates, and `server/` translates between MCP and Python
- **One IO thread.** All storage access serializes through a single executor, so the one SQLite connection is never used concurrently
- **Facade pattern.** `cache/` exposes a clean API, and callers never touch `storage/` directly
- **Performance first in hot paths.** Hashing, chunking, and tokenization are optimized, and everything else favors clarity

---

## Storage (`storage/docstore/`)

### ContentStorage (vendored SQLite + FTS5)

The storage backend is a small SQLite store with FTS5, vendored into the package as `DocStore`. It holds text and metadata only:

- **FTS5 full-text search.** BM25 keyword ranking powers `search`, and `grep`'s fallback prefilter
- **Grep trigram index.** A contentless FTS5 `trigram` table (`grams`, keyed by `gram_files`) holds each file's whole text with positions, so a phrase query is an exact substring test across chunk boundaries. Each row records the content hash it was built from. `put` never touches it: `grep` scans every file whose row is missing or stale, then indexes the ones whose loaded text hashes to their entry. Rows are dropped with their path on delete, eviction or clear. Needs SQLite 3.43+; without it `grep` uses the BM25 prefilter
- **Raw text storage.** File contents are stored as plain text in the `text` column, with no compression
- **Indexed paths.** Each row's `path` sits in its own indexed column as well as in the metadata JSON, so a lookup by path never scans the table. A store created without the column gains and backfills it on open. Other metadata filters go through `json_extract`
- **Head rows.** The row with no `parent_id` (a chunked file's parent, or the only row of any other file) carries the fields that change on reads: `mtime` and `access_history`. `get`, `record_access` and `update_mtime` read or write that one row and never load chunk text
- **Past versions.** A `versions` table keeps the last 4 versions of each path (2,000 overall), zlib-compressed. A version is kept when a `put` replaces content of at most `MAX_CONTENT_SIZE` bytes whose hash checks out, and dropped with its path on delete, eviction or clear
- **Text-free projections.** `get_metadata`, `get_document_ids`, `distinct_paths`, `file_stats` and `count_files` read metadata without selecting the text column, so a caller inspecting one field does not re-materialize the cached corpus
- **Shutdown compaction.** Closing merges the FTS5 and grep-trigram indexes (discarding the delete markers a deletion leaves behind), returns the freed pages to the filesystem, and checkpoints the WAL with `wal_checkpoint(TRUNCATE)`. Stores run in incremental auto-vacuum mode; opening a populated store in another mode rewrites it once with `VACUUM`. The WAL is capped at 16 MB (`journal_size_limit`). See [performance.md](performance.md#cache-footprint-on-disk)

### Document Model

Files are stored as `Document` rows:

```
Small file (< 8KB):
  └── Single document: page_content=full_text

Large file (≥ 8KB):
  ├── Parent document: page_content="", is_parent=True
  └── Child documents (per CDC chunk):
      ├── page_content=chunk_text, chunk_index=0
      ├── page_content=chunk_text, chunk_index=1
      └── ...
```

Large files are split via HyperCDC (content-defined chunking) into multiple child documents. The parent holds file-level metadata and a `manifest` (the ordered chunk hashes); children hold raw text for content retrieval, search, and grep. A rewrite keeps the child rows whose `chunk_hash` is unchanged and replaces the rest. A file that would produce more than 500 chunks is stored as one document instead (`storage_mode` `single_doc_fallback`).

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
| `access_history` | `JSON` | Recent access timestamps, used by W-TinyLFU eviction. Head row only |
| `is_parent` | `bool` | Parent document marker (large files only) |
| `preview` | `str` | First 200 chars of file content, pre-stored at index time. `search` shows a 200-char window around the first query term and falls back to this when no term is found in the matched chunk |
| `created_at` | `float` | Time the entry was stored |
| `storage_mode` | `str` | `single_doc`, `chunked` or `single_doc_fallback` |
| `chunk_hash` | `str` | BLAKE3 hash of the chunk's bytes (child documents only) |
| `manifest` | `list[str]` | Ordered chunk hashes of the file (parent document only) |

### W-TinyLFU Eviction

When the number of cached files exceeds `MAX_CACHE_ENTRIES` (default 10,000), `put` evicts 10% of the files (at least one) with a TinyLFU-inspired policy (`_tinylfu.py`) that scores entries by both frequency and recency:

- Frequency comes from a 4-bit Count-Min sketch that ages by halving one row at a time, so a file read many times is kept even when it was not the most recent
- Recency comes from an LRU ordering. Eviction takes the oldest 2x-victim-count files from the LRU tail, ranks them by (frequency, last access) ascending, and removes the lowest. There is no admission window
- The in-memory index bootstraps lazily, the first time the count exceeds the cap, from each entry's `access_history` metadata (the last 5 accesses), so it survives a restart without a separate table
- This keeps a large one-time read (for example a wide grep seed) from pushing out the files you actually work on

### Session Metrics (`storage/sqlite.py`)

Separate SQLite database (`metrics.db`) for token savings, cache hits/misses, and tool call counts. `SessionMetrics` accumulates in memory and persists on shutdown. Not used for file content.

---

## Core Algorithms

### Chunking (`core/chunking/`)

Chunk sizes are 2KB minimum, ~8KB average (13 mask bits), 64KB maximum. Key property: similar files produce identical chunks even when bytes shift position, enabling efficient re-chunking on file changes.

#### NumPy-vectorized CDC (`_simd.py`)

The chunker `ContentStorage` uses (`get_optimal_chunker(prefer_simd=True)`). It runs in one thread; the speedup comes from NumPy vectorization, not parallel workers:

1. Map every byte through a 256-entry table of 64-bit fingerprints
2. XOR each 4-byte window into one fingerprint
3. Mark positions where `(fingerprint & mask) == 0` as boundary candidates
4. Walk the candidates sequentially, enforcing the 2KB minimum and 64KB maximum

#### Gear HyperCDC (`_gear.py`)

The fallback when NumPy is unavailable (`prefer_simd=False` also selects it). Content-defined chunking using a Gear hash rolling window:

- Pre-computed 256-entry gear table for O(1) byte lookups
- Rolling hash: `h = ((h << 1) + gear[byte]) & MASK_64`
- Boundary when `(h & mask) == 0`
- Skip-min: no boundary checks in first 2KB per chunk

The two chunkers place boundaries differently for the same input, so all callers go through `get_optimal_chunker`. `_gear.py` also holds `hypercdc_boundaries` (entropy-adaptive stepping, semantic boundary snapping) and `hierarchical_hypercdc_chunks`, which the store does not call.

---

### Hashing (`core/hashing/`)

BLAKE3 primary, BLAKE2b fallback. LRU-cached to avoid re-hashing identical data.
The caches are keyed on the buffer they hashed, so they retain it: 1,024 chunk
entries and 512 content entries, with content over 64 KiB hashed uncached, which
bounds retention at about 64 MiB and 32 MiB.

- **Content hash freshness.** Detects mtime changes with identical content (touch, git checkout)
- **Change detection.** Cached versus current content hash
- **Chunk identity.** `hash_chunk` gives each CDC chunk a stable id, which is what lets a re-write reinsert only the chunks that actually changed
- **Possession proofs.** `keyed_hash` signs the `coverage_token` a ranged read hands back, under a key generated per process
- **Wire form.** `short_hash` cuts a content hash to 16 hex characters for responses; `hash_matches` accepts that length or the full digest and nothing shorter

---

### Tokenizer (`core/tokenizer/`)

GPT-4o (o200k_base) BPE tokenizer; counts match tiktoken exactly for texts up to
256,000 characters. The vocabulary is downloaded once, SHA-256 verified and cached
under `TOKENIZER_CACHE_DIR`; the server loads it at startup. If it cannot be
loaded, `count_tokens` falls back to a word-count heuristic.

- o200k's pre-tokenizer ported to stdlib `re`, its `\p{..}` classes built from
  `unicodedata` over the BMP only, since `re` scans classes above U+FFFF range
  by range. Each character above U+FFFF is matched through a BMP stand-in of
  the same category; ASCII text uses ASCII classes. Both patterns are built
  when the tokenizer loads
- **O(N log M)** priority-queue merge; a pair ranks as its concatenation's token id
- Merge results (4,096 entries) and the last 8 counts are memoized; the count memo
  keys on the text's hash and length, so it holds no text
- Texts over 256,000 characters are estimated from 16 evenly spaced 4,000-character
  windows. Such files exceed the default `max_size` (100,000 characters), so the
  estimate only feeds metrics
- `tokens_exceed(text, limit)` counts exactly but stops at `limit`; the
  `max_response_tokens` checks use it

---

### Text (`core/text/`)

#### `_diff.py`: Unified Diffs

- Unified diffs via Python `difflib`, without the `---`/`+++` file headers. A last line with no line terminator is followed by `\ No newline at end of file`, as in `git diff`
- `diff_with_stats()`: computes the unified diff and its statistics (insertions, deletions, modifications) from a single line-matcher pass, shared by the write/edit/batch-edit and compare paths that need both. Its diff text matches `generate_diff`
- `rebase_diff_hunks()`: shifts a slice's hunk headers onto whole-file line numbers
- Adaptive context width: files under 100 lines use 2 lines of diff context instead of 3, where the extra line is a large share of a small payload; larger files keep 3
- `_suppress_large_diff()` (in `cache/_helpers.py`): suppresses a diff over 8,000 tokens or at least 90% of the file's token count, to prevent context overflow; files of 200 tokens or fewer always keep theirs. A middle tier keeps the per-hunk `@@` headers (which regions changed and by how much) when a diff is over budget but has few hunks, so the caller can fetch specifics with a ranged `read`; beyond a hunk cap it falls back to a bare count summary

#### `_outline.py`: Outline and Large-File Map

`extract_outline` scans line by line for definition forms and returns each with its 1-based line number; `render_outline` prints them as `<line>: <signature>`. It serves `read(outline=true)`.

`large_file_map` is what a full read of a file over `max_size` returns instead of the text:

1. The rendered outline, capped at 2000 entries and 4000 tokens (deepest entries dropped first, with an omitted count).
2. If no definitions are found: the first and last 20 lines under `// L<a>-<b>` anchors, each side clipped to 2000 characters.

The output is deterministic, delivers no file body, and every line number is a valid `offset`. The read reports `truncated`, `total_lines` and a `partial:` `file_hash`.

---

## Process Model

The console command (`launcher.py`) imports almost nothing. Unless `SCMCP_SHARED_SERVER` is off, it connects to the daemon, sends one JSON preamble line (`{"cwd": ..., "token": ...}`) and waits up to 10 s for the daemon's `{"scmcp": "ready"}` line. Only then does it copy stdin to the socket and the socket to stdout, until either side closes. A connection dropped before `ready` (a daemon shutting down closes the connections still in its backlog) is retried once, which starts a fresh daemon. When no daemon is listening it starts one (`python -m semantic_cache_mcp.server serve`, detached: a new session on POSIX, `DETACHED_PROCESS` on Windows) and waits up to 60 s for the endpoint, or until the new daemon exits with an error. Any failure before `ready` falls back to the in-process server.

The daemon (`server/_daemon.py`) holds an exclusive lock on its endpoint's lock file (`flock` on POSIX, `msvcrt.locking` on Windows), so a second daemon for the same endpoint exits after waiting 0.5 s for it. The wait covers a launcher probing the lock at the same instant. It enters the FastMCP lifespan once, so one `ToolProcessSupervisor`, tool worker and cache serve every connection. Each connection is read through the SDK's `stdio_server` adapter and run as its own `Server.run`, which reuses the reference-counted lifespan. The preamble's directory becomes `CONNECTION_CWD` for that connection's tasks. The client root is cached per MCP session (a `WeakKeyDictionary` keyed by the session), never per process; a client that names no root resolves relative paths against `CONNECTION_CWD`.

- **Key:** a hash of the version, the install location and every configuration variable (`PATH`, `HOME`, `USERPROFILE` and `SCMCP_*` included), so the transport is part of it too.
- **`unix` endpoint** (POSIX default): `<runtime dir>/semantic-cache-mcp-<uid>/<key>.sock`. The directory must be owned by the user with mode `0700`; the socket is `0600`. The preamble carries no token: the directory's permissions are the access check.
- **`tcp` endpoint** (Windows, or `SCMCP_SHARED_TRANSPORT=tcp`): a random `127.0.0.1` port. The daemon writes `{"port", "token"}` atomically to `<key>.port` (mode `0600`; on Windows under `%LOCALAPPDATA%\semantic-cache-mcp\run\`, whose ACL admits only the user). The token is 32 random bytes per daemon. A connection whose preamble token does not match (`hmac.compare_digest`) is closed before any MCP message is read, since any local user can reach a loopback port. The launcher trusts a port file only while the endpoint lock is held. A free lock means the daemon died, and the port may now belong to another process.
- **Idle exit:** after `SCMCP_IDLE_TIMEOUT` seconds with no connection (at least 10 s before the first one, so the launcher that started it can always connect), or on SIGTERM/SIGINT (POSIX only), the daemon stops accepting, removes its socket or port file and releases the lock, so a new daemon can start at once. Then it finishes the connections still open and closes the lifespan.
- **Tool worker:** the lifespan starts `ToolProcessSupervisor`, which spawns one `spawn`-context subprocess (`semantic-cache-tool-worker`). The worker loads the tokenizer, opens the `SemanticCache` and runs each tool call the server forwards over a pipe. A call that exceeds `TOOL_TIMEOUT` (default 30 s), errors, or breaks the protocol discards the worker; the next call starts a fresh one. Startup is allowed 30 s; closing sends `shutdown`, waits 8 s, then terminates and kills.
- **Concurrency:** `_serialized` and the single worker run one tool call at a time across all clients.

## Threading Model & Graceful Shutdown

### Thread Pools

The daemon and the tool worker each run a single asyncio event loop. The `SemanticCache`, and with it every executor below, lives in the tool worker. Blocking operations are offloaded to thread pools:

| Executor | Workers | Used for |
|----------|---------|----------|
| **IO executor** (`DetachedExecutor`) | 1 | All `ContentStorage` reads and writes, file I/O, and `large_file_map()`. One daemon thread, so the one SQLite connection is never touched concurrently and a wedged call cannot pin interpreter exit |
| **Default executor** | N (OS-dependent) | Other work, such as base64-encoding image reads |
| **Async subprocess** | n/a | `_format_file()` (ruff, prettier, etc.) |

After a tool timeout, `reset_executor()` abandons the stuck IO thread and rebinds storage to a fresh single-thread executor.

### Graceful Shutdown

On SIGTERM/SIGINT (POSIX) or idle timeout, the daemon:

1. Stops accepting, removes its endpoint, releases the lock, and waits for open connections to close (see Process Model)
2. Leaves the lifespan, whose `finally` calls `ToolProcessSupervisor.async_close()`: it sends `shutdown` to the worker, waits up to 8 seconds, then terminates and kills it
3. In the worker, the loop ends and `SemanticCache.async_close()` runs:
   - Sets `_shutting_down`, so new `begin_operation()` calls return `False`
   - Waits up to 8 seconds for in-flight operations to drain (`_drained` event), and catches `CancelledError` during the drain so close always proceeds
   - Persists session metrics, then closes ContentStorage, then the SQLite pool, then the IO executor
4. Write and edit tool handlers use `asyncio.shield()` via `_shielded_write()`, so the inner write completes even if the handler is cancelled. After a cancellation the write gets 2 more seconds before the handler gives up; the write itself keeps running
5. All `ContentStorage` async methods guard `_closed` and return safe defaults instead of crashing
6. `ContentStorage.close()` runs the final save on a daemon thread with a 5-second deadline, so a hung SQLite call cannot block exit

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
   ~0.2 ms latency)       ▼                    ▼
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
and the hash compute, plus an indexed lookup by path, keeps single-file unchanged
reads to **~0.2 ms** in-process.

A diff is returned only when the file is at least 200 tokens (`_DIFF_MIN_TOKENS`)
and the diff is under 90% of the file's tokens (`_DIFF_MAX_RATIO`); otherwise the
changed file is sent in full. The `unchanged` marker is used only when it costs
fewer tokens than the file, so a tiny file is sent in full from cache instead. A
full read of a file over `max_size` (default `MAX_CONTENT_SIZE`, 100,000
characters) returns the `large_file_map` instead of the text.

**The possession gate.** The diagram above is `smart_read`, which answers
from cache state alone. That answer is a claim about the *server*: the store is on
disk and outlives the process, the client session, and the caller's context window,
so it can never establish that the caller still holds a file. The tool layer
therefore re-gates it on evidence the caller supplies — `known_hash` on `read`, a
`known_hashes` entry on `batch_read`. Only a matching hash turns the fast path into
an `unchanged` reply; without one the caller gets the file, cheaply, from cache.
The same rule governs diffs, which are just as unusable without the base content.
A claim that names a kept past version of the path, rather than the current
entry, is the same proof for that version: the reply is a diff from those bytes,
verified against the claimed hash before use, or `unchanged` when the file is
back to exactly that version.

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
                                 is left alone when already fresh;
                                 a diff when the claim names a
                                 kept past version
         └──────┬─────────────────────┘
                │
         4. Return BatchReadResult with per-file status + content_hash
```

At most 50 files are read per call. Priority paths go first, then the rest smallest-first, against a shared token budget (`max_total_tokens`, default 50,000, capped at 200,000). A file that does not fit is `skipped` with an `est_tokens` cost, binary files are skipped, and a mapped (`truncated`) file carries no `content_hash`.

### Write / Edit

```
Client ──→ smart_write(path, content)
                │
      read existing file from disk, hash it
      (cached text is reused only if the hash matches the entry)
                │
         apply new content (append: old + new)
                │
        ┌───────┴──────────┐
        ▼                  ▼
   atomic write       refresh_path: store in ContentStorage
   to disk            (old text kept as a past version)
        │
   return diff (not full content)
```

Every mutation reads the file from disk and never trusts mtime, since a stale base would be written back over the real file. `smart_edit` and `smart_batch_edit` follow the same path with anchor edits: find/replace, find/replace scoped to a line range, or a line-range replacement. A batch applies up to 50 edits independently, and the ones that fail do not block the rest. The diff is suppressed under the same token rules as a read diff.

### Search (BM25)

```
query ──→ semantic_search(cache, query, k, directory)
              │
       ┌──────┴──────────────────────────┐
       ▼                                 ▼
 in-session result cache hit?      MISS, BM25 retrieve
 (LRU keyed on q,k,dir)                  │
   YES → return immediately       BM25 keyword search (FTS5 full-text)
   (< 0.01 ms vs 0.72 ms                 │
    for a cold search)             deduplicate by path
                                         │
                                  store in result LRU
                                         │
                                  top-k results
                                  (path, preview, relevance)
```

The query is cut to 8,000 characters and `k` is clamped to 100. The `directory` filter is applied in SQL as a path prefix before `LIMIT`. The store fetches `2k` rows, skips parent rows and keeps the best row per path. Relevance is the BM25 score divided by the best score (best = 1.0), and the preview is a 200-character window around the first query term.

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

`grep` pulls the literal runs every match must contain from the pattern
(`_grams.required_literals`; a pattern with `|?*{[`, or a letter or digit
escape other than `\b \B \d \D \s \S \w \W \A \Z`, has none) and asks the trigram index for the files holding all of them, plus every
file not yet indexed. Then it runs a compiled `re` pattern over those
candidates. A case-insensitive query uses only the ASCII runs of each literal,
and both index and query map dotless/dotted I, long s and the Kelvin sign to
their ASCII partner, the letters `re.IGNORECASE` equates with ASCII but SQLite's
fold does not. Without the index, or for a pattern it cannot narrow, a sound
BM25 prefilter (required tokens expanded to the FTS5 vocabulary terms containing
them) or a full scan takes its place. A pattern that cannot be used is an
error, not an empty answer, so a typo'd pattern is never mistaken for a genuine
miss: an invalid regex, one over `GREP_MAX_PATTERN_LEN` (1,000 characters), or one
with a repeated group holding an unbounded quantifier (`(a+)+`) raises
`ValueError` and the tool layer surfaces it. The length and quantifier checks
bound the ReDoS surface. `fixed_string=true` escapes the pattern instead, for
anything meant to match literally, and skips those checks.

---

[← Back to README](../README.md)
