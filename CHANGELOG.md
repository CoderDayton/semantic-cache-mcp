# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.1] - 2026-07-07: Correctness and token-efficiency fixes

A focused follow-up to 0.5.0 from a full-repo audit. It fixes three data-safety
bugs in the write, edit, and summarize paths, tightens input validation and
error handling across the server and storage layers, and trims a little more
from diff payloads. There are no public API changes and no cache-format change,
so upgrading is a drop-in with no migration.

### Fixed

- **`batch_edit` no longer corrupts a file when two edits target overlapping
  line ranges.** Successful edits are applied back-to-front against a running
  copy of the file, so a later edit whose line range overlaps an earlier one
  read shifted offsets and could splice garbage or raise mid-batch after other
  edits had already been written. Overlapping ranges are now detected up front;
  the later edit fails with a clear "overlaps another edit in this batch" message
  and the file ends up exactly as the surviving edits alone would leave it.
- **`batch_edit` degrades gracefully on invalid UTF-8.** When a file with
  non-UTF-8 bytes had no cache entry, `batch_edit` raised where `write` and
  `edit` had long since learned to retry with replacement characters and log a
  warning. It now follows the same try-strict, fall-back-to-replace pattern in
  both of its disk-read branches.
- **`read` validates `offset`/`limit` before forwarding.** A negative `offset`
  or a `limit` below 1 is now rejected locally instead of being forwarded to the
  worker unchecked.
- **Summarization fallback covers whole paragraphs.** The blank-line paragraph
  splitter used when a file has no function or header boundaries could drop the
  overflow lines of any paragraph longer than the segment limit. This path isn't
  reached today — the boundary splitter always fires first — so this is a latent
  fix, but the fallback is now correct and covered by a test.
- **An invalid `LOG_LEVEL` no longer crashes startup.** An unrecognized value
  now falls back to `INFO` instead of raising during import, matching how the
  other environment settings tolerate bad input.
- **Closed a race in the eviction index.** A file re-write landing while the
  in-memory index was rebuilding itself from disk could merge stale document IDs
  into the fresh entry. The rebuild now skips any path that was written during
  the rebuild window.

### Changed

- **Leaner diffs.** Files under 100 lines now use 2 lines of surrounding context
  in a diff instead of 3, where the third line is a large share of a small
  payload; larger files keep the usual 3. When a diff is too large to send in
  full, the suppressed summary now includes the per-hunk `@@` headers (which
  regions changed, and by how much) up to a limit, so you can pull the specifics
  with a ranged `read` instead of re-reading the whole file.
- **One diff pass instead of two.** `write`, `edit`, `batch_edit`, and
  `compare` used to run the line-matcher twice over the same two texts — once for
  the diff, once for its statistics. They now share a single pass. The output is
  identical; large-file writes and edits are measurably faster (a 360 KB chunked
  write dropped from about 21 ms to about 8 ms).

### Internal

- Removed dead unreachable guards, narrowed a few over-broad `except` blocks,
  added an identifier allowlist on the one SQL column name that is interpolated
  rather than bound, added missing failure logging, and added direct
  (non-mocked) test coverage for the async file-I/O helpers.

## [0.5.0] - 2026-06-09: Biggest release yet, a near-complete rewrite

This is the biggest release so far. The embedding and vector search code is gone,
the third-party vector database is replaced by a small SQLite + FTS5 store we
vendor ourselves, the storage package is renamed, reads get a new hash check that
lets the cache skip work, all the tool descriptions are rewritten, and the MCP
framework is bumped. Nothing that saves tokens was lost: chunking, chunk
reassembly, the content cache, `grep`, `glob`, `diff`, and summarization all work
the same as before. The server uses about 400 MB less memory now that it never
loads an ONNX model, it starts faster, and it depends on only three packages at
runtime.

### Added

- Incremental chunk updates. Editing a large file now rewrites only the chunks
  that actually changed instead of re-chunking and re-storing the whole file.
  Each chunk carries its own BLAKE3 hash and the file keeps a manifest of those
  hashes, so on a re-write the cache keeps every chunk whose bytes are unchanged
  — no row rewrite, no search re-index — and writes only the few that differ. A
  one-line edit to a file that splits into 43 chunks now touches about 2 of them.
  It stays crash-safe by ordering: the file's `content_hash` is written last, so
  a write that fails partway is caught by the next read's freshness check and
  re-stored, and the eviction index rebuilds itself from disk on failure. The
  token-savings benchmark now reports chunk economics — how many files chunk,
  how much chunk content repeats, and the share of per-edit chunk writes this
  avoids — so the win is measured, not assumed.
- Hash-driven read freshness. Every `read` returns a `content_hash`, and `read`
  takes an optional `known_hash`. Send back the hash you already have and the
  server answers `unchanged: true` instead of resending the file. The caller
  knows that hash for sure, so there is no guessing about what was sent earlier
  in the session. `write`, `edit`, and `batch_edit` return the new `content_hash`
  too, so right after changing a file you can pass it as `known_hash` and skip
  the re-read. Ranged reads with `offset`/`limit` answer `unchanged` from a stat
  alone when the hash matches, and when they do need the lines they cut them from
  the cached copy instead of re-reading the whole file from disk. The stats count
  only the lines a ranged read returns, not the whole file.

### Changed

- `search` is BM25 keyword only. It ranks cached files by how well their words
  match the query and returns a score from 0 to 1, where the best match is 1.0.
  Punctuation in a query is treated as plain text, so a term like `in-flight` or
  a stray `*` still matches instead of coming back empty. It matches on words,
  not meaning, so use `grep` for exact strings and `batch_read` to pull more
  files into the cache.
- A small SQLite + FTS5 store replaces `simplevecdb`. A focused `DocStore` and
  `AsyncDocStore` now back storage, using FTS5 `bm25()` ranking and JSON metadata
  filters copied straight from the old catalog code. There is no embedding
  column, no stub vector, no usearch index, and no crash-recovery sidecar files,
  since SQLite WAL handles crash safety on its own.
- The storage package was renamed. `storage/vector` is now `storage/docstore`,
  `VectorStorage` is now `ContentStorage`, `VECDB_PATH` is now `CONTENT_DB_PATH`,
  and the cache file `vecdb.db` is now `docstore.db`.
- Diffs do more of the work now. The `read` diff gate went from 0.6 to 0.9, with
  a floor at 200 tokens, so a small edit to a medium or large file comes back as
  a diff with the changed line numbers instead of the whole file. Tiny files
  still come back in full. The diff itself is leaner too: it drops the
  `--- old`/`+++ new` file headers and the prose prefix and keeps just the `@@`
  hunks, which already carry the line numbers.
- All 13 tool descriptions were rewritten so they read as one workflow: `glob` to
  find files, `batch_read` to cache them, `search` or `grep` to look inside,
  `read` to open, then `edit` or `write` to change. They share the same wording
  for errors and statuses, and they describe what the tools actually do,
  including the BM25 fix for `search`.
- `fastmcp` was upgraded to 3.2 or newer (3.4.2). Parameter docs now show up as
  real per-argument descriptions instead of one long blob.

### Removed

- Embedding and vector search. Deleted `core/embeddings` (FastEmbed/ONNX, the
  OpenAI-compatible provider, and the HuggingFace model registry) and
  `core/similarity` (cosine). Vector similarity (`find_similar`, `search_hybrid`,
  per-file embeddings) is gone, and `diff` no longer reports a similarity score.
- Dependencies. Dropped `fastembed`, `openai`, the gpu extra (`fastembed-gpu` and
  `onnxruntime`), and now `simplevecdb`, `usearch`, and `sqlcipher3-binary` too.
  The runtime now needs only `blake3`, `fastmcp`, and `numpy`.
- Config. Removed `EMBEDDING_DEVICE`, `EMBEDDING_MODEL`,
  `OPENAI_EMBEDDINGS_ENABLED`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`,
  `OPENAI_EMBEDDING_MODEL`, and `OPENAI_EMBEDDING_DIMENSIONS`.
- Stats. The embedding block (`model`, `provider`, `ready`) is gone from the
  `stats` payload. Process RSS is still reported.

### Migration

- The first time you start after upgrading, the cache runs a one-time cleanup
  that deletes the old `vecdb.db` files (simplevecdb plus usearch, and the
  short-lived FTS build) and their sidecars, guarded by a `.docstore_v1` marker.
  The cache rebuilds itself on demand into `docstore.db`.
- Upgrading to chunk-level content addressing clears any existing `docstore.db`
  the first time you start, so the cache repopulates in the new chunk format.
  This runs once, guarded by a `.docstore_manifest_v1` marker.

## [0.4.9] - 2026-05-30

Fixes a correctness bug in line-addressed reads that made fresh-but-summarized
output look like a stale cache, plus internal hardening, a vector-storage
refactor, and a round of hot-path performance work. No public API changes.

### Fixed

- **`read` with `offset`/`limit` no longer summarizes large files.** For files
  over `MAX_CONTENT_SIZE` (100 KB default), ranged reads sliced over
  *semantically summarized* content, so `lines.total` reported the summarized
  line count and the emitted line numbers did not map to disk. Callers saw
  `read` and `grep` disagree (e.g. `total: 2322` vs a real line 5352) and
  mistook the fresh-but-summarized result for a stale cache. `smart_read` now
  takes a `summarize` flag (default `True`); the offset/limit path passes
  `summarize=False` to slice literal disk lines with real line numbers and a
  true total. Side benefit: ranged reads of large files skip the embed/
  summarize step entirely.
- **`read` offset past EOF returns a coherent empty window.** An out-of-range
  `offset` (or an empty file) previously reported `lines.start > lines.end`;
  it now reports `start == end == total`.

### Changed

- **Remote-forwarding tools now forward their full parameter set automatically.**
  In supervisor/remote mode, each forwarding tool (`read`, `grep`, `search`,
  `batch_read`, …) previously hand-listed the kwargs it relayed to the remote
  peer, so a newly added parameter could be silently dropped. A new
  `_forward_kwargs` helper derives the forwarded set from the *calling tool's*
  own signature — every parameter except `ctx`, including keyword-only ones —
  and fails loudly on `*args`/`**kwargs` tools or unknown overrides. Guarded by
  `tests/test_remote_forward.py`.
- **Vector storage split into focused modules.** The monolithic
  `storage/vector/__init__.py` (−442 lines) is now a thin package surface over
  new `_grep.py` (pattern/vocab/phonetic grep) and `_search.py` (semantic and
  hybrid search) modules. Pure refactor — same public symbols and behavior.
- **Response-contract guard.** `tests/test_response_contract.py` asserts every
  key a tool emits is declared in its response model, failing loudly if a tool
  ever returns an undeclared key.

### Performance

- **`_is_binary_content` non-printable scan** now uses a single
  `bytes.translate` C pass instead of a per-byte Python comprehension.
- **`_extract_line_range`** computes char offsets in two non-overlapping passes
  with O(1) extra memory, dropping the redundant prefix sum.
- **`summarize_semantic`** fills a pre-allocated row buffer in place instead of
  re-`np.stack`-ing the whole selection on every accept (was O(k²)); the
  `_simple_embedding` fallback uses `np.bincount` over a single index array.
- **`cosine_similarity` matrix build** fast-paths homogeneous `array.array("f")`
  inputs by concatenating into one contiguous f32 buffer, skipping the per-row
  Python assignment loop (typecode-guarded).
- **`compute_delta`** sizes its estimate via `itertools.chain` instead of
  building a temporary concatenated list.

## [0.4.8] - 2026-05-24

`read_image` hardening: guard the on-the-wire payload against Anthropic's
~5 MB upload cap, and move base64 encoding off the event loop.

### Added

- **`SCMCP_MAX_ENCODED_IMAGE_BYTES`** (default 5,000,000) — wire-side cap on
  the base64-encoded image payload. The existing raw cap of 5 MiB expands to
  ~6.99 MB on the wire, which upstream rejects with an opaque 400. The
  encoded-size guard catches this pre-encode and surfaces a clear tool-level
  error naming the env var. Validated against the actual `base64.b64encode`
  length for every residue class mod 3.

### Changed

- **`read_image` base64 runs off the event loop** — encoding moves to the
  default `ThreadPoolExecutor` under `asyncio.wait_for(_TOOL_TIMEOUT)`, so a
  multi-MB encode no longer blocks every other coroutine and a runaway
  buffer can't hang the tool indefinitely.

## [0.4.7] - 2026-05-21

DX & feedback-loop hardening based on a 24h behavioral audit of production
traffic. Closes the most common wasted-call shapes (silent grep empties,
unactionable `unchanged:true`, opaque edit timeouts, alias confusion) and
adds the `edit_preview` probe.

### Added

- **`edit_preview` tool** — Read-only probe returning `{found, match_count,
  line_numbers, context}` for a given `old_string` against a file. Lets
  callers verify an anchor is unique before committing to a 30s `edit`.
  Response budget ≈ 200 tokens.
- **`read_image` tool** — Pass-through for image files. Returns an MCP
  image content block (base64 + mime) alongside a JSON metadata sidecar,
  so vision-capable models see the actual pixels. Format is verified by
  magic bytes, not by file extension: PNG, JPEG, GIF, TIFF, BMP, and
  WebP are accepted regardless of filename, and a mis-named file (text
  saved as `.png`) is refused. Bypasses the semantic cache (no
  embedding/description). Capped at 5 MiB; override via
  `SCMCP_MAX_IMAGE_BYTES`. Use `read` for non-image files.
- **Per-phase timing in edit timeouts** — `edit` and `batch_edit` now thread
  a `_PhaseTimer` through `smart_edit` (input_validation, binary_check,
  cache_lookup, anchor_search, diff_gen, atomic_write, format_subprocess,
  cache_refresh). Timeout errors name the phase that was running and report
  elapsed seconds.
- **Fuzzy edit-miss hints** — When `old_string` doesn't match, the
  ValueError now appends up to 3 nearest-line suggestions (via
  `difflib.SequenceMatcher`). Skipped on files over 5000 lines.
- **Grep cache-miss reason** — `grep` with a `path=` that has no cached
  files under it now returns `reason: "no_files_cached_under_path"` and a
  `hint` pointing at `batch_read`/`glob`, instead of returning `[]`
  silently.
- **Structured binary file responses** — Reading a binary file no longer
  raises. The read tool returns `{ok: true, is_binary: true, size, mime}`
  so callers can branch without parsing error strings. Mime is sniffed
  from extension + a small magic-byte table.
- **Did-you-mean for unknown parameters** — A new FastMCP middleware
  silently rewrites common aliases (`abs_path`/`paths`/`file` → `path`,
  `query`/`q` → `pattern`) and replaces unknown-param `-32602` errors with
  a clean ToolError plus a `difflib` close-match suggestion.
- **Per-session unchanged tracking** — `read` now consults a process-wide
  LRU keyed by `(session_id, abs_path)`. The first read in a session
  always sends full content; subsequent reads return `unchanged: true`
  with `content_hash` and `total_lines` so the model can decide locally
  whether a ranged re-read is warranted. Mutations (`write`, `edit`,
  `batch_edit`, `delete`) invalidate the entry; `clear` resets the
  tracker.

### Changed

- **`read.offset=0` accepted** — Previously rejected with
  "offset must be >= 1"; now treated as from-start (equivalent to
  omitting). Negative offsets still rejected.
- **Formatter timeout default 10s → 15s** — Configurable via the
  `SCMCP_FORMAT_TIMEOUT_S` environment variable.
- **`edit`/`batch_edit` descriptions** — `edit` now leads with the
  recommendation to use `batch_edit` for multiple changes on the same
  file. `batch_edit` description drops the "for one change, prefer edit"
  softener that contradicted the audit signal (270 single edits vs 35
  batch in production).
- **`search` description rewritten** — Repositions semantic search as the
  first move for concept-level queries ("where is rate limiting handled")
  rather than a grep alternative, after an audit found the tool was never
  called. Drops the failure-first "empty results usually mean..." framing.
- **`write` description** — Adds a behavior block (overwrite vs. `append`,
  `created`/`updated` status, diff-on-update) so the tool's return shape is
  documented alongside `edit`/`batch_edit`, instead of jumping straight from
  summary to arguments.

### Removed

- **`similar` tool** — Removed end to end: the MCP tool, the
  `find_similar_files()` function, the `SimilarFilesResult`/`SimilarFile`
  and `SimilarResponse` types, and `MAX_SIMILAR_K`. The tool went unused
  in production — agents always reached for `grep` or `search`. The
  vector index it shared with `search` and `read`'s diff-against-similar
  path is unaffected.
- **`diff` tool** — Removed the MCP tool for explicit two-file comparison.
  Agents reach for `git diff` instead, and `read` already returns a unified
  diff for "what changed since I last read this file". The `compare_files()`
  core function is retained as a library API.

### Fixed

- **TinyLFU bootstrap race** — A `remove()` landing while
  `TinyLFUIndex.ensure_loaded()` awaited its loader could not see the
  half-built index, so a path deleted mid-bootstrap was resurrected by a
  loader snapshot taken before the delete committed. Such removals are
  now recorded and replayed onto the rebuilt index.
- **`read_image` size recheck** — The size limit is re-checked against
  the bytes actually read, closing a race where a file growing (or a
  swapped symlink target) between the `stat` and the read could exceed
  `SCMCP_MAX_IMAGE_BYTES`.
- **`edit_preview` error mapping** — A non-regular-file or unreadable
  target now surfaces as a clean `ToolError` instead of leaking an
  internal `-32603`, matching `read`/`read_image`.
- **Defensive `access_history` parsing** — A corrupt or non-list
  `access_history` value in DB metadata no longer crashes a cache-hit
  read; non-numeric entries are dropped, matching `TinyLFUIndex`.
- **Stale mtime persisted after writes** — `write`, `edit`, and
  `batch_edit` refreshed the cache with the pre-write mtime captured for
  the freshness check, so the next read saw cache-mtime < disk-mtime and
  needlessly re-read and re-hashed the file. The cache now stores the
  post-write mtime.
- **First read could deliver a bare marker** — On the first read of a
  session, a file already warm in the cache returned the
  `// File unchanged` marker instead of real content, and truncated reads
  were marked fully "seen" — so a follow-up read collapsed to
  `unchanged:true` for a file the model never received in full. The first
  read now re-fetches real content, and a file is marked seen only when
  the complete file was sent.

## [0.4.6] - 2026-05-06

### Changed

- **simplevecdb 2.6.0** — Bumped minimum dependency from 2.5.0. Inherits upstream review-pass-3 fixes: hybrid-search RRF rank symmetry under metadata filters, RRF deduplication keyed by document ID instead of text (no more silent merge of distinct docs sharing text), per-connection lock on every catalog read path (`get_documents_by_ids`, `keyword_search`, `count`, …) closing a known sqlite3 thread-safety gap, atomic `UsearchIndex.save` via sibling `.tmp` + `os.replace` + parent-dir fsync, atomic `delete_collection` with a tightened TOCTOU window, NaN/Inf rejection at insert before the catalog row commits, and softened INT8 quantization range checks (clip + one-shot `DeprecationWarning` instead of `ValueError`) so embeddings drifting marginally outside [-1, 1] no longer crash inserts.

### Fixed

- **`batch_read` no longer stalls the event loop** — `SemanticCache.get_embeddings_batch` was a sync method that ran ONNX inference on the calling thread. For `batch_smart_read` that thread was the asyncio event loop, freezing every concurrent MCP call for the duration of the batch embed and bypassing the dedicated single-thread ONNX executor (which can segfault under concurrent inference). The method is now async and dispatches through `cache._io_executor`. Programmatic callers must add `await`.
- **Write timeouts no longer pin the shutdown drain** — `_shielded_write` previously skipped `end_operation()` whenever `asyncio.timeout` fired, leaking the inflight counter forever because the shielded task kept running in the background. After the first write timeout, every subsequent shutdown blocked the full 8-second drain window for nothing. `end_operation()` is now wired as a `Task.add_done_callback`, so it fires exactly once when the inner task actually finishes — success, error, cancellation, or post-timeout completion.
- **`glob` no longer blocks the event loop** — `glob_with_cache_status` walked the filesystem with `Path.glob()` directly on the loop. On NFS, FUSE, or large repos that walk could stall every concurrent MCP call for seconds. The walk now runs on the IO executor with the existing deadline guard applied inside the worker.
- **Eviction no longer scans the full collection on every write** — `_evict_if_needed` (called on every `put`) now short-circuits via a cheap `count()` check before doing the LRU-K scan. Drops the per-write O(N) scan that touched 50K+ rows on chunked-file workloads.
- **Pre-fetched stats reused in `batch_read`** — the pre-scan loop in `batch_smart_read` now reuses the `_stat_map` collected via the prefetch gather instead of issuing a second sync `stat()` / `is_file()` per file on the event loop.
- **`save()` ↔ `close()` race window closed** — `VectorStorage.save()` and the `close()` daemon thread now share a `threading.Lock`, eliminating the narrow race where eviction-driven save and the final close save could call usearch's not-thread-safe save concurrently.
- **`_format_file` bounded after SIGKILL** — the post-kill `proc.wait()` now has a 2-second timeout so a wedged formatter child cannot hang the call indefinitely.
- **Per-event-loop tool lock** — `_tool_lock` rebinds when the running event loop changes, removing a stale-lock failure mode under pytest-asyncio function-scoped loops.
- **GPU VRAM leak with `EMBEDDING_DEVICE=cpu`** — When `onnxruntime-gpu` is installed but `EMBEDDING_DEVICE=cpu`, fastembed no longer auto-selects CUDA. The ONNX session now receives an explicit `providers=["CPUExecutionProvider"]`, preventing ~2GB of phantom VRAM allocation.
- **Guard fastembed init when OpenAI provider is active** — `_get_model()` now raises immediately if called with `OPENAI_EMBEDDINGS_ENABLED=true`, making it impossible to accidentally load the local ONNX model when embeddings are routed through Ollama/OpenAI.
- **CUDA fallback preserves CPU constraint** — When CUDA initialization fails at runtime, the retry path now explicitly sets `CPUExecutionProvider` instead of removing the `providers` kwarg (which let ONNX Runtime auto-select CUDA again).

## [0.4.5] - 2026-04-28

### Added

- **OpenAI-compatible embeddings** — Added an opt-in remote embedding provider controlled by `OPENAI_EMBEDDINGS_ENABLED`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_EMBEDDING_MODEL`. Local FastEmbed remains the default path, while the OpenAI-compatible path defaults to Ollama at `http://localhost:11434/v1`.

### Changed

- **Inferred remote embedding dimensions** — `OPENAI_EMBEDDING_DIMENSIONS` is now optional. When unset, semantic-cache infers and records the vector dimension from the first successful remote embedding; when set, the value is sent as the provider `dimensions` parameter and validated against the response.
- **Embedding provider docs** — README and environment-variable docs now include Ollama/OpenAI-compatible setup examples, including `ollama pull nomic-embed-text` for the default local remote-provider path.

## [0.4.4] - 2026-04-20

### Fixed

- **macOS process-exit hang** — `DetachedExecutor.shutdown(wait=False, cancel_futures=True)` no longer risks pinning interpreter shutdown on macOS when a worker is stuck. The executor now runs its worker on a truly detached low-level thread while preserving `wait=True` semantics via an internal stop event.

## [0.4.3] - 2026-04-20

### Added

- **Explicit verbosity toggles** — `write`, `edit`, and `batch_edit` now accept `show_diff`, and `search` now accepts `show_preview`, so large payloads are opt-in when they materially affect the next decision.

### Changed

- **Lean default mutation responses** — Clean deterministic `write`, `edit`, and `batch_edit` results no longer return full diffs by default. They now expose machine-readable `diff_state` metadata and reserve full diffs for partial applies, debug mode, or explicit requests.
- **Compressed batch/search/glob/stats payloads** — `batch_read` now returns `unchanged_count` by default instead of full unchanged path lists, skipped-file guidance moved to a summary hint, `search` omits previews by default, `glob` omits per-match `tokens`/`mtime` outside debug, and `stats` text output is shorter while preserving structured data.
- **Estimated token impact** — In representative local simulations, the new defaults cut response size by about **67.2%** across compact-mode edit/read/batch cases, **53.7%** across the normal-mode `search`/`glob` cases, and **59.9%** across the combined sample workload.

### Fixed

- **Warm-cache read safety** — `read` no longer drops the `content` field on unchanged cache hits, so a first read in a new client session still receives a body even when the persistent cache is already warm.
- **Diff contract stability** — Truncated responses now preserve diff metadata instead of silently dropping it.
- **Diff state accuracy** — Unchanged writes no longer misreport `diff_omitted=true`, and diff-bearing tools now distinguish `full`, `unchanged`, and `omitted` states consistently.

## [0.4.2] - 2026-04-10

### Changed

- **simplevecdb 2.5.0** — Bumped minimum dependency to pick up the new
  `delete_collection`, `store_embeddings`, and pagination APIs along with
  fixes to delete ordering, FTS retries, and connection health probes.
- **`store_embeddings=True`** — VectorStorage now opts into SQLite-side
  embedding storage. simplevecdb 2.5.0 changed the default to `False` to save
  ~2× storage; without opting in, `get_embeddings_by_ids` would return `None`
  and break embedding-aware similarity reuse in `SemanticCache.get()`.
- **Atomic collection reset** — `clear()` and `clear_if_model_changed()` now
  call `delete_collection()`, which drops the SQLite tables, FTS index, and
  usearch file in one call, replacing the previous per-id loop and manual
  file unlinks. The new helper `_reset_collection_sync()` handles the
  startup-path (no event loop) variant.
- **Sync VectorDB + manual async wrapper** — Replaced `AsyncVectorDB` with a
  direct sync `VectorDB` plus a manually-built `AsyncVectorCollection`
  wrapper. `AsyncVectorDB.collection()` does not expose `store_embeddings`
  in 2.5.0 (no kwargs forwarding, no setter), so we need the sync collection
  factory anyway. Going through the public sync `VectorDB` deletes every
  remaining `simplevecdb` private-attribute access from the project: no more
  `_db._db`, `_db._executor`, or `_collection._collection` reach-throughs.
  A new `VectorStorage.rebind_executor()` method gives `SemanticCache.reset_executor`
  a public seam to swap the IO executor after a hung worker.

## [0.4.1] - 2026-04-02

### Changed

- **Automatic cache behavior** — Removed `diff_mode` parameter from `read` and `batch_read`. The server now automatically detects whether a file is new, unchanged, or modified and returns the optimal response (full content, `"unchanged":true` marker, or unified diff). No configuration needed.

### Fixed

- **Embedding dimension mismatch guard** — `_resolve_embedding` validates vector dimensions before passing to usearch, raising `ValueError` instead of segfaulting on model change mid-session.
- **Runtime dimension check** — `clear_if_model_changed` now verifies the live index dimension matches the model, catching stale indexes even when the sidecar metadata is missing.
- **Save race condition** — `save()` skips if `close()` is already running on the daemon thread, preventing concurrent usearch saves that caused heap corruption.
- **Oversized file truncation** — Files producing >500 CDC chunks now fall back to single-doc storage instead of silently truncating content.
- **ReDoS mitigation** — Grep rejects regex patterns longer than 1,000 characters.
- **Stats crash on missing DB** — `get_stats()` handles deleted database files gracefully.

## [0.4.0] - 2026-03-30

### Added

- **`delete` tool** — Added a narrow cache-aware delete operation for one file or one symlink path, with `dry_run` support and immediate cache eviction.
- **Path-filtered `grep`** — Exact cached-content search can now be scoped to one file, suffix, or glob path filter to reduce noise and token spend.

### Changed

- **LLM tool routing prompts** — Rewrote tool docstrings and README guidance so models choose the right cache-first tool more reliably and recover cleanly from empty or unchanged results.
- **Relative path resolution** — Tool paths now resolve against the client project root instead of the server process cwd.
- **FastMCP 3.1 alignment** — Normalized tool outputs and remote dispatch behavior to match current FastMCP response handling.

### Fixed

- **Tool hangs under concurrent access** — Blocking file I/O, SQLite catalog work, and all ONNX inference paths are isolated from the event loop and serialized safely, eliminating the GPU-spin / no-response hang class under load.
- **Timeout recovery** — Added a supervised tool worker that drops and restarts wedged executors after tool timeouts or worker protocol failures without stretching the caller's timeout budget.
- **Embedding dimension detection** — Removed the hardcoded 384-dimension fallback so non-default embedding models no longer corrupt vector storage shape.
- **Stats consistency** — Internal stats counters now stay coherent across clears, rewrites, and cache refreshes.

### Performance

- **Cache hit ratio** — `read` and `batch_read` now block `diff_mode=false` for unchanged cached full-file reads so callers reuse the cached version instead of forcing redundant disk I/O.
- **Embedding reuse** — Small edits reuse cached embeddings when possible, and `similar` avoids recomputing source embeddings for fresh cached files.
- **Freshness checks** — `diff` now uses the same mtime-plus-content-hash freshness logic as read/write paths, avoiding cache misses on touch-only changes.
- **Adaptive refresh timeout** — Cache refreshes now choose a timeout based on remaining work, reducing unnecessary executor resets after slow but healthy write/edit refreshes.
- **Lower startup churn** — Removed the embedding keepalive task and unnecessary cache rewrites during worker initialization.

## [0.3.4] - 2026-03-15

### Fixed

- **Event loop blocking** — ONNX embedding inference, SQLite catalog operations, and subprocess formatter calls were running synchronously on the asyncio event loop, causing the server to hang under load. All blocking calls now run via `asyncio.to_thread()`.
- **Graceful shutdown** — SIGTERM/SIGINT handlers cancel all tasks so lifespan cleanup runs. Write/edit operations are shielded from `CancelledError` via `asyncio.shield()` to prevent file corruption. `async_close()` drains in-flight operations (8s timeout) before closing storage.
- **Use-after-close crashes** — All VectorStorage async methods now guard against closed state, returning safe defaults instead of crashing during shutdown.
- **Embedding dimension mismatch** — `_resolve_embedding` now queries the actual model dimension instead of hardcoding 384, preventing `Vector dimension 384 != index dimension N` errors with non-default models (e.g. `Snowflake/snowflake-arctic-embed-m-v2.0`).
- **`_format_file` blocking** — Replaced `subprocess.run()` with `asyncio.create_subprocess_exec()` so auto-formatting no longer freezes the server.
- **`_expand_globs` unbounded** — Added 5-second deadline to prevent recursive `**` glob patterns from blocking indefinitely.
- **Connection pool timeout** — Reduced SQLite pool wait from 10s to 5s to surface exhaustion faster.

### Performance

- **Dedicated embedding executor** — ONNX calls use a single-thread `ThreadPoolExecutor` so concurrent embeddings don't starve the default thread pool (used by storage I/O).
- **Parallel cache lookups** — `batch_smart_read` gathers all `cache.get()` calls via `asyncio.gather()` instead of N serial awaits, and reuses results in the pre-scan loop (eliminates ~N redundant SQLite queries per batch).
- **No double-fetch on diff path** — `smart_read` saves the cache entry before the sentinel-null and restores it for diff generation (eliminates 1 SQLite query per changed-file read).
- **Embedding reuse** — `find_similar_files` reuses `cached.embedding` when available instead of calling ONNX (saves 20–100ms per cached file).

## [0.3.3] - 2026-03-10

### Fixed

- **Eviction miscounting** — LRU-K eviction counted documents instead of files, under-evicting at cache capacity.
- **Semantic boundary snapping** — Zero-distance sentinel allowed worse candidates to overwrite perfect matches.
- **`HierarchicalHasher.finalize_content`** — Always returned empty chunk list due to clearing before copy.
- **SQLite connection leak** — Migration helper leaked connection on query exception.
- **Duplicate log handlers** — Module re-import added redundant stderr handlers.
- **Batch edit crash** — Non-UTF-8 files caused unhandled `UnicodeDecodeError`.
- **Shutdown hang** — Graceful shutdown could block indefinitely on client disconnect.
- **Input validation** — Hardened storage layer against missing/malformed inputs.
- **`close()` blocking** — Cache close could hang when background save was stuck.

### Changed

- Stripped padding, repetition, and template prose across all `.py` and `.md` (net −1,350 lines).

### Removed

- Dead code: `_myers_diff`, `_unified_diff_fast`, `generate_diff_streaming`, `invert_diff`, `apply_delta`, `_fit_content_to_max_size`, `save_session`, `_zero_embedding`, stale singleton re-exports.

### Performance

- `estimate_min_tokens` returns cached token counts instead of re-reading full files.
- `find_similar_files` no longer double-computes embeddings for uncached files.
- `grep` skips fetching context lines in compact mode.

## [0.3.2] - 2026-03-08

### Added

- **Custom embedding model support** — Set `EMBEDDING_MODEL` to any HuggingFace model with an ONNX export. Models not in fastembed's built-in list are automatically downloaded and registered from HuggingFace Hub on first startup.
- **SHA256 verification** — Downloaded ONNX model files are verified against HuggingFace-reported hashes to prevent tampering.
- **Clear error messages** — Specific errors for models without ONNX exports and for network failures when downloading custom models.

## [0.3.1] - 2026-03-08

### Changed

- **Removed explicit `onnxruntime` dependency** — `fastembed` now owns the ONNX Runtime dependency. Users with `fastembed-gpu` get `onnxruntime-gpu` automatically instead of being forced to CPU.

### Added

- **`[gpu]` optional extra** — Install with `semantic-cache-mcp[gpu]` to get NVIDIA GPU acceleration via `fastembed-gpu`.
- **`gpu` alias for `EMBEDDING_DEVICE`** — `EMBEDDING_DEVICE=gpu` now accepted as an alias for `cuda`.
- **Startup warning on missing CUDA** — When `EMBEDDING_DEVICE=gpu/cuda` but `CUDAExecutionProvider` is unavailable, logs a warning with install instructions before falling back to CPU.

## [0.3.0] - 2026-03-08 — Storage Rewrite

Complete storage backend rewrite from compressed chunks (SQLiteStorage) to raw text + vector embeddings (VectorStorage via simplevecdb). Simpler data path, better search, same caching semantics.

### Changed

- **Storage backend: SQLiteStorage → VectorStorage** — Files stored as plain text with HNSW embedding vectors. Eliminates compression/decompression overhead.
- **Small files** (< 8KB) stored as a single document; large files split via HyperCDC into content-defined chunks, each with its own embedding.
- **Thread safety** — `threading.RLock` on all public VectorStorage methods for safe concurrent access.
- **Dependencies** — Replaced `fastembed-gpu` (broken Rust rewrite) with `fastembed`. Removed `onnxruntime-gpu` (fastembed handles provider selection).
- **Stats tool** — Now returns token savings, hit/miss ratio, DB size, and session uptime in a flat JSON structure.
- **Search scores** — Normalized to 0–1 range (best result = 1.0) instead of raw RRF scores.

### Added

- **Content hash freshness** — BLAKE3 hash comparison when mtime changes but content is identical (touch, git checkout, editor re-save). Returns "unchanged" instead of re-reading. Applied across all 7 freshness check locations.
- **Truncation hints** — `read`/`batch_read` responses include `hint` with offset to continue reading.
- **Configurable embedding model** — `EMBEDDING_MODEL` env var (default: `BAAI/bge-small-en-v1.5`).
- **`grep` tool** — Regex/literal pattern search across cached files with line numbers and context.
- **`docs/env_variables.md`** — Full reference for all configurable env vars.
- **Auto-migration** — Detects and removes legacy v0.2.0 `cache.db` on first startup.

### Fixed

- **Stale cache** — `touch`, `git checkout`, editor re-saves no longer invalidate cache when content is identical.
- **`find_similar_files` returning 0 results** — Always computes embedding via `cache.get_embedding()` instead of relying on VectorStorage.get().
- **`stats` key mismatch** — Fixed `total_files` → `files_cached` in 3 locations.

### Removed

- Compressed chunk storage (ZSTD/LZ4/Brotli layer)
- File locking (`filelock`) — replaced by in-process `threading.RLock`
- Dead code: `_backtrack()` in `_diff.py`

## [0.2.0] - 2026-03-02

### Added

- **Cross-process file locking** — `filelock` serializes database access across concurrent MCP instances (e.g. Cursor + Claude Desktop sharing the same cache). Lock timeout produces a clear `RuntimeError` instead of cryptic SQLite crashes.
- **Atomic file writes** — All `write`/`edit`/`batch_edit` operations use temp-file + rename to prevent data loss on crash or signal interruption.
- **Thread-safe connection pool** — `threading.Lock` around pool counter prevents connection overflow under concurrent access.
- **Thread-safe tokenizer init** — Double-checked locking prevents duplicate downloads when multiple threads call `get_tokenizer()` simultaneously. Download is now atomic (temp file + rename).
- **Thread-safe ZSTD compressor cache** — Double-checked locking on lazy compressor/decompressor initialization.

### Fixed

- **Directory filter bypass** — `search(directory=...)` used `startswith()` which matched `/project_evil` when filtering for `/project`. Now uses `Path.is_relative_to()`.
- **Special files passed to formatter** — `_format_file` now rejects char devices, pipes, and `/proc` entries via `stat.S_ISREG` before spawning subprocess.
- **Startup crash on init failure** — `UnboundLocalError` when `SemanticCache()` or `warmup()` raised during lifespan. `cache` is now initialized to `None` with proper guards.
- **Negative offset/limit silently wrapping** — `read` tool now validates `offset >= 1` and `limit >= 1`; `max_size` clamped to prevent unbounded reads.
- **`executescript` breaking transactions** — `clear()` used `executescript` which auto-commits, defeating the connection pool's transaction management. Replaced with separate `execute()` calls.
- **O(N) eviction loading all metadata** — Eviction now uses `ORDER BY json_extract(...) LIMIT ?` in SQL instead of loading all rows + JSON parsing in Python.
- **LRU cache memory bloat** — Content hash cache now bypasses `@lru_cache` for files > 64KB, bounding worst-case retention to ~128MB instead of ~20GB.
- **`k=0` / `k<0` passing search guards** — `min(k, MAX)` now wrapped with `max(1, ...)` for both `search` and `similar`.
- **`compare_files` crash on missing/binary files** — Now validates file existence and catches `UnicodeDecodeError` with clean error messages.
- **`assert` used for control flow** — Three `assert` statements in `write.py` replaced with `TypeError` raises (assertions are stripped by `-O`).
- **Symlink traversal in glob** — `glob_with_cache_status` now skips symlinks that resolve outside the base directory.
- **`SEMANTIC_CACHE_DIR` env var not resolved** — Now calls `.expanduser().resolve()` on the override path.
- **Operator precedence ambiguity** — Added explicit parentheses in `_summarize.py` for `or`/`and` expression.
- **Redundant `ORDER BY` in `find_similar`** — Removed wasted sort; similarity search already ranks results.
- **Double chunking pass in `put()`** — Removed chunk counting loop that duplicated work done by storage layer.

### Performance

- **Vectorized hamming distance** — `hamming_distance_batch` now uses `np.unpackbits` on uint8 view of XOR results instead of Python-level popcount loops. Scalar `hamming_distance` uses Kernighan's bit-counting algorithm.
- **Vectorized SimHash bit packing** — `compute_simhash` replaces Python loop with `np.uint64` power-of-two dot product. `compute_simhash_batch` uses pre-allocated matrix instead of `np.vstack`.
- **O(N) top-K selection** — `np.argpartition` replaces `np.argsort` in similarity ranking (2 call sites), reducing top-K from O(N log N) to O(N).
- **O(N) pruning threshold** — `np.partition` replaces `np.percentile` for dimension pruning cutoff in cosine similarity.
- **Native binary quantization** — `np.packbits`/`np.unpackbits` replaces Python bit-manipulation loops in `quantize_binary`/`dequantize_binary`.
- **Buffer protocol blob deserialization** — Single `b"".join()` + `np.frombuffer` reshape replaces per-row `struct.unpack` loop in batch cosine similarity.
- **Pre-allocated matrices** — `np.empty` + fill replaces `np.vstack` with list comprehension in 3 hot paths (LSH batch, cosine batch ×2).

### Changed

- **Stdout redirect uses `contextlib.redirect_stdout`** — Replaces manual `sys.stdout` swap for thread-safety and re-entrancy.
- **Explicit stderr logging handler** — `logging.StreamHandler(sys.stderr)` instead of `basicConfig()` to guard against third-party reconfiguration.
- Type annotations tightened: `dict[str, Any]` → `dict[str, bool | int]` in `get_hash_stats`, `-> list` → `-> list[float]` in `cosine_similarity_batch`, `params: list` → `params: list[str]` in `find_similar`.
- README: added `uvx` vs `uv tool install` explanation, cross-platform cache paths, `SEMANTIC_CACHE_DIR` env var.

## [0.1.1] - 2026-02-21

### Fixed

- **macOS/Windows installation** — `fastembed-gpu` and `onnxruntime-gpu` (Linux-only wheels) replaced with platform-conditional dependencies. CPU variants install on macOS/Windows; GPU variants remain on Linux.
- **Cross-platform cache directory** — respects `$SEMANTIC_CACHE_DIR` env override, then uses platform-appropriate defaults: `$XDG_CACHE_HOME` on Linux, `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows.
- **Cross-platform RSS memory stats** — `/proc/self/status` replaced with platform-aware helper: `resource.getrusage` on macOS, `K32GetProcessMemoryInfo` on Windows, graceful `None` on unsupported platforms.
- **UTF-16/32 files falsely detected as binary** — BOM-aware check (UTF-32 LE/BE, UTF-16 LE/BE, UTF-8 BOM) now runs before the null-byte heuristic.
- **Inline binary checks consolidated** — `read.py` now uses the shared `_is_binary_content()` helper instead of duplicating null-byte checks.

### Changed

- Installation docs updated to use `uvx` instead of `uv tool install`.
- CI: action versions bumped (checkout v6, setup-uv v7, codecov v5, upload-artifact v6), macOS added to test matrix.

## [0.1.0] - 2026-02-21

### Added

- Initial release
- Session metrics: per-session and lifetime tracking of tokens saved, cache hits/misses, files read/written/edited, diffs served, and tool call counts. Persisted to SQLite on shutdown and aggregated across sessions via the `stats` tool.
- 11 MCP tools: `read`, `write`, `edit`, `batch_edit`, `search`, `similar`, `glob`, `batch_read`, `diff`, `stats`, `clear`
- Smart file reading with diff-mode — unchanged files cost ~5 tokens, modified files return unified diffs (80–95% savings)
- Semantic similarity search via local ONNX embeddings (BAAI/bge-small-en-v1.5, no API keys)
- Persistent LSH index for O(1) similarity lookups; serialized to SQLite, survives restarts
- Batch embedding — all new/changed files in a `batch_read` are embedded in a single model call
- Line-range editing for `edit` and `batch_edit` — scoped find/replace and direct line replacement
- int8 quantized embedding storage (388 bytes/vector, 22x smaller than float32)
- SIMD-parallel content-defined chunking (~70–95 MB/s), BLAKE3 hashing, ZSTD compression
- LRU-K eviction with 10,000-entry default; DoS limits on write size, match count, and glob scope
- `diff_mode=false` on `batch_read` for full content recovery after LLM context compression
- `append=true` on `write` for chunked large file writes
- `cached_only=true` on `glob` to filter to already-cached files

[Unreleased]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.3.3...HEAD
[0.3.3]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/CoderDayton/semantic-cache-mcp/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/CoderDayton/semantic-cache-mcp/releases/tag/v0.1.0
