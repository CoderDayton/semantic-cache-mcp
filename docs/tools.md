# Tools

Every tool runs through one cache. Each response that delivers a file carries its `content_hash`; pass it back as `known_hash` (or a `known_hashes` entry on `batch_read`) to prove you still hold the file.

| Tool | What it does |
|------|--------------|
| `read` | Cache-aware read: full file, `unchanged`, or a diff. `offset`/`limit` for line ranges, `outline=true` for a map of a large file. |
| `batch_read` | Many files at once under a token budget, with globs and priorities. |
| `read_image` | Sends an image to vision models as an MCP image block. |
| `write` | Create or replace a file, with `dry_run`, `append` and formatters. |
| `edit` | Exact find/replace, scoped to a line range, or a plain line-range replace. |
| `batch_edit` | Many edits to one file, applied atomically. |
| `edit_preview` | Shows where an `old_string` matches before you edit. |
| `delete` | Deletes one file or symlink and drops it from the cache. No globs, no recursion. |
| `warm` | Indexes files for `grep` and `search`, and returns counts only. |
| `grep` | Regex or literal search over cached files. |
| `search` | BM25 keyword ranking of cached files. |
| `glob` | Finds files and shows which are cached. |
| `stats` | Tokens saved, cache size, and session metrics. |
| `clear` | Empties the cache. |

## read

```
read path="/src/app.py"                        # full, unchanged, or diff
read path="/src/app.py" known_hash="847f..."   # prove you hold it
read path="/src/app.py" offset=120 limit=80    # lines 120 to 199 only
read path="/src/app.py" outline=true           # one line per definition
```

| State | Response | Token cost |
|-------|----------|------------|
| First read | Full content plus a `content_hash` | Normal |
| Unchanged | `unchanged: true`, when you pass back a matching `known_hash` | A few tokens |
| Modified | A unified diff, when you pass back the `known_hash` of the version you hold | 5 to 20% of the file |
| Over `max_size` | An outline (or the first and last 20 lines), flagged `truncated` with `total_lines` | A fraction of the file |

- Files under 200 tokens, and diffs at least 90% of the file's size, come back in full.
- A partial or mapped read reports `file_hash` with a `partial:` prefix. It names the file but never counts as proof you hold it.
- A ranged read returns a `coverage_token` for the lines sent. Pass it back as `known_hash`: a window you hold answers `unchanged`, and windows that add up to the whole file earn a full `content_hash`.
- `line_numbers=true` adds a line-number gutter to a ranged read. It costs about 17% more tokens; the range is always in `lines`.

## read_image

Returns an MCP image content block (base64 + mime). The format is checked by magic bytes (PNG, JPEG, GIF, TIFF, BMP, WebP), not the extension. Bypasses the cache. Capped at 5 MiB raw (`SCMCP_MAX_IMAGE_BYTES`) and 5,000,000 bytes encoded (`SCMCP_MAX_ENCODED_IMAGE_BYTES`).

## write

```
write path="/src/new.py" content="..."
write path="/src/new.py" content="..." auto_format=true
write path="/src/large.py" content="...chunk1..." append=false   # first chunk
write path="/src/large.py" content="...chunk2..." append=true    # later chunks
```

Returns `created` or `updated`, and the diff only with `show_diff=true`. Creates missing parent directories with `create_parents=true`. A full write returns a usable `content_hash`; an append needs `known_hash` to earn one. With `dry_run=true` the status is `would_create` or `would_update` and nothing is written.

## edit

```
# Mode A: find/replace across the file
edit path="/src/app.py" old_string="def foo():" new_string="def foo(x: int):"
edit path="/src/app.py" old_string="..." new_string="..." replace_all=true auto_format=true

# Mode B: find/replace inside a line range (a shorter old_string works)
edit path="/src/app.py" old_string="pass" new_string="return x" start_line=42 end_line=42

# Mode C: replace the whole range, no old_string needed (fewest tokens)
edit path="/src/app.py" new_string="    return result\n" start_line=80 end_line=83
```

| Mode | Parameters | Best for |
|------|-----------|----------|
| Find/replace | `old_string` + `new_string` | Unique strings, no line numbers known |
| Scoped | `old_string` + `new_string` + `start_line`/`end_line` | Shorter anchors when `read` gave you line numbers |
| Line replace | `new_string` + `start_line`/`end_line` | Fewest tokens when line numbers are known |

Pass `known_hash` to get a usable `content_hash` back and skip the read afterwards. `dry_run=true` reports `would_edit` and writes nothing.

## batch_edit

```
# Mode A: [old, new]
batch_edit path="/src/app.py" edits='[["old1","new1"],["old2","new2"]]'

# Mode B: [old, new, start_line, end_line]
batch_edit path="/src/app.py" edits='[["pass","return x",42,42]]'

# Mode C: [null, new, start_line, end_line]
batch_edit path="/src/app.py" edits='[[null,"    return result\n",80,83]]'

# Mixed modes (object syntax also works)
batch_edit path="/src/app.py" edits='[
  ["old1", "new1"],
  {"old": "pass", "new": "return x", "start_line": 42, "end_line": 42},
  {"old": null, "new": "    return result\n", "start_line": 80, "end_line": 83}
]' auto_format=true
```

All edits apply together or not at all, with a result per edit. An ambiguous anchor, an anchor inside another edit's range, and overlapping ranges are rejected, never guessed. Takes `known_hash` on the same terms as `edit`.

## batch_read

```
batch_read paths="/src/a.py,/src/b.py" max_total_tokens=50000
batch_read paths='["/src/a.py","/src/b.py"]' priority="/src/main.py"
batch_read paths="/src/*.py" max_total_tokens=30000
batch_read paths="/src/a.py,/src/b.py" known_hashes='{"/src/a.py":"8f3c..."}'
```

Expands simple globs, reads `priority` files first, stays under `max_total_tokens`, and lists skipped files with how to get them. Files you prove you hold through `known_hashes` collapse into an `unchanged` count.

## warm, grep, search, glob

```
warm paths="src/**/*.py"                        # index a tree, returns counts only
search query="authentication middleware" k=5 show_preview=true
grep pattern="class Cache" path="src/**/*.py"
grep pattern="content_hash" output="paths"
grep pattern="TODO" output="count"
glob pattern="**/*.py" directory="./src" cached_only=true
```

The usual first move in an unfamiliar repo: `warm`, then `grep` for an exact string or `search` for a concept, then `read` only the files those name.

- `warm` takes up to `max_files` per call (default 200, maximum 2,000) and reports every skipped file with a reason (`not_found`, `not_a_file`, `binary`, `too_large`, `unreadable`, `timeout`).
- `grep` returns hits as `"<line>:<text>"` strings grouped by file, context lines as `"<line>-<text>"`, with paths relative to the `root` it names. An invalid, over-long or catastrophically backtracking pattern is an error, never an empty result. Use `fixed_string=true` for literal text.
- `search` ranks with BM25 over the current project, or over `directory` when given. Terms join with `OR`. Previews are opt-in (`show_preview=true`) and centre on the matching term.
- Repeated searches hit an in-session LRU in under 0.01 ms. Every cache write invalidates it.
- Every response says whether the scan finished, so a capped result never reads as a total.
