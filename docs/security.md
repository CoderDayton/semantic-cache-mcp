# Security Considerations

## Threat Model

Single-user, local-only. Cached content lives in `~/.cache/semantic-cache-mcp/`. No multi-user or network-accessible deployment support.

**In scope:** local filesystem caching and SQLite on disk; the per-user shared daemon's local endpoint (see [Shared daemon](#shared-daemon)).
**Out of scope:** multi-tenant, network exposure, auth (defers to OS permissions).

**No path sandbox.** The server reads and writes whatever the user running it can, at any absolute path. A relative path resolves against the client's root. Treat the tools as having the same file access as the client that launches them.

---

## Security Controls

### File Access

**Path resolution.** Paths are normalized with `Path.resolve()` (`~` expanded) before the cache looks them up, so `../` segments and symlinks collapse to one canonical key. This is normalization, not confinement: an absolute path outside the client root is accepted. Relative paths join the client's first root, or the directory the client launched the command from. `grep` and `search` default to that root, but an explicit absolute path or directory reaches anywhere.

**Symlink handling.** Symlinks are followed and resolved to their target, which is intentional for developer workflows. Read and write log the resolution at DEBUG level. `glob` skips a symlink whose target lies outside the searched directory.

**Binary file detection.** Non-text files are rejected before any caching attempt, using layered checks on the first 8KB:
1. A Unicode BOM (UTF-8/16/32) marks the file as text and skips the rest
2. Null byte scan
3. Magic number signatures: PNG, JPEG, GIF, ZIP, GZIP, BZIP2, ELF, MZ/PE, PDF, OLE2
4. High-entropy heuristic: >30% non-printable characters

**File type gating.** Only regular files are accepted. Directories, devices, sockets, and other special files are rejected immediately.

### DoS Protection

Size limits prevent memory exhaustion from oversized inputs:

| Limit            | Default | Applies to                             |
|------------------|---------|----------------------------------------|
| `MAX_WRITE_SIZE` | 10M characters | `write` tool content             |
| `MAX_EDIT_SIZE`  | 10M characters | `edit` and `batch_edit` file size; a file over 40 MB on disk is refused by `stat` before it is read |
| `MAX_SEARCH_QUERY_LEN` | 8,000 chars | `search` query              |
| `MAX_BATCH_FILES` / `MAX_BATCH_EDITS` | 50 | files per `batch_read`, edits per `batch_edit` |
| `SCMCP_MAX_IMAGE_BYTES` | 5 MiB | `read_image` file size (see [env_variables.md](env_variables.md)) |
| `MAX_CONTENT_SIZE` | 100 KB | Default max bytes returned by `read`  |
| `MAX_MATCHES`    | 10,000  | `replace_all` match count in `edit`   |
| `GREP_MAX_PATTERN_LEN` | 1,000 chars | `grep` regex source, before compilation |
| `DOWNLOAD_TIMEOUT_S` | 30 s | Tokenizer bootstrap download |
| `DOWNLOAD_MAX_BYTES` | 32 MB | Tokenizer bootstrap download |

Limits are checked before the content is hashed, diffed or stored, so they fail fast.

**Search.** Results are capped at 100, and glob is capped at 1,000 matches with a 5-second timeout.

**Regex.** See [Regular expression safety](#regular-expression-safety) below.

### Regular expression safety

`grep` compiles a caller-supplied pattern, which makes catastrophic backtracking a denial-of-service surface. It is a real one, not a theoretical one: before 0.5.3, `(a+)+$` against 28 characters of input ran for **11.22 seconds** and 40 characters exceeded two minutes — both under a 1-second timeout budget that never fired, because the timeout could not interrupt a running match.

Two defences that sound plausible do not work here, and both were measured before being discarded:

- **A length cap is not sufficient.** The blow-up is driven by the length of the *subject*, not the pattern. `(a+)+$` is six characters.
- **Offloading to a thread does not help.** CPython holds the GIL for the duration of a `re` match, so a scan moved to the executor starves the event loop anyway — while also occupying the single thread the store serialises its I/O through.

So the pattern is rejected by shape, before `re.compile`, in about 0.01 ms: a repeatable group (`*`, `+`, `{n,}`, or `{n,m}` with m ≥ 2) enclosing an unbounded quantifier is refused, and the error names a safe equivalent. Character classes are parsed rather than scanned, so a quantifier inside `[...]` is not mistaken for a nested one.

This is a conservative shape check, not a proof of termination — it rejects the constructs that cause exponential backtracking in practice, and does not claim to catch every pathological pattern expressible in the language. Alongside it:

- `GREP_MAX_PATTERN_LEN` (1,000 chars) bounds the pattern source.
- An over-long, uncompilable, or rejected pattern raises an error rather than returning an empty result set, so a refusal can never be mistaken for a search that found nothing.
- `fixed_string=true` escapes the pattern and skips the regex engine entirely — the escape hatch for anything meant to match literally.

### SQL Injection

All SQL queries use parameterized statements. The only dynamic SQL construction is `IN` clauses where the placeholder string (`?,?,?`) is built from a count, never from user-supplied data. User values are always passed as bound parameters.

```python
# Safe: placeholder count from len(), values as parameters
placeholders = ",".join("?" * len(paths))
conn.execute(f"DELETE FROM files WHERE path IN ({placeholders})", paths)
```

### Input Validation

All inputs are validated before I/O:
- Empty string checks for `edit` operations (prevents accidental full-file deletion)
- `old_string == new_string` detection (no-op guard)
- Path existence and file-type checks before access
- Content type validation (binary detection)

### Data Storage

**Local only.** All data is stored in `~/.cache/semantic-cache-mcp/`. The server creates that directory with the process umask and does not restrict it to `700`; see [Recommendations](#for-users) to do so yourself.

**No network transmission.** The only outbound request is a one-time download on first use: the BPE tokenizer from `openaipublic.blob.core.windows.net` (~3.5MB). It is SHA256-verified, and a corrupted download is discarded. The request is streamed under a 30-second timeout and a 32 MB ceiling, and the partial file is removed on any failure. Before 0.5.3 it used `urlretrieve`, which accepts no timeout at all, while holding a lock during server startup — an endpoint that accepted the connection and then stalled hung the server indefinitely.

**Durable writes.** Every file write goes through an atomic temp-file-then-rename. Since 0.5.3 the temp file and its parent directory are both `fsync`ed before the rename, so a crash cannot land the rename ahead of the bytes it points at — which would otherwise leave a file that exists, is the wrong size, and that the cache believes it wrote correctly. This costs roughly 1.1 ms per write on ext4/NVMe.

**SQLite WAL mode.** Crash recovery, with no data corruption on abrupt termination. The WAL is capped at 16 MB and truncated on close.

**Possession proofs.** `unchanged`, diffs and claimable `content_hash` values are produced only for a caller that echoes a matching hash; this guards against a client being told it holds bytes it does not, not against a client reading a file. A ranged read's `coverage_token` carries a 64-bit MAC under a signing key generated per process and never persisted; a token from another process or a restarted worker is rejected and costs a re-read, never a false claim.

### Shared daemon

By default the `semantic-cache-mcp` command relays stdio to one per-user daemon, which any process able to reach its endpoint can talk to. The endpoint is therefore restricted to the owning user:

- **Unix socket (POSIX default).** The socket lives in `$XDG_RUNTIME_DIR/semantic-cache-mcp-<uid>/` (the system temp dir if unset), created `0700`; the socket is `0600`. If that directory already exists but is not a directory owned by the current user with no group/other permission bits, the launcher raises `PermissionError` and serves in-process rather than use it.
- **TCP loopback (Windows, or `SCMCP_SHARED_TRANSPORT=tcp`).** The daemon listens on `127.0.0.1` on a random port. Any local user can connect to a loopback port, so each connection must open with a preamble carrying a 256-bit random token. The token and port are written to a `0600` port file; the file is replaced atomically. The token is compared with `hmac.compare_digest`, and a connection without it is closed before any MCP message is read. A connection that sends no preamble within 10 s is dropped.
- **Stale endpoints.** The launcher trusts a port file only while the daemon's endpoint lock is held, so a file left by a crashed daemon is ignored.
- **No cross-configuration leakage.** The daemon is keyed by version, install path, `PATH`, `HOME`/`USERPROFILE`, the cache and log variables, and every `SCMCP_` variable, so clients with different settings never share a daemon. See [env_variables.md](env_variables.md#shared-server).
- **Opt out.** `SCMCP_SHARED_SERVER=0` runs one in-process server per client with no listener.

Clients of the same daemon share one cache, but each connection keeps its own MCP session and roots. On Windows the runtime directory is under `%LOCALAPPDATA%`, whose ACL admits only the user; the launcher does not check it.

### Local Processing

Keyword search (BM25) and outlining run entirely on the local machine. There is no model download and no inference service, and no file content is sent to any external API.

---

## Recommendations

### For Users

**Sensitive files.** Avoid caching files that contain secrets, credentials, or PII. Clear the cache when switching to or from sensitive projects:

```bash
# Via MCP tool
clear()

# Via filesystem (drops the cache, metrics, and tokenizer)
rm -rf ~/.cache/semantic-cache-mcp/
```

**Cache location permissions.** The cache directory is created with your umask (typically `755`), so other local users may be able to read it. Verify:

```bash
ls -la ~/.cache/semantic-cache-mcp/
```

It should be `drwx------` (700). If it is not, restrict it:

```bash
chmod 700 ~/.cache/semantic-cache-mcp/
```

**No encryption.** Cached content is stored unencrypted. Use filesystem-level encryption (for example macOS FileVault or Linux LUKS) if you cache sensitive projects.

### For Deployment

**Single-user only.** No authentication layer. Do not expose in multi-user environments.

**Container isolation.** Mount only the directories the cache needs, and avoid mounting `/` or sensitive paths.

**Audit logging.** File accesses are logged at DEBUG level (path plus token counts), to stderr and a daily file under the cache dir's `logs/` (or `LOG_DIR`). This is not designed as a security audit trail.

---

## Known Limitations

| Limitation                | Notes                                              |
|---------------------------|----------------------------------------------------|
| No encryption at rest     | Use filesystem-level encryption if needed          |
| No access control         | Relies entirely on OS filesystem permissions       |
| No path sandbox           | Any absolute path the user can access is readable and writable |
| Loopback TCP is reachable by local users | Mode `tcp` relies on the 256-bit token; Unix socket mode relies on the `0700` directory |
| No audit trail            | Logging is operational, not security-grade         |
| Single-user design        | Multi-tenant use is not supported                  |
| Network on first use only | Tokenizer download only, hash-verified, bounded by timeout and size |
| Regex guard is heuristic  | Rejects the shapes that backtrack catastrophically in practice, not a termination proof |

---

## Reporting Security Issues

If you discover a security vulnerability, please report it privately via [GitHub Security Advisories](https://github.com/CoderDayton/semantic-cache-mcp/security/advisories) or by email to coderdayton14@gmail.com.

Please do not open a public GitHub issue for security vulnerabilities.

---

[← Back to README](../README.md)
