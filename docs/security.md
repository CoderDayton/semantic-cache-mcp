# Security Considerations

## Threat Model

Single-user, local-only. Cached content lives in `~/.cache/semantic-cache-mcp/`. No multi-user or network-accessible deployment support.

**In scope:** local filesystem caching and SQLite on disk.
**Out of scope:** multi-tenant, network exposure, auth (defers to OS permissions).

---

## Security Controls

### File Access

**Path resolution.** All paths are resolved via `Path.resolve()` before any I/O, which prevents directory traversal through `../` sequences or symlink tricks that escape the intended base directory.

**Symlink handling.** Symlinks are followed and resolved to their target, which is intentional for developer workflows. Symlink resolution is logged at DEBUG level.

**Binary file detection.** Non-text files are rejected before any caching attempt, using several layered checks:
1. Null byte scan in the first 8KB
2. Magic number signatures: PNG, JPEG, GIF, ZIP, GZIP, ELF, MZ/PE, PDF, OLE2
3. High-entropy heuristic: >30% non-printable characters

**File type gating.** Only regular files are accepted. Directories, devices, sockets, and other special files are rejected immediately.

### DoS Protection

Size limits prevent memory exhaustion from oversized inputs:

| Limit            | Default | Applies to                             |
|------------------|---------|----------------------------------------|
| `MAX_WRITE_SIZE` | 10 MB   | `write` tool content                   |
| `MAX_EDIT_SIZE`  | 10 MB   | `edit` and `batch_edit` file size      |
| `MAX_CONTENT_SIZE` | 100 KB | Default max bytes returned by `read`  |
| `MAX_MATCHES`    | 10,000  | `replace_all` match count in `edit`   |
| `GREP_MAX_PATTERN_LEN` | 1,000 chars | `grep` regex source, before compilation |
| `DOWNLOAD_TIMEOUT_S` | 30 s | Tokenizer bootstrap download |
| `DOWNLOAD_MAX_BYTES` | 32 MB | Tokenizer bootstrap download |

All limits are enforced **before** any I/O operation, so they fail fast.

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

**Local only.** All data is stored in `~/.cache/semantic-cache-mcp/` with `700` permissions.

**No network transmission.** The only outbound request is a one-time download on first use: the BPE tokenizer from `openaipublic.blob.core.windows.net` (~3.5MB). It is SHA256-verified, and a corrupted download is discarded. The request is streamed under a 30-second timeout and a 32 MB ceiling, and the partial file is removed on any failure. Before 0.5.3 it used `urlretrieve`, which accepts no timeout at all, while holding a lock during server startup — an endpoint that accepted the connection and then stalled hung the server indefinitely.

**Durable writes.** Every file write goes through an atomic temp-file-then-rename. Since 0.5.3 the temp file and its parent directory are both `fsync`ed before the rename, so a crash cannot land the rename ahead of the bytes it points at — which would otherwise leave a file that exists, is the wrong size, and that the cache believes it wrote correctly. This costs roughly 1.1 ms per write on ext4/NVMe.

**SQLite WAL mode.** Crash recovery, with no data corruption on abrupt termination.

### Local Processing

Keyword search (BM25) and summarization run entirely on the local machine. There is no model download and no inference service, and no file content is sent to any external API.

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

**Cache location permissions.** The cache directory is created with user-only permissions, but the result depends on your umask. Verify:

```bash
ls -la ~/.cache/semantic-cache-mcp/
```

It should be `drwx------` (700). If it is world-readable, restrict it:

```bash
chmod 700 ~/.cache/semantic-cache-mcp/
```

**No encryption.** Cached content is stored unencrypted. Use filesystem-level encryption (for example macOS FileVault or Linux LUKS) if you cache sensitive projects.

### For Deployment

**Single-user only.** No authentication layer. Do not expose in multi-user environments.

**Container isolation.** Mount only the directories the cache needs, and avoid mounting `/` or sensitive paths.

**Audit logging.** File accesses are logged at INFO level (path plus token counts). This is not designed as a security audit trail.

---

## Known Limitations

| Limitation                | Notes                                              |
|---------------------------|----------------------------------------------------|
| No encryption at rest     | Use filesystem-level encryption if needed          |
| No access control         | Relies entirely on OS filesystem permissions       |
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
