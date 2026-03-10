# Security Considerations

## Threat Model

Single-user, local-only. Cached content lives in `~/.cache/semantic-cache-mcp/`. No multi-user or network-accessible deployment support.

**In scope:** local filesystem caching, SQLite on disk, local embedding inference.
**Out of scope:** multi-tenant, network exposure, auth (defers to OS permissions).

---

## Security Controls

### File Access

**Path resolution** — All paths are resolved via `Path.resolve()` before any I/O, preventing directory traversal via `../` sequences or symlink tricks that escape the intended base directory.

**Symlink handling** — Symlinks are followed and resolved to their target (intentional for developer workflows). Symlink resolution is logged at DEBUG level.

**Binary file detection** — Non-text files are rejected before any caching attempt, using multiple layered checks:
1. Null byte scan in the first 8KB
2. Magic number signatures: PNG, JPEG, GIF, ZIP, GZIP, ELF, MZ/PE, PDF, OLE2
3. High-entropy heuristic: >30% non-printable characters

**File type gating** — Only regular files are accepted. Directories, devices, sockets, and other special files are rejected immediately.

### DoS Protection

Size limits prevent memory exhaustion from oversized inputs:

| Limit            | Default | Applies to                             |
|------------------|---------|----------------------------------------|
| `MAX_WRITE_SIZE` | 10 MB   | `write` tool content                   |
| `MAX_EDIT_SIZE`  | 10 MB   | `edit` and `batch_edit` file size      |
| `MAX_CONTENT_SIZE` | 100 KB | Default max bytes returned by `read`  |
| `MAX_MATCHES`    | 10,000  | `replace_all` match count in `edit`   |

All limits are enforced **before** any I/O operation — fail-fast.

**Search and similar** — Results are capped at 100 and 50 respectively; glob is capped at 1,000 matches with a 5-second timeout.

### SQL Injection

All SQL queries use parameterized statements. The only dynamic SQL construction is `IN` clauses where the placeholder string (`?,?,?`) is built from a count — never from user-supplied data. User values are always passed as bound parameters.

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

**Local only** — All data stored in `~/.cache/semantic-cache-mcp/` with `700` permissions.

**No network transmission** — The only outbound requests are one-time downloads on first use:
1. Embedding model from HuggingFace Hub (~130MB)
2. Tokenizer from `openaipublic.blob.core.windows.net` (~3.5MB)

Both are SHA256-verified; corrupted downloads are discarded.

**SQLite WAL mode** — crash recovery, no data corruption on abrupt termination.

### Embedding Model

Inference runs locally via ONNX Runtime ([FastEmbed](https://github.com/qdrant/fastembed), [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)). No file content is sent to any external API.

---

## Recommendations

### For Users

**Sensitive files** — Avoid caching files containing secrets, credentials, or PII. Clear the cache when switching to or from sensitive projects:

```bash
# Via MCP tool
clear()

# Via filesystem
rm -rf ~/.cache/semantic-cache-mcp/cache.db
```

**Cache location permissions** — The cache directory is created with user-only permissions, but depends on your umask. Verify:

```bash
ls -la ~/.cache/semantic-cache-mcp/
```

It should be `drwx------` (700). If it is world-readable, restrict it:

```bash
chmod 700 ~/.cache/semantic-cache-mcp/
```

**No encryption** — Cached content is stored unencrypted. Use filesystem-level encryption (e.g., macOS FileVault, Linux LUKS) if you cache sensitive projects.

### For Deployment

**Single-user only** — No authentication layer. Do not expose in multi-user environments.

**Container isolation** — Mount only the directories the cache needs; avoid mounting `/` or sensitive paths.

**Audit logging** — File accesses logged at INFO level (path + token counts). Not designed as a security audit trail.

---

## Known Limitations

| Limitation                | Notes                                              |
|---------------------------|----------------------------------------------------|
| No encryption at rest     | Use filesystem-level encryption if needed          |
| No access control         | Relies entirely on OS filesystem permissions       |
| No audit trail            | Logging is operational, not security-grade         |
| Single-user design        | Multi-tenant use is not supported                  |
| Network on first use only | Model and tokenizer downloads, both hash-verified  |

---

## Reporting Security Issues

If you discover a security vulnerability, please report it privately via [GitHub Security Advisories](https://github.com/CoderDayton/semantic-cache-mcp/security/advisories) or by email to coderdayton14@gmail.com.

Please do not open a public GitHub issue for security vulnerabilities.

---

[← Back to README](../README.md)
