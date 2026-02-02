# Security Considerations

This document outlines security considerations for Semantic Cache MCP.

## Threat Model

Semantic Cache MCP is designed as a **single-user, local cache** for Claude Code. It operates on your local filesystem and stores cached data in `~/.cache/semantic-cache-mcp/`.

### In Scope

- File content caching and retrieval
- Local SQLite database operations
- Embedding generation for semantic similarity

### Out of Scope

- Multi-user or multi-tenant scenarios
- Network-accessible deployment
- Authentication or authorization (relies on filesystem permissions)

## Security Controls

### File Access

1. **Path Resolution**: All file paths are resolved via `Path.resolve()` before access, preventing directory traversal via `../` sequences.

2. **Symlink Handling**: Symlinks are followed and resolved to their target. This is intentional for developer workflows but logged at DEBUG level.

3. **Binary File Detection**: Binary files (containing null bytes) are rejected to prevent unintended data exposure or processing issues.

4. **File Type Validation**: Only regular files are cached. Directories, devices, and other special files are rejected.

### Data Storage

1. **Local Storage Only**: Cache data is stored in `~/.cache/semantic-cache-mcp/` with standard user permissions.

2. **No Network Transmission**: Cached content and embeddings are never transmitted over the network.

3. **SQLite WAL Mode**: Uses Write-Ahead Logging for data integrity and crash recovery.

### Embedding Model

1. **Local Execution**: Embeddings are generated locally using FastEmbed (nomic-embed-text-v1.5). No API calls are made.

2. **Model Download**: The embedding model is downloaded once from HuggingFace Hub on first use.

## Recommendations

### For Users

1. **Sensitive Files**: Avoid caching files containing secrets, credentials, or sensitive data. The cache stores full file content.

2. **Cache Location**: The cache directory (`~/.cache/semantic-cache-mcp/`) contains plaintext file content. Ensure appropriate filesystem permissions.

3. **Clear Cache**: Use the `clear` tool to remove cached content when working with sensitive projects:
   ```
   clear()
   ```

### For Deployment

1. **Single User Only**: Do not deploy Semantic Cache MCP in multi-user environments without additional access controls.

2. **Filesystem Permissions**: Rely on filesystem permissions for access control. The cache respects file readability.

3. **Container Isolation**: If running in containers, mount only necessary directories to limit cache scope.

## Known Limitations

1. **No Encryption**: Cached content is stored unencrypted. Use filesystem-level encryption if required.

2. **No Access Logging**: File accesses are logged at DEBUG level but not designed for audit purposes.

3. **No Rate Limiting**: No built-in rate limiting for cache operations.

## Reporting Security Issues

If you discover a security vulnerability, please report it privately via GitHub Security Advisories or email to coderdayton14@gmail.com.
