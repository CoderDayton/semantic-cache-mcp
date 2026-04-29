# Environment Variables

All environment variables are optional. Defaults are tuned for typical usage.

## Cache & Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMANTIC_CACHE_DIR` | Platform-specific\* | Override cache/database directory path. All data (database, models, metrics) lives under this directory. |
| `MAX_CACHE_ENTRIES` | `10000` | Maximum cached file entries before LRU-K eviction kicks in. Higher values use more memory and disk. |
| `MAX_CONTENT_SIZE` | `100000` | Maximum bytes returned by a single read operation. Files larger than this are truncated with a hint to use `offset`/`limit`. |

\* Linux: `~/.cache/semantic-cache-mcp/`, macOS: `~/Library/Caches/semantic-cache-mcp/`, Windows: `%LOCALAPPDATA%\semantic-cache-mcp\`

## Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model name for semantic search/similarity. Changing this clears and rebuilds all cached embeddings on next use. See [supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/). |
| `EMBEDDING_DEVICE` | `cpu` | Hardware for embedding inference. Options: `cpu` (no GPU needed), `gpu` or `cuda` (NVIDIA GPU via ONNX Runtime), `auto` (detect available). Requires `fastembed-gpu` for GPU — see below. |
| `OPENAI_EMBEDDINGS_ENABLED` | `false` | Route embeddings through an OpenAI-compatible API instead of loading the local FastEmbed model. |
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible embeddings endpoint. Default targets Ollama. Use `https://api.openai.com/v1` for hosted OpenAI. |
| `OPENAI_API_KEY` | `ollama` | API key passed to the OpenAI-compatible client. Ollama accepts any non-empty value; hosted OpenAI requires a real key. |
| `OPENAI_EMBEDDING_MODEL` | `nomic-embed-text` | Remote embedding model name. Changing this clears and rebuilds stored embeddings on next use. |
| `OPENAI_EMBEDDING_DIMENSIONS` | *(inferred)* | Optional requested/expected remote embedding dimension. Leave unset to infer it from the first returned vector. |

### GPU acceleration

The default install uses CPU inference via `fastembed`. To enable NVIDIA GPU acceleration:

```bash
uv tool install "semantic-cache-mcp[gpu]"
# or: pip install "semantic-cache-mcp[gpu]"
```

Then set `EMBEDDING_DEVICE=gpu` in your MCP config. If CUDA is unavailable at runtime, the server logs a warning and falls back to CPU automatically.

### Choosing an embedding model

The default `BAAI/bge-small-en-v1.5` (33M params, 384 dimensions) is fast on CPU and sufficient for code similarity. Consider alternatives if you need:

- **Higher quality**: `BAAI/bge-base-en-v1.5` (110M params, 768D) — better retrieval at ~3x slower inference
- **Longer context**: `nomic-ai/nomic-embed-text-v1.5` (137M params, 768D, 8192 token context) — better for large code files
- **Multilingual**: `BAAI/bge-small-zh-v1.5` or `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Smallest footprint**: `BAAI/bge-micro-v2` (17M params) — faster startup, slightly lower quality

Any HuggingFace sentence-transformer model with an ONNX export will work — if the model isn't in fastembed's built-in list, it's automatically registered from HuggingFace Hub on first use (requires network access for initial download).

> **Warning**: Changing the model invalidates all cached embeddings. The cache will rebuild as files are re-read.

### OpenAI-compatible embeddings

Set `OPENAI_EMBEDDINGS_ENABLED=true` to skip local FastEmbed startup and send embedding requests to an OpenAI-compatible service. The defaults target Ollama:

```json
"env": {
  "OPENAI_EMBEDDINGS_ENABLED": "true",
  "OPENAI_BASE_URL": "http://localhost:11434/v1",
  "OPENAI_API_KEY": "ollama",
  "OPENAI_EMBEDDING_MODEL": "nomic-embed-text"
}
```

Run `ollama pull nomic-embed-text` first if the model is not installed. For hosted OpenAI, override the URL, key, and model:

```json
"env": {
  "OPENAI_EMBEDDINGS_ENABLED": "true",
  "OPENAI_BASE_URL": "https://api.openai.com/v1",
  "OPENAI_API_KEY": "sk-...",
  "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small"
}
```

`OPENAI_EMBEDDING_DIMENSIONS` is optional. If unset, semantic-cache infers and records the dimension from the first returned vector. If set, the value is sent as OpenAI's `dimensions` request parameter and the response is validated against it.

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`. Set to `DEBUG` for troubleshooting embedding/storage issues. |

## Tool Response

| Variable | Default | Description |
|----------|---------|-------------|
| `TOOL_OUTPUT_MODE` | `compact` | Response detail level. Options: `compact` (minimal metadata, best for token savings), `normal` (includes context lines in grep, extra diagnostics), `debug` (full diagnostics including timing and internal state). |
| `TOOL_MAX_RESPONSE_TOKENS` | `0` | Global cap on response tokens per tool call. `0` disables the cap. Useful for constraining token budget on large operations. |
| `TOOL_TIMEOUT` | `30` | Seconds before a tool call times out and returns an error. On timeout, the executor is automatically reset so subsequent calls work without restarting. Lower for fast machines, raise for slow I/O or large files. |

## Example: MCP Server Config with Custom Env

```json
{
  "mcpServers": {
    "semantic-cache": {
      "command": "uvx",
      "args": ["semantic-cache-mcp"],
      "env": {
        "EMBEDDING_MODEL": "BAAI/bge-base-en-v1.5",
        "EMBEDDING_DEVICE": "gpu",
        "LOG_LEVEL": "DEBUG",
        "MAX_CACHE_ENTRIES": "20000"
      }
    }
  }
}
```
