# Client Setup

All clients run the same command: `uvx semantic-cache-mcp`. You need [uv](https://docs.astral.sh/uv/getting-started/installation/).

## Claude Code

```bash
claude mcp add semantic-cache --scope user -- uvx semantic-cache-mcp
```

## Cursor

`~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "semantic-cache": {
      "command": "uvx",
      "args": ["semantic-cache-mcp"]
    }
  }
}
```

## VS Code (Copilot)

`.vscode/mcp.json`:

```json
{
  "servers": {
    "semantic-cache": {
      "type": "stdio",
      "command": "uvx",
      "args": ["semantic-cache-mcp"]
    }
  }
}
```

## Codex

`~/.codex/config.toml`:

```toml
[mcp_servers.semantic-cache]
command = "uvx"
args = ["semantic-cache-mcp"]
```

## OpenCode

`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "semantic-cache": {
      "type": "local",
      "command": ["uvx", "semantic-cache-mcp"],
      "enabled": true
    }
  }
}
```

## Claude Desktop and other clients

Any client that takes an `mcpServers` JSON block:

```json
{
  "mcpServers": {
    "semantic-cache": {
      "command": "uvx",
      "args": ["semantic-cache-mcp"]
    }
  }
}
```

## Pinning a version

`uvx` reuses its cached copy. To control the version, install it once and point the client at the command:

```bash
uv tool install semantic-cache-mcp    # then "command": "semantic-cache-mcp", no args
```

## Make the agent use it

Agents reach for their built-in file tools by habit. Turn those off so every read goes through the cache.

**Claude Code**: `~/.claude/settings.json`

```json
{
  "permissions": {
    "deny": ["Read", "Edit", "Write"]
  }
}
```

**OpenCode**: `~/.config/opencode/opencode.json`

```json
{
  "permission": {
    "read": "deny",
    "edit": "deny",
    "write": "deny"
  }
}
```

Then tell the agent why, in `CLAUDE.md`, `AGENTS.md` or your client's rules file:

```markdown
## Tools

- Use `semantic-cache` for all file reads, writes, edits, grep and search instead of built-in file tools.
- Keep each `content_hash` you receive and pass it back as `known_hash`, so re-reads cost a few tokens.
```

## Settings

Pass environment variables in the client's `env` block. Every variable is listed in [env_variables.md](env_variables.md).
