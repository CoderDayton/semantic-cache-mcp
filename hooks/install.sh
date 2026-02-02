#!/bin/bash
# Semantic Cache MCP - Hook Installer
# Installs PreToolUse hook for automatic Read tool caching
#
# Usage: ./hooks/install.sh
#        ./hooks/install.sh --uninstall

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

CLAUDE_DIR="${HOME}/.claude"
HOOKS_DIR="${CLAUDE_DIR}/hooks"
SETTINGS_FILE="${CLAUDE_DIR}/settings.json"
HOOK_SCRIPT="pre-tool-use.sh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

check_dependencies() {
    local missing=()

    for cmd in sqlite3 jq bc; do
        if ! command -v "$cmd" &>/dev/null; then
            missing+=("$cmd")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        error "Missing required dependencies: ${missing[*]}"
        echo ""
        echo "Install them with:"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            echo "  brew install ${missing[*]}"
        else
            echo "  sudo apt install ${missing[*]}"
        fi
        exit 1
    fi

    success "Dependencies verified: sqlite3, jq, bc"
}

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------

install_hook_script() {
    info "Installing hook script..."

    # Create hooks directory
    if [ ! -d "$HOOKS_DIR" ]; then
        mkdir -p "$HOOKS_DIR"
        success "Created $HOOKS_DIR"
    else
        info "Hooks directory exists"
    fi

    # Copy script
    local src="${SCRIPT_DIR}/${HOOK_SCRIPT}"
    local dst="${HOOKS_DIR}/${HOOK_SCRIPT}"

    if [ ! -f "$src" ]; then
        error "Source script not found: $src"
        error "Run this script from the semantic-cache-mcp directory"
        exit 1
    fi

    if [ -f "$dst" ]; then
        # Check if identical
        if cmp -s "$src" "$dst"; then
            info "Hook script already installed and up to date"
            return 0
        fi
        warn "Existing hook script will be overwritten"
    fi

    cp "$src" "$dst"
    chmod +x "$dst"
    success "Installed ${dst}"
}

configure_settings() {
    info "Configuring Claude settings..."

    # Create settings file if doesn't exist
    if [ ! -f "$SETTINGS_FILE" ]; then
        echo '{}' > "$SETTINGS_FILE"
        info "Created new settings file"
    fi

    # Validate JSON
    if ! jq empty "$SETTINGS_FILE" 2>/dev/null; then
        error "Invalid JSON in $SETTINGS_FILE"
        error "Please fix the JSON syntax and retry"
        exit 1
    fi

    # Check if hook already configured
    local existing_hook
    existing_hook=$(jq -r '.hooks.PreToolUse // empty' "$SETTINGS_FILE" 2>/dev/null)

    if [ -n "$existing_hook" ]; then
        # Check if our hook is already in the array
        local has_our_hook
        has_our_hook=$(jq -r '.hooks.PreToolUse[]? | select(.matcher == "Read") | select(.hooks[]?.command | contains("pre-tool-use.sh")) | "yes"' "$SETTINGS_FILE" 2>/dev/null | head -1)

        if [ "$has_our_hook" = "yes" ]; then
            info "Semantic cache hook already configured"
            return 0
        fi

        warn "Existing PreToolUse hooks found - adding semantic cache hook"
    fi

    # Add our hook configuration
    local hook_config='{
        "matcher": "Read",
        "hooks": [
            {
                "type": "command",
                "command": "~/.claude/hooks/pre-tool-use.sh"
            }
        ]
    }'

    # Create backup
    cp "$SETTINGS_FILE" "${SETTINGS_FILE}.bak"
    info "Backup created: ${SETTINGS_FILE}.bak"

    # Add hook to settings
    local updated
    if [ -z "$existing_hook" ]; then
        # No PreToolUse hooks - create the structure
        updated=$(jq --argjson hook "$hook_config" '.hooks.PreToolUse = [$hook]' "$SETTINGS_FILE")
    else
        # Append to existing PreToolUse array
        updated=$(jq --argjson hook "$hook_config" '.hooks.PreToolUse += [$hook]' "$SETTINGS_FILE")
    fi

    echo "$updated" > "$SETTINGS_FILE"
    success "Updated $SETTINGS_FILE"
}

# -----------------------------------------------------------------------------
# Uninstallation
# -----------------------------------------------------------------------------

uninstall() {
    info "Uninstalling semantic cache hook..."

    # Remove hook from settings
    if [ -f "$SETTINGS_FILE" ]; then
        if jq -e '.hooks.PreToolUse' "$SETTINGS_FILE" &>/dev/null; then
            # Remove our specific hook
            local updated
            updated=$(jq 'if .hooks.PreToolUse then .hooks.PreToolUse |= map(select(.matcher != "Read" or (.hooks | all(.command | contains("pre-tool-use.sh") | not)))) else . end' "$SETTINGS_FILE")

            # Clean up empty arrays
            updated=$(echo "$updated" | jq 'if .hooks.PreToolUse == [] then del(.hooks.PreToolUse) else . end')
            updated=$(echo "$updated" | jq 'if .hooks == {} then del(.hooks) else . end')

            echo "$updated" > "$SETTINGS_FILE"
            success "Removed hook from settings"
        else
            info "No PreToolUse hooks in settings"
        fi
    fi

    # Remove hook script
    local hook_path="${HOOKS_DIR}/${HOOK_SCRIPT}"
    if [ -f "$hook_path" ]; then
        rm "$hook_path"
        success "Removed $hook_path"
    else
        info "Hook script not found"
    fi

    success "Uninstallation complete"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

show_usage() {
    echo "Semantic Cache MCP - Hook Installer"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --uninstall    Remove the hook"
    echo "  --help         Show this message"
    echo ""
    echo "This installs a PreToolUse hook that intercepts Read tool calls"
    echo "and returns cached content when available, saving 80%+ tokens."
}

main() {
    case "${1:-}" in
        --uninstall|-u)
            uninstall
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        "")
            echo "╔════════════════════════════════════════════════════════════╗"
            echo "║        Semantic Cache MCP - Hook Installer                 ║"
            echo "╚════════════════════════════════════════════════════════════╝"
            echo ""

            check_dependencies
            install_hook_script
            configure_settings

            echo ""
            success "Installation complete!"
            echo ""
            echo "Next steps:"
            echo "  1. Restart Claude Code for changes to take effect"
            echo "  2. Read files normally - caching happens automatically"
            echo "  3. Check cache hits in: ~/.cache/semantic-cache-mcp/hook.log"
            echo ""
            echo "To uninstall: $0 --uninstall"
            ;;
        *)
            error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
