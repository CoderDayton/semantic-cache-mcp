"""Cache package - re-exports all public symbols for backward compatibility."""

from __future__ import annotations

from ..core.embeddings import embed, embed_query
from ._helpers import (
    MAX_DIFF_TO_FULL_RATIO,
    MAX_EDIT_SIZE,
    MAX_MATCHES,
    MAX_RETURN_DIFF_TOKENS,
    MAX_WRITE_SIZE,
    _choose_min_token_content,
    _find_match_line_numbers,
    _format_file,
    _is_binary_content,
    _suppress_large_diff,
)
from .read import MAX_BATCH_FILES, MAX_BATCH_TOKENS, batch_smart_read, smart_read
from .search import (
    MAX_GLOB_MATCHES,
    MAX_SEARCH_K,
    MAX_SEARCH_QUERY_LEN,
    MAX_SIMILAR_K,
    compare_files,
    find_similar_files,
    glob_with_cache_status,
    semantic_search,
)
from .store import SemanticCache
from .write import MAX_BATCH_EDITS, smart_batch_edit, smart_edit, smart_write

__all__ = [
    # Store
    "SemanticCache",
    # Read
    "smart_read",
    "batch_smart_read",
    "MAX_BATCH_FILES",
    "MAX_BATCH_TOKENS",
    # Write
    "smart_write",
    "smart_edit",
    "smart_batch_edit",
    "MAX_BATCH_EDITS",
    # Search
    "semantic_search",
    "compare_files",
    "find_similar_files",
    "glob_with_cache_status",
    "MAX_SEARCH_K",
    "MAX_SEARCH_QUERY_LEN",
    "MAX_SIMILAR_K",
    "MAX_GLOB_MATCHES",
    # Core embeddings re-exported so tests can patch semantic_cache_mcp.cache.embed
    "embed",
    "embed_query",
    # Helpers (some used by tests)
    "_suppress_large_diff",
    "_is_binary_content",
    "_find_match_line_numbers",
    "_format_file",
    "_choose_min_token_content",
    "MAX_WRITE_SIZE",
    "MAX_EDIT_SIZE",
    "MAX_MATCHES",
    "MAX_RETURN_DIFF_TOKENS",
    "MAX_DIFF_TO_FULL_RATIO",
]
