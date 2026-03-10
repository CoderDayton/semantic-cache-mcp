"""BPE tokenizer package — o200k_base encoding with O(N log M) merge tracking."""

from __future__ import annotations

from ._bpe import (
    TOKENIZER_CACHE_DIR,
    BPETokenizer,
    _ensure_tokenizer,
    _tokenizer,
    _tokenizer_loaded,
    count_tokens,
    get_tokenizer,
)

__all__ = [
    "BPETokenizer",
    "TOKENIZER_CACHE_DIR",
    "count_tokens",
    "get_tokenizer",
    "_ensure_tokenizer",
    "_tokenizer",
    "_tokenizer_loaded",
]
