"""Self-contained BPE tokenizer supporting o200k_base encoding.

Based on Sebastian Raschka's BPE from scratch implementation:
https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/05_bpe-from-scratch/bpe-from-scratch.ipynb
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Set

# Cache directory for tokenizer files
TOKENIZER_CACHE_DIR = Path.home() / ".cache" / "semantic-cache-mcp" / "tokenizer"

# o200k_base encoding URL (GPT-4o tokenizer with 199,997 tokens)
O200K_BASE_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
O200K_BASE_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


class BPETokenizer:
    """Self-contained BPE tokenizer compatible with tiktoken o200k_base encoding.

    Supports:
    - Loading pretrained vocab/merges from tiktoken files
    - GPT-4o style tokenization (o200k_base)
    - Special token handling
    """

    __slots__ = ("vocab", "inverse_vocab", "bpe_ranks", "special_tokens", "_pat")

    # GPT-4 style pattern (requires regex module for Unicode properties)
    _GPT4_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    # Simplified fallback pattern for stdlib re
    _SIMPLE_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d|[A-Za-z]+|\d+|[^\sA-Za-z0-9]+|\s+"""

    def __init__(self) -> None:
        # Maps token_id -> token_bytes
        self.vocab: dict[int, bytes] = {}
        # Maps token_bytes -> token_id
        self.inverse_vocab: dict[bytes, int] = {}
        # BPE merge ranks: {(bytes_A, bytes_B): rank} - lower rank = higher priority
        self.bpe_ranks: dict[tuple[bytes, bytes], int] = {}
        # Special tokens: {token_str: token_id}
        self.special_tokens: dict[str, int] = {}
        # Regex pattern for pre-tokenization (re.Pattern or regex.Pattern)
        self._pat: Any = None

    def load_tiktoken_file(self, path: Path | str) -> None:
        """Load vocab and merges from a tiktoken .tiktoken file.

        The tiktoken format is: base64(token_bytes) + " " + rank
        Each line represents a token and its rank/id.

        Args:
            path: Path to .tiktoken file
        """
        path = Path(path)
        self.vocab.clear()
        self.inverse_vocab.clear()
        self.bpe_ranks.clear()

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                token_b64, rank_str = parts
                token_bytes = base64.b64decode(token_b64)
                rank = int(rank_str)

                self.vocab[rank] = token_bytes
                self.inverse_vocab[token_bytes] = rank

        # Build BPE merge ranks from vocab
        # For each multi-byte token, infer the merge that created it
        self._build_bpe_ranks()

    def _build_bpe_ranks(self) -> None:
        """Build BPE merge ranks from vocabulary.

        For tokens > 1 byte, the merge rank is the token's rank.
        The merge is: (prefix, suffix) where prefix+suffix = token.
        We use the split that has the highest-ranked prefix (greedy).
        """
        # Sort by rank to process in order
        sorted_tokens = sorted(self.vocab.items(), key=lambda x: x[0])

        for rank, token_bytes in sorted_tokens:
            if len(token_bytes) < 2:
                continue

            # Find best split: prefer the one where both parts exist
            best_split = None
            for i in range(1, len(token_bytes)):
                prefix = token_bytes[:i]
                suffix = token_bytes[i:]
                if prefix in self.inverse_vocab and suffix in self.inverse_vocab:
                    # Prefer split with higher-ranked (earlier) prefix
                    if best_split is None:
                        best_split = (prefix, suffix)
                    elif self.inverse_vocab[prefix] < self.inverse_vocab[best_split[0]]:
                        best_split = (prefix, suffix)

            if best_split:
                self.bpe_ranks[best_split] = rank

    def add_special_tokens(self, tokens: dict[str, int]) -> None:
        """Add special tokens to the vocabulary.

        Args:
            tokens: Mapping of special token strings to their IDs
        """
        self.special_tokens.update(tokens)
        for token_str, token_id in tokens.items():
            token_bytes = token_str.encode("utf-8")
            self.vocab[token_id] = token_bytes
            self.inverse_vocab[token_bytes] = token_id

    def _bpe_merge(self, token_bytes: bytes) -> list[bytes]:
        """Apply BPE merges to a byte sequence.

        Args:
            token_bytes: Raw bytes to tokenize

        Returns:
            List of merged byte sequences
        """
        if len(token_bytes) <= 1:
            return [token_bytes] if token_bytes else []

        # Start with individual bytes
        parts = [bytes([b]) for b in token_bytes]

        while len(parts) > 1:
            # Find pair with lowest rank (highest priority)
            min_rank = float("inf")
            min_idx = -1

            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                rank = self.bpe_ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_idx = i

            # No more merges possible
            if min_idx == -1 or min_rank == float("inf"):
                break

            # Merge the pair
            merged = parts[min_idx] + parts[min_idx + 1]
            parts = parts[:min_idx] + [merged] + parts[min_idx + 2 :]

        return parts

    def encode(self, text: str, *, allowed_special: Set[str] | None = None) -> list[int]:
        """Encode text into token IDs.

        Args:
            text: Text to encode
            allowed_special: Set of special token strings allowed in text

        Returns:
            List of token IDs
        """
        if allowed_special is None:
            allowed_special = set()

        token_ids: list[int] = []

        # Handle special tokens by splitting around them
        if self.special_tokens and allowed_special:
            special_pattern = (
                "(" + "|".join(re.escape(t) for t in sorted(allowed_special, key=len, reverse=True)) + ")"
            )

            last_idx = 0
            for match in re.finditer(special_pattern, text):
                # Encode prefix
                prefix = text[last_idx : match.start()]
                if prefix:
                    token_ids.extend(self._encode_chunk(prefix))

                # Add special token
                special = match.group(0)
                if special in self.special_tokens:
                    token_ids.append(self.special_tokens[special])

                last_idx = match.end()

            # Encode remainder
            text = text[last_idx:]

        if text:
            token_ids.extend(self._encode_chunk(text))

        return token_ids

    def _encode_chunk(self, text: str) -> list[int]:
        """Encode a chunk of text (no special tokens).

        Args:
            text: Text chunk to encode

        Returns:
            List of token IDs
        """
        token_ids: list[int] = []

        # Pre-tokenize using regex pattern (lazy compile)
        if self._pat is None:
            try:
                import regex
                self._pat = regex.compile(self._GPT4_PATTERN)
            except ImportError:
                self._pat = re.compile(self._SIMPLE_PATTERN, re.UNICODE)

        matches = self._pat.findall(text)

        for piece in matches:
            piece_bytes = piece.encode("utf-8")

            # Check if whole piece is in vocab
            if piece_bytes in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[piece_bytes])
                continue

            # Apply BPE
            for part in self._bpe_merge(piece_bytes):
                if part in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[part])
                    continue
                # Unknown - encode individual bytes
                token_ids.extend(self.inverse_vocab[bytes([b])] for b in part if bytes([b]) in self.inverse_vocab)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        return b"".join(self.vocab[tid] for tid in token_ids if tid in self.vocab).decode("utf-8", errors="replace")

    def count(self, text: str, *, allowed_special: Set[str] | None = None) -> int:
        """Return number of tokens in text.

        Args:
            text: Text to count tokens for
            allowed_special: Set of allowed special tokens

        Returns:
            Token count
        """
        return len(self.encode(text, allowed_special=allowed_special))


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_tokenizer: BPETokenizer | None = None
_tokenizer_loaded: bool = False

# Special tokens for o200k_base
_SPECIAL_TOKENS = {"<|endoftext|>": 199999, "<|endofprompt|>": 200018}


def _init_tokenizer(cache_file: Path) -> BPETokenizer:
    """Initialize tokenizer from cache file."""
    tok = BPETokenizer()
    tok.load_tiktoken_file(cache_file)
    tok.add_special_tokens(_SPECIAL_TOKENS)
    return tok


def _verify_hash(path: Path, expected: str) -> bool:
    """Verify SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected


def _ensure_tokenizer() -> BPETokenizer | None:
    """Lazily load the o200k_base tokenizer. Downloads if not cached."""
    global _tokenizer, _tokenizer_loaded

    if _tokenizer_loaded:
        return _tokenizer

    _tokenizer_loaded = True
    cache_file = TOKENIZER_CACHE_DIR / "o200k_base.tiktoken"

    # Try to load from cache (with hash verification)
    if cache_file.exists():
        try:
            if _verify_hash(cache_file, O200K_BASE_SHA256):
                _tokenizer = _init_tokenizer(cache_file)
                return _tokenizer
            cache_file.unlink()  # Remove corrupted file
        except Exception:
            pass

    # Try to download and verify
    try:
        import urllib.request
        TOKENIZER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(O200K_BASE_URL, cache_file)

        if not _verify_hash(cache_file, O200K_BASE_SHA256):
            cache_file.unlink()
            return None

        _tokenizer = _init_tokenizer(cache_file)
        return _tokenizer
    except Exception:
        return None


def count_tokens(content: str) -> int:
    """Count tokens using o200k_base BPE tokenizer.

    Falls back to heuristic if tokenizer unavailable.

    Args:
        content: Text to count tokens for

    Returns:
        Token count
    """
    if not content:
        return 0

    tokenizer = _ensure_tokenizer()

    if tokenizer and tokenizer.vocab:
        try:
            return tokenizer.count(content)
        except Exception:
            pass

    # Fallback heuristic: words * 1.3 + chars * 0.1
    spaces = content.count(" ") + content.count("\n") + content.count("\t")
    words = spaces + 1
    return int(words * 1.3 + len(content) * 0.1)


# Public alias (already memoized via _tokenizer_loaded flag)
get_tokenizer = _ensure_tokenizer
