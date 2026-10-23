"""Self-contained BPE tokenizer supporting o200k_base encoding.

Optimizations:
- O(N log M) merge tracking via priority queue (not O(N²) naive search)
- Merge caching: memoize piece bytes → merged parts
- o200k pre-tokenizer ported to stdlib ``re`` over the BMP; astral characters
  match through same-class BMP stand-ins
- Streaming token encoding for unbounded text
- Vectorized hash-based vocab lookups
- Fast path for single-byte tokens (no BPE needed)
- Lazy merge deduplication (avoid redundant merges)

Reference: Zouhar et al. (2023) "A Formal Perspective on Byte-Pair Encoding"
"""

from __future__ import annotations

import base64
import hashlib
import heapq
import logging
import re
import sys
import threading
import unicodedata
from collections import OrderedDict
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from ...config import CACHE_DIR

if TYPE_CHECKING:
    from collections.abc import Iterator, Set

logger = logging.getLogger(__name__)

TOKENIZER_CACHE_DIR = CACHE_DIR / "tokenizer"
O200K_BASE_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
O200K_BASE_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"

# Bounds on the one-time tokenizer fetch. urlretrieve honours no timeout at
# all, so an endpoint that accepts the connection and then stalls wedges the
# first count_tokens() call forever — and the server primes that at startup,
# under a lock every other thread queues behind. The size cap bounds a
# hijacked or misconfigured endpoint: the real file is ~4 MB, and anything
# larger fails the hash check anyway, so stop before it reaches the disk.
DOWNLOAD_TIMEOUT_S = 30.0
DOWNLOAD_MAX_BYTES = 32 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 1 << 16

_EXACT_COUNT_MAX_CHARS = 256_000
_SAMPLE_WINDOWS = 16
_SAMPLE_CHARS = 4_000
_COUNT_CACHE_SIZE = 8

type CountKey = tuple[int, int, frozenset[str] | None]

_ASCII_WHITE_SPACE = (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20)
_WHITE_SPACE = "".join(
    map(
        chr,
        (
            *_ASCII_WHITE_SPACE,
            0x85,
            0xA0,
            0x1680,
            *range(0x2000, 0x200B),
            0x2028,
            0x2029,
            0x202F,
            0x205F,
            0x3000,
        ),
    )
)


def _class_body(chars: list[str]) -> str:
    """Regex character-class body for *chars* (sorted), as escaped ranges."""
    parts: list[str] = []
    i = 0
    while i < len(chars):
        j = i
        while j + 1 < len(chars) and ord(chars[j + 1]) == ord(chars[j]) + 1:
            j += 1
        lo, hi = re.escape(chars[i]), re.escape(chars[j])
        parts.append(lo if i == j else f"{lo}-{hi}")
        i = j + 1
    return "".join(parts)


def _compile(letter: str, number: str, upper: str, lower: str, ws: str) -> re.Pattern[str]:
    contraction = "(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
    lead = f"[^\\r\\n{letter}{number}]?"
    source = "|".join(
        (
            f"{lead}[{upper}]*[{lower}]+{contraction}",
            f"{lead}[{upper}]+[{lower}]*{contraction}",
            f"[{number}]{{1,3}}",
            f" ?[^{ws}{letter}{number}]+[\\r\\n/]*",
            f"[{ws}]*[\\r\\n]+",
            f"[{ws}]+(?![^{ws}])",
            f"[{ws}]+",
        )
    )
    return re.compile(source)


@cache
def ascii_pattern() -> re.Pattern[str]:
    """``o200k_pattern`` restricted to ASCII; splits ASCII text identically."""
    ws = _class_body([chr(c) for c in _ASCII_WHITE_SPACE])
    return _compile("A-Za-z", "0-9", "A-Z", "a-z", ws)


_BMP_END = 0x10000
_ASTRAL = re.compile(f"[{chr(_BMP_END)}-{chr(sys.maxunicode)}]")
_STAND_INS = {
    "Lu": "A",
    "Ll": "a",
    "Lt": chr(0x01C5),
    "Lm": chr(0x02B0),
    "Lo": chr(0x00AA),
    "M": chr(0x0300),
    "N": "0",
}


@cache
def _stand_in(char: str) -> str:
    """A BMP character in the same pattern classes as *char*.

    No astral character is White_Space, so every other category maps to
    ``!``, which only the punctuation alternative matches.
    """
    cat = unicodedata.category(char)
    return _STAND_INS.get(cat) or _STAND_INS.get(cat[0], "!")


def _stand_in_match(match: re.Match[str]) -> str:
    return _stand_in(match.group())


def _probe(text: str) -> tuple[re.Pattern[str], str]:
    """The pattern for *text* and the string to match it on.

    ``re`` matches classes above U+FFFF about 9x slower, so astral characters
    are swapped one for one with BMP stand-ins. Match spans then index *text*.
    """
    if text.isascii():
        return ascii_pattern(), text
    if len(text.encode("utf-16-le", "surrogatepass")) != 2 * len(text):
        return o200k_pattern(), _ASTRAL.sub(_stand_in_match, text)
    return o200k_pattern(), text


def split(text: str) -> list[str]:
    """Pre-tokenize *text* into the pieces o200k_base encodes separately."""
    pat, probe = _probe(text)
    if probe is text:
        return pat.findall(text)
    return [text[m.start() : m.end()] for m in pat.finditer(probe)]


def iter_split(text: str) -> Iterator[str]:
    """``split``, lazily."""
    pat, probe = _probe(text)
    for m in pat.finditer(probe):
        yield text[m.start() : m.end()]


@cache
def o200k_pattern() -> re.Pattern[str]:
    """tiktoken's o200k_base pre-tokenizer, ported to stdlib ``re``, for BMP text.

    The source pattern uses ``\\p{..}`` Unicode categories, which ``re`` lacks,
    so each class is built from ``unicodedata``. Its ``\\s`` is Unicode
    White_Space, which excludes U+001C..U+001F that Python's ``\\s`` matches.
    """
    by_class: dict[str, list[str]] = {"L": [], "N": [], "U": [], "W": []}
    for cp in range(_BMP_END):
        cat = unicodedata.category(chr(cp))
        major = cat[0]
        if major == "L":
            by_class["L"].append(chr(cp))
        elif major == "N":
            by_class["N"].append(chr(cp))
        if cat in ("Lu", "Lt", "Lm", "Lo") or major == "M":
            by_class["U"].append(chr(cp))
        if cat in ("Ll", "Lm", "Lo") or major == "M":
            by_class["W"].append(chr(cp))
    return _compile(
        _class_body(by_class["L"]),
        _class_body(by_class["N"]),
        _class_body(by_class["U"]),
        _class_body(by_class["W"]),
        _class_body(sorted(_WHITE_SPACE)),
    )


class BPETokenizer:
    """Self-contained BPE tokenizer compatible with tiktoken o200k_base encoding.

    Optimized for speed and memory efficiency:
    - Priority queue for O(N log M) merge tracking
    - Memoized merge operations
    - Compiled regex pattern caching
    - Streaming decode for large token lists
    """

    __slots__ = (
        "vocab",
        "inverse_vocab",
        "special_tokens",
        "_merge_cache",
        "_merge_cache_maxsize",
        "_count_cache",
    )

    def __init__(self) -> None:
        self.vocab: dict[int, bytes] = {}
        self.inverse_vocab: dict[bytes, int] = {}
        self.special_tokens: dict[str, int] = {}

        # LRU merge cache: bounded by entry count (4096, least recently
        # used evicted), not by bytes — entries hold token byte sequences
        # of arbitrary length, so actual memory scales with token sizes.
        self._merge_cache: OrderedDict[bytes, list[bytes]] = OrderedDict()
        self._merge_cache_maxsize: int = 4096
        self._count_cache: OrderedDict[CountKey, int] = OrderedDict()

    def load_tiktoken_file(self, path: Path | str) -> None:
        path = Path(path)
        self.vocab.clear()
        self.inverse_vocab.clear()
        self._merge_cache.clear()
        self._count_cache.clear()

        with open(path, encoding="utf-8") as f:
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

    def add_special_tokens(self, tokens: dict[str, int]) -> None:
        self._count_cache.clear()
        self.special_tokens.update(tokens)
        for token_str, token_id in tokens.items():
            token_bytes = token_str.encode("utf-8")
            self.vocab[token_id] = token_bytes
            self.inverse_vocab[token_bytes] = token_id

    def _bpe_merge_optimized(self, token_bytes: bytes) -> list[bytes]:
        """Apply BPE merges with O(N log M) complexity via priority queue.

        Uses a doubly-linked list to avoid O(N) list slicing per merge.
        Each merge is O(1) pointer update + O(log M) heap operation.
        """
        if len(token_bytes) <= 1:
            return [token_bytes] if token_bytes else []

        # Check cache first (move to end for LRU ordering)
        if token_bytes in self._merge_cache:
            self._merge_cache.move_to_end(token_bytes)
            return self._merge_cache[token_bytes]

        # Fast path: if whole token is in vocab, return as-is
        if token_bytes in self.inverse_vocab:
            self._merge_cache[token_bytes] = [token_bytes]
            return [token_bytes]

        # Doubly-linked list: nodes[i] = [value, prev_idx, next_idx]
        # Using list-of-lists instead of dataclass for speed in hot path
        n = len(token_bytes)
        nodes: list[list] = [[bytes([token_bytes[i]]), i - 1, i + 1] for i in range(n)]
        nodes[0][1] = -1  # head has no prev
        nodes[-1][2] = -1  # tail has no next

        ranks = self.inverse_vocab
        heap: list[tuple[int, int, int]] = []
        for i in range(n - 1):
            rank = ranks.get(nodes[i][0] + nodes[i + 1][0])
            if rank is not None:
                heap.append((rank, i, i + 1))
        heapq.heapify(heap)

        while heap:
            rank, li, ri = heapq.heappop(heap)

            if (
                nodes[li][2] != ri
                or nodes[ri][1] != li
                or nodes[li][0] is None
                or nodes[ri][0] is None
                or ranks.get(nodes[li][0] + nodes[ri][0]) != rank
            ):
                continue

            # Merge: absorb right into left
            merged = nodes[li][0] + nodes[ri][0]
            nodes[li][0] = merged

            # Unlink right node
            right_next = nodes[ri][2]
            nodes[li][2] = right_next
            if right_next != -1:
                nodes[right_next][1] = li
            nodes[ri][0] = None  # mark removed

            # Check new left pair
            left_prev = nodes[li][1]
            if left_prev != -1 and nodes[left_prev][0] is not None:
                r = ranks.get(nodes[left_prev][0] + merged)
                if r is not None:
                    heapq.heappush(heap, (r, left_prev, li))

            # Check new right pair
            if right_next != -1 and nodes[right_next][0] is not None:
                r = ranks.get(merged + nodes[right_next][0])
                if r is not None:
                    heapq.heappush(heap, (r, li, right_next))

        # Collect results by walking the linked list
        parts: list[bytes] = []
        # Find head (first non-removed node)
        i = 0
        while i < n and nodes[i][0] is None:
            i += 1
        while i != -1 and i < n:
            if nodes[i][0] is not None:
                parts.append(nodes[i][0])
            i = nodes[i][2]

        # LRU insert: evict oldest if at capacity
        self._merge_cache[token_bytes] = parts
        if len(self._merge_cache) > self._merge_cache_maxsize:
            self._merge_cache.popitem(last=False)
        return parts

    def encode(self, text: str, *, allowed_special: Set[str] | None = None) -> list[int]:
        if allowed_special is None:
            allowed_special = set()

        token_ids: list[int] = []

        # Split around special tokens
        if self.special_tokens and allowed_special:
            special_pattern = (
                "("
                + "|".join(re.escape(t) for t in sorted(allowed_special, key=len, reverse=True))
                + ")"
            )

            last_idx = 0
            for match in re.finditer(special_pattern, text):
                prefix = text[last_idx : match.start()]
                if prefix:
                    token_ids.extend(self._encode_chunk(prefix))

                special = match.group(0)
                if special in self.special_tokens:
                    token_ids.append(self.special_tokens[special])

                last_idx = match.end()

            text = text[last_idx:]

        if text:
            token_ids.extend(self._encode_chunk(text))

        return token_ids

    def _encode_chunk(self, text: str) -> list[int]:
        token_ids: list[int] = []
        token_ids_append = token_ids.append
        inverse_vocab_get = self.inverse_vocab.get
        bpe_merge = self._bpe_merge_optimized

        for piece in split(text):
            piece_bytes = piece.encode("utf-8")

            if len(piece_bytes) == 1:
                token_ids_append(inverse_vocab_get(piece_bytes, 0))
                continue

            tid = inverse_vocab_get(piece_bytes)
            if tid is not None:
                token_ids_append(tid)
                continue

            for part in bpe_merge(piece_bytes):
                token_ids_append(inverse_vocab_get(part, 0))

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        result = []
        for tid in token_ids:
            if tid in self.vocab:
                result.append(self.vocab[tid])
        return b"".join(result).decode("utf-8", errors="replace")

    def count(self, text: str, *, allowed_special: Set[str] | None = None) -> int:
        """Return the exact token count, estimated by sampling only above
        ``_EXACT_COUNT_MAX_CHARS``. Recent counts are memoized by text hash."""
        key = count_key(text, allowed_special)
        result = self._count_cache.pop(key, None)
        if result is None:
            if len(text) > _EXACT_COUNT_MAX_CHARS:
                result = self._estimate_tokens(text)
            else:
                result = len(self.encode(text, allowed_special=allowed_special))
        self._count_cache[key] = result
        if len(self._count_cache) > _COUNT_CACHE_SIZE:
            self._count_cache.popitem(last=False)
        return result

    def exceeds(self, text: str, limit: int) -> bool:
        """Whether the exact token count of *text* exceeds *limit*.

        Counting stops once it does, so the cost follows *limit*, not the text.
        """
        if len(text) <= _EXACT_COUNT_MAX_CHARS:
            cached = self._count_cache.get(count_key(text, None))
            if cached is not None:
                return cached > limit
        inverse_vocab = self.inverse_vocab
        bpe_merge = self._bpe_merge_optimized
        total = 0
        for piece in iter_split(text):
            piece_bytes = piece.encode("utf-8")
            if len(piece_bytes) == 1 or piece_bytes in inverse_vocab:
                total += 1
            else:
                total += len(bpe_merge(piece_bytes))
            if total > limit:
                return True
        return False

    def _estimate_tokens(self, text: str) -> int:
        """Token count extrapolated from evenly spaced windows of *text*."""
        if not self.vocab:
            # Heuristic fallback
            spaces = text.count(" ") + text.count("\n")
            return int((spaces + 1) * 1.3)

        try:
            step = len(text) // _SAMPLE_WINDOWS
            windows = [text[i * step : i * step + _SAMPLE_CHARS] for i in range(_SAMPLE_WINDOWS)]
            sampled = sum(len(self._encode_chunk(w)) for w in windows)
            return round(len(text) * sampled / sum(map(len, windows)))
        except (KeyError, ValueError, IndexError) as e:
            logger.debug(f"Token estimation fallback: {e}")
            spaces = text.count(" ") + text.count("\n")
            return int((spaces + 1) * 1.3)


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_tokenizer: BPETokenizer | None = None
_tokenizer_loaded: bool = False

_SPECIAL_TOKENS = {"<|endoftext|>": 199999, "<|endofprompt|>": 200018}


def _init_tokenizer(cache_file: Path) -> BPETokenizer:
    logger.debug(f"Loading o200k_base tokenizer from {cache_file}")
    tok = BPETokenizer()
    tok.load_tiktoken_file(cache_file)
    tok.add_special_tokens(_SPECIAL_TOKENS)
    ascii_pattern()
    o200k_pattern()
    logger.info(f"Tokenizer initialized with {len(tok.vocab)} tokens")
    return tok


def _download_tokenizer(url: str, dest: Path) -> None:
    """Stream *url* to *dest* under a socket timeout and a hard size cap.

    Raises OSError (which covers socket timeouts) on a stalled or unreachable
    endpoint, and ValueError when the response exceeds ``DOWNLOAD_MAX_BYTES`` —
    both already handled by the caller as "no tokenizer, use the heuristic".
    """
    import urllib.request  # noqa: PLC0415

    written = 0
    # URL is a module-level https constant and the result is hash-verified by
    # the caller before it is ever loaded.
    with (
        urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response,  # nosec B310
        open(dest, "wb") as out,
    ):
        while True:
            chunk = response.read(_DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            written += len(chunk)
            if written > DOWNLOAD_MAX_BYTES:
                raise ValueError(
                    f"tokenizer download exceeded {DOWNLOAD_MAX_BYTES} bytes; aborting"
                )
            out.write(chunk)


def _verify_hash(path: Path, expected: str) -> bool:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected


_tokenizer_lock = threading.Lock()


def _ensure_tokenizer() -> BPETokenizer | None:
    """Lazy-load with automatic download on first call; marks loaded on failure to prevent retry."""
    global _tokenizer, _tokenizer_loaded

    if _tokenizer_loaded:
        return _tokenizer

    with _tokenizer_lock:
        # Double-checked locking: re-check after acquiring lock
        if _tokenizer_loaded:
            return _tokenizer

        cache_file = TOKENIZER_CACHE_DIR / "o200k_base.tiktoken"

        if cache_file.exists():
            try:
                if _verify_hash(cache_file, O200K_BASE_SHA256):
                    _tokenizer = _init_tokenizer(cache_file)
                    _tokenizer_loaded = True
                    return _tokenizer
                logger.warning("Hash verification failed, re-downloading")
                cache_file.unlink()
            except (OSError, ValueError) as e:
                logger.warning(f"Failed to load cached tokenizer: {e}")

        try:
            import urllib.error
            import urllib.request

            TOKENIZER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Download to temp file then rename atomically. URL is a
            # compile-time constant and the result is hash-verified below.
            tmp_file = cache_file.with_suffix(".tmp")
            try:
                _download_tokenizer(O200K_BASE_URL, tmp_file)
            except BaseException:
                tmp_file.unlink(missing_ok=True)
                raise

            if not _verify_hash(tmp_file, O200K_BASE_SHA256):
                tmp_file.unlink(missing_ok=True)
                _tokenizer_loaded = True  # Don't retry on hash mismatch
                return None

            tmp_file.replace(cache_file)  # Atomic on POSIX
            _tokenizer = _init_tokenizer(cache_file)
            _tokenizer_loaded = True
            return _tokenizer
        except (OSError, urllib.error.URLError, ValueError) as e:
            logger.warning(f"Failed to download tokenizer: {e}")
            _tokenizer_loaded = True  # Don't retry on failure
            return None


def count_tokens(content: str) -> int:
    """Count tokens; falls back to word-count heuristic if tokenizer unavailable."""
    if not content:
        return 0

    tokenizer = _ensure_tokenizer()
    if tokenizer and tokenizer.vocab:
        try:
            return tokenizer.count(content)
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Token counting failed, using heuristic: {e}")

    # Fallback: words * 1.3 + chars * 0.1
    spaces = content.count(" ") + content.count("\n") + content.count("\t")
    words = spaces + 1
    return int(words * 1.3 + len(content) * 0.1)


get_tokenizer = _ensure_tokenizer


def count_key(text: str, allowed_special: Set[str] | None) -> CountKey:
    """Memo key for *text*: its hash and length, so the memo keeps no text alive."""
    return (hash(text), len(text), frozenset(allowed_special) if allowed_special else None)


def tokens_exceed(content: str, limit: int) -> bool:
    """Whether the exact token count of *content* exceeds *limit*, without
    counting past *limit*.

    Byte-level tokens let one character cost up to 4 tokens, so only text
    with at most ``limit // 4`` characters is known to fit uncounted.
    """
    if len(content) <= limit // 4:
        return False
    tokenizer = _ensure_tokenizer()
    if tokenizer and tokenizer.vocab:
        try:
            return tokenizer.exceeds(content, limit)
        except (KeyError, ValueError, IndexError) as e:
            logger.warning(f"Token counting failed, using heuristic: {e}")
    return count_tokens(content) > limit
