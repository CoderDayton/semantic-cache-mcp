"""Tests for the BPE tokenizer module."""

from __future__ import annotations

import time

import pytest

from semantic_cache_mcp.core.tokenizer import (
    BPETokenizer,
    _bpe,
    count_tokens,
    get_tokenizer,
    tokens_exceed,
)
from semantic_cache_mcp.core.tokenizer._bpe import o200k_pattern


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_empty_string_returns_zero(self) -> None:
        """Empty string should return 0 tokens."""
        assert count_tokens("") == 0

    def test_simple_text_returns_positive(self) -> None:
        """Simple text should return positive token count."""
        result = count_tokens("Hello, world!")
        assert result > 0
        assert isinstance(result, int)

    def test_whitespace_only_returns_positive(self) -> None:
        """Whitespace-only text should return positive count."""
        result = count_tokens("   \n\t  ")
        assert result > 0

    def test_unicode_chinese_text(self) -> None:
        """Chinese text should be tokenized correctly."""
        result = count_tokens("hello")
        assert result > 0

    def test_unicode_emoji_text(self) -> None:
        """Emoji text should be tokenized correctly."""
        result = count_tokens("hello world")
        assert result > 0

    def test_mixed_unicode_text(self) -> None:
        """Mixed unicode (ASCII + Chinese + emoji) should work."""
        result = count_tokens("Hello world test 123")
        assert result > 0

    def test_large_text_performance(self) -> None:
        """Large text should complete in reasonable time."""
        large_text = "word " * 10000  # ~50KB of text
        result = count_tokens(large_text)
        assert result > 1000  # Should have many tokens

    def test_deterministic_output(self) -> None:
        """Same input should always produce same token count."""
        text = "The quick brown fox jumps over the lazy dog."
        count1 = count_tokens(text)
        count2 = count_tokens(text)
        count3 = count_tokens(text)
        assert count1 == count2 == count3

    def test_special_characters(self) -> None:
        """Special characters should be handled."""
        result = count_tokens("!@#$%^&*()_+-=[]{}|;':\",./<>?")
        assert result > 0


class TestBPETokenizer:
    """Tests for BPETokenizer class."""

    def test_empty_tokenizer_vocab(self) -> None:
        """New tokenizer should have empty vocab."""
        tok = BPETokenizer()
        assert len(tok.vocab) == 0
        assert len(tok.inverse_vocab) == 0

    def test_add_special_tokens(self) -> None:
        """Special tokens should be added to vocab."""
        tok = BPETokenizer()
        tok.add_special_tokens({"<|test|>": 999})
        assert tok.special_tokens["<|test|>"] == 999
        assert 999 in tok.vocab

    def test_encode_empty_string(self) -> None:
        """Encoding empty string should return empty list."""
        tok = get_tokenizer()
        if tok is None:
            pytest.skip("Tokenizer not available")
        result = tok.encode("")
        assert result == []

    def test_encode_decode_roundtrip(self) -> None:
        """Encoding then decoding should return original text."""
        tok = get_tokenizer()
        if tok is None:
            pytest.skip("Tokenizer not available")

        original = "Hello, world! This is a test."
        encoded = tok.encode(original)
        decoded = tok.decode(encoded)
        assert decoded == original

    def test_encode_decode_roundtrip_unicode(self) -> None:
        """Unicode text should roundtrip correctly."""
        tok = get_tokenizer()
        if tok is None:
            pytest.skip("Tokenizer not available")

        original = "Hello world test"
        encoded = tok.encode(original)
        decoded = tok.decode(encoded)
        assert decoded == original

    def test_count_method(self) -> None:
        """count() should return same as len(encode())."""
        tok = get_tokenizer()
        if tok is None:
            pytest.skip("Tokenizer not available")

        text = "Test counting tokens"
        assert tok.count(text) == len(tok.encode(text))

    def test_count_survives_memo_eviction_between_get_and_touch(self) -> None:
        """A hit evicted by another thread mid-lookup still returns its count."""
        from collections import OrderedDict

        class EvictOnGet(OrderedDict):
            def get(self, key, default=None):  # type: ignore[override]
                value = super().get(key, default)
                self.pop(key, None)
                return value

        tok = BPETokenizer()
        tok._count_cache = EvictOnGet({_bpe.count_key("abc", None): 7})
        assert tok.count("abc") == 7

    def test_add_special_tokens_invalidates_count_memo(self) -> None:
        """A count memoized before a special token exists must not be reused."""
        tok = BPETokenizer()
        text = "<|x|>"
        before = tok.count(text, allowed_special={text})
        tok.add_special_tokens({text: 5})
        assert tok.count(text, allowed_special={text}) == 1 != before


_O200K_GOLDEN: list[tuple[str, list[int]]] = [
    ("Hello, world!", [13225, 11, 2375, 0]),
    (
        "def parseHTTPResponse(self, rawBytes):\n    return self._cache[rawBytes]\n",
        [
            1314,
            8420,
            17893,
            3186,
            2402,
            11,
            11428,
            10629,
            1883,
            271,
            622,
            1051,
            2697,
            18949,
            58,
            1618,
            10629,
            1592,
        ],
    ),
    (
        "I'm sure they'll say we've DONE it, haven't they?",
        [15390, 3239, 57956, 2891, 24716, 113799, 480, 11, 23399, 1023, 30],
    ),
    (
        "x = 1234567 + 89; y=0x1F",
        [87, 314, 220, 7633, 19354, 22, 659, 220, 7479, 26, 342, 28, 15, 87, 16, 37],
    ),
    ("    \n\n\t  trailing   \n", [24372, 7758, 57985, 10190]),
    ("naïve café Größe ÉCOLE", [1503, 9954, 737, 30469, 74095, 10055, 8310, 1400]),
    (
        "日本語のテキストと中文混合",
        [9048, 40909, 3385, 16056, 18368, 38236, 5330, 10667, 85591, 4377],
    ),
    ("emoji 🎉🚀 and ✓ marks", [75339, 139786, 231, 112927, 222, 326, 81033, 22891]),
    (
        "path/to/file.py // comment /* block */",
        [4189, 72231, 51766, 17311, 602, 5375, 2785, 4355, 932],
    ),
    ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", [117525, 117525, 117525, 117525, 45037]),
    (
        "camelCaseWord XMLHttpRequest getHTTPStatusCode",
        [178067, 6187, 12929, 100497, 2303, 717, 17893, 3233, 2836],
    ),
    (
        "\N{NO-BREAK SPACE}nbsp\N{EM SPACE}em\N{IDEOGRAPHIC SPACE}ideo",
        [5310, 9431, 33203, 347, 1397, 2451],
    ),
    ("\x1cfs\x1f", [216, 6469, 219]),
]


_O200K_ASTRAL_GOLDEN: list[tuple[str, list[int]]] = [
    (
        f"{chr(0x1D400)}{chr(0x1D41B)}{chr(0x1D41C)} Bold {chr(0x1D41A)}{chr(0x1D41B)}",
        [91362, 222, 91362, 249, 91362, 250, 83138, 220, 91362, 248, 91362, 249],
    ),
    (
        f"{chr(0x10400)}{chr(0x10428)}{chr(0x10429)} deseret {chr(0x10428)}x",
        [172, 238, 238, 222, 172, 238, 238, 101, 172, 238, 238, 102]
        + [731, 64794, 220, 172, 238, 238, 101, 87],
    ),
    (
        "n" + "".join(map(chr, range(0x1D7CE, 0x1D7D2))) + " = 5",
        [77, 43120, 253, 236, 43120, 253, 237, 43120, 253, 238, 43120, 253, 239, 314, 220, 20],
    ),
    (
        f"{chr(0x20000)}{chr(0x20001)}中文 {chr(0x20002)}",
        [172, 62313, 222, 172, 62313, 223, 10667, 220, 172, 62313, 224],
    ),
    (
        f"ka{chr(0x11100)}{chr(0x11101)} mark",
        [1854, 172, 239, 226, 222, 172, 239, 226, 223, 1983],
    ),
    (
        f"a{chr(0x1F389)}b's {chr(0x1F680)}{chr(0x1F680)}!!\n\n",
        [64, 71344, 231, 65, 885, 169883, 222, 112927, 222, 19886],
    ),
    (
        f"x {chr(0x1F389)} \n  {chr(0x1F389)}\t",
        [87, 139786, 231, 793, 220, 139786, 231, 197],
    ),
]


class TestO200kExactness:
    """Encoding matches tiktoken's o200k_base id for id."""

    @pytest.mark.parametrize(("text", "expected"), _O200K_GOLDEN + _O200K_ASTRAL_GOLDEN)
    def test_encode_matches_tiktoken(self, text: str, expected: list[int]) -> None:
        tok = get_tokenizer()
        if tok is None:
            pytest.skip("Tokenizer not available")
        assert tok.encode(text) == expected

    def test_long_text_count_is_exact(self) -> None:
        if get_tokenizer() is None:
            pytest.skip("Tokenizer not available")
        text = "\n".join(t for t, _ in _O200K_GOLDEN) * 200
        assert len(text) == 76_600
        assert count_tokens(text) == 23_400

    def test_pattern_has_no_astral_classes(self) -> None:
        """Astral text splits through the BMP pattern, never a slower full one."""
        assert max(o200k_pattern().pattern) < chr(0x10000)

    def test_tokenizer_load_builds_pattern(self) -> None:
        cache_file = _bpe.TOKENIZER_CACHE_DIR / "o200k_base.tiktoken"
        if not cache_file.exists():
            pytest.skip("Tokenizer not available")
        o200k_pattern.cache_clear()
        _bpe._init_tokenizer(cache_file)
        assert o200k_pattern.cache_info().currsize == 1


class TestCountCost:
    """Counting stays bounded in time and memory."""

    def test_count_memo_holds_no_text(self) -> None:
        tok = get_tokenizer()
        if tok is None:
            pytest.skip("Tokenizer not available")
        text = "memo text " * 1000
        tok.count(text)
        assert all(text not in key for key in tok._count_cache)

    def test_estimate_samples_whole_text(self) -> None:
        tok = get_tokenizer()
        if tok is None:
            pytest.skip("Tokenizer not available")
        sparse = "word " * 80_000
        dense = "".join(f"{i * 2654435761 % 2**32:08x}," for i in range(44_445))
        text = sparse + dense + sparse
        exact = len(tok.encode(text))
        assert abs(tok.count(text) - exact) / exact < 0.03

    @pytest.mark.parametrize("delta", [-1, 0, 1])
    def test_tokens_exceed_matches_count(self, delta: int) -> None:
        text = f"x {chr(0x1F389)} " * 40 + "naïve café " * 30
        limit = count_tokens(text) + delta
        assert tokens_exceed(text, limit) is (count_tokens(text) > limit)

    def test_tokens_exceed_counts_past_char_length(self) -> None:
        if get_tokenizer() is None:
            pytest.skip("Tokenizer not available")
        text = "".join(chr(0x1F300 + i) for i in range(64))
        assert count_tokens(text) > len(text)
        assert tokens_exceed(text, len(text))

    def test_tokens_exceed_stops_at_limit(self) -> None:
        if get_tokenizer() is None:
            pytest.skip("Tokenizer not available")
        text = "alpha beta gamma, delta! " * 400_000
        start = time.perf_counter()
        assert tokens_exceed(text, 100)
        assert time.perf_counter() - start < 0.5


class TestTokenizerEdgeCases:
    """Edge case tests for tokenizer."""

    def test_single_byte_input(self) -> None:
        """Single character should tokenize."""
        result = count_tokens("a")
        assert result >= 1

    def test_very_long_word(self) -> None:
        """Very long word (no spaces) should tokenize."""
        result = count_tokens("a" * 1000)
        assert result > 0

    def test_repeated_pattern(self) -> None:
        """Repeated pattern should tokenize efficiently."""
        result = count_tokens("abc" * 1000)
        assert result > 0

    def test_newlines_only(self) -> None:
        """Newlines only should tokenize."""
        result = count_tokens("\n" * 100)
        assert result > 0

    def test_mixed_whitespace(self) -> None:
        """Mixed whitespace should tokenize."""
        result = count_tokens(" \t\n \t\n " * 50)
        assert result > 0
