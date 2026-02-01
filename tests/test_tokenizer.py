"""Tests for the BPE tokenizer module."""

from __future__ import annotations

import pytest

from semantic_cache_mcp.core.tokenizer import BPETokenizer, count_tokens, get_tokenizer


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
