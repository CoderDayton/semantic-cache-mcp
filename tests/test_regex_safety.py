"""grep must refuse patterns it cannot bound.

`re` offers no match budget and holds the GIL while matching, so once a
catastrophically backtracking pattern starts, nothing stops it: not the tool
timeout (its timer is an event-loop callback that never gets to run), not a
worker thread, not cancellation. Measured before this guard existed, a single
28-character line and the pattern `(a+)+$` pinned the whole server for 11
seconds, and 40 characters for over two minutes.

The only available defense is to not start. These tests pin both halves of
that: the exponential shapes are rejected, and ordinary patterns are not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from semantic_cache_mcp.cache import SemanticCache
from semantic_cache_mcp.cache.read import smart_read
from semantic_cache_mcp.storage.docstore._grep import grep, has_nested_quantifier

# A repeated group wrapping an unbounded quantifier: the engine can split one
# run of input between the inner and outer repeat in exponentially many ways.
EXPONENTIAL_PATTERNS = [
    r"(a+)+$",
    r"(a*)*$",
    r"(a+)*b",
    r"([a-z]+)+@",
    r"(\w*)*!",
    r"([0-9]{2,})+x",
    r"^(\s*\w+)+$",
    r"(x+x+)+y",
    r"((a+)+)+",
    r"(a+){2,}",
    r"(a+){3}",
]

# Patterns a caller would plausibly send, including every regex this repo's own
# source and tests use. None may be refused.
ORDINARY_PATTERNS = [
    r"def \w+\(",
    r"(abc)+",
    r"(foo|bar)*",
    r"[a-z]+",
    r"\d{2,4}-\d{2}",
    r"a+b+c+",
    r"(TODO|FIXME):",
    r"^\s*class\s+\w+",
    r"(a+)?",
    r"(a+){0,1}",
    r"import\s+(\w+)",
    r"\[(\w+)\]",
    r"(?:https?)://\S+",
    r"[\[\](){}]+",
    r"a{2}b{3}",
    r"(a)(b)(c)+",
    r"^#{1,3}\s+",
    r"x[abcd]y",
    r"(export\s+)?(class|interface)",
    r"error: code \d+",
    r"WARN: code \d+",
    r"[A-Z][a-z]+",
    r"^\s*(async\s+)?function\s+\w+",
]


class TestNestedQuantifierDetection:
    @pytest.mark.parametrize("pattern", EXPONENTIAL_PATTERNS)
    def test_flags_exponential_shapes(self, pattern: str) -> None:
        assert has_nested_quantifier(pattern) is True

    @pytest.mark.parametrize("pattern", ORDINARY_PATTERNS)
    def test_leaves_ordinary_patterns_alone(self, pattern: str) -> None:
        assert has_nested_quantifier(pattern) is False
        re.compile(pattern)  # the fixtures must be real regexes

    @pytest.mark.parametrize(
        "pattern",
        ["", "(", ")", "[", "\\", "a{", "a{,}", "a{x}", "(((", ")))", "[^]", "[]]", "\\("],
    )
    def test_never_raises_on_malformed_input(self, pattern: str) -> None:
        """It runs before re.compile, so it sees uncompilable patterns too."""
        assert has_nested_quantifier(pattern) in (True, False)

    def test_nested_groups_without_repetition_are_fine(self) -> None:
        """Nesting alone is harmless — it is the outer *repeat* that compounds."""
        assert has_nested_quantifier("(" * 200 + "a+" + ")" * 200) is False

    def test_escaped_metacharacters_are_literals(self) -> None:
        assert has_nested_quantifier(r"\(a\+\)\+") is False


class TestGrepRefusesUnboundedPatterns:
    async def test_exponential_pattern_is_rejected_immediately(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        target = temp_dir / "victim.txt"
        target.write_text("a" * 40 + "!\n")
        await smart_read(semantic_cache, str(target), force_full=True)

        with pytest.raises(ValueError, match="unsafe regex pattern"):
            await grep(semantic_cache._storage, r"(a+)+$")

    async def test_rejection_names_both_escape_hatches(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        target = temp_dir / "victim.txt"
        target.write_text("aaa!\n")
        await smart_read(semantic_cache, str(target), force_full=True)

        with pytest.raises(ValueError) as excinfo:
            await grep(semantic_cache._storage, r"(a+)+$")
        message = str(excinfo.value)
        assert "fixed_string=True" in message
        assert "repeat" in message

    async def test_the_safe_equivalent_still_works(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """`(a+)+` means the same as `a+`, so the guard costs no expressiveness."""
        target = temp_dir / "victim.txt"
        target.write_text("a" * 40 + "!\n")
        await smart_read(semantic_cache, str(target), force_full=True)

        results = await grep(semantic_cache._storage, r"a+!$")
        assert sum(len(r["matches"]) for r in results) == 1

    async def test_fixed_string_bypasses_the_guard(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """Literal mode compiles the pattern escaped, so it cannot backtrack."""
        target = temp_dir / "literal.txt"
        target.write_text("value = (a+)+$ here\n")
        await smart_read(semantic_cache, str(target), force_full=True)

        results = await grep(semantic_cache._storage, "(a+)+$", fixed_string=True)
        assert sum(len(r["matches"]) for r in results) == 1

    async def test_invalid_regex_still_reports_as_invalid(
        self, semantic_cache: SemanticCache, temp_dir: Path
    ) -> None:
        """The new guard must not swallow the existing bad-pattern error."""
        target = temp_dir / "any.txt"
        target.write_text("text\n")
        await smart_read(semantic_cache, str(target), force_full=True)

        with pytest.raises(ValueError, match="invalid regex pattern"):
            await grep(semantic_cache._storage, "[unterminated(")
