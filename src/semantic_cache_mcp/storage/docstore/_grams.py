"""Literal extraction and trigram queries for the grep index.

The index is an FTS5 ``trigram`` table that folds case and keeps positions,
so a phrase query is a substring test. A grep pattern maps to the literals
every match must contain, so the query returns a superset of the files the
pattern can match — never fewer.

Python's ``re.IGNORECASE`` equates a few non-ASCII letters with ASCII ones
that SQLite's fold keeps apart (dotless and dotted I with ``i``, long s with
``s``, the Kelvin sign with ``k``). :func:`prefold` maps those to their ASCII
partner on both sides, and a case-insensitive query uses ASCII text only:
with that, an exhaustive check over the BMP finds no ASCII pattern character
whose case-insensitive matches the index would miss.
"""

from __future__ import annotations

import re

GRAM_LEN = 3

_PREFOLD_MAP = {chr(0x131): "i", chr(0x130): "i", chr(0x17F): "s", chr(0x212A): "k"}
_PREFOLD = str.maketrans(_PREFOLD_MAP)
_ASCII_RUN = re.compile("[" + chr(0) + "-" + chr(0x7F) + "]+")

_OPTIONAL = frozenset("|?*{[")
_BREAKS = frozenset(".^$()+")
_CLASS_ESCAPES = frozenset("bBdDsSwWAZ")


def prefold(text: str) -> str:
    if text.isascii() or not any(ch in text for ch in _PREFOLD_MAP):
        return text
    return text.translate(_PREFOLD)


def required_literals(pattern: str, *, fixed_string: bool) -> list[str] | None:
    """Literal runs every match of *pattern* contains, or None.

    None means no run is provably required: the pattern has an optional part
    (``|?*{[``), or an escape (numeric, hex, backreference) this parser does
    not model. ``.^$()+`` and the class escapes ``\\b \\d \\s \\w`` and kin
    end a run but leave the runs around them required.
    """
    if fixed_string:
        return [pattern] if pattern else None
    if any(ch in _OPTIONAL for ch in pattern):
        return None
    literals: list[str] = []
    run: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "\\":
            escaped = pattern[index + 1 : index + 2]
            if not escaped:
                return None
            if escaped.isalnum():
                if escaped not in _CLASS_ESCAPES:
                    return None
                literals.append("".join(run))
                run = []
            else:
                run.append(escaped)
            index += 2
            continue
        if char in _BREAKS:
            literals.append("".join(run))
            run = []
        else:
            run.append(char)
        index += 1
    literals.append("".join(run))
    found = [literal for literal in literals if literal]
    return found or None


def gram_query(literals: list[str], *, case_sensitive: bool) -> str | None:
    """An FTS5 MATCH expression requiring every literal as a substring, or
    None when no literal is long enough to query.

    Each literal of at least ``GRAM_LEN`` characters becomes one phrase, which
    the trigram index answers as an exact substring test. A case-insensitive
    query keeps only the ASCII runs of each literal.
    """
    phrases: dict[str, None] = {}
    for literal in literals:
        folded = prefold(literal)
        for run in [folded] if case_sensitive else _ASCII_RUN.findall(folded):
            if len(run) >= GRAM_LEN:
                phrases[run] = None
    if not phrases:
        return None
    return " AND ".join('"' + phrase.replace('"', '""') + '"' for phrase in phrases)
