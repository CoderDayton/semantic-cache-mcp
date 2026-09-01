"""The form a content hash takes on the wire, and the rule for accepting one.

A possession claim is only ever checked against the cache entry for the path it
names, so the hash has to separate two *versions of one file* — not every file
the server has ever seen. Sixteen hex characters (64 bits) does that with room
to spare, and saves ~14 tokens on every hash delivered: ~500 on a 30-file
`batch_read`, plus ~24 per ranged read inside a `coverage_token`.

Accepting a prefix is where this could go wrong. A rule of "the stored hash
starts with what you sent" would accept a single character and match every
version of the file, handing out `unchanged` for content the caller never held.
So acceptance is exact-length: the wire form or the whole digest, hex only,
nothing in between.
"""

from __future__ import annotations

import hmac
from typing import Final

# Hex characters of a content hash put on the wire. 64 bits: two versions of
# one file colliding is not a scenario worth defending against, and a wrong
# answer would cost a re-read rather than corrupt anything.
WIRE_HASH_LENGTH: Final = 16

_HEX_DIGITS: Final = frozenset("0123456789abcdef")


def short_hash(full_hash: str) -> str:
    """Return the wire form of *full_hash* (idempotent on an already-short one)."""
    return full_hash[:WIRE_HASH_LENGTH]


def _is_hex(value: str) -> bool:
    return bool(value) and all(char in _HEX_DIGITS for char in value)


def hash_matches(claimed: str | None, stored: str | None) -> bool:
    """True when *claimed* is the caller echoing *stored* back.

    Accepts the wire form (`WIRE_HASH_LENGTH` hex characters) or the full
    digest, and only those two lengths. Anything else — a shorter prefix, a
    `partial:` file hash, an empty string, uppercase, a non-hex string — is
    refused, so a malformed or fabricated claim costs the caller a re-read and
    never buys an `unchanged` it did not earn.
    """
    if not claimed or not stored:
        return False
    if not _is_hex(claimed) or not _is_hex(stored):
        return False
    if len(claimed) == len(stored):
        return hmac.compare_digest(claimed, stored)
    if len(claimed) == WIRE_HASH_LENGTH and len(stored) > WIRE_HASH_LENGTH:
        return hmac.compare_digest(claimed, stored[:WIRE_HASH_LENGTH])
    return False


__all__ = ["WIRE_HASH_LENGTH", "hash_matches", "short_hash"]
