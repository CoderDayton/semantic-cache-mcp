"""Caller-carried proof of possession for ranged reads.

A ranged read shows the caller one window of a file, so it earns no claimable
`content_hash`: nobody can vouch for bytes they were never sent. But the caller
*can* vouch for the window it was shown, and that claim is worth serving — a
re-read of the same window has nothing left to send.

A coverage token carries exactly that claim. It names the file version the
windows came from and the line spans delivered so far, and it is signed so the
server can tell a token it minted from one that was invented. The caller echoes
it back as `known_hash` on the next ranged read; the server widens the coverage
with the newly delivered window and returns a fresh token. Once the accumulated
spans reach every line, the caller has been shown the whole file and the reply
upgrades to a real `content_hash`.

Two properties this module exists to guarantee:

- Coverage is carried by the caller, never tracked server-side. Server-side
  accumulation would let a compaction between two windows go unnoticed and
  certify possession of bytes the caller had already dropped — which is the
  failure the whole possession design exists to prevent.
- The signing key is generated per process and never persisted. A restarted
  worker rejects every token its predecessor minted, costing a re-read and
  never a false claim. Every failure mode here degrades to "send the bytes".
"""

from __future__ import annotations

import hmac
import secrets
from dataclasses import dataclass
from typing import Final

from ..core.hashing import KEYED_HASH_KEY_SIZE, keyed_hash

# Wire format: "<version>:<content hash>:<spans>:<mac>". The separator never
# appears inside a field — the hash is hex, and spans are digits, '-' and ','.
_FIELD_SEPARATOR: Final = ":"
_SPAN_SEPARATOR: Final = "-"
_RANGE_SEPARATOR: Final = ","
_TOKEN_VERSION: Final = "pcov1"
# version, content hash, spans — the MAC is split off before the count is checked.
_TOKEN_FIELD_COUNT: Final = 3

# 64-bit tag. The threat here is a fabricated or mangled token, not an attacker
# with oracle access, and every rejection costs only a re-read; 8 bytes keeps
# the token short while making an accidental match vanishingly unlikely.
_MAC_BYTES: Final = 8

# Ceiling on tracked spans, so a caller reading many scattered windows cannot
# grow the token without bound. Overflow drops spans, which only ever
# under-claims coverage (see `merge`).
MAX_TRACKED_SPANS: Final = 16

# Per-process signing key. Never persisted, never logged: a token is only
# meaningful to the process that minted it.
_SIGNING_KEY: Final = secrets.token_bytes(KEYED_HASH_KEY_SIZE)


def _sign(body: str) -> str:
    """Keyed MAC over a token body.

    Which primitive signs is `core.hashing`'s business — BLAKE3 where the
    wheel is present, BLAKE2b otherwise — and it does not need to be settled
    here: a token is only ever verified by the process that minted it, so both
    sides of a comparison always agree.
    """
    return keyed_hash(body.encode("utf-8"), _SIGNING_KEY, _MAC_BYTES)


@dataclass(frozen=True, slots=True)
class LineSpans:
    """Merged, sorted, half-open ``[start, end)`` line spans, 0-based.

    Kept normalized on every construction path: sorted, non-overlapping, and
    coalesced across touching spans, so a window that is covered at all is
    covered by exactly one span.
    """

    spans: tuple[tuple[int, int], ...] = ()

    def merge(self, start: int, end: int) -> LineSpans:
        """Return coverage widened by the ``[start, end)`` window."""
        if end <= start:
            return self
        merged: list[tuple[int, int]] = []
        for span_start, span_end in sorted((*self.spans, (start, end))):
            if merged and span_start <= merged[-1][1]:
                previous_start, previous_end = merged[-1]
                merged[-1] = (previous_start, max(previous_end, span_end))
            else:
                merged.append((span_start, span_end))
        if len(merged) > MAX_TRACKED_SPANS:
            # Forgetting a span only ever under-claims: the caller is asked to
            # re-read a window it actually still holds, never credited with one
            # it does not. Keep the widest, which are the ones worth not
            # re-sending.
            widest = sorted(merged, key=lambda span: span[1] - span[0], reverse=True)
            merged = sorted(widest[:MAX_TRACKED_SPANS])
        return LineSpans(tuple(merged))

    def covers(self, start: int, end: int) -> bool:
        """True when every line of ``[start, end)`` has been delivered."""
        if end <= start:
            return True
        return any(span_start <= start and end <= span_end for span_start, span_end in self.spans)

    def covers_all(self, total_lines: int) -> bool:
        """True when the spans account for every line in the file."""
        return total_lines <= 0 or self.covers(0, total_lines)

    def encode(self) -> str:
        return _RANGE_SEPARATOR.join(f"{start}{_SPAN_SEPARATOR}{end}" for start, end in self.spans)

    @classmethod
    def decode(cls, text: str) -> LineSpans | None:
        """Parse an encoded span list, or ``None`` if it is malformed."""
        if not text:
            return cls()
        spans: list[tuple[int, int]] = []
        for raw_span in text.split(_RANGE_SEPARATOR):
            raw_start, separator, raw_end = raw_span.partition(_SPAN_SEPARATOR)
            if not separator:
                return None
            try:
                start, end = int(raw_start), int(raw_end)
            except ValueError:
                return None
            if start < 0 or end <= start:
                return None
            spans.append((start, end))
        return cls(tuple(spans))


EMPTY_SPANS: Final = LineSpans()


def encode_coverage_token(content_hash: str, spans: LineSpans) -> str:
    """Mint a signed token asserting *spans* were delivered from *content_hash*."""
    body = _FIELD_SEPARATOR.join((_TOKEN_VERSION, content_hash, spans.encode()))
    return f"{body}{_FIELD_SEPARATOR}{_sign(body)}"


def decode_coverage_token(token: str | None) -> tuple[str, LineSpans] | None:
    """Return ``(content_hash, spans)`` for a token this process signed.

    ``None`` for anything else, which deliberately includes every other value a
    caller might pass as `known_hash`: a bare `content_hash`, a `partial:`
    file hash, a token from a previous process, or an invented string. A
    rejected token claims nothing, so the read simply sends the window.
    """
    if not token:
        return None
    body, separator, mac = token.rpartition(_FIELD_SEPARATOR)
    if not separator or not hmac.compare_digest(_sign(body), mac):
        return None
    fields = body.split(_FIELD_SEPARATOR)
    if len(fields) != _TOKEN_FIELD_COUNT:
        return None
    version, content_hash, raw_spans = fields
    if version != _TOKEN_VERSION or not content_hash:
        return None
    spans = LineSpans.decode(raw_spans)
    if spans is None:
        return None
    return content_hash, spans
