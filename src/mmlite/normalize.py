"""String normalization — a faithful port of what MetaMapLite actually does.

Ported from (metamaplite master):
* ``gov.nih.nlm.nls.metamap.lite.Normalization.normalizeUtf8AsciiString`` — the query-side
  normalizer used by ``NormalizedStringCache`` / ``EntityLookup4``;
* ``gov.nih.nlm.nls.nlp.nlsstrings.NLSStrings.removeLeftParentheticals / stripPossessives /
  removeExtraBlanks``;
* ``gov.nih.nlm.nls.nlp.nlsstrings.MetamapTokenization.removePossessives / TOKEN_DELIMITERS``;
* ``irutils.MappedMultiKeyIndexGeneration`` — the index key is simply ``STR.toLowerCase()``.

MetaMapLite does **no** uninflection, punctuation stripping or word reordering: it relies on
MRCONSO containing the variant strings.  At lookup time ``EntityLookup4`` queries the index
with both the original span and ``normalize(span)``, and ``MappedMultiKeyIndexLookup.lookup``
lower-cases every query, so the effective keys are ``span.lower()`` and ``normalize(span)``;
see :func:`lookup_keys`.

``NORM_VERSION`` is stamped into the index so a mismatch between builder and code is detected.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .greek import greek_to_ascii

NORM_VERSION = "mml-1"  # bump when index_key() or normalize() changes

# NLSStrings.left_parenthetical — stripped only when the string *starts* with one of them
LEFT_PARENTHETICALS = ("[X]", "[V]", "[D]", "[M]", "[EDTA]", "[SO]", "[Q]")

# MetamapTokenization.TOKEN_DELIMITERS = " \t\n\r\f$|~"
_TOKEN_DELIM_RE = re.compile(r"[ \t\n\r\f$|~]+")


def index_key(term: str) -> str:
    """Key under which a MRCONSO string is stored (MappedMultiKeyIndexGeneration: toLowerCase)."""
    return term.lower()


def remove_left_parentheticals(s: str) -> str:
    """NLSStrings.removeLeftParentheticals: strip a leading tag, else collapse runs of blanks."""
    for tag in LEFT_PARENTHETICALS:
        if s.startswith(tag):
            return s[len(tag) :]
    return remove_extra_blanks(s)


def remove_extra_blanks(s: str) -> str:
    """NLSStrings.removeExtraBlanks: tokenize on ' ' and rejoin.

    Java appends a trailing blank; it is irrelevant because :func:`strip_possessives`
    re-tokenizes afterwards, so we do not reproduce it.
    """
    return " ".join(t for t in s.split(" ") if t)


def remove_possessives(token: str) -> str:
    """MetamapTokenization.removePossessives, applied to a single token."""
    n = len(token)
    pos = token.rfind("'s")
    if pos >= 0 and pos == n - 2:
        if pos - 1 >= 0 and token[pos - 1].isalnum():
            return token[:pos]
        return token
    pos = token.rfind("'")
    if pos >= 0 and pos == n - 1 and pos != 0 and token[pos - 1] == "s":
        return token[:pos]
    return token


def strip_possessives(s: str) -> str:
    """NLSStrings.stripPossessives: split on TOKEN_DELIMITERS, strip each, rejoin with ' '."""
    return " ".join(remove_possessives(t) for t in _TOKEN_DELIM_RE.split(s) if t)


@lru_cache(maxsize=200_000)
def normalize(text: str) -> str:
    """Normalization.normalizeUtf8AsciiString.

    Greek to ASCII, left parentheticals, lower-case, then possessives.
    """
    s = greek_to_ascii(text)
    s = remove_left_parentheticals(s)
    s = s.lower()
    return strip_possessives(s)


def lookup_keys(text: str) -> tuple[str, ...]:
    """The index keys EntityLookup4 tries for a span: ``lower(span)`` and ``normalize(span)``."""
    low = index_key(text)
    norm = normalize(text)
    return (low,) if norm == low else (low, norm)
