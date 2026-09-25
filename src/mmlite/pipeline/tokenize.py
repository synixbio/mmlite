"""MetaMapLite tokenization — port of ``metamap.prefix.Tokenize.mmPosTokenize`` (KEEP_WHITE_SPACE),
``Scanner.classifyToken`` / ``addOffsets`` and ``CharUtils.isPunct``.

Behaviour: a *word* is a maximal run of characters that are neither whitespace nor MetaMap
punctuation; every whitespace or punctuation character is its own one-character token.
Whitespace tokens are kept (``EntityLookup4`` joins token texts to rebuild the matched span),
and offsets are cumulative from the sentence start, so ``"".join(t.text for t in tokens)``
reproduces the input exactly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# CharUtils.isPunct
PUNCT = set("~!@#$%^&*()_+-=|\\<>?/,.`';:[]{}\"")

# Java Character.isWhitespace || Character.isSpaceChar.  Python str.isspace() is the same set
# plus U+0085 (NEL), which Java treats as a control character, not whitespace.
JAVA_WHITESPACE = frozenset(chr(c) for c in range(0x10000) if chr(c).isspace() and c != 0x85)

_SEP_CLASS = "".join(re.escape(c) for c in sorted(PUNCT | JAVA_WHITESPACE))
# One token per match: a word run, or a single separator character.
_TOKEN_RE = re.compile(rf"[^{_SEP_CLASS}]+|[{_SEP_CLASS}]")

# Scanner patterns, in the order Scanner.classifyToken tests them.  Java's \s and [A-Za-z]
# are ASCII-only; we mirror that.
_CLASS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ws", re.compile(r"^[ \t\n\x0b\f\r]$")),
    ("an", re.compile(r"^[A-Za-z]+[0-9]+$")),
    ("uc", re.compile(r"^[A-Z][A-Z0-9]+$")),
    ("lc", re.compile(r"^[a-z]+$")),
    ("ic", re.compile(r"^[A-Za-z]+$")),
    ("nu", re.compile(r"^[0-9]+$")),
    ("gr", re.compile(r"^[Ͱ-Ͽ]+$")),
    ("op", re.compile(r"^\($")),
    ("cp", re.compile(r"^\)$")),
    ("ob", re.compile(r"^\[$")),
    ("cb", re.compile(r"^\]$")),
    ("cm", re.compile(r"^,$")),
    ("pd", re.compile(r"^\.$")),
    ("pn", re.compile(r"^[()!@#$%^&*+=\-_\[\]{}.,?/']+$")),
)


@dataclass
class Token:
    """An ``ERToken``: text, absolute character offset, Scanner class, and (later) POS tag."""

    text: str
    start: int
    token_class: str
    pos: str = ""  # "" = not tagged (EntityLookup4 accepts it); "WS" for whitespace tokens
    extra: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def end(self) -> int:
        return self.start + len(self.text)

    @property
    def is_whitespace(self) -> bool:
        return self.token_class == "ws"


def is_punct(ch: str) -> bool:
    return ch in PUNCT


def is_java_space(ch: str) -> bool:
    return ch in JAVA_WHITESPACE


def classify(text: str) -> str:
    """Scanner.classifyToken."""
    for name, pat in _CLASS_PATTERNS:
        if pat.match(text):
            return name
    return "unknown"


def tokenize(text: str, base_offset: int = 0) -> list[Token]:
    """Scanner.analyzeText: tokenize, classify, and attach cumulative offsets."""
    tokens: list[Token] = []
    pos = base_offset
    for m in _TOKEN_RE.finditer(text):
        s = m.group(0)
        tokens.append(Token(s, pos, classify(s)))
        pos += len(s)
    return tokens


def tokens_text(tokens: list[Token]) -> str:
    """Tokenize.getTextFromTokenList: concatenation of token texts (whitespace included)."""
    return "".join(t.text for t in tokens)


def remove_whitespace_tokens(tokens: list[Token]) -> list[Token]:
    """Scanner.removeWhiteSpaceTokens."""
    return [t for t in tokens if not t.is_whitespace]
