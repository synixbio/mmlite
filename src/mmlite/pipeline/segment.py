"""Text segmentation — port of ``MetaMapLite.processPassage``'s segmentation switch.

Methods (``metamaplite.segmentation.method``):
* ``SENTENCES`` (default): a sentence detector.  Java uses the OpenNLP maximum-entropy model
  ``en-sent.bin``; we provide a rule-based default (:class:`RegexSentenceDetector`) and an
  optional spaCy-backed one (:class:`mmlite.pipeline.spacy_nlp.SpacySentenceDetector`).
  Boundary agreement with Java is measured in ``tests/parity/README.md``.
* ``BLANKLINES``: split on ``"\\n\\n"``.
* ``LINES``: split on ``"\\n"``, dropping blank lines.

Offsets: Java computes segment offsets with ``indexOf`` and an ``offset = segment.length()``
slip that only matters for repeated segments; we track exact offsets instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

SEGMENTATION_METHODS = ("SENTENCES", "BLANKLINES", "LINES")


@dataclass(frozen=True)
class Sentence:
    text: str
    offset: int  # absolute character offset of ``text`` in the document

    @property
    def end(self) -> int:
        return self.offset + len(self.text)


class SentenceDetector(Protocol):
    def spans(self, text: str) -> list[tuple[int, int]]:
        """Return (start, end) character spans of sentences in ``text``."""
        ...


# Tokens that end with a period but do not end a sentence (compared lower-cased, period stripped).
_ABBREVIATIONS = frozenset(
    [
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "cf",
        "fig",
        "figs",
        "eq",
        "eqs",
        "ref",
        "refs",
        "no",
        "nos",
        "vol",
        "approx",
        "dept",
        "univ",
        "inc",
        "ltd",
        "co",
        "corp",
        "al",
        "ca",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "sat",
        "sun",
    ]
)

# Candidate boundary: terminator(s), optional closing quote/bracket, whitespace, then an
# upper-case letter, digit, or opening quote/bracket.
_BOUNDARY_RE = re.compile(r"""([.?!]+)(["')\]]*)(\s+)(?=["'(\[]*[A-Z0-9])""")
_WORD_BEFORE_RE = re.compile(r"(\S+)$")


class RegexSentenceDetector:
    """Rule-based sentence splitter with an abbreviation list and single-letter-initial guard."""

    def __init__(self, abbreviations: frozenset[str] = _ABBREVIATIONS):
        self.abbreviations = abbreviations

    def spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start = 0
        for m in _BOUNDARY_RE.finditer(text):
            if m.group(1)[0] == ".":
                before = _WORD_BEFORE_RE.search(text, 0, m.start(1))
                word = before.group(1).lower() if before else ""
                if word in self.abbreviations or (len(word) == 1 and word.isalpha()):
                    continue
            end = m.end(2)
            spans.append((start, end))
            start = m.end(3)
        if start < len(text):
            spans.append((start, len(text)))
        return _trim(text, spans)


def _trim(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            out.append((s, e))
    return out


def segment(
    text: str,
    method: str = "SENTENCES",
    base_offset: int = 0,
    detector: SentenceDetector | None = None,
) -> list[Sentence]:
    method = method.upper()
    if method == "SENTENCES":
        det = detector or RegexSentenceDetector()
        return [Sentence(text[s:e], base_offset + s) for s, e in det.spans(text)]
    if method == "BLANKLINES":
        return _split_on(text, _BLANKLINE_RE, base_offset, skip_blank=False)
    if method == "LINES":
        return _split_on(text, _LINE_RE, base_offset, skip_blank=True)
    raise ValueError(
        f"unknown segmentation method {method!r}; expected one of {SEGMENTATION_METHODS}"
    )


# Java splits on the literal "\n\n" / "\n"; matching CR too keeps CRLF and old-Mac files from
# collapsing into one segment (and off-by-one offsets from a trailing "\r" in every line).
_LINE_RE = re.compile(r"\r\n|\r|\n")
# Atomic, so one CRLF is never re-read as a CR plus an LF and counted as two line breaks.
_BLANKLINE_RE = re.compile(r"(?>\r\n|\r|\n){2,}")


def _split_on(
    text: str, sep: re.Pattern[str], base_offset: int, skip_blank: bool
) -> list[Sentence]:
    out: list[Sentence] = []
    pos = 0
    for m in sep.finditer(text):
        piece = text[pos : m.start()]
        if piece and (not skip_blank or piece.strip()):
            out.append(Sentence(piece, base_offset + pos))
        pos = m.end()
    piece = text[pos:]
    if piece and (not skip_blank or piece.strip()):
        out.append(Sentence(piece, base_offset + pos))
    return out
