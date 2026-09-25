"""Abbreviations — ports of ``bioc.tool.ExtractAbbrev`` (Schwartz & Hearst 2003),
``metamap.lite.MarkAbbreviations`` and ``metamap.lite.UserDefinedAcronym``.

MetaMapLite's main path runs the extractor on each sentence, then for every entity whose
matched text equals a detected *long form* it copies the entity onto the short form's
definition site and onto every token in the passage equal to the short form
(``MarkAbbreviations.markAbbreviations`` / ``findMatches``).  The copies share the original
``Ev`` objects, so their concept evidence keeps the long form's offsets — reproduced as is.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from ..normalize import normalize
from ..types import ENTITY_DEFAULTS, ConceptInfo, Entity, Ev
from .tokenize import Token, tokenize

log = logging.getLogger(__name__)

# --- Schwartz-Hearst ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AbbrInfo:
    short_form: str
    short_form_index: int  # absolute char offset
    long_form: str
    long_form_index: int


def _has_letter(s: str) -> bool:
    return any(c.isalpha() for c in s)


def _has_capital(s: str) -> bool:
    return any(c.isupper() for c in s)


def _is_valid_short_form(s: str) -> bool:
    return _has_letter(s) and (s[0].isalnum() or s[0] == "(")


def find_best_long_form(short_form: str, long_form: str) -> int | None:
    """Return the start index (within ``long_form``) of the best long form, or None."""
    s_index = len(short_form) - 1
    l_index = len(long_form) - 1
    while s_index >= 0:
        ch = short_form[s_index].lower()
        if not ch.isalnum():
            s_index -= 1
            continue
        while (l_index >= 0 and long_form[l_index].lower() != ch) or (
            s_index == 0 and l_index > 0 and long_form[l_index - 1].isalnum()
        ):
            l_index -= 1
        if l_index < 0:
            return None
        l_index -= 1
        s_index -= 1
    return long_form.rfind(" ", 0, max(l_index + 1, 0)) + 1


def _accept_pair(short_form: str, best_long_form: str) -> bool:
    if len(short_form) == 1:
        return False
    long_size = len([t for t in best_long_form.replace("-", " ").split() if t])
    short_size = len(short_form)
    for c in short_form:
        if not c.isalnum():
            short_size -= 1
    return not (
        len(best_long_form) < len(short_form)
        or (short_form + " ") in best_long_form
        or best_long_form.endswith(short_form)
        or long_size > short_size * 2
        or long_size > short_size + 5
        or short_size > 10
    )


def extract_abbr_pairs(text: str, base_offset: int = 0) -> list[AbbrInfo]:
    """Schwartz-Hearst abbreviation pairs in ``text`` with absolute offsets."""
    out: list[AbbrInfo] = []
    cur = text
    consumed = 0  # chars of ``text`` dropped from the front of ``cur``
    while (open_i := cur.find(" (")) > -1:
        close_i = cur.find(")", open_i)
        short_form = long_form = ""
        if close_i > -1:
            long_form = cur[:open_i]
            short_form = cur[open_i + 2 : close_i]
        if short_form or long_form:
            if short_form and long_form and len(short_form) > 1 and len(long_form) > 1:
                if "(" in short_form:
                    new_close = cur.find(")", close_i + 1)
                    if new_close > -1:
                        short_form = cur[open_i + 2 : new_close]
                        close_i = new_close
                for sep in (", ", "; "):
                    k = short_form.find(sep)
                    if k > -1:
                        short_form = short_form[:k]
                short_start = open_i + 2
                long_start = 0
                if len(short_form.split()) > 2 or len(short_form) > len(long_form):
                    # long form is inside the parentheses; short form is the word before them
                    tmp = cur.rfind(" ", 0, open_i)
                    tmp_str = cur[tmp + 1 : open_i]
                    long_form, long_start = short_form, short_start
                    short_form, short_start = tmp_str, tmp + 1
                    if not _has_capital(short_form):
                        short_form = ""
                if short_form and _is_valid_short_form(short_form):
                    sf = short_form.strip()
                    lf = long_form.strip()
                    sf_start = short_start + (len(short_form) - len(short_form.lstrip()))
                    lf_start = long_start + (len(long_form) - len(long_form.lstrip()))
                    best = find_best_long_form(sf, lf)
                    if best is not None:
                        best_lf = lf[best:]
                        if _accept_pair(sf, best_lf):
                            out.append(
                                AbbrInfo(
                                    sf,
                                    base_offset + consumed + sf_start,
                                    best_lf,
                                    base_offset + consumed + lf_start + best,
                                )
                            )
            consumed += close_i + 1
            cur = cur[close_i + 1 :]
        else:
            break
    return out


# --- MarkAbbreviations -------------------------------------------------------------------------


def _copy_at(entity: Entity, text: str, start: int) -> Entity:
    """``new Entity(entity)``: copies docid, fieldid, lexical category, score and the Ev set
    (sharing the Ev objects); sentence number, negation and location fall back to defaults."""
    e = copy.copy(entity)
    e.evs = list(entity.evs)
    e.text = text
    e.start = start
    e.length = len(text)
    e.sentence_number = ENTITY_DEFAULTS["sentence_number"]
    e.negated = False
    e.location_position = 0
    return e


def token_index(text: str, base_offset: int = 0) -> dict[str, list[Token]]:
    """Tokens of ``text`` grouped by their text, each group in document order."""
    index: dict[str, list[Token]] = {}
    for t in tokenize(text, base_offset):
        index.setdefault(t.text, []).append(t)
    return index


def find_matches(
    text: str,
    entity: Entity,
    base_offset: int = 0,
    index: dict[str, list[Token]] | None = None,
) -> list[Entity]:
    """MarkAbbreviations.findMatches: a copy of ``entity`` at every token equal to its text.

    Java re-tokenizes the whole passage on every call. That is the same token list every time, so
    callers matching many entities against one passage pass ``index`` (from :func:`token_index`)
    and pay for the tokenization once; the result is identical either way.
    """
    if index is None:
        index = token_index(text, base_offset)
    return [_copy_at(entity, t.text, t.start) for t in index.get(entity.text, ())]


def mark_abbreviations(
    text: str, abbr_infos: Iterable[AbbrInfo], entities: Iterable[Entity], base_offset: int = 0
) -> list[Entity]:
    """MarkAbbreviations.markAbbreviations (BioC variant): new entities for abbreviated forms.

    ``abbr_infos`` carry absolute offsets; ``text`` is the passage text starting at
    ``base_offset``.  Returns only the *added* entities (callers union them with the input).
    """
    long_to_short: dict[str, str] = {}
    short_form_map: dict[str, list[AbbrInfo]] = {}
    for info in abbr_infos:
        short_form_map.setdefault(info.short_form, []).append(info)
        long_to_short.setdefault(info.long_form, info.short_form)
    added: list[Entity] = []
    if not long_to_short:
        return added
    # Built on first use: tokenizing the passage once instead of once per match was the single
    # largest cost in annotating a long note.
    index: dict[str, list[Token]] | None = None
    for entity in entities:
        short = long_to_short.get(entity.text)
        if short is None or short not in short_form_map:
            continue
        for info in short_form_map[short]:
            loc = info.short_form_index
            if loc <= 0 or not info.short_form:
                continue  # Java: location.getOffset() > 0
            begin = max(0, loc - base_offset)
            end = min(begin + len(info.short_form), len(text))
            if text[begin:end] != info.short_form:
                continue
            new = _copy_at(entity, info.short_form, loc)
            added.append(new)
            if index is None:
                index = token_index(text, base_offset)
            added.extend(find_matches(text, new, base_offset, index))
    return added


# --- User-defined acronyms ---------------------------------------------------------------------


@dataclass
class UserDefinedAcronym:
    short_form: str
    long_form: str
    concepts: list[ConceptInfo]


class UserDefinedAcronyms:
    """``metamaplite.uda.filename``: lines of ``acronym|long form``; the long form is looked up
    in the dictionary (verbatim lower-cased, else normalized) when the file is loaded."""

    def __init__(self, udas: Iterable[UserDefinedAcronym] = ()):
        self.map: dict[str, UserDefinedAcronym] = {u.short_form: u for u in udas}

    @classmethod
    def from_file(
        cls, path: Path | str, lookup_concepts: Callable[[str, str], list[ConceptInfo]]
    ) -> UserDefinedAcronyms:
        """``lookup_concepts(key, norm_term) -> list[ConceptInfo]`` (EntityLookup._lookup)."""
        path = Path(path)
        if not path.exists():
            log.warning("user-defined acronym file %s does not exist", path)
            return cls()
        udas = []
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("|")
            if len(fields) > 1 and fields[0].strip() and fields[1].strip():
                short, long_ = fields[0], fields[1]
                norm = normalize(long_)
                concepts = lookup_concepts(long_.lower(), norm) or lookup_concepts(norm, norm)
                if not concepts:
                    log.warning(
                        "user-defined acronym %r: long form %r is not in the index", short, long_
                    )
                udas.append(UserDefinedAcronym(short, long_, concepts))
        return cls(udas)

    def generate_entities(self, docid: str, tokens: list[Token]) -> list[Entity]:
        """UserDefinedAcronym.generateEntities: an entity for every token equal to an acronym."""
        out = []
        for t in tokens:
            uda = self.map.get(t.text)
            if uda is None:
                continue
            evs = [
                Ev(ci, uda.short_form, ci.concept_string, t.start, len(t.text), 100.0, "")
                for ci in uda.concepts
            ]
            # Java calls Entity(id="UDA", docid, text, start, length, score, evSet), leaving
            # fieldId/lexicalCategory/sentenceNumber at their field defaults.
            out.append(
                Entity(
                    docid,
                    ENTITY_DEFAULTS["fieldid"],
                    uda.short_form,
                    ENTITY_DEFAULTS["lexical_category"],
                    ENTITY_DEFAULTS["sentence_number"],
                    t.start,
                    len(t.text),
                    100.0,
                    evs,
                )
            )
        return out

    def __len__(self) -> int:
        return len(self.map)
