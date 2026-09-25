"""Result types — ports of ``metamap.lite.types.ConceptInfo``, ``Ev`` and ``Entity``.

Java field defaults that show up in output: ``Entity.fieldId = "TXT"``,
``lexicalCategory = "UNK"``, ``sentenceNumber = 1`` (see :data:`ENTITY_DEFAULTS`).

Identity rules mirror the Java ``equals``/``hashCode``:
* ``ConceptInfo``: by CUI;
* ``Ev``: by (start, length, CUI);
* ``Entity``: by (start, length) — a span; it holds one ``Ev`` per CUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


@dataclass(frozen=True, eq=False)
class ConceptInfo:
    cui: str
    preferred_name: str
    concept_string: str  # the dictionary string that matched (Java: docStr / normTerm for custom)
    # Every SAB the CUI appears in (CuiSourceSetIndex), in MRCONSO first-appearance order.  A
    # tuple rather than a frozenset because that order is what Java inserts into its HashSet,
    # and hence what decides the order sources print in (see IndexLookup.sources).
    sources: tuple[str, ...]
    semantic_types: frozenset[str]  # abbreviations, e.g. {"dsyn"}; order is not observable

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ConceptInfo) and other.cui == self.cui

    def __hash__(self) -> int:
        return hash(self.cui)


@dataclass(eq=False)
class Ev:
    """One concept evidence for a span (``Ev``)."""

    concept: ConceptInfo
    matched_text: str  # originalTerm
    norm_term: str
    start: int
    length: int
    # Always 0.0: EntityLookup4 never scores evidence, and neither does this port.  It is kept
    # because Java's JSON output carries it.  Not a confidence; see the mmi score instead.
    score: float = 0.0
    part_of_speech: str = ""

    @property
    def cui(self) -> str:
        return self.concept.cui

    @property
    def end(self) -> int:
        return self.start + self.length

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Ev)
            and other.start == self.start
            and other.length == self.length
            and other.concept == self.concept
        )

    def __hash__(self) -> int:
        return hash((self.start, self.length, self.concept.cui))


class _EntityDefaults(TypedDict):
    fieldid: str
    lexical_category: str
    sentence_number: int


ENTITY_DEFAULTS: _EntityDefaults = {
    "fieldid": "TXT",
    "lexical_category": "UNK",
    "sentence_number": 1,
}


@dataclass(eq=False)
class Entity:
    """A matched span with its candidate concepts (``Entity``)."""

    docid: str
    fieldid: str | None
    text: str  # matched text (originalTerm, whitespace included)
    lexical_category: str | None  # POS of the first token
    sentence_number: int  # Java: index of the 15-token window that produced the span (!)
    start: int
    length: int
    score: float = 0.0  # always 0.0, as on Ev: EntityLookup4 does not score entities
    evs: list[Ev] = field(default_factory=list)
    negated: bool = False
    # Set only by ConText (``--usecontext``); NegEx, the default detector, leaves them None.
    # Java's Entity has a temporality field and no experiencer one, so its ConText wrapper
    # computes the experiencer and throws it away; we keep it.
    temporality: str | None = None
    experiencer: str | None = None
    # ConText's full negation verdict: "Affirmed", "Negated" or "Possible".  ``negated`` is the
    # boolean Java keeps, which collapses Possible into false -- so "rule out pneumonia" and
    # "pneumonia" are indistinguishable there.  Java computes the verdict and discards it, as it
    # does the experiencer; this keeps it, without changing what ``negated`` means.
    assertion: str | None = None
    location_position: int = 0  # sentence index within the document

    @property
    def end(self) -> int:
        return self.start + self.length

    @property
    def cuis(self) -> list[str]:
        return [ev.cui for ev in self.evs]

    def add_evs(self, new_evs: list[Ev]) -> None:
        """EntityLookup4.addEvSetToSpanMap: merge, skipping CUIs already present."""
        have = {ev.cui for ev in self.evs}
        for ev in new_evs:
            if ev.cui not in have:
                self.evs.append(ev)
                have.add(ev.cui)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Entity) and other.start == self.start and other.length == self.length
        )

    def __hash__(self) -> int:
        return hash((self.start, self.length))

    def __repr__(self) -> str:
        return f"Entity({self.text!r}@{self.start}:{self.length} {self.cuis})"
