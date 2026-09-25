"""The document model — ``bioc.BioCDocument`` / ``bioc.BioCPassage`` as this port needs them.

MetaMapLite's input loaders all produce the same shape::

    BioCDocument{ id, infons, passages[ BioCPassage{ text, offset, infons } ] }

and ``MetaMapLite.processDocument`` walks its passages, annotating each with the passage's
``offset`` as the base for every offset it reports.  These dataclasses mirror those fields under
their BioC names — not an invented vocabulary — because the loaders, the ``infons`` maps and any
future BioC *output* formatter all speak BioC.

Two infons carry meaning to the annotator, exactly as in Java:

* ``docid``  — becomes ``Entity.docid``.  ``processDocument`` fills it in from the document id
  when a passage does not set one; a passage that sets it wins.
* ``section`` — becomes ``Entity.fieldid``, and MMI's "field" column.  It may legitimately be
  absent (``FreeText`` sets none when loading a *file*), leaving ``fieldid`` as ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# MetaMapLite.processDocument: used when a document's id is None or blank.  Seven zeros and an
# upper-case extension -- not a typo for FreeText's eight-zero "00000000.tx", which is a
# different default set by a different class.
DEFAULT_DOCUMENT_ID = "0000000.TXT"

# MetaMapLite.processPassage: used when a passage carries no "docid" infon.  Reachable only when
# a passage is annotated directly rather than through a document, since processDocument fills the
# infon in first.  Eight zeros, no extension -- a third distinct default.
DEFAULT_PASSAGE_DOCID = "00000000"


@dataclass
class Passage:
    """One annotatable region of a document (``BioCPassage``).

    ``offset`` is the position of ``text`` within the source the loader read, and becomes the
    base offset for entities found here, so ``entity.start`` indexes that source rather than this
    passage.  Loaders may produce passages whose offsets do not describe any single string —
    Java's ``PubMedXMLDocument`` gives both the title and the abstract offset 0 — so ``text`` is
    the only thing an offset is guaranteed to be relative to.  Hence :meth:`text_at`.
    """

    text: str
    offset: int = 0
    infons: dict[str, str] = field(default_factory=dict)

    @property
    def docid(self) -> str | None:
        return self.infons.get("docid")

    @property
    def section(self) -> str | None:
        """The ``section`` infon — ``Entity.fieldid``.  ``None`` when the loader sets none."""
        return self.infons.get("section")

    @property
    def end(self) -> int:
        return self.offset + len(self.text)

    def text_at(self, start: int, length: int) -> str:
        """The passage text under an absolute ``(start, length)``, as an entity reports them.

        ``passage.text_at(e.start, e.length) == e.text`` is the invariant every loader must
        satisfy, and the one check that catches the offset mistakes this port has made twice
        already (DEVELOPMENT_PLAN §6.6 and §6 finding 6).
        """
        lo = start - self.offset
        return self.text[lo : lo + length]


@dataclass
class Document:
    """A loaded input document (``BioCDocument``).

    Deliberately has no ``text`` property: with loaders that give several passages the same
    offset there is no single string that all of its entity offsets index.  Work per passage.
    """

    id: str | None = None
    passages: list[Passage] = field(default_factory=list)
    infons: dict[str, str] = field(default_factory=dict)

    def resolved_id(self) -> str:
        """``processDocument``: a missing or blank id becomes :data:`DEFAULT_DOCUMENT_ID`."""
        return self.id if self.id and self.id.strip() else DEFAULT_DOCUMENT_ID

    def with_docids(self) -> Document:
        """Fill in ``docid`` infons the way ``processDocument`` does, in place.

        The resolved id goes into the document's own infons unconditionally, and into each
        passage's only where that passage does not already carry one.
        """
        self.id = self.resolved_id()
        self.infons["docid"] = self.id
        for passage in self.passages:
            passage.infons.setdefault("docid", self.id)
        return self
