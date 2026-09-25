"""PubMed/MEDLINE loaders — ports of ``metamap.document.PubMedXMLDocument`` (``pubmed``) and
``MedlineDocument`` (``medline``).

**Passage offsets.** Every passage here is at offset 0, including Medline's, although Java's
``MedlineDocument`` computes ``len(PMID) + 13`` for the title and ``len(PMID) + len(title) + 19``
for the abstract.  Java never reports those offsets: in its default ``SENTENCES`` segmentation
``OpenNLPSentenceExtractor.createSentences`` overwrites the passage offset with each sentence's
position in the passage, so entity offsets come out passage-relative; and in ``LINES`` /
``BLANKLINES`` it passes the passage offset to ``text.indexOf(segment, offset)`` over
passage-relative text, which returns -1 and yields **negative** entity offsets.  Both verified
against MetaMapLite 3.6.2rc8 on a Medline file.  Offset 0 reproduces what Java's default mode
actually prints; DEVELOPMENT_PLAN §16 has the detail.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from .corpora import title_abstract_document
from .model import Document, Passage
from .singleline import java_read_lines

log = logging.getLogger(__name__)


def _local(tag: str) -> str:
    """``XMLStreamReader.getLocalName``: the tag without any ``{namespace}`` prefix."""
    return tag.rsplit("}", 1)[-1]


def element_text(elem: ET.Element) -> str:
    """All text inside ``elem``, inline markup included.

    Java calls ``XMLStreamReader.getElementText()``, which throws on any child element — so a
    title or abstract containing ``<i>``, ``<sup>`` or ``<sub>`` (routine in real PubMed records)
    aborts the entire run with ``elementGetText() function expects text only elment``.  Verified.
    For markup-free elements, which are the only ones Java can process at all, this returns
    exactly what Java does.
    """
    return "".join(elem.itertext())


class PubMedXMLLoader:
    """``pubmed`` — a ``PubmedArticleSet`` file, one document per ``PubmedArticle``.

    A streaming port of ``readAsBioCDocumentList``, and like it, matches element names anywhere
    in the tree.  Two consequences are reproduced because they shape output:

    * Every ``PMID`` element sets the document's id, including the ones inside
      ``CommentsCorrections`` that follow the abstract — so ``Document.id`` ends up as the *last*
      PMID in the article.  Passages capture the id current *when they are created*, though, and
      the title and abstract come before those references, so ``Entity.docid`` is the article's
      own PMID.  Verified: Java reports ``11111111``, not the trailing cited ``99999999``.
    * Each ``AbstractText`` is its own ``abstract`` passage, so a structured abstract
      (``BACKGROUND``, ``RESULTS``, ...) becomes several passages, each at offset 0.

    Content before the first ``PubmedArticle`` goes into a document Java never adds to its list,
    so it is dropped; a ``PubmedBookArticle`` is not a ``PubmedArticle`` and its passages join
    the preceding article's document.  Both follow from the same streaming state and are kept.
    """

    name = "pubmed"

    def read(self, text: str, docid: str | None = None) -> list[Document]:
        return self._parse(text.encode("utf-8"), "<input>")

    def load_file(
        self,
        path: Path | str,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
        docid: str | None = None,
    ) -> list[Document]:
        # Bytes, so the XML declaration decides the encoding.  Java's FileReader uses the JVM's
        # default charset instead -- UTF-8 on Java 18+, the Windows code page before that.
        return self._parse(Path(path).read_bytes(), str(path))

    def _parse(self, data: bytes, source: str) -> list[Document]:
        import io

        documents: list[Document] = []
        current = Document()  # Java's initial document, which is never added to the list
        try:
            for event, elem in ET.iterparse(io.BytesIO(data), events=("start", "end")):
                name = _local(elem.tag)
                if event == "start":
                    if name == "PubmedArticle":
                        current = Document()
                        documents.append(current)
                    continue
                if name == "PMID":
                    current.id = element_text(elem)
                elif name in ("ArticleTitle", "AbstractText"):
                    section = "title" if name == "ArticleTitle" else "abstract"
                    current.passages.append(
                        Passage(
                            text=element_text(elem),
                            offset=0,
                            # Java writes doc.getID() as it stands right now.  Before any PMID
                            # it is null (Java then reports "00000000"; here the document id is
                            # filled in instead) -- unreachable in real PubMed XML, where PMID
                            # is the first child of MedlineCitation.
                            infons={"section": section}
                            | ({"docid": current.id} if current.id is not None else {}),
                        )
                    )
                elif name == "PubmedArticle":
                    elem.clear()  # keep memory flat on large PubMed baseline files
        except ET.ParseError as e:
            raise ValueError(f"{source}: not well-formed XML: {e}") from None
        return documents


def java_trim(s: str) -> str:
    """``String.trim()``: strips characters at or below U+0020 — not Unicode whitespace."""
    start, end = 0, len(s)
    while start < end and s[start] <= " ":
        start += 1
    while end > start and s[end - 1] <= " ":
        end -= 1
    return s[start:end]


class MedlineLoader:
    """``medline`` — MEDLINE/PubMed text format (``.nbib``): ``PMID- ``, ``TI  - ``, ``AB  - ``.

    A port of ``readAsBioCDocumentList``:

    * a field is ``line[0:4]`` (the tag) and ``line[6:]`` (the value); a line whose tag is blank
      continues the previous field, so multi-line titles and abstracts are joined, each line
      followed by a space — which leaves a trailing space on both;
    * only ``PMID``, ``TI`` and ``AB`` are kept, as sections ``TI`` and ``AB`` — and ``TI`` is one
      of the two values MMI ranks as a title;
    * a blank line ends a record, resetting the title and abstract but **not** the PMID, and one
      more record is always emitted at end of input.  So a file ending in a blank line yields a
      trailing empty document that repeats the last PMID; it has no text and contributes nothing.

    Java's ``line.substring(0, 4)`` / ``substring(6)`` throw on a line shorter than that, aborting
    the run; slicing here simply yields a shorter tag or an empty value.
    """

    name = "medline"

    def read(self, text: str, docid: str | None = None) -> list[Document]:
        documents: list[Document] = []
        pmid = ""
        title: list[str] = []
        abstract: list[str] = []
        key = ""
        for line in java_read_lines(text):
            if not java_trim(line):
                documents.append(self._document(pmid, title, abstract))
                title, abstract = [], []
                continue
            header, content = line[0:4], line[6:]
            if java_trim(header):
                key = java_trim(header)
            if key == "PMID":
                pmid = content
            elif key == "TI":
                title.append(content + " ")
            elif key == "AB":
                abstract.append(content + " ")
        documents.append(self._document(pmid, title, abstract))
        return documents

    def load_file(
        self,
        path: Path | str,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
        docid: str | None = None,
    ) -> list[Document]:
        from ..api import read_document

        return self.read(read_document(Path(path), encoding, errors))

    def _document(self, pmid: str, title: list[str], abstract: list[str]) -> Document:
        document = title_abstract_document(pmid, "".join(title), "".join(abstract), self.name)
        for passage, section in zip(document.passages, ("TI", "AB"), strict=True):
            passage.infons["section"] = section
            del passage.infons["inputformat"]  # MedlineDocument does not set one
        return document
