"""BioC XML loader — port of ``metamap.document.BioCDocumentLoaderImpl`` (``bioc``).

Java reads the file with the ``bioc`` library into ``BioCDocument`` objects and hands them to
``processDocument`` unchanged, so the BioC elements map onto :mod:`.model` directly:
``document/id`` → ``Document.id``, ``document/infon`` → ``Document.infons``, ``passage/infon`` →
``Passage.infons`` (so a ``docid`` or ``section`` infon in the file reaches ``Entity.docid`` /
``Entity.fieldid``, and a ``type`` infon does not).  Verified against MetaMapLite 3.6.2rc8; the
DTD declaration and the ``source``/``date``/``key`` header make no difference.

**Offsets.** Java ignores a passage's ``<offset>`` for its ``<text>`` — §16's finding, confirmed
again here: a passage at offset 100 reports its concepts from 0 — but a ``<sentence>`` element
already in the file is passed straight to the annotator **with its own offset**, and a passage
holding both text and sentences yields entities from both.  So a passage's text becomes a
:class:`Passage` at offset 0 and each of its sentences becomes a further ``Passage`` at the
sentence's offset, all carrying the passage's infons.  One difference follows: Java treats such a
sentence as a single sentence, whereas here it is re-segmented like any other passage text, so a
``<sentence>`` that contains several sentences may be negated or abbreviation-expanded slightly
differently.  Not observable on well-formed BioC.

No ``bioc`` dependency: the subset MetaMapLite consumes is a dozen element names, and
:mod:`xml.etree.ElementTree` reads it.  Annotations and relations already in the file are ignored,
as Java ignores them.
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from pathlib import Path

from .model import Document, Passage
from .pubmed import _local, element_text


def _infons(elem: ET.Element) -> dict[str, str]:
    return {
        i.get("key", ""): element_text(i) for i in elem if _local(i.tag) == "infon" and i.get("key")
    }


def _offset(elem: ET.Element) -> int:
    node = next((c for c in elem if _local(c.tag) == "offset"), None)
    try:
        return int(element_text(node).strip()) if node is not None else 0
    except ValueError:
        return 0


def _text(elem: ET.Element) -> str | None:
    node = next((c for c in elem if _local(c.tag) == "text"), None)
    return element_text(node) if node is not None else None


class BioCLoader:
    """``bioc`` — a BioC XML ``collection``, one document per ``document`` element."""

    name = "bioc"

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
        # Java's BioC loader reads with an explicit UTF-8 charset; bytes let the XML declaration
        # say so and agree with it.
        return self._parse(Path(path).read_bytes(), str(path))

    def _parse(self, data: bytes, source: str) -> list[Document]:
        try:
            root = ET.parse(io.BytesIO(data)).getroot()
        except ET.ParseError as e:
            raise ValueError(f"{source}: not well-formed XML: {e}") from None
        if _local(root.tag) == "document":
            elements = [root]  # a lone document rather than a collection; harmless to accept
        else:
            elements = [d for d in root if _local(d.tag) == "document"]
        return [self._document(d) for d in elements]

    def _document(self, elem: ET.Element) -> Document:
        id_node = next((c for c in elem if _local(c.tag) == "id"), None)
        document = Document(
            id=element_text(id_node) if id_node is not None else None, infons=_infons(elem)
        )
        for p in (c for c in elem if _local(c.tag) == "passage"):
            infons = _infons(p)
            text = _text(p)
            if text is not None:
                document.passages.append(Passage(text=text, offset=0, infons=dict(infons)))
            for s in (c for c in p if _local(c.tag) == "sentence"):
                stext = _text(s)
                if stext is not None:
                    document.passages.append(
                        Passage(text=stext, offset=_offset(s), infons={**infons, **_infons(s)})
                    )
        return document
