"""Free text loader — port of ``metamap.document.FreeText``.

Java has two document builders and they do **not** agree, which is visible in output:

``instantiateBioCDocument(text)`` — the stdin / ``--pipe`` path — sets ``docid``, ``inputformat``
and ``section="text"``, and ids the document ``00000000.tx``.

``instantiateBioCDocument(text, filename)`` — the file path — sets ``docid`` and ``inputformat``
but **no** ``section``, so a file-loaded document yields entities whose ``fieldid`` is ``None``.
Confirmed against MetaMapLite 3.6.2rc8: ``--outputformat=json`` on a file emits no ``fieldid``
key at all, ``org.json`` having dropped the null.

Both are reproduced.  DEVELOPMENT_PLAN §12 records why ``MetaMapLite.process_text`` and
``process_file`` still pass ``fieldid="text"`` rather than routing through this loader.
"""

from __future__ import annotations

from pathlib import Path

from .model import Document, Passage

# The document id Java gives free text with no filename.  Eight zeros; the sibling defaults in
# model.py differ from it and from each other.
FREETEXT_DOCID = "00000000.tx"


def java_basename(filename: str) -> str:
    """``FreeText``'s basename: the last element of ``filename.split("/")``.

    Java splits on ``/`` only, so a Windows backslash path has no separator to split on and the
    whole path *is* the basename — which MetaMapLite then reports as ``docid`` (confirmed by
    passing an absolute Windows path: the entire path comes back as the document id).
    Reproduced rather than corrected, per DEVELOPMENT_PLAN §11 rule 3; pass ``docid=`` to
    :meth:`FreeTextLoader.load_file` for anything else.
    """
    return filename.split("/")[-1]


class FreeTextLoader:
    """``bioc.document.loader.freetext`` — one passage holding the whole text."""

    name = "freetext"

    def read(self, text: str, docid: str | None = None) -> list[Document]:
        """``instantiateBioCDocument(text)`` — the no-filename form, which sets ``section``."""
        did = docid or FREETEXT_DOCID
        passage = Passage(
            text=text,
            offset=0,
            infons={"docid": did, "inputformat": self.name, "section": "text"},
        )
        return [Document(id=did, passages=[passage])]

    def load_file(
        self,
        path: Path | str,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
        docid: str | None = None,
    ) -> list[Document]:
        """``loadFileAsBioCDocumentList`` — the filename form, which sets **no** ``section``.

        Read through :func:`mmlite.api.read_document`, so line endings survive and
        offsets index the file verbatim.  Java's own file reader does neither: it sizes a
        ``char[]`` from the file length in *bytes*, so any multi-byte character leaves trailing
        NULs on the text, and its ``--pipe`` reader rewrites every line ending to ``\\n``.
        Neither is reproduced — they corrupt input rather than shape output.
        """
        from ..api import read_document

        # Java takes the basename of the filename *as given*; Path() would normalise a
        # forward-slash path to backslashes on Windows first and change the answer.
        given = str(path)
        text = read_document(Path(path), encoding, errors)
        did = docid if docid is not None else java_basename(given)
        passage = Passage(text=text, offset=0, infons={"docid": did, "inputformat": self.name})
        # Java ids the document with the *full* filename here while the passage keeps the
        # basename; it is the passage infon that reaches Entity.docid.
        return [Document(id=given, passages=[passage])]
