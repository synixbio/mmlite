"""One-document-per-line loaders — ports of ``metamap.document.SingleLineInput`` and
``SingleLineDelimitedInputWithID``.

Java registers three names over these two classes: ``sli`` and ``sldi`` are *both*
``SingleLineInput`` (an upstream duplicate registration), and ``sldiwi`` is the delimited one.

Every line becomes its own :class:`~mmlite.documents.model.Document` with a single passage
at **offset 0**, so entity offsets are line-relative.  That is Java's behaviour, not an oversight:
these formats hold unrelated documents that happen to share a file.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .freetext import java_basename
from .model import Document, Passage

log = logging.getLogger(__name__)

# SingleLineInput.instantiateBioCDocument(docText): five zeros.  A fourth distinct default id,
# alongside FreeText's "00000000.tx" and model.py's two -- none of them typos for another.
SLI_DOCID = "00000.txt"

# SingleLineDelimitedInputWithID: a *regular expression*, overridable in Java with the system
# property below.  The property name says "sldiwd" where the format is called "sldiwi"; that
# typo is upstream's, and is recorded here so nobody re-derives the name and gets it wrong.
DEFAULT_SLDIWI_DELIMITER = r"\|"
SLDIWI_DELIMITER_PROPERTY = "metamaplite.sldiwd.delimiter.regexp"

_LINE_BREAK = re.compile(r"\r\n|\n|\r")


def java_read_lines(text: str) -> list[str]:
    """Split as ``BufferedReader.readLine()`` does: on ``\\n``, ``\\r\\n`` or ``\\r``.

    Not :meth:`str.splitlines`, which also breaks on form feed, vertical tab, ``\\x1c``-``\\x1e``
    and ``U+2028``/``U+2029`` — a line containing any of those would become two documents here
    and one in Java.  A terminator at the end of the text does not start a further line.
    """
    if not text:
        return []
    lines = _LINE_BREAK.split(text)
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def java_split(pattern: str, text: str) -> list[str]:
    """``String.split(regex)`` — like :func:`re.split` but with trailing empties removed.

    The difference is load-bearing for ``sldiwi``: ``"id|"`` splits to ``["id"]`` in Java, not
    ``["id", ""]``, so a line with an id and no text has too few fields and is rejected.
    """
    parts = re.split(pattern, text)
    while parts and parts[-1] == "":
        parts.pop()
    return parts


class SingleLineInputLoader:
    """``sli`` / ``sldi`` — one document per line, no id in the line itself."""

    name = "sli"

    def read(self, text: str, docid: str | None = None) -> list[Document]:
        """``bioCLoadFile(Reader)``: passages get **no** infons at all, not even ``section``.

        So these entities end up with ``fieldid`` of ``None`` — the opposite asymmetry to
        :class:`~mmlite.documents.freetext.FreeTextLoader`, whose *read* path is the one
        that sets ``section`` and whose file path is the one that does not.
        """
        did = docid or SLI_DOCID
        return [
            self._document(line, did, Passage(text=line, offset=0), i)
            for i, line in enumerate(java_read_lines(text))
        ]

    def load_file(
        self,
        path: Path | str,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
        docid: str | None = None,
    ) -> list[Document]:
        """``bioCLoadFile(filename)``: passages carry ``docid``, ``inputformat`` and ``section``."""
        from ..api import read_document

        given = str(path)
        did = docid if docid is not None else java_basename(given)
        documents = []
        for i, line in enumerate(java_read_lines(read_document(Path(path), encoding, errors))):
            passage = Passage(
                text=line,
                offset=0,
                infons={"docid": did, "inputformat": self.name, "section": "text"},
            )
            documents.append(self._document(line, did, passage, i))
        return documents

    @staticmethod
    def _document(line: str, docid: str, passage: Passage, i: int) -> Document:
        """One line's document, including the per-line id Java computes and then discards.

        Java writes ``%08d.TX`` into the *document's* infons, but ``processDocument`` immediately
        overwrites that key with the document's own id -- so the sequence number never reaches
        ``Entity.docid``.  Reproduced (``Document.with_docids`` performs the same overwrite), so
        that a caller using the loader directly sees what Java's loader returns.
        """
        return Document(id=docid, passages=[passage], infons={"docid": f"{i:08d}.TX"})


class SingleLineDelimitedWithIdLoader:
    """``sldiwi`` — ``id|text`` per line."""

    name = "sldiwi"

    def __init__(self, delimiter: str = DEFAULT_SLDIWI_DELIMITER):
        self.delimiter = delimiter

    def read(self, text: str, docid: str | None = None) -> list[Document]:
        """``docid`` is ignored: every line carries its own id, which is the point of the format."""
        return [self._document(line) for line in java_read_lines(text)]

    def load_file(
        self,
        path: Path | str,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
        delimiter: str | None = None,
        docid: str | None = None,
    ) -> list[Document]:
        from ..api import read_document

        text = read_document(Path(path), encoding, errors)
        return [self._document(line, delimiter) for line in java_read_lines(text)]

    def _document(self, line: str, delimiter: str | None = None) -> Document:
        """``instantiateBioCDocument``: ``fields[0]`` is the id, ``fields[1]`` is the text.

        Only ``fields[1]`` — a line with more delimiters than one silently loses everything after
        the second field.  A line that does not split into at least two fields produces an
        **empty document** (no passages) and a warning, which then contributes no entities.
        """
        fields = java_split(delimiter or self.delimiter, line)
        if len(fields) <= 1:
            log.warning("too few fields in line: %s, returning an empty document", line)
            return Document()
        passage = Passage(
            text=fields[1],
            offset=0,
            infons={
                "docid": fields[0],
                # Java sets a "text" infon whose value is also "text"; nothing reads it, but it
                # is part of what the loader produces.
                "text": "text",
                "section": "text",
                "inputformat": self.name,
            },
        )
        return Document(id=fields[0], passages=[passage])
