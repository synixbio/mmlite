"""Corpus formats that carry a title and an abstract per record — ports of
``metamap.document.ChemDNER``, ``ChemDNERSLDI``, ``NCBICorpusDocument`` and ``PubTator``.

All four build the same two-passage document: a ``title`` passage and an ``abstract`` passage,
both with the record's id as their ``docid`` infon.  The ``section`` infon reaches
``Entity.fieldid``, and ``title`` is one of the two values MMI treats as a title
(:mod:`mmlite.output.mmi` ranks title concepts without the frequency factor), so these are
the first loaders for which that branch is reachable.

**Every passage sits at offset 0** except PubTator's abstract, so entity offsets are
passage-relative here.  Verified against MetaMapLite 3.6.2rc8, whose ``chemdner`` output reports
the abstract's first concept at ``start=0``.

Malformed lines are skipped with a warning.  Java instead indexes ``docFields[1]`` and
``docFields[2]`` without checking the field count and throws ``ArrayIndexOutOfBoundsException``
mid-file — the same class of defect as the input-corrupting ones §12 declined to reproduce, and
aborting a batch on one bad line is not behaviour worth copying.  (``ChemDNERSLDI`` even reads
``docFields[1]`` *before* the guard that was meant to prevent exactly that.)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .model import Document, Passage
from .singleline import java_read_lines, java_split

log = logging.getLogger(__name__)

_CATEGORY_TAGS = (re.compile(r'<category="[A-Za-z]+">'), re.compile(r"</category>"))


def remove_category_tags(text: str) -> str:
    """``NCBICorpusDocument.removeCategoryTags``: strip the corpus's inline annotation tags.

    Only ``[A-Za-z]+`` category names, exactly as Java's regex has it — a tag such as
    ``<category="Modifier_2">`` is left in place, tags and all.
    """
    for pattern in _CATEGORY_TAGS:
        text = pattern.sub("", text)
    return text


def title_abstract_document(
    docid: str, title: str, abstract: str, inputformat: str, abstract_offset: int = 0
) -> Document:
    """The two-passage document all four loaders build."""
    return Document(
        id=docid,
        passages=[
            Passage(
                text=title,
                offset=0,
                infons={"docid": docid, "section": "title", "inputformat": inputformat},
            ),
            Passage(
                text=abstract,
                offset=abstract_offset,
                infons={"docid": docid, "section": "abstract", "inputformat": inputformat},
            ),
        ],
    )


class _LineCorpusLoader:
    """Shared plumbing: one record per line, malformed lines skipped."""

    name = "chemdner"

    def read(self, text: str, docid: str | None = None) -> list[Document]:
        documents = []
        for line in java_read_lines(text):
            document = self._parse_line(line)
            if document is not None:
                documents.append(document)
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

    def _parse_line(self, line: str) -> Document | None:
        raise NotImplementedError

    def _too_few(self, line: str) -> None:
        log.warning("too few fields in %s line: %s, skipping", self.name, line)


class ChemDNERLoader(_LineCorpusLoader):
    """``chemdner`` — ``id<TAB>title<TAB>abstract``."""

    name = "chemdner"

    def _clean(self, text: str) -> str:
        return text

    def _parse_line(self, line: str) -> Document | None:
        fields = line.split("\t")
        if len(fields) < 3:
            self._too_few(line)
            return None
        return title_abstract_document(
            fields[0], self._clean(fields[1]), self._clean(fields[2]), self.name
        )


class NCBICorpusLoader(ChemDNERLoader):
    """``ncbicorpus`` — ChemDNER's layout, with the NCBI Disease Corpus's tags stripped."""

    name = "ncbicorpus"

    def _clean(self, text: str) -> str:
        return remove_category_tags(text)


class ChemDNERSLDILoader(_LineCorpusLoader):
    """``chemdnersldi`` — ``id|title<TAB>abstract``: a pipe for the id, a tab inside the body."""

    name = "chemdnersldi"

    def _parse_line(self, line: str) -> Document | None:
        fields = java_split(r"\|", line)
        if len(fields) < 2:
            self._too_few(line)
            return None
        body = fields[1].split("\t")
        if len(body) < 2:
            self._too_few(line)
            return None
        return title_abstract_document(fields[0], body[0], body[1], self.name)


class PubTatorLoader:
    """``pubtator`` — ``id|t|title`` and ``id|a|abstract`` lines, annotation lines ignored.

    **This loader is a correction, not a port, and has no Java oracle.** MetaMapLite 3.6.2rc8's
    ``PubTator.loadFileAsBioCDocumentList`` calls ``br.close()`` *inside* its read loop, so
    ``--inputformat=pubtator`` throws ``java.io.IOException: Stream closed`` on any file at all —
    verified, including on a single-line file. Its single-document method works but folds the
    whole file into one document built from the last ``t`` and ``a`` lines it saw, and its stdin
    path emits one document per *line*, each a snapshot of the accumulator so far.

    None of that is behaviour worth reproducing, so this groups lines into one document per
    record id, in first-appearance order, which is what the format means. Recorded as a
    deliberate divergence in DEVELOPMENT_PLAN §15.
    """

    name = "pubtator"

    def read(self, text: str, docid: str | None = None) -> list[Document]:
        titles: dict[str, str] = {}
        abstracts: dict[str, str] = {}
        for line in java_read_lines(text):
            # An annotation line is tab-delimited and has no pipes, so it never splits into
            # three fields and is skipped -- the same test Java applies.
            fields = line.split("|")
            if len(fields) <= 2:
                continue
            record, kind, body = fields[0].strip(), fields[1], fields[2]
            if kind == "t":
                titles[record] = body
            elif kind == "a":
                abstracts[record] = body
        documents = []
        for record in dict.fromkeys([*titles, *abstracts]):
            title = titles.get(record, "")
            documents.append(
                title_abstract_document(
                    record,
                    title,
                    abstracts.get(record, ""),
                    self.name,
                    # Java's own instantiateBioCDocument puts the abstract at len(title); it is
                    # the one passage in any loader with a non-zero offset.  PubTator's gold
                    # annotation files are believed to use len(title) + 1 instead, the title and
                    # abstract being joined by a newline, but no file was available to check and
                    # Java's formula is the only specification in reach -- DEVELOPMENT_PLAN §21
                    # records the decision and what would settle it.
                    abstract_offset=len(title),
                )
            )
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
