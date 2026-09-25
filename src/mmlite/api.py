"""Python API — the counterpart of ``gov.nih.nlm.nls.ner.MetaMapLite``.

from mmlite import MetaMapLite
mml = MetaMapLite(index_directory="ivf")
for entity in mml.process_text("The patient has type 2 diabetes mellitus."):
    print(entity.start, entity.text, entity.cuis)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Self

from .config import Settings
from .documents import DEFAULT_PASSAGE_DOCID, Document, Passage, get_loader
from .index.lookup import IndexLookup
from .output import ResultFormatter, get_formatter
from .pipeline import TextPipeline
from .pipeline.abbreviations import UserDefinedAcronyms
from .pipeline.entity_lookup import CustomTerms, EntityLookup, SpecialTerms
from .pipeline.postag import NullPosTagger, PosTagger
from .pipeline.segment import SentenceDetector
from .types import Entity

DEFAULT_DOCID = "00000000.tx"  # FreeText.instantiateBioCDocument for text without a file


def read_document(path: Path | str, encoding: str = "utf-8", errors: str = "replace") -> str:
    """Read a text document the way the annotator expects it.

    Line endings are **not** translated: every offset an annotation reports indexes the string
    returned here, so a CRLF file must keep its CRs or a ``brat`` ``.ann`` will not line up with
    its own ``.txt``.  Undecodable bytes become ``U+FFFD`` rather than raising, so one bad byte
    in a clinical note does not abort a batch.

    A leading byte-order mark is dropped: Notepad and PowerShell's ``Set-Content -Encoding utf8``
    both write one, editors don't show it, and ``str.strip()`` doesn't remove it - left in, it
    would be character 0 of the text (shifting every offset by one against what the editor
    shows) and would make a BOM-only file look non-empty.
    """
    with open(path, encoding=encoding, errors=errors, newline="") as fh:
        text = fh.read()
    return text.removeprefix("﻿")


class MetaMapLite:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        index_directory: Path | str | None = None,
        properties_file: Path | str | None = None,
        preload_index: bool = False,
        tagger: PosTagger | None = None,
        sentence_detector: SentenceDetector | None = None,
        excluded_terms: SpecialTerms | None = None,
        custom_terms: CustomTerms | None = None,
        udas: UserDefinedAcronyms | None = None,
        remove_subsumed: bool = True,
    ):
        if settings is None:
            settings = Settings.load(
                Path(properties_file) if properties_file else None,
                index_directory=Path(index_directory) if index_directory else None,
            )
        elif index_directory is not None:
            settings.index_directory = Path(index_directory)
        self.settings = settings
        self.index = IndexLookup(settings.index_directory)
        if preload_index:
            self.index.load_in_memory()
        if tagger is None and not settings.enable_postagging:
            tagger = NullPosTagger()
        self.pipeline = TextPipeline(settings, tagger=tagger, sentence_detector=sentence_detector)
        self.lookup = EntityLookup(
            self.index,
            settings,
            excluded_terms=excluded_terms,
            custom_terms=custom_terms,
            udas=udas,
            remove_subsumed=remove_subsumed,
        )
        self._formatters: dict[str, ResultFormatter] = {}

    def process_text(
        self,
        text: str,
        *,
        docid: str = DEFAULT_DOCID,
        fieldid: str | None = "text",
        restrict_to_sts: set[str] | None = None,
        restrict_to_sources: set[str] | None = None,
        detect_negations: bool | None = None,
    ) -> list[Entity]:
        """Annotate free text.  Restriction sets default to the configured settings.

        ``fieldid`` becomes ``Entity.fieldid`` and MMI's field column.  It defaults to ``"text"``,
        which is *not* what Java produces for a file: its free-text loader sets no ``section``
        infon, so its entities carry ``None`` and its ``json`` output omits the key entirely.
        Pass ``fieldid=None`` for that behaviour — it is what ``annotate`` and
        :meth:`process_document` do.  The default is kept as it is so existing callers' output
        does not change; see DEVELOPMENT_PLAN §21.
        """
        sts = restrict_to_sts if restrict_to_sts is not None else set(self.settings.semantic_types)
        srcs = (
            restrict_to_sources if restrict_to_sources is not None else set(self.settings.sourceset)
        )
        return self.lookup.process_text(
            text, self.pipeline, docid, fieldid, sts, srcs, detect_negations=detect_negations
        )

    def process_file(
        self,
        path: Path | str,
        *,
        docid: str | None = None,
        encoding: str = "utf-8",
        errors: str = "replace",
        **kwargs: Any,
    ) -> list[Entity]:
        """Annotate a text file; ``docid`` defaults to its name, as the CLI's ``--input`` does.

        Reads via :func:`read_document`, so reported offsets index the file verbatim.  Remaining
        keyword arguments are passed to :meth:`process_text` — including ``fieldid``, which
        defaults to ``"text"`` here and to ``None`` on the Java-faithful ``load`` +
        :meth:`process_document` path (DEVELOPMENT_PLAN §21).
        """
        path = Path(path)
        return self.process_text(
            read_document(path, encoding, errors), docid=docid or path.name, **kwargs
        )

    def process_passage(
        self,
        passage: Passage,
        *,
        docid: str | None = None,
        restrict_to_sts: set[str] | None = None,
        restrict_to_sources: set[str] | None = None,
        detect_negations: bool | None = None,
    ) -> list[Entity]:
        """Annotate one passage (``EntityLookup4.processPassage``).

        Offsets are reported relative to whatever the passage's ``offset`` is relative to — the
        source the loader read — not to ``passage.text``.  ``docid`` defaults to the passage's
        own ``docid`` infon, then to Java's :data:`~mmlite.documents.DEFAULT_PASSAGE_DOCID`;
        ``fieldid`` is the ``section`` infon, which may be ``None``.
        """
        sts = restrict_to_sts if restrict_to_sts is not None else set(self.settings.semantic_types)
        srcs = (
            restrict_to_sources if restrict_to_sources is not None else set(self.settings.sourceset)
        )
        if docid is None:
            # Java tests the infon against null, not for truthiness: a Medline record with no
            # PMID carries docid "" and is reported as "", not as the default.
            docid = passage.docid if passage.docid is not None else DEFAULT_PASSAGE_DOCID
        return self.lookup.process_sentences(
            self.pipeline.process(passage.text, passage.offset),
            passage.text,
            docid,
            passage.section,
            sts,
            srcs,
            base_offset=passage.offset,
            detect_negations=detect_negations,
        )

    def process_document(self, document: Document, **kwargs: Any) -> list[Entity]:
        """Annotate every passage of a document (``MetaMapLite.processDocument``).

        Missing ``docid`` infons are filled from the document id first, exactly as Java does, so
        a passage that sets its own keeps it.  Remaining keyword arguments go to
        :meth:`process_passage`.
        """
        document.with_docids()
        entities: list[Entity] = []
        for passage in document.passages:
            entities.extend(self.process_passage(passage, **kwargs))
        return entities

    def process_documents(self, documents: Iterable[Document], **kwargs: Any) -> list[Entity]:
        """``processDocumentList``: annotate each document, concatenating the results."""
        entities: list[Entity] = []
        for document in documents:
            entities.extend(self.process_document(document, **kwargs))
        return entities

    def load(
        self, path: Path | str, inputformat: str = "freetext", **kwargs: Any
    ) -> list[Document]:
        """Load an input file with a registered loader (``--inputformat``).

        ``mmlite.documents.loader_names()`` lists what is registered: ``freetext`` plus
        the corpus formats (``bioc``, ``chemdner``, ``chemdnersldi``, ``medline``,
        ``ncbicorpus``, ``pubmed``, ``pubtator``, ``sldi``, ``sldiwi``, ``sli``).
        """
        return get_loader(inputformat).load_file(path, **kwargs)

    def formatter(self, name: str = "mmi") -> ResultFormatter:
        """A result formatter by ``--outputformat`` name (mmi, json, brat, cuilist, full).

        Formatters are stateless and cached per name, so formatting many documents in a loop
        does not rebuild one each time.
        """
        key = name.lower()
        fmt = self._formatters.get(key)
        if fmt is None:
            fmt = get_formatter(
                key,
                treecode_lookup=self.index.mesh_treecodes,
                brat_type=self.settings.brat_typename,
                java_order=self.settings.strict_parity,
            )
            self._formatters[key] = fmt
        return fmt

    def format(self, entities: list[Entity], name: str = "mmi") -> str:
        return self.formatter(name).format(entities)

    def lookup_term(self, term: str) -> list[Entity]:
        """Match a bare term without segmentation, tagging, filtering or subsumption removal."""
        return self.lookup.lookup_term(term)

    def close(self) -> None:
        self.index.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
