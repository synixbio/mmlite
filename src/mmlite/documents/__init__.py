"""Input documents and their loaders — port of ``metamap.document``.

The model is in :mod:`~mmlite.documents.model`; this module is the loader protocol and
the name registry, mirroring ``BioCDocumentLoaderRegistry``.  Registered names are
``--inputformat`` values.

Every input format MetaMapLite registers is ported: ``freetext``, ``sli``/``sldi``, ``sldiwi``,
``chemdner``, ``chemdnersldi``, ``ncbicorpus``, ``pubmed``, ``medline``, ``bioc`` and ``pubtator``
(the last a deliberate correction — see :class:`~mmlite.documents.corpora.PubTatorLoader`).
A new loader slots in through :func:`register` without changing anything here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .bioc import BioCLoader
from .corpora import (
    ChemDNERLoader,
    ChemDNERSLDILoader,
    NCBICorpusLoader,
    PubTatorLoader,
)
from .freetext import FREETEXT_DOCID, FreeTextLoader
from .model import DEFAULT_DOCUMENT_ID, DEFAULT_PASSAGE_DOCID, Document, Passage
from .pubmed import MedlineLoader, PubMedXMLLoader
from .singleline import SLI_DOCID, SingleLineDelimitedWithIdLoader, SingleLineInputLoader

__all__ = [
    "DEFAULT_DOCUMENT_ID",
    "DEFAULT_PASSAGE_DOCID",
    "FREETEXT_DOCID",
    "SLI_DOCID",
    "BioCLoader",
    "ChemDNERLoader",
    "ChemDNERSLDILoader",
    "Document",
    "DocumentLoader",
    "FreeTextLoader",
    "MedlineLoader",
    "NCBICorpusLoader",
    "Passage",
    "PubMedXMLLoader",
    "PubTatorLoader",
    "SingleLineDelimitedWithIdLoader",
    "SingleLineInputLoader",
    "get_loader",
    "loader_names",
    "register",
]


@runtime_checkable
class DocumentLoader(Protocol):
    """``BioCDocumentLoader``.

    Java declares three methods; two suffice here, since its single-document variant is the
    first element of the list one.  Both return a list because several formats (ChemDNER,
    PubTator, the single-line formats) hold many documents per file.
    """

    name: str

    def load_file(
        self,
        path: Path | str,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
        docid: str | None = None,
    ) -> list[Document]: ...

    def read(self, text: str, docid: str | None = None) -> list[Document]: ...


_LOADERS: dict[str, DocumentLoader] = {}


def register(name: str, loader: DocumentLoader) -> None:
    """``BioCDocumentLoaderRegistry.register``.  Re-registering a name replaces it.

    Names map to loaders many-to-one — Java registers both ``sli`` and ``sldi`` for one class —
    so nothing here assumes a loader has exactly one name.
    """
    _LOADERS[name.lower()] = loader


def get_loader(name: str) -> DocumentLoader:
    """The loader registered under ``name``, case-insensitively."""
    try:
        return _LOADERS[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown input format {name!r}; expected one of {loader_names()}"
        ) from None


def loader_names() -> tuple[str, ...]:
    return tuple(sorted(_LOADERS))


register("freetext", FreeTextLoader())
# Java registers "sli" and "sldi" to one and the same SingleLineInput instance.
_single_line = SingleLineInputLoader()
register("sli", _single_line)
register("sldi", _single_line)
register("sldiwi", SingleLineDelimitedWithIdLoader())
register("chemdner", ChemDNERLoader())
register("chemdnersldi", ChemDNERSLDILoader())
register("ncbicorpus", NCBICorpusLoader())
register("pubtator", PubTatorLoader())
register("pubmed", PubMedXMLLoader())
register("medline", MedlineLoader())
register("bioc", BioCLoader())
