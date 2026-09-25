"""REST API over :class:`mmlite.MetaMapLite`.

    mmlite serve --indexdir ivf --port 8000
    # or: uvicorn mmlite.server:app --port 8000   (configured from MMLITE_* env vars)

The index and the spaCy model are loaded once at start-up.  Annotation is CPU-bound and the
underlying objects (SQLite connection, spaCy pipeline, per-concept caches) are not thread-safe,
so requests are serialised behind a lock and run in a worker thread; scale out with processes
(``uvicorn --workers N``), not threads.

Optional hardening, both off unless set in the environment:

- ``MMLITE_API_TOKEN``: every endpoint except ``/health`` then requires
  ``Authorization: Bearer <token>``.
- ``MMLITE_CORS_ORIGINS``: comma-separated origins allowed to call the API from a browser.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Literal

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from . import __version__
from .api import DEFAULT_DOCID, MetaMapLite
from .config import Settings
from .documents import get_loader, loader_names
from .output import FORMATS
from .pipeline import NEGATION_DETECTORS, get_negation_detector, negation_detector_names
from .pipeline.segment import SEGMENTATION_METHODS
from .semtypes import TUI_TO_ABBREV, TUI_TO_NAME
from .types import Entity

log = logging.getLogger(__name__)

TEXT_FORMATS = FORMATS  # mmi | json | brat | cuilist | full


def _env_limit(name: str, default: int) -> int:
    """A positive integer request limit from the environment.

    Bad values log a warning and fall back to ``default``.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        log.warning("ignoring %s=%r: expected a positive integer, using %d", name, raw, default)
        return default
    return value


# Per document.  Under strict-parity ConText annotation time is quadratic in document length
# (50 KB is ~2 s with NegEx but minutes with ConText), so the default bounds the worst single
# request rather than trusting clients (docs/OPERATIONS_MANUAL.md §8.2).
MAX_TEXT_CHARS = _env_limit("MMLITE_MAX_TEXT_CHARS", 50_000)
MAX_BATCH_DOCS = _env_limit("MMLITE_MAX_BATCH_DOCS", 256)
# The raw body of POST /annotate/formatted, which may hold up to MAX_BATCH_DOCS documents plus
# markup (PubMed XML, BioC); each document in it is still held to MAX_TEXT_CHARS.
MAX_FORMATTED_CHARS = _env_limit("MMLITE_MAX_FORMATTED_CHARS", 1_000_000)

# A shared secret clients send as "Authorization: Bearer <token>".  Unset (the default) leaves
# the API open, as before; a reverse proxy doing its own authentication needs nothing here.
API_TOKEN = os.environ.get("MMLITE_API_TOKEN") or None


def _cors_origins() -> list[str]:
    """``MMLITE_CORS_ORIGINS`` as a list; empty means no CORS headers at all."""
    raw = os.environ.get("MMLITE_CORS_ORIGINS", "")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# --- schema ------------------------------------------------------------------------------------


class Concept(BaseModel):
    cui: str = Field(examples=["C0011860"])
    preferred_name: str = Field(examples=["Diabetes Mellitus, Non-Insulin-Dependent"])
    semantic_types: list[str] = Field(examples=[["dsyn"]])
    sources: list[str] = Field(examples=[["MSH", "SNOMEDCT_US"]])
    matched_string: str = Field(
        description="the dictionary string that matched", examples=["Type 2 diabetes mellitus"]
    )


class EntityOut(BaseModel):
    start: int = Field(description="character offset in the input text")
    length: int
    text: str = Field(description="the matched span, verbatim")
    part_of_speech: str | None = Field(description="POS tag of the span's first token")
    sentence: int = Field(description="index of the sentence the span was found in")
    negated: bool
    # ConText only (--usecontext / metamaplite.negation.detector=context); null under NegEx.
    temporality: str | None = None
    experiencer: str | None = None
    assertion: str | None = Field(
        default=None,
        description=(
            "ConText's verdict: 'Affirmed', 'Negated' or 'Possible'. `negated` is the boolean "
            "MetaMapLite keeps and collapses 'Possible' into false, so a hedged mention "
            "('rule out pneumonia') reads as affirmed there; this field separates the two"
        ),
        examples=["Possible"],
    )
    score: float = Field(
        description=(
            "always 0.0: MetaMapLite's EntityLookup4 does not score entities, so this is not a "
            "confidence and 0.0 does not mean a weak match. The `mmi` format's score column is "
            "a separate ranking and is non-zero"
        )
    )
    concepts: list[Concept]


class AnnotateRequest(BaseModel):
    text: str = Field(examples=["No history of type 2 diabetes mellitus."])
    docid: str = DEFAULT_DOCID
    restrict_to_sts: list[str] | None = Field(
        default=None, description="semantic type abbreviations or TUIs", examples=[["dsyn", "sosy"]]
    )
    restrict_to_sources: list[str] | None = Field(
        default=None, description="UMLS source abbreviations", examples=[["MSH"]]
    )
    detect_negations: bool | None = Field(
        default=None, description="override the server's negation setting"
    )
    detector: str | None = Field(
        default=None,
        description=(
            "negation detector for this request: 'negex' (default) or 'context', which also "
            "reports temporality and experiencer"
        ),
        examples=["context"],
    )


class Document(BaseModel):
    docid: str = DEFAULT_DOCID
    text: str


class BatchRequest(BaseModel):
    documents: list[Document]
    restrict_to_sts: list[str] | None = None
    restrict_to_sources: list[str] | None = None
    detect_negations: bool | None = None
    detector: str | None = None


class FormattedRequest(BaseModel):
    """One blob of text in any input format MetaMapLite reads, annotated as a whole."""

    text: str = Field(
        description="the document(s), in `inputformat`",
        examples=["D1\tAspirin in myocardial infarction.\tPatients received aspirin."],
    )
    inputformat: str = Field(
        default="freetext",
        description="any registered loader; see GET /formats",
        examples=["chemdner"],
    )
    restrict_to_sts: list[str] | None = None
    restrict_to_sources: list[str] | None = None
    detect_negations: bool | None = None
    detector: str | None = None


class SemanticType(BaseModel):
    tui: str = Field(examples=["T047"])
    abbreviation: str = Field(examples=["dsyn"])
    name: str = Field(examples=["Disease or Syndrome"])


class FormatsResponse(BaseModel):
    input_formats: list[str] = Field(description="values accepted by `inputformat`")
    output_formats: list[str] = Field(description="values accepted by the `format` query")
    segmentation_methods: list[str]
    negation_detectors: list[str] = Field(description="values accepted by `detector`")


class AnnotateResponse(BaseModel):
    docid: str
    entities: list[EntityOut]
    elapsed_ms: float


class BatchResponse(BaseModel):
    results: list[AnnotateResponse]
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    index_directory: str
    index: dict[str, str | int]
    postagging: bool
    postag_model: str | None  # None when tagging is off
    detect_negations: bool
    negation_detector: str
    strict_parity: bool  # False: known MetaMapLite bugs fixed, output diverges from Java
    auth_required: bool  # MMLITE_API_TOKEN is set
    segmentation_method: str
    uptime_s: float


def to_entity_out(e: Entity) -> EntityOut:
    return EntityOut(
        start=e.start,
        length=e.length,
        text=e.text,
        part_of_speech=e.lexical_category,
        sentence=e.location_position,
        negated=e.negated,
        temporality=e.temporality,
        experiencer=e.experiencer,
        assertion=e.assertion,
        score=e.score,
        concepts=[
            Concept(
                cui=ev.cui,
                preferred_name=ev.concept.preferred_name,
                semantic_types=sorted(ev.concept.semantic_types),
                sources=sorted(ev.concept.sources),
                matched_string=ev.concept.concept_string,
            )
            for ev in e.evs
        ],
    )


# --- service -----------------------------------------------------------------------------------


STRICT_CONTEXT_WARNING = (
    "ConText is running with strict_parity on: a later sentence can overwrite an earlier "
    "mention's experiencer and temporality (a relative's history marks the patient's own "
    "diagnosis Historical/Other), and annotation time is quadratic in document length. Set "
    "MMLITE_STRICT_PARITY=false unless you need Java-identical output."
)


def _is_context(name: str | None) -> bool:
    return name is not None and NEGATION_DETECTORS.get(name.strip().lower()) == "context"


class Service:
    """Holds the single MetaMapLite instance and serialises access to it."""

    def __init__(self, mml: MetaMapLite):
        self.mml = mml
        self.lock = threading.Lock()
        self.started = time.monotonic()
        # The library default stays Java-exact; a service is where strict ConText hurts, so say
        # so once, at start-up or on the first request that asks for ConText.
        self._warned_strict_context = False
        if mml.settings.detect_negations and _is_context(mml.settings.negation_detector):
            self._warn_strict_context()

    def _warn_strict_context(self) -> None:
        if self.mml.settings.strict_parity and not self._warned_strict_context:
            self._warned_strict_context = True
            log.warning(STRICT_CONTEXT_WARNING)

    @contextmanager
    def _detector(self, name: str | None) -> Iterator[None]:
        """Swap the negation detector for one request, under the lock.

        The detector is built once at start-up; a request asking for the other one borrows it
        and puts the original back, so a failure cannot leave the server on the wrong detector.
        """
        if name is None:
            yield
            return
        try:
            detector = get_negation_detector(name, strict_parity=self.mml.settings.strict_parity)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from None
        if _is_context(name):
            self._warn_strict_context()
        previous = self.mml.lookup.negex
        self.mml.lookup.negex = detector
        try:
            yield
        finally:
            self.mml.lookup.negex = previous

    def annotate(
        self,
        text: str,
        docid: str,
        sts: list[str] | None,
        sources: list[str] | None,
        negations: bool | None,
        detector: str | None = None,
    ) -> list[Entity]:
        with self.lock, self._detector(detector):
            return self.mml.process_text(
                text,
                docid=docid,
                restrict_to_sts=set(sts) if sts is not None else None,
                restrict_to_sources=set(sources) if sources is not None else None,
                detect_negations=negations,
            )

    def annotate_formatted(
        self,
        text: str,
        inputformat: str,
        sts: list[str] | None,
        sources: list[str] | None,
        negations: bool | None,
        detector: str | None = None,
    ) -> list[tuple[str, list[Entity]]]:
        """Load ``text`` with a registered loader and annotate every document it yields."""
        try:
            loader = get_loader(inputformat)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from None
        with self.lock, self._detector(detector):
            try:
                documents = loader.read(text)
            except ValueError as e:  # e.g. PubMed XML that is not well-formed
                raise HTTPException(status_code=422, detail=str(e)) from None
            # Counted before annotating, not after: checking the result meant doing all the
            # work under the lock and then throwing it away.
            if len(documents) > MAX_BATCH_DOCS:
                raise HTTPException(
                    status_code=413,
                    detail=f"input yields {len(documents)} documents, "
                    f"over the {MAX_BATCH_DOCS} limit",
                )
            for document in documents:
                size = sum(len(passage.text) for passage in document.passages)
                if size > MAX_TEXT_CHARS:
                    raise HTTPException(
                        status_code=413,
                        detail=f"document {document.resolved_id()!r} exceeds {MAX_TEXT_CHARS} "
                        f"characters ({size})",
                    )
            out = []
            for document in documents:
                document.with_docids()
                out.append(
                    (
                        document.resolved_id(),
                        self.mml.process_document(
                            document,
                            restrict_to_sts=set(sts) if sts is not None else None,
                            restrict_to_sources=set(sources) if sources is not None else None,
                            detect_negations=negations,
                        ),
                    )
                )
            return out

    def lookup_term(self, term: str) -> list[Entity]:
        with self.lock:
            return self.mml.lookup_term(term)

    def format(self, entities: list[Entity], name: str) -> str:
        with self.lock:
            return self.mml.format(entities, name)


_service: Service | None = None


def get_service() -> Service:
    if _service is None:  # pragma: no cover - set by the lifespan handler
        raise HTTPException(status_code=503, detail="index is not loaded")
    return _service


def build_service(settings: Settings | None = None, preload_index: bool = False) -> Service:
    settings = settings or Settings.load()
    return Service(MetaMapLite(settings, preload_index=preload_index))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _service
    if _service is None:
        preload = os.environ.get("MMLITE_PRELOAD_INDEX", "").lower() in ("1", "true", "yes")
        _service = build_service(preload_index=preload)
    yield
    if _service is not None:
        _service.mml.close()
        _service = None


_bearer = HTTPBearer(auto_error=False, description="set when MMLITE_API_TOKEN is")


def require_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)],
) -> None:
    """Reject the request unless it carries the configured bearer token (if one is configured)."""
    if API_TOKEN is None:
        return
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(), API_TOKEN.encode()
    ):
        raise HTTPException(
            status_code=401,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


app = FastAPI(
    title="mmlite",
    version=__version__,
    summary="UMLS concept recognition (a Python port of NLM MetaMapLite)",
    lifespan=lifespan,
)
if cors_origins := _cors_origins():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )

ServiceDep = Annotated[Service, Depends(get_service)]
Authorized = [Depends(require_token)]


def _check_text(text: str, limit: int | None = None, what: str = "text") -> None:
    limit = MAX_TEXT_CHARS if limit is None else limit
    if not text.strip():
        raise HTTPException(status_code=422, detail=f"{what} is empty")
    if len(text) > limit:
        raise HTTPException(
            status_code=413, detail=f"{what} exceeds {limit} characters ({len(text)})"
        )


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(service: ServiceDep) -> HealthResponse:
    s = service.mml.settings
    stats = service.mml.index.meta
    return HealthResponse(
        status="ok",
        version=__version__,
        index_directory=str(s.index_directory),
        index={
            k: stats.get(k, "")
            for k in ("built_at", "sources", "norm_version", "n_cuisourceinfo", "n_cuist")
        },
        postagging=s.enable_postagging,
        postag_model=s.postag_model if s.enable_postagging else None,
        detect_negations=s.detect_negations,
        negation_detector=s.negation_detector,
        strict_parity=s.strict_parity,
        auth_required=API_TOKEN is not None,
        segmentation_method=s.segmentation_method,
        uptime_s=round(time.monotonic() - service.started, 1),
    )


@app.post("/annotate", response_model=AnnotateResponse, tags=["annotate"], dependencies=Authorized)
async def annotate(
    service: ServiceDep, req: Annotated[AnnotateRequest, Body()]
) -> AnnotateResponse:
    """Find UMLS concepts in one document."""
    _check_text(req.text)
    t0 = time.perf_counter()
    entities = await asyncio.to_thread(
        service.annotate,
        req.text,
        req.docid,
        req.restrict_to_sts,
        req.restrict_to_sources,
        req.detect_negations,
        req.detector,
    )
    return AnnotateResponse(
        docid=req.docid,
        entities=[to_entity_out(e) for e in entities],
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.post(
    "/annotate/batch", response_model=BatchResponse, tags=["annotate"], dependencies=Authorized
)
async def annotate_batch(
    service: ServiceDep, req: Annotated[BatchRequest, Body()]
) -> BatchResponse:
    """Annotate several documents in one request (processed sequentially)."""
    if not req.documents:
        raise HTTPException(status_code=422, detail="documents is empty")
    if len(req.documents) > MAX_BATCH_DOCS:
        raise HTTPException(
            status_code=413,
            detail=f"batch exceeds {MAX_BATCH_DOCS} documents ({len(req.documents)})",
        )
    for doc in req.documents:
        _check_text(doc.text)
    t0 = time.perf_counter()
    results = []
    for doc in req.documents:
        d0 = time.perf_counter()
        entities = await asyncio.to_thread(
            service.annotate,
            doc.text,
            doc.docid,
            req.restrict_to_sts,
            req.restrict_to_sources,
            req.detect_negations,
            req.detector,
        )
        results.append(
            AnnotateResponse(
                docid=doc.docid,
                entities=[to_entity_out(e) for e in entities],
                elapsed_ms=round((time.perf_counter() - d0) * 1000, 2),
            )
        )
    return BatchResponse(results=results, elapsed_ms=round((time.perf_counter() - t0) * 1000, 2))


@app.post(
    "/annotate/text",
    response_class=PlainTextResponse,
    tags=["annotate"],
    responses={200: {"content": {"text/plain": {"example": "doc|MMI|11.05|...|"}}}},
    dependencies=Authorized,
)
async def annotate_text(
    service: ServiceDep,
    req: Annotated[AnnotateRequest, Body()],
    format: Annotated[str, Query(description=f"one of: {', '.join(TEXT_FORMATS)}")] = "mmi",
) -> str:
    """Annotate one document and return MetaMapLite's own output formats as plain text."""
    if format not in TEXT_FORMATS:
        raise HTTPException(
            status_code=422, detail=f"unknown format {format!r}; expected {TEXT_FORMATS}"
        )
    _check_text(req.text)
    entities = await asyncio.to_thread(
        service.annotate,
        req.text,
        req.docid,
        req.restrict_to_sts,
        req.restrict_to_sources,
        req.detect_negations,
        req.detector,
    )
    return await asyncio.to_thread(service.format, entities, format)


@app.post(
    "/annotate/formatted", response_model=BatchResponse, tags=["annotate"], dependencies=Authorized
)
async def annotate_formatted(
    service: ServiceDep, req: Annotated[FormattedRequest, Body()]
) -> BatchResponse:
    """Annotate text in any of MetaMapLite's input formats.

    The blob is parsed by the named loader and every document it yields is annotated, so one
    request can carry a ChemDNER file, a PubMed XML set, a BioC collection or a plain note.
    Offsets are relative to each document's own passage, as they are everywhere else.
    """
    _check_text(req.text, MAX_FORMATTED_CHARS)
    t0 = time.perf_counter()
    annotated = await asyncio.to_thread(
        service.annotate_formatted,
        req.text,
        req.inputformat,
        req.restrict_to_sts,
        req.restrict_to_sources,
        req.detect_negations,
        req.detector,
    )
    return BatchResponse(
        results=[
            AnnotateResponse(
                docid=docid, entities=[to_entity_out(e) for e in entities], elapsed_ms=0.0
            )
            for docid, entities in annotated
        ],
        elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.get("/formats", response_model=FormatsResponse, tags=["service"], dependencies=Authorized)
def formats() -> FormatsResponse:
    """Every value this server accepts for `inputformat`, `format`, segmentation and `detector`."""
    return FormatsResponse(
        input_formats=list(loader_names()),
        output_formats=list(TEXT_FORMATS),
        segmentation_methods=list(SEGMENTATION_METHODS),
        negation_detectors=negation_detector_names(),
    )


@app.get(
    "/semantic-types",
    response_model=list[SemanticType],
    tags=["dictionary"],
    dependencies=Authorized,
)
def semantic_types() -> list[SemanticType]:
    """The UMLS semantic types, for building `restrict_to_sts` values.

    Either the abbreviation or the TUI is accepted in a restriction, in any case.
    """
    return [
        SemanticType(tui=tui, abbreviation=abbrev, name=TUI_TO_NAME[tui])
        for tui, abbrev in TUI_TO_ABBREV.items()
    ]


@app.get("/lookup", tags=["dictionary"], dependencies=Authorized)
async def lookup(
    service: ServiceDep,
    term: Annotated[str, Query(description="a term to look up verbatim (no segmentation/tagging)")],
) -> list[Concept]:
    """Dictionary lookup for a single term — the counterpart of ``mmlite lookup``."""
    if not term.strip():
        raise HTTPException(status_code=422, detail="term is empty")
    entities = await asyncio.to_thread(service.lookup_term, term)
    seen: dict[str, Concept] = {}
    for e in entities:
        for ev in e.evs:
            seen.setdefault(
                ev.cui,
                Concept(
                    cui=ev.cui,
                    preferred_name=ev.concept.preferred_name,
                    semantic_types=sorted(ev.concept.semantic_types),
                    sources=sorted(ev.concept.sources),
                    matched_string=ev.concept.concept_string,
                ),
            )
    return list(seen.values())


def serve(
    settings: Settings | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    preload_index: bool = False,
    reload: bool = False,
) -> None:  # pragma: no cover - thin uvicorn wrapper
    """Run the API with uvicorn, loading the index before the first request."""
    import uvicorn

    global _service
    if not reload:
        _service = build_service(settings, preload_index=preload_index)
    elif settings is not None:
        _export_settings(settings)
    uvicorn.run("mmlite.server:app" if reload else app, host=host, port=port, reload=reload)


def _export_settings(settings: Settings) -> None:
    """Pass settings to a reloaded worker process through the environment.

    Covers every :class:`Settings` field, so ``--reload`` behaves like a normal start-up
    instead of silently dropping the restriction and formatting settings.
    """
    for attr, value in (
        ("INDEX_DIRECTORY", settings.index_directory),
        ("SOURCESET", settings.sourceset),
        ("SEMANTIC_TYPES", settings.semantic_types),
        ("SEGMENTATION_METHOD", settings.segmentation_method),
        ("ENABLE_POSTAGGING", settings.enable_postagging),
        ("POSTAG_LIST", settings.postag_list),
        ("POSTAG_MODEL", settings.postag_model),
        ("POSTAG_VERB_RESCUE", settings.postag_verb_rescue),
        ("DETECT_NEGATIONS", settings.detect_negations),
        ("NEGATION_DETECTOR", settings.negation_detector),
        ("STRICT_PARITY", settings.strict_parity),
        ("PRECISION_FILTER", settings.precision_filter),
        ("BRAT_TYPENAME", settings.brat_typename),
        ("NORMALIZED_STRING_CACHE_SIZE", settings.normalized_string_cache_size),
        ("EXCLUDED_TERMS_FILE", settings.excluded_terms_file),
        ("UDA_FILE", settings.uda_file),
        ("CUI_TERM_LIST_FILE", settings.cui_term_list_file),
    ):
        if value is None or value == []:
            continue
        os.environ[f"MMLITE_{attr}"] = ",".join(value) if isinstance(value, list) else str(value)


__all__ = ["app", "build_service", "serve"]
