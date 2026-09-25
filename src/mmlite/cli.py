"""Command-line interface, the counterpart of MetaMapLite's ``metamaplite.sh``.

Commands: ``annotate`` (the main one, mirroring ``metamaplite.sh``'s flags), ``build-index``,
``lookup``, ``normalize``, ``tokenize``, ``index-stats`` and ``serve``.  Run
``mmlite --help`` for the full flag list, or see ``docs/USER_GUIDE.md`` section 5.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import typer

from .config import Settings
from .index.build import build_index
from .index.lookup import IndexLookup
from .normalize import lookup_keys, normalize
from .output import FORMATS
from .pipeline.segment import SEGMENTATION_METHODS
from .semtypes import unknown_semantic_types

# Everything that can go wrong while opening an index or building a pipeline: a missing index
# (OSError), a corrupt one (DatabaseError), a normalizer-version mismatch (RuntimeError), a
# missing optional dependency (ImportError), or a bad settings value (ValueError).
_STARTUP_ERRORS = (ImportError, OSError, sqlite3.DatabaseError, RuntimeError, ValueError)

app = typer.Typer(help="mmlite: Python port of NLM MetaMapLite", no_args_is_help=True)


def _split(csv: str | None) -> set[str] | None:
    return {s.strip() for s in csv.split(",") if s.strip()} if csv else None


def _detector_name(usecontext: bool, name: str | None) -> str | None:
    """``--usecontext`` and ``--negation-detector``, which must not disagree."""
    if not usecontext:
        return name
    if name is not None and name.strip().lower() != "context":
        _fail(f"--usecontext conflicts with --negation-detector {name}")
    return "context"


def _fail(message: str) -> None:
    """Report a missing optional dependency (or model) without a traceback."""
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _fail_startup(error: Exception, postag: bool) -> None:
    """Report a start-up failure, suggesting ``--no-postag`` only when that would help.

    The hint is for one failure: the tagger is unavailable, which is what
    ``spacy_nlp.TAGGER_UNAVAILABLE`` names. Every other start-up failure --
    a missing index above all, which is an ``OSError`` just as a missing model is -- has nothing
    to do with tagging, and offering that advice sends a newcomer after the wrong problem at the
    moment they can least afford it. The suggestion is equally pointless once ``--no-postag`` is
    already in effect.
    """
    from .pipeline.spacy_nlp import TAGGER_UNAVAILABLE

    relevant = postag and isinstance(error, TAGGER_UNAVAILABLE)
    hint = "\n(or pass --no-postag to skip part-of-speech tagging)" if relevant else ""
    _fail(f"{error}{hint}")


def _use_utf8(stream: Any, **kwargs: Any) -> None:
    """Read/write UTF-8 regardless of the console code page.

    UMLS preferred names contain non-ASCII characters, so on a Windows console (cp1252 by
    default) writing MMI output would otherwise raise ``UnicodeEncodeError``, and piped UTF-8
    input would be mis-decoded.  Streams a test runner replaced have no ``reconfigure``.
    """
    # Which of these fires is console- and test-runner-dependent; none of them is worth failing on.
    with contextlib.suppress(AttributeError, ValueError, OSError):
        stream.reconfigure(encoding="utf-8", errors="replace", **kwargs)


@app.callback()
def _main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log progress to stderr"),
) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@app.command("build-index")
def build_index_cmd(
    mrconso: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to MRCONSO.RRF"),
    mrsty: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to MRSTY.RRF"),
    out: Path = typer.Option(Path("ivf"), help="Output index directory"),
    mrsat: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="Path to MRSAT.RRF (MeSH tree codes)"
    ),
    sources: str | None = typer.Option(
        None, help="Comma-separated SABs to keep, e.g. MSH,SNOMEDCT_US"
    ),
    include_suppressed: bool = typer.Option(False, help="Keep SUPPRESS=O/E/Y rows"),
    overwrite: bool = typer.Option(False, help="Replace an existing index"),
) -> None:
    """Build the index from UMLS RRF files (equivalent of Java CreateIndexes)."""
    db = build_index(
        mrconso=mrconso,
        mrsty=mrsty,
        out_dir=out,
        mrsat=mrsat,
        sources=_split(sources),
        include_suppressed=include_suppressed,
        overwrite=overwrite,
    )
    typer.echo(f"index written to {db}")


@app.command()
def lookup(
    term: str = typer.Argument(..., help="Term to look up (normalized before matching)"),
    indexdir: Path = typer.Option(None, help="Index directory (default: settings / ivf)"),
    restrict_to_sources: str | None = typer.Option(None, help="Comma-separated SABs"),
    restrict_to_sts: str | None = typer.Option(
        None, help="Comma-separated semantic types (dsyn,hops or TUIs)"
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table"),
) -> None:
    """Look up a single term (equivalent of Java MappedMultiKeyIndexLookup)."""
    settings = Settings.load(index_directory=indexdir)
    try:
        with IndexLookup(settings.index_directory) as ix:
            hits = ix.lookup(
                term, sources=_split(restrict_to_sources), semantic_types=_split(restrict_to_sts)
            )
    except _STARTUP_ERRORS as e:
        _fail(str(e))
    if as_json:
        typer.echo(json.dumps([h.__dict__ for h in hits], indent=2))
        return
    if not hits:
        typer.echo("no matches")
        raise typer.Exit(code=1)
    for h in hits:
        typer.echo(
            f"{h.cui}\t{h.preferred_name}\t[{','.join(h.semantic_types)}]\t{h.sab}/{h.tty}\t{h.matched_string}"
        )


@app.command("normalize")
def normalize_cmd(
    text: list[str] = typer.Argument(..., help="Text to normalize (words are joined)"),
) -> None:
    """Show the normalized form and index keys for a span.

    Port of Java's ``Normalization.normalizeUtf8AsciiString``.
    """
    span = " ".join(text)
    typer.echo(f"normalized: {normalize(span)}")
    typer.echo(f"lookup keys: {list(lookup_keys(span))}")


@app.command("tokenize")
def tokenize_cmd(
    text: list[str] = typer.Argument(..., help="Text to segment and tokenize (words are joined)"),
    segmentation: str = typer.Option("SENTENCES", help="SENTENCES | BLANKLINES | LINES"),
    postag: bool = typer.Option(True, help="Run the spaCy POS tagger (--no-postag to skip)"),
    postag_model: str | None = typer.Option(
        None, help="spaCy model for POS tags (default: settings / en_core_web_sm)"
    ),
    show_ws: bool = typer.Option(False, help="Include whitespace tokens in the listing"),
) -> None:
    """Show sentences and MetaMapLite tokens (class, offset, POS) for a span of text."""
    from .pipeline import TextPipeline
    from .pipeline.postag import NullPosTagger

    if segmentation.upper() not in SEGMENTATION_METHODS:
        _fail(
            f"unknown segmentation method {segmentation!r}; expected one of {SEGMENTATION_METHODS}"
        )
    settings = Settings.load(
        segmentation_method=segmentation, enable_postagging=postag, postag_model=postag_model
    )
    try:
        pipe = TextPipeline(settings, tagger=None if postag else NullPosTagger())
    except _STARTUP_ERRORS as e:
        _fail_startup(e, postag)
    for ts in pipe.process(" ".join(text)):
        typer.echo(f"[{ts.index}] @{ts.sentence.offset}: {ts.sentence.text}")
        for t in ts.tokens:
            if show_ws or not t.is_whitespace:
                typer.echo(f"    {t.start:>5} {t.token_class:<8} {t.pos:<5} {t.text!r}")


@app.command("annotate")
def annotate_cmd(
    text: list[str] = typer.Argument(None, help="Text to annotate (words are joined)"),
    pipe: bool = typer.Option(False, "--pipe", help="Read the text from stdin"),
    input_file: Path | None = typer.Option(
        None, "--input", "-i", exists=True, dir_okay=False, help="Read the text from a file"
    ),
    indexdir: Path = typer.Option(None, help="Index directory (default: settings / ivf)"),
    inputformat: str = typer.Option(
        "freetext",
        help=(
            "freetext | sli | sldi | sldiwi | chemdner | chemdnersldi | ncbicorpus | pubtator"
            " | pubmed | medline | bioc"
        ),
    ),
    outputformat: str = typer.Option(
        "mmi",
        help="mmi | json | brat | cuilist | full | bc (aliases: fulljson, bc-evaluate, cdi, bioc)",
    ),
    restrict_to_sts: str | None = typer.Option(
        None, help="Comma-separated semantic types, e.g. dsyn,sosy"
    ),
    restrict_to_sources: str | None = typer.Option(
        None, help="Comma-separated SABs, e.g. MSH,SNOMEDCT_US"
    ),
    segmentation: str = typer.Option("SENTENCES", help="SENTENCES | BLANKLINES | LINES"),
    postag: bool = typer.Option(True, help="Run the spaCy POS tagger (--no-postag to skip)"),
    postag_model: str | None = typer.Option(
        None, help="spaCy model for POS tags (default: settings / en_core_web_sm)"
    ),
    negation: bool = typer.Option(True, help="Run negation detection"),
    usecontext: bool = typer.Option(
        False, "--usecontext", help="Use ConText instead of NegEx (adds temporality/experiencer)"
    ),
    negation_detector: str | None = typer.Option(
        None,
        help="Negation detector by name: negex, context, or a registered plugin "
        "(default: settings / negex; --usecontext is short for 'context')",
    ),
    strict_parity: bool | None = typer.Option(
        None,
        "--strict-parity/--no-strict-parity",
        help="Reproduce MetaMapLite exactly, bugs included (default: settings / on). "
        "--no-strict-parity fixes ConText's cross-sentence overwrites and trigger regexes, "
        "and sorts output lists instead of using Java hash order",
    ),
    precision_filter: bool | None = typer.Option(
        None,
        "--precision-filter/--no-precision-filter",
        help="Drop case-mismatched acronym concepts ('Plan' -> PLAN) and one-word function "
        "words ('for', 'with') (default: settings / off; not in Java MetaMapLite)",
    ),
    excluded_terms: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="specialterms.txt (CUI:term / *:term lines)"
    ),
    cui_term_list: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="Custom concepts file (CUI|term lines)"
    ),
    uda: Path | None = typer.Option(
        None, exists=True, dir_okay=False, help="User-defined acronyms file (acronym|long form)"
    ),
    preload: bool = typer.Option(
        False,
        help="Load the term map into memory first (~4 s, +1.6 GB RAM). Measured no faster than "
        "SQLite on local disk; only worth trying if the index sits on slow storage",
    ),
    keep_subsumed: bool = typer.Option(False, help="Keep spans contained in longer matches"),
) -> None:
    """Find UMLS concepts in text (equivalent of metamaplite.sh)."""
    import sys

    from .api import MetaMapLite
    from .documents import get_loader

    try:
        loader = get_loader(inputformat)
    except ValueError as e:
        _fail(str(e))

    # Read before loading the index: an empty input should fail in milliseconds, not after a
    # ten-second index load.  ``docid`` is passed explicitly for a file because the loaders
    # reproduce Java's basename logic, which splits on "/" only and would keep a Windows path
    # whole (see documents.freetext.java_basename).
    try:
        if input_file is not None:
            documents = loader.load_file(input_file, docid=input_file.name)
        else:
            if pipe:
                _use_utf8(sys.stdin, newline="")
                raw = sys.stdin.read()
            else:
                raw = " ".join(text or [])
            documents = loader.read(raw)
    except ValueError as e:  # e.g. PubMed XML that is not well-formed
        _fail(str(e))
    if not any(p.text.strip() for d in documents for p in d.passages):
        raise typer.BadParameter("no text given")
    if outputformat.lower() not in FORMATS:
        _fail(f"unknown output format {outputformat!r}; expected one of {FORMATS}")
    if segmentation.upper() not in SEGMENTATION_METHODS:
        _fail(
            f"unknown segmentation method {segmentation!r}; expected one of {SEGMENTATION_METHODS}"
        )
    if bad := unknown_semantic_types(_split(restrict_to_sts)):
        _fail(f"unknown semantic type(s) in --restrict-to-sts: {', '.join(bad)}")
    settings = Settings.load(
        index_directory=indexdir,
        segmentation_method=segmentation,
        enable_postagging=postag,
        postag_model=postag_model,
        detect_negations=negation,
        negation_detector=_detector_name(usecontext, negation_detector),
        strict_parity=strict_parity,
        precision_filter=precision_filter,
        excluded_terms_file=excluded_terms,
        cui_term_list_file=cui_term_list,
        uda_file=uda,
    )
    try:
        mml = MetaMapLite(settings, preload_index=preload, remove_subsumed=not keep_subsumed)
    except _STARTUP_ERRORS as e:
        _fail_startup(e, postag)
    try:
        # Java's processDocumentList concatenates every document's entities and formats them
        # once, so a multi-document input yields one output stream, not one per document.
        entities = mml.process_documents(
            documents,
            restrict_to_sts=_split(restrict_to_sts),
            restrict_to_sources=_split(restrict_to_sources),
        )
        _use_utf8(sys.stdout)
        sys.stdout.write(mml.format(entities, outputformat))
    finally:
        mml.close()


@app.command("serve")
def serve_cmd(
    indexdir: Path = typer.Option(None, help="Index directory (default: settings / ivf)"),
    host: str = typer.Option(
        "127.0.0.1", help="Bind address; use 0.0.0.0 to accept remote clients"
    ),
    port: int = typer.Option(8000),
    segmentation: str = typer.Option("SENTENCES", help="SENTENCES | BLANKLINES | LINES"),
    postag: bool = typer.Option(True, help="Run the spaCy POS tagger"),
    postag_model: str | None = typer.Option(
        None, help="spaCy model for POS tags (default: settings / en_core_web_sm)"
    ),
    negation: bool = typer.Option(True, help="Run negation detection"),
    usecontext: bool = typer.Option(
        False, "--usecontext", help="Use ConText instead of NegEx (adds temporality/experiencer)"
    ),
    negation_detector: str | None = typer.Option(
        None,
        help="Negation detector by name: negex, context, or a registered plugin "
        "(default: settings / negex; --usecontext is short for 'context')",
    ),
    strict_parity: bool | None = typer.Option(
        None,
        "--strict-parity/--no-strict-parity",
        help="Reproduce MetaMapLite exactly, bugs included (default: settings / on). "
        "--no-strict-parity fixes ConText's cross-sentence overwrites and trigger regexes, "
        "and sorts output lists instead of using Java hash order",
    ),
    precision_filter: bool | None = typer.Option(
        None,
        "--precision-filter/--no-precision-filter",
        help="Drop case-mismatched acronym concepts ('Plan' -> PLAN) and one-word function "
        "words ('for', 'with') (default: settings / off; not in Java MetaMapLite)",
    ),
    excluded_terms: Path | None = typer.Option(None, exists=True, dir_okay=False),
    uda: Path | None = typer.Option(None, exists=True, dir_okay=False),
    cui_term_list: Path | None = typer.Option(None, exists=True, dir_okay=False),
    preload: bool = typer.Option(
        False,
        help="Load the term map into memory at start-up (+1.6 GB per worker; measured no "
        "faster on local disk, see OPERATIONS_MANUAL §8.4)",
    ),
    reload: bool = typer.Option(False, help="Auto-reload on code changes (development)"),
) -> None:
    """Serve the REST API (docs at /docs). The index loads once, before the first request."""
    try:
        from .server import serve
    except ImportError as e:
        _fail(f"{e}\nthe REST API needs extra packages: pip install 'mmlite[server]'")

    settings = Settings.load(
        index_directory=indexdir,
        segmentation_method=segmentation,
        enable_postagging=postag,
        postag_model=postag_model,
        detect_negations=negation,
        negation_detector=_detector_name(usecontext, negation_detector),
        strict_parity=strict_parity,
        precision_filter=precision_filter,
        excluded_terms_file=excluded_terms,
        uda_file=uda,
        cui_term_list_file=cui_term_list,
    )
    typer.echo(f"loading index from {settings.index_directory} ...")
    try:
        serve(settings, host=host, port=port, preload_index=preload, reload=reload)
    except (ImportError, OSError) as e:
        _fail(str(e))


@app.command("index-stats")
def index_stats(
    indexdir: Path = typer.Option(None, help="Index directory (default: settings / ivf)"),
) -> None:
    """Print index metadata and row counts."""
    settings = Settings.load(index_directory=indexdir)
    try:
        with IndexLookup(settings.index_directory, check_norm_version=False) as ix:
            stats = ix.stats()
    except _STARTUP_ERRORS as e:
        _fail(str(e))
    for k, v in stats.items():
        typer.echo(f"{k}: {v}")


if __name__ == "__main__":
    app()
