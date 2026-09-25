#!/usr/bin/env python
"""Shared row shape for the flat one-row-per-(span, concept) annotation table.

Not a script -- import it. Every script that writes or reads this table (`notes_of_interest.py`,
and the exports `load_to_sqlite.py` loads) must agree on the same rows for the same corpus, and
that only holds if the row shape has **one** definition: several copies of the same column list
drift silently.

So this module owns four things those scripts share: the column list, the mapping from a
:class:`~mmlite.types.Entity` to rows, the common command-line flags, and the rule for
which assertion attributes a given pipeline actually assessed.

## One row per (span, concept)

mmlite returns one :class:`Entity` per matched span, each carrying one `Ev` per candidate
CUI -- "cold" is one span with several concepts. These exports are flat, so a span with three
candidate CUIs becomes three rows that share `document`, `start`, `end` and `text`. Count spans
with ``COUNT(DISTINCT document, start, end)``, not ``COUNT(*)``.

## Assessed vs. asserted false

The assertion columns are nullable **on purpose**, and the distinction is the reason this module
asks the settings rather than the data:

* ``negated = 0`` means the negation detector ran and found no negation.
* ``negated`` empty/NULL means nothing assessed it -- the run passed ``--no-negation``.

Those are different facts and a query that lumps them together is comparing an absence of evidence
with evidence of absence. The same holds for `assertion`, `temporality` and `experiencer`, which
**only ConText sets**: under the default NegEx detector Java's pipeline never computes them, so
they are NULL rather than false. Pass ``--usecontext`` to get them (see `Entity.temporality` in
`mmlite/types.py`).

An attribute nothing in the run assesses gets **no column at all**, rather than a column that is
empty in every row: that would state a fact about the pipeline once per row. `--all-attributes`
keeps every column regardless, for a downstream schema that must not change shape between runs.
"""

from __future__ import annotations

import argparse
import codecs
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from mmlite import read_document
from mmlite.config import Settings
from mmlite.types import Entity

#: Every column an export can carry, in order. `semantic_types` replaces what a semantic-group
#: pipeline would call `group`: mmlite resolves semantic *types* (`dsyn`, `sosy`) and has
#: no group layer, and a concept can hold several, so it is a list -- comma-joined for the flat
#: formats (CSV, SQLite) and a real list where the format has one (JSONL, Parquet).
COLUMNS: tuple[str, ...] = (
    "document",
    "cui",
    "preferred_name",
    # The dictionary string that matched, which is what explains a surprising hit -- `term` is
    # how you find the `--excluded-terms` line that would drop it. Kept in every format: dropping
    # it from one made the same corpus look different depending on which script wrote it.
    "term",
    "semantic_types",
    "negated",
    "assertion",
    "temporality",
    "experiencer",
    "start",
    "end",
    "text",
)

#: The assertion columns, in COLUMNS order. `negated` is the only one NegEx sets.
ASSERTION_COLUMNS: tuple[str, ...] = ("negated", "assertion", "temporality", "experiencer")


def assessed_attributes(settings: Settings) -> frozenset[str]:
    """Which assertion attributes this configuration actually computes.

    Asked of the settings, not of the output. The alternative -- write every column, then drop the
    ones that came out empty -- cannot be done while streaming, and would also be wrong: a column
    belongs here because something assessed it, not because some mention happened to have it. A
    corpus where nothing is negated is not the same as a run with negation switched off.
    """
    if not settings.detect_negations:
        return frozenset()
    if settings.negation_detector == "context":
        return frozenset(ASSERTION_COLUMNS)
    return frozenset({"negated"})


def columns_for(
    assessed: frozenset[str], *, no_text: bool = False, all_attributes: bool = False
) -> list[str]:
    """:data:`COLUMNS` minus what this run would leave empty in every row."""
    skip = set() if all_attributes else set(ASSERTION_COLUMNS) - assessed
    if no_text:
        skip.add("text")
    return [c for c in COLUMNS if c not in skip]


def dropped_columns(columns: Iterable[str]) -> list[str]:
    """Assertion columns absent from `columns`, for the "not written" notice."""
    return [c for c in ASSERTION_COLUMNS if c not in set(columns)]


# --- corpus -----------------------------------------------------------------------------------


def document_label(path: Path) -> str:
    """The note's **file name**, which is what every export keys documents by.

    Never the path: a path records where the corpus happened to be mounted, differs between
    machines, and stops two exports of one corpus joining on `document`.
    """
    return path.name


def iter_text_files(input_dir: Path, pattern: str = "*.txt", recursive: bool = False) -> list[Path]:
    glob = input_dir.rglob if recursive else input_dir.glob
    return sorted(glob(pattern))


def duplicate_label_warning(paths: Iterable[Path]) -> str | None:
    """Warn when two notes share a file name -- they would merge into one `document` silently."""
    seen: dict[str, int] = {}
    for p in paths:
        seen[document_label(p)] = seen.get(document_label(p), 0) + 1
    dupes = sorted(name for name, n in seen.items() if n > 1)
    if not dupes:
        return None
    shown = ", ".join(dupes[:5]) + (f" (+{len(dupes) - 5} more)" if len(dupes) > 5 else "")
    return (
        f"warning: {len(dupes)} file name(s) occur more than once, so their rows will merge "
        f"under one `document` key: {shown}"
    )


def read_text(path: Path, encoding: str = "utf-8") -> tuple[str, str | None]:
    """Read `path` for annotation; return (text, warning).

    Decodes strictly first so a wrong `--encoding` is reported rather than silently turning words
    into U+FFFD, then falls back to replacing the bad bytes so the rest of the note is still
    annotated. Mirrors `analyze_folder.py`, and `read_document` keeps line endings intact so the
    offsets in these exports index the original file verbatim.
    """
    try:
        return read_document(path, encoding, "strict"), None
    except UnicodeDecodeError as e:
        text = read_document(path, encoding, "replace")
        return text, (
            f"not valid {encoding} (first bad byte at offset {e.start}); "
            f"{text.count(chr(0xFFFD))} undecodable byte(s) replaced, so words containing them "
            f"cannot match - rerun with --encoding (e.g. cp1252)"
        )


# --- rows -------------------------------------------------------------------------------------


def annotation_rows(entities: list[Entity], document: str) -> Iterator[dict[str, Any]]:
    """Flatten a document's entities into one row per (span, concept).

    Values are kept in their natural Python types -- `negated` is a bool or None, `semantic_types`
    is a list -- and each exporter spells them for its own format. Doing the spelling here would
    force CSV's "" and Parquet's null to be the same string.
    """
    for e in entities:
        common = {
            "document": document,
            "negated": e.negated,
            "assertion": e.assertion,
            "temporality": e.temporality,
            "experiencer": e.experiencer,
            "start": e.start,
            "end": e.end,
            "text": e.text,
        }
        for ev in e.evs:
            yield {
                **common,
                "cui": ev.cui,
                "preferred_name": ev.concept.preferred_name,
                "term": ev.concept.concept_string,
                # Sorted so two exports of one corpus compare equal; the set's own order is not
                # observable (see ConceptInfo.semantic_types in mmlite/types.py).
                "semantic_types": sorted(ev.concept.semantic_types),
            }


def row_for(
    row: dict[str, Any], columns: Iterable[str], assessed: frozenset[str]
) -> dict[str, Any]:
    """Restrict a row to `columns`, blanking attributes this run never assessed.

    A kept-but-unassessed column (``--all-attributes``) must be None, not False: it is the "not
    assessed" case, and writing 0 there would claim an assessment that never happened.
    """
    out: dict[str, Any] = {}
    for c in columns:
        out[c] = None if c in ASSERTION_COLUMNS and c not in assessed else row[c]
    return out


# --- command line -----------------------------------------------------------------------------


def add_pipeline_args(ap: argparse.ArgumentParser, *, workers: bool = False) -> None:
    """The flags every exporter shares, spelled as `analyze_folder.py` spells them."""
    ap.add_argument("--pattern", default="*.txt", help="Glob for input files (default: *.txt)")
    ap.add_argument("--recursive", action="store_true", help="Also search subdirectories")
    ap.add_argument(
        "--index-dir", type=Path, default=Path("ivf"), help="Index directory (default: ivf)"
    )
    ap.add_argument(
        "--restrict-to-sts",
        help="Comma-separated semantic type abbreviations or TUIs, e.g. dsyn,sosy",
    )
    ap.add_argument(
        "--restrict-to-sources", help="Comma-separated UMLS source abbreviations, e.g. MSH"
    )
    ap.add_argument("--no-negation", action="store_true", help="Disable negation detection")
    ap.add_argument("--no-postag", action="store_true", help="Skip POS tagging (no spaCy needed)")
    ap.add_argument(
        "--usecontext",
        action="store_true",
        help="Use ConText instead of NegEx, which also assesses assertion, temporality and "
        "experiencer (they are NULL otherwise, meaning not assessed)",
    )
    ap.add_argument(
        "--excluded-terms", type=Path, help="specialterms.txt-style exclusion list (see README)"
    )
    ap.add_argument("--encoding", default="utf-8", help="Input text encoding (default: utf-8)")
    ap.add_argument(
        "--preload",
        action="store_true",
        help="Load the whole term index into memory first: ~4 s and ~1.6 GB per process, and "
        "measured slower overall on every batch tried - see examples/README.md",
    )
    ap.add_argument(
        "--all-attributes",
        action="store_true",
        help="Keep every assertion column, including ones this run does not assess (they are "
        "empty/NULL in every row)",
    )
    if workers:
        ap.add_argument("--workers", type=int, default=1, help="Process pool size (default: 1)")


def split_csv(value: str | None) -> set[str] | None:
    return {s.strip() for s in value.split(",") if s.strip()} if value else None


def build_settings(args: argparse.Namespace) -> Settings:
    """`Settings` for the common flags. `--usecontext` selects the ConText detector."""
    return Settings.load(
        index_directory=args.index_dir,
        enable_postagging=not args.no_postag,
        detect_negations=not args.no_negation,
        negation_detector="context" if args.usecontext else "negex",
        excluded_terms_file=args.excluded_terms,
    )


def validate_common(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    """Fail fast on flags that would otherwise just quietly match nothing."""
    from mmlite.semtypes import unknown_semantic_types

    if args.usecontext and args.no_negation:
        ap.error("--usecontext conflicts with --no-negation: ConText *is* the detector")
    if bad := unknown_semantic_types(split_csv(args.restrict_to_sts)):
        # Otherwise an unknown abbreviation just matches nothing: 0 rows everywhere, exit 0.
        ap.error(f"unknown semantic type(s) in --restrict-to-sts: {', '.join(bad)}")
    if args.excluded_terms is not None and not args.excluded_terms.is_file():
        ap.error(f"{args.excluded_terms} is not a file")
    try:
        codecs.lookup(args.encoding)
    except LookupError:
        ap.error(f"unknown --encoding {args.encoding!r}")
