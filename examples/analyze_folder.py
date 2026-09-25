#!/usr/bin/env python
"""Annotate every text file in a folder with UMLS concepts using mmlite.

Usage:
    python examples/analyze_folder.py notes/
    python examples/analyze_folder.py notes/ --output-dir out/ --outputformat json
    python examples/analyze_folder.py notes/ --pattern "*.txt" --recursive --workers 4
    python examples/analyze_folder.py notes/ --restrict-to-sts dsyn,sosy,phsu --workers 4

The index is loaded once and reused across every file (opening it per file would dominate the
run time). One output file is written per input file, plus a summary printed at the end: files
processed, files that errored, total concepts found, and the most frequent concepts across the
whole folder.

Requires a built index (see `mmlite build-index` / docs/USER_GUIDE.md, section 3).

Files are read as UTF-8 unless --encoding says otherwise. A file that isn't valid in that encoding
is still annotated (bad bytes replaced by U+FFFD) but flagged with a warning in its status line
and counted in the summary - the replaced words can no longer match anything.

--workers > 1 uses a process pool instead of one process, since the underlying index/pipeline
objects are not thread-safe (see docs/USER_GUIDE.md, section 11) — each worker process builds
its own MetaMapLite instance and index copy, so only use it for a folder large enough that the
extra RAM and per-worker start-up cost pay for themselves.
"""

from __future__ import annotations

import argparse
import codecs
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from mmlite import MetaMapLite, read_document
from mmlite.config import Settings
from mmlite.semtypes import unknown_semantic_types
from mmlite.types import Entity

OUTPUT_EXTENSIONS = {
    "mmi": ".mmi",
    "json": ".json",
    "brat": ".ann",
    "cuilist": ".cuis",
    "full": ".full",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_dir", type=Path, help="Folder of text files to annotate")
    ap.add_argument(
        "--pattern", default="*.txt", help="Glob pattern for input files (default: *.txt)"
    )
    ap.add_argument("--recursive", action="store_true", help="Also search subdirectories")
    ap.add_argument(
        "--output-dir", type=Path, help="Write results here (default: alongside each input file)"
    )
    ap.add_argument(
        "--index-dir", type=Path, default=Path("ivf"), help="Index directory (default: ivf)"
    )
    ap.add_argument(
        "--outputformat",
        default="mmi",
        choices=("mmi", "json", "brat", "cuilist", "full"),
        help="Output format (default: mmi)",
    )
    ap.add_argument(
        "--restrict-to-sts",
        help="Comma-separated semantic type abbreviations or TUIs, e.g. dsyn,sosy",
    )
    ap.add_argument(
        "--restrict-to-sources", help="Comma-separated UMLS source abbreviations, e.g. MSH"
    )
    ap.add_argument("--no-negation", action="store_true", help="Disable NegEx negation detection")
    ap.add_argument("--no-postag", action="store_true", help="Skip POS tagging (no spaCy needed)")
    ap.add_argument(
        "--preload",
        action="store_true",
        help="Load the whole term index into memory first (§11): ~4 s and ~1.6 GB per process, "
        "and measured slower overall on every batch tried - see examples/README.md",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process pool size for parallel annotation (default: 1)",
    )
    ap.add_argument(
        "--top", type=int, default=10, help="How many top concepts to summarize (default: 10)"
    )
    ap.add_argument(
        "--excluded-terms",
        type=Path,
        help="Path to a specialterms.txt-style exclusion list: lines of `CUI:normalized term` "
        "(or `*:term` for every CUI), keyed on the term shown in the summary's 'matched as' "
        "column - NOT the preferred name. Java applies its own 43-entry list by default and "
        "this port does not; treat it as a starting point, since it has no entry for the "
        "matches that dominate clinical text (e.g. 'for' -> WWOX gene)",
    )
    ap.add_argument(
        "--encoding",
        default="utf-8",
        help="Input text encoding, e.g. cp1252 or latin-1 (default: utf-8)",
    )
    return ap.parse_args(argv)


def _split(csv: str | None) -> set[str] | None:
    return {s.strip() for s in csv.split(",") if s.strip()} if csv else None


@dataclass
class FileResult:
    path: Path
    n_spans: int = 0
    n_negated: int = 0
    concepts: Counter | None = None  # CUI -> occurrence count within this file
    names: dict[str, str] | None = None  # CUI -> preferred name, for the summary
    # CUI -> the normalized terms that matched it. These, not the preferred name, are what an
    # --excluded-terms line has to say (`CUI:normalized term`), so the summary can print them.
    terms: dict[str, Counter] | None = None
    error: str | None = None
    skipped: str | None = None  # why the file was deliberately not annotated (not a failure)
    warning: str | None = None  # annotated, but the output is suspect (e.g. bad encoding)


def _concept_counts(entities: list[Entity]) -> Counter:
    return Counter(ev.cui for e in entities for ev in e.evs)


def _matched_as(counts: Counter | None, width: int = 22) -> str:
    """Render a CUI's normalized terms for the summary, commonest first, truncated to `width`."""
    if not counts:
        return ""
    ordered = [t for t, _ in counts.most_common()]
    shown = ordered[0]
    if len(ordered) > 1:
        shown += f" +{len(ordered) - 1}"
    return shown if len(shown) <= width else shown[: width - 1] + "…"


def _concept_terms(entities: list[Entity]) -> dict[str, Counter]:
    """CUI -> Counter of the normalized terms that matched it (the --excluded-terms key)."""
    terms: dict[str, Counter] = {}
    for e in entities:
        for ev in e.evs:
            terms.setdefault(ev.cui, Counter())[ev.norm_term] += 1
    return terms


def read_text(path: Path, encoding: str) -> tuple[str, str | None]:
    """Read `path` for annotation; return (text, warning). Decodes strictly first so a wrong
    encoding is reported instead of silently turning words into U+FFFD, then falls back to
    replacing the bad bytes so the rest of the file is still annotated."""
    try:
        return read_document(path, encoding, "strict"), None
    except UnicodeDecodeError as e:
        text = read_document(path, encoding, "replace")
        return text, (
            f"not valid {encoding} (first bad byte at offset {e.start}); "
            f"{text.count(chr(0xFFFD))} undecodable byte(s) replaced, so words containing them "
            f"cannot match - rerun with --encoding (e.g. cp1252)"
        )


def _annotate_one(
    mml: MetaMapLite,
    path: Path,
    output_dir: Path | None,
    outputformat: str,
    restrict_to_sts: set[str] | None,
    restrict_to_sources: set[str] | None,
    encoding: str = "utf-8",
) -> FileResult:
    # read_document keeps line endings intact, so reported offsets index the original bytes.
    text, warning = read_text(path, encoding)
    if not text.strip():
        return FileResult(path, skipped="empty file")
    entities = mml.process_text(
        text,
        docid=path.name,
        restrict_to_sts=restrict_to_sts,
        restrict_to_sources=restrict_to_sources,
    )
    out_dir = output_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (path.stem + OUTPUT_EXTENSIONS[outputformat])
    out_path.write_text(mml.format(entities, outputformat), encoding="utf-8")
    return FileResult(
        path,
        n_spans=len(entities),
        n_negated=sum(1 for e in entities if e.negated),
        concepts=_concept_counts(entities),
        names={ev.cui: ev.concept.preferred_name for e in entities for ev in e.evs},
        terms=_concept_terms(entities),
        warning=warning,
    )


# --- worker-process entry point (only used when --workers > 1) ---------------------------------

_worker_mml: MetaMapLite | None = None
_worker_args: argparse.Namespace | None = None


def _init_worker(settings: Settings, args: argparse.Namespace) -> None:
    global _worker_mml, _worker_args
    _worker_mml = MetaMapLite(settings, preload_index=args.preload)
    _worker_args = args


def _worker_annotate(path: Path) -> FileResult:
    assert _worker_mml is not None and _worker_args is not None
    try:
        return _annotate_one(
            _worker_mml,
            path,
            _worker_args.output_dir,
            _worker_args.outputformat,
            _split(_worker_args.restrict_to_sts),
            _split(_worker_args.restrict_to_sources),
            _worker_args.encoding,
        )
    except Exception as e:  # report the failure per file; never abort the whole batch
        return FileResult(path, error=str(e))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input_dir.is_dir():
        print(f"error: {args.input_dir} is not a directory", file=sys.stderr)
        return 1

    glob = args.input_dir.rglob if args.recursive else args.input_dir.glob
    files = sorted(glob(args.pattern))
    if not files:
        print(f"no files matching {args.pattern!r} under {args.input_dir}", file=sys.stderr)
        return 1
    # an unknown abbreviation would otherwise just match nothing: 0 spans everywhere, exit 0
    if bad := unknown_semantic_types(_split(args.restrict_to_sts)):
        print(
            f"error: unknown semantic type(s) in --restrict-to-sts: {', '.join(bad)}",
            file=sys.stderr,
        )
        return 1
    if args.excluded_terms is not None and not args.excluded_terms.is_file():
        print(f"error: {args.excluded_terms} is not a file", file=sys.stderr)
        return 1
    try:
        codecs.lookup(args.encoding)
    except LookupError:
        print(f"error: unknown --encoding {args.encoding!r}", file=sys.stderr)
        return 1

    settings = Settings.load(
        index_directory=args.index_dir,
        enable_postagging=not args.no_postag,
        detect_negations=not args.no_negation,
        excluded_terms_file=args.excluded_terms,
    )

    t0 = time.time()
    results: list[FileResult] = []
    try:
        if args.workers > 1:
            with ProcessPoolExecutor(
                max_workers=args.workers, initializer=_init_worker, initargs=(settings, args)
            ) as pool:
                results = list(pool.map(_worker_annotate, files))
        else:
            with MetaMapLite(settings, preload_index=args.preload) as mml:
                sts, srcs = _split(args.restrict_to_sts), _split(args.restrict_to_sources)
                for path in files:
                    try:
                        results.append(
                            _annotate_one(
                                mml,
                                path,
                                args.output_dir,
                                args.outputformat,
                                sts,
                                srcs,
                                args.encoding,
                            )
                        )
                    except Exception as e:  # one bad file must not abort the batch
                        results.append(FileResult(path, error=str(e)))
    except (FileNotFoundError, RuntimeError, ImportError, OSError) as e:
        # index failed to load at all - nothing downstream can succeed either
        print(f"error: {e}", file=sys.stderr)
        return 1

    ok = [r for r in results if r.error is None and r.skipped is None]
    failed = [r for r in results if r.error is not None]
    skipped = [r for r in results if r.skipped is not None]
    for r in results:
        if r.error:
            status = f"ERROR: {r.error}"
        elif r.skipped:
            status = f"skipped ({r.skipped})"
        else:
            status = f"{r.n_spans} spans, {r.n_negated} negated"
            if r.warning:
                status += f" - WARNING: {r.warning}"
        print(f"{r.path.name}: {status}", file=sys.stderr)

    mentions: Counter = Counter()  # CUI -> occurrences across the corpus
    doc_freq: Counter = Counter()  # CUI -> number of files it appears in
    names: dict[str, str] = {}
    terms: dict[str, Counter] = {}  # CUI -> normalized terms that matched it
    for r in ok:
        if r.concepts:
            mentions.update(r.concepts)
            doc_freq.update(r.concepts.keys())
        names.update(r.names or {})
        for cui, counts in (r.terms or {}).items():
            terms.setdefault(cui, Counter()).update(counts)
    warned = [r for r in ok if r.warning]

    elapsed = time.time() - t0
    print(
        f"\n{len(ok)}/{len(files)} files processed ({len(failed)} failed, {len(skipped)} "
        f"skipped) in {elapsed:.1f}s, {len(mentions)} distinct concepts across the corpus",
        file=sys.stderr,
    )
    if warned:
        print(
            f"{len(warned)} file(s) had encoding problems (see WARNING lines above)",
            file=sys.stderr,
        )
    if mentions and args.top > 0:
        print(f"top {args.top} concepts by number of mentions:", file=sys.stderr)
        print(
            f"  {'mentions':>8}  {'files':>5}  {'cui':<8}  {'matched as':<22}  preferred name",
            file=sys.stderr,
        )
        for cui, count in mentions.most_common(args.top):
            print(
                f"  {count:>8}  {doc_freq[cui]:>5}  {cui:<8}  "
                f"{_matched_as(terms.get(cui)):<22}  {names.get(cui, '')}",
                file=sys.stderr,
            )
        # These counts are raw frequency, not clinical relevance, and the words that dominate
        # them ("for", "with") are rarely what anyone meant. The exclusion list is the lever -
        # and it keys on the "matched as" term above, never on the preferred name.
        print(
            "  (to drop one: add a `CUI:matched as` line to an --excluded-terms file, "
            "e.g. `C0521125:for`)",
            file=sys.stderr,
        )

    return 1 if failed and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
