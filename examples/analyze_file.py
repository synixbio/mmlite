#!/usr/bin/env python
"""Annotate a single text file with UMLS concepts using mmlite.

Usage:
    python examples/analyze_file.py note.txt
    python examples/analyze_file.py note.txt --index-dir ivf --outputformat json -o note.json
    python examples/analyze_file.py note.txt --restrict-to-sts dsyn,sosy,phsu --no-negation

Prints a one-line summary to stderr (so it doesn't pollute piped stdout output) and writes the
formatted annotation to stdout, or to the file given with -o/--output (creating its folder if
needed).

The input is read as UTF-8 unless --encoding says otherwise. A file that isn't valid in that
encoding (e.g. a Windows-1252 export containing "Sjögren") is still annotated, with the bad bytes
replaced by U+FFFD, but a warning is printed: the replaced words can no longer match anything,
so rerun with the right --encoding rather than trusting that output.

Requires a built index (see `mmlite build-index` / docs/USER_GUIDE.md, section 3).
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import os
import sys
from pathlib import Path

from mmlite import MetaMapLite, read_document
from mmlite.config import Settings
from mmlite.semtypes import unknown_semantic_types


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, help="Text file to annotate")
    ap.add_argument(
        "-o", "--output", type=Path, help="Write formatted output here (default: stdout)"
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
        "--excluded-terms",
        type=Path,
        help="Path to a specialterms.txt-style exclusion list: lines of `CUI:normalized term` "
        "(or `*:term` for every CUI), keyed on the normalized matched term - NOT the preferred "
        "name. Java applies its own 43-entry list by default and this port does not; treat it "
        "as a starting point, since it has no entry for the matches that dominate clinical "
        "text (e.g. 'for' -> WWOX gene). analyze_folder.py prints the term to use",
    )
    ap.add_argument(
        "--encoding",
        default="utf-8",
        help="Input text encoding, e.g. cp1252 or latin-1 (default: utf-8)",
    )
    return ap.parse_args(argv)


def _split(csv: str | None) -> set[str] | None:
    return {s.strip() for s in csv.split(",") if s.strip()} if csv else None


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.input.is_file():
        print(f"error: {args.input} is not a file", file=sys.stderr)
        return 1
    # an unknown abbreviation would otherwise just match nothing: 0 spans, exit 0, no hint why
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

    try:
        # read_document keeps line endings intact, so reported offsets index the original text.
        text, warning = read_text(args.input, args.encoding)
        if warning:
            print(f"warning: {args.input.name}: {warning}", file=sys.stderr)
        with MetaMapLite(settings) as mml:
            entities = mml.process_text(
                text,
                docid=args.input.name,
                restrict_to_sts=_split(args.restrict_to_sts),
                restrict_to_sources=_split(args.restrict_to_sources),
            )
            formatted = mml.format(entities, args.outputformat)
    except (FileNotFoundError, RuntimeError, ImportError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    n_negated = sum(1 for e in entities if e.negated)
    n_concepts = len({ev.cui for e in entities for ev in e.evs})
    print(
        f"{args.input.name}: {len(entities)} spans, {n_concepts} distinct concepts, "
        f"{n_negated} negated",
        file=sys.stderr,
    )

    if args.output:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(formatted, encoding="utf-8")
        except OSError as e:
            print(f"error: cannot write {args.output}: {e.strerror or e}", file=sys.stderr)
            return 1
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        try:
            sys.stdout.write(formatted)
            sys.stdout.flush()
        except BrokenPipeError:
            # A downstream reader closed early (`| head`, `| grep -m1`, quitting `less`). That's
            # normal use, not a failure - but Python would still flush stdout at interpreter exit
            # and print "Exception ignored ... BrokenPipeError" to stderr. Pointing our fd at
            # devnull makes that final flush a no-op, so we exit quietly.
            # Suppressed because stdout may not be a real file descriptor (replaced by a test
            # harness, or already closed - io.UnsupportedOperation subclasses both of these);
            # then there is no pending fd-level flush to silence and this is moot anyway.
            with contextlib.suppress(OSError, ValueError):
                os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
