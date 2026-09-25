#!/usr/bin/env python
"""Find documents that affirm a clinical concept — simple phenotyping.

Usage:
    python examples/notes_of_interest.py free_texts/synthetic "diabetes mellitus"
    python examples/notes_of_interest.py free_texts/synthetic --cui C0011860 C0018802
    python examples/notes_of_interest.py free_texts/synthetic "chest pain" --show-negated
    python examples/notes_of_interest.py free_texts/synthetic "chest pain" --save cohort.sqlite

**Why this needs NLP and not grep.** A keyword search for "chest pain" matches "patient denies
chest pain" — the exact opposite of what you want. This script counts a document only when the
concept is mentioned *and not negated*, and `--show-negated` exists so you can see how often that
distinction fires. On a real corpus it fires a lot.

Concepts are matched by CUI, so morphological and synonym variants ("diabetes mellitus type 2",
"T2DM", "DM2") collapse to one concept automatically. That is the other thing grep cannot do.

A bare phrase is resolved against the index first (the same lookup `mmlite lookup` does)
and every CUI it resolves to becomes a target — which is why the resolved concepts are printed
before the scan: a phrase that resolves to something you did not mean is the most common way a
cohort goes wrong. `--cui` skips resolution when you already know what you want.

## Saving a search

`--save PATH` writes the search to SQLite, and **appends** rather than overwrites, so one file
accumulates a searchable history of what was asked. Four tables:

  * `search` — one row per run: what was asked, over what corpus, with what totals
  * `concepts` — what the phrase resolved to, for that run
  * `documents` — one row per file scanned, **including the ones with no mention**
  * `console` — the printed report, verbatim, one row per line

`documents` carries every file, not only the hits, because "absent" is a result: a cohort
denominator cannot be reconstructed from a list of matches. The console text is stored beside the
structured tables rather than instead of them — the tables are what you query, the transcript is
what you show someone who asks what you actually ran.

Nothing is written when the search fails early (an unresolvable phrase, an empty corpus), so a
`search` row always means a search that ran.

Requires a built index (see `mmlite build-index`).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _annotations import (
    add_pipeline_args,
    build_settings,
    document_label,
    duplicate_label_warning,
    iter_text_files,
    read_text,
    split_csv,
    validate_common,
)

from mmlite import MetaMapLite

SCHEMA = """
CREATE TABLE IF NOT EXISTS search (
    search_id   INTEGER PRIMARY KEY,
    ran_at      TEXT NOT NULL,
    term        TEXT,               -- NULL when --cui was used
    cuis        TEXT NOT NULL,      -- comma-joined targets, resolved or given
    corpus      TEXT NOT NULL,
    n_documents INTEGER NOT NULL,
    n_affirmed  INTEGER NOT NULL,
    n_negated   INTEGER NOT NULL,   -- documents mentioning it *only* as negated
    n_absent    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS concepts (
    search_id      INTEGER NOT NULL REFERENCES search(search_id),
    cui            TEXT NOT NULL,
    preferred_name TEXT,
    semantic_types TEXT
);

CREATE TABLE IF NOT EXISTS documents (
    search_id  INTEGER NOT NULL REFERENCES search(search_id),
    source     TEXT NOT NULL,
    path       TEXT NOT NULL,
    status     TEXT NOT NULL,       -- 'affirmed' | 'negated_only' | 'absent'
    n_affirmed INTEGER NOT NULL,
    n_negated  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS console (
    search_id INTEGER NOT NULL REFERENCES search(search_id),
    line_no   INTEGER NOT NULL,
    line      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_doc_search ON documents(search_id, status);
CREATE INDEX IF NOT EXISTS ix_con_search ON concepts(search_id, cui);
"""


class Console:
    """Print to stdout and keep a verbatim transcript for the `console` table."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.extend(text.split("\n"))


def resolve_term(mml: MetaMapLite, term: str) -> list[tuple[str, str, str]]:
    """(cui, preferred_name, semantic_types) for every concept `term` resolves to."""
    seen: dict[str, tuple[str, str, str]] = {}
    for entity in mml.lookup_term(term):
        for ev in entity.evs:
            seen.setdefault(
                ev.cui,
                (ev.cui, ev.concept.preferred_name, ",".join(sorted(ev.concept.semantic_types))),
            )
    return sorted(seen.values())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input_dir", type=Path, help="Folder of text files to search")
    ap.add_argument("term", nargs="?", default=None, help="Concept phrase to resolve and search")
    ap.add_argument("--cui", nargs="+", default=None, help="Search these CUIs instead of a phrase")
    ap.add_argument(
        "--show-negated", action="store_true", help="List the negated-only documents too"
    )
    ap.add_argument("--save", type=Path, default=None, help="Append this search to a SQLite file")
    add_pipeline_args(ap)
    # intermixed: 3.11 argparse rejects the optional "term" positional once a flag precedes it
    args = ap.parse_intermixed_args(argv)
    validate_common(args, ap)

    if not args.term and not args.cui:
        ap.error("give a search term or --cui")
    if args.term and args.cui:
        ap.error("give a search term or --cui, not both")
    if args.no_negation:
        # The whole point of this script is the affirmed/negated split.
        ap.error("--no-negation would make every mention count as affirmed")
    if not args.input_dir.is_dir():
        print(f"error: {args.input_dir} is not a directory", file=sys.stderr)
        return 1

    files = iter_text_files(args.input_dir, args.pattern, args.recursive)
    if not files:
        print(f"no files matching {args.pattern!r} under {args.input_dir}", file=sys.stderr)
        return 1
    if collision := duplicate_label_warning(files):
        print(collision, file=sys.stderr)

    say = Console()
    settings = build_settings(args)
    sts, srcs = split_csv(args.restrict_to_sts), split_csv(args.restrict_to_sources)
    started = time.time()

    try:
        with MetaMapLite(settings, preload_index=args.preload) as mml:
            if args.cui:
                targets = {c.upper() for c in args.cui}
                concepts: list[tuple[str, str, str]] = []
                say(f"Searching for CUIs: {', '.join(sorted(targets))}")
            else:
                concepts = resolve_term(mml, args.term)
                if not concepts:
                    print(
                        f"No concept matches {args.term!r} in the index.\n"
                        "Try the concept's canonical name, or search by --cui.",
                        file=sys.stderr,
                    )
                    return 1
                targets = {cui for cui, _, _ in concepts}
                say(f"{args.term!r} resolves to {len(concepts)} concept(s):")
                for cui, pref, types in concepts:
                    say(f"  {cui}  {types:<12} {pref}")

            affirmed: list[tuple[str, int]] = []
            negated_only: list[str] = []
            # Every file, including the ones with no mention: "absent" is a result, and a
            # denominator cannot be recovered from a list of hits.
            rows: list[tuple[str, str, str, int, int]] = []

            for path in files:
                text, _ = read_text(path, args.encoding)
                entities = mml.process_text(
                    text,
                    docid=document_label(path),
                    restrict_to_sts=sts,
                    restrict_to_sources=srcs,
                )
                # One span can carry several CUIs; a span counts once if any of them is a target.
                hits = [e for e in entities if targets & set(e.cuis)]
                yes = sum(1 for e in hits if not e.negated)
                no = sum(1 for e in hits if e.negated)
                if yes:
                    status = "affirmed"
                    affirmed.append((document_label(path), yes))
                elif no:
                    status = "negated_only"
                    negated_only.append(document_label(path))
                else:
                    status = "absent"
                rows.append((document_label(path), str(path), status, yes, no))
    except (FileNotFoundError, RuntimeError, ImportError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    n_absent = len(files) - len(affirmed) - len(negated_only)
    say(f"\nScanned {len(files)} documents in {time.time() - started:.1f}s.")
    say(f"  {len(affirmed)} affirm the concept")
    say(f"  {len(negated_only)} mention it only as negated")
    say(f"  {n_absent} do not mention it\n")

    if affirmed:
        say("Affirmed in:")
        for name, n in sorted(affirmed, key=lambda kv: (-kv[1], kv[0])):
            say(f"  {name:<52} {n} mention(s)")

    if args.show_negated and negated_only:
        say("\nNegated-only (excluded from the cohort):")
        for name in sorted(negated_only):
            say(f"  {name}")
    elif negated_only:
        say(f"\n({len(negated_only)} negated-only documents hidden; --show-negated to list)")

    if args.save is None:
        return 0

    # Written only now: an early return above means the search never ran, and a `search` row
    # should not exist for one that did not.
    args.save.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.save)
    try:
        conn.executescript(SCHEMA)
        cur = conn.execute(
            "INSERT INTO search (ran_at, term, cuis, corpus, n_documents, n_affirmed, "
            "n_negated, n_absent) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(timespec="seconds"),
                args.term,
                ",".join(sorted(targets)),
                str(args.input_dir),
                len(files),
                len(affirmed),
                len(negated_only),
                n_absent,
            ),
        )
        search_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO concepts (search_id, cui, preferred_name, semantic_types) "
            "VALUES (?, ?, ?, ?)",
            [(search_id, *c) for c in concepts],
        )
        conn.executemany(
            "INSERT INTO documents (search_id, source, path, status, n_affirmed, n_negated) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(search_id, *r) for r in rows],
        )
        conn.executemany(
            "INSERT INTO console (search_id, line_no, line) VALUES (?, ?, ?)",
            [(search_id, i, line) for i, line in enumerate(say.lines, 1)],
        )
        conn.commit()
    finally:
        conn.close()

    # Flush stdout first, or a redirected (block-buffered) report lands after this line.
    sys.stdout.flush()
    print(f"\n-> saved as search {search_id} in {args.save}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
