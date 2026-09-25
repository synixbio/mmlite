#!/usr/bin/env python
"""Load mmlite JSON annotation output into a SQLite database for querying.

Usage:
    mmlite annotate --input free_texts/synthetic/note01.txt --outputformat json > note01.json
    python examples/json_to_sqlite.py note01.json
    python examples/json_to_sqlite.py out/*.json --db examples/annotations.sqlite
    python examples/json_to_sqlite.py out/ --pattern "*.json" --db examples/annotations.sqlite

Reads one or more files produced by `analyze_file.py --outputformat json` / `analyze_folder.py
--outputformat json` (see docs/USER_GUIDE.md, section 6 "json") and loads them into a normalized
SQLite database:

    entities                one row per matched span (docid, text, start, length, negated)
    concepts                one row per distinct CUI (cui, preferred_name)
    concept_semantic_types  CUI -> semantic type abbreviation (many-to-many)
    concept_sources         CUI -> UMLS source vocabulary (many-to-many)
    evidence                one row per (entity, CUI) match — links entities to concepts

Re-running for a docid already in the database replaces that docid's rows first, so this script
is safe to re-run after re-annotating a file. Wildcards on the command line are expanded by this
script itself (via glob), so `*.json` works the same whether your shell would have expanded it
or not (PowerShell, unlike POSIX shells, does not expand `*` before handing it to a program).
Files may be UTF-8 with or without a byte-order mark, or UTF-16 - what PowerShell's `>` writes
(`mmlite annotate ... --outputformat json > note.json`) depending on its version. If
every input fails to load and the database didn't exist before, it is not left behind empty.

Example queries once loaded:

    sqlite3 examples/annotations.sqlite "select cui, preferred_name from concepts limit 5"
    sqlite3 examples/annotations.sqlite "select docid, count(*) from entities group by docid"
    sqlite3 examples/annotations.sqlite \\
        "select c.preferred_name, count(*) from evidence e
         join concepts c on c.cui = e.cui
         group by e.cui order by count(*) desc limit 10"
"""

from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    docid TEXT NOT NULL,
    matched_text TEXT NOT NULL,
    fieldid TEXT,
    start INTEGER NOT NULL,
    length INTEGER NOT NULL,
    negated INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS concepts (
    cui TEXT PRIMARY KEY,
    preferred_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS concept_semantic_types (
    cui TEXT NOT NULL REFERENCES concepts(cui),
    semantic_type TEXT NOT NULL,
    PRIMARY KEY (cui, semantic_type)
);

CREATE TABLE IF NOT EXISTS concept_sources (
    cui TEXT NOT NULL REFERENCES concepts(cui),
    source TEXT NOT NULL,
    PRIMARY KEY (cui, source)
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY,
    entity_id INTEGER NOT NULL REFERENCES entities(id),
    cui TEXT NOT NULL REFERENCES concepts(cui),
    matched_text TEXT NOT NULL,
    concept_string TEXT NOT NULL,
    score REAL NOT NULL,
    start INTEGER NOT NULL,
    length INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_entities_docid ON entities(docid);
CREATE INDEX IF NOT EXISTS ix_evidence_entity ON evidence(entity_id);
CREATE INDEX IF NOT EXISTS ix_evidence_cui ON evidence(cui);
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "inputs", nargs="+", help="JSON file(s), glob pattern(s), or a directory (with --pattern)"
    )
    ap.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern used when an input is a directory (default: *.json)",
    )
    ap.add_argument(
        "--db",
        type=Path,
        default=Path(__file__).parent / "annotations.sqlite",
        help="SQLite database to write (default: examples/annotations.sqlite)",
    )
    return ap.parse_args(argv)


def _resolve_inputs(raw: list[str], pattern: str) -> list[Path]:
    """Expand directories (via --pattern) and glob wildcards a shell may not have expanded."""
    paths: list[Path] = []
    for item in raw:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob(pattern)))
        elif p.is_file():
            paths.append(p)
        elif any(ch in item for ch in "*?["):
            # glob.glob, not Path().glob: the latter rejects absolute patterns (C:\out\*.json),
            # which is exactly what a PowerShell user tab-completes.
            paths.extend(Path(p) for p in sorted(glob.glob(item)))
        else:
            print(
                f"warning: {item!r} is not a file, directory, or glob match; skipping",
                file=sys.stderr,
            )
    return paths


def load_file(con: sqlite3.Connection, path: Path) -> tuple[int, int]:
    """Insert one JSON annotation file, replacing any existing rows for the same docid(s).

    Wrapped in a savepoint: this replaces a docid by deleting its rows first, so a file that
    turns out to be malformed half-way through must not leave the old rows deleted and the new
    ones half-written.  Either the whole file loads or the database is untouched by it.
    """
    con.execute("SAVEPOINT load_file")
    try:
        counts = _load_file(con, path)
    except BaseException:
        con.execute("ROLLBACK TO load_file")
        raise
    finally:
        con.execute("RELEASE load_file")
    return counts


def _load_file(con: sqlite3.Connection, path: Path) -> tuple[int, int]:
    # bytes, not text: json.loads then detects UTF-8 (with or without BOM) and UTF-16/32 itself
    entities = json.loads(path.read_bytes())
    if not isinstance(entities, list):
        raise TypeError(f"expected a list of entities, got {type(entities).__name__}")
    docids = {e["docid"] for e in entities}
    for docid in docids:
        entity_ids = [
            row[0] for row in con.execute("SELECT id FROM entities WHERE docid = ?", (docid,))
        ]
        if entity_ids:
            con.executemany("DELETE FROM evidence WHERE entity_id = ?", [(i,) for i in entity_ids])
            con.execute("DELETE FROM entities WHERE docid = ?", (docid,))

    n_entities = n_evidence = 0
    for e in entities:
        cur = con.execute(
            "INSERT INTO entities (docid, matched_text, fieldid, start, length, negated) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                e["docid"],
                e["matchedtext"],
                e.get("fieldid"),
                e["start"],
                e["length"],
                int(e["negated"]),
            ),
        )
        entity_id = cur.lastrowid
        n_entities += 1
        for ev in e["evlist"]:
            ci = ev["conceptinfo"]
            con.execute(
                "INSERT OR IGNORE INTO concepts (cui, preferred_name) VALUES (?, ?)",
                (ci["cui"], ci["preferredname"]),
            )
            con.executemany(
                "INSERT OR IGNORE INTO concept_semantic_types (cui, semantic_type) VALUES (?, ?)",
                [(ci["cui"], st) for st in ci["semantictypes"]],
            )
            con.executemany(
                "INSERT OR IGNORE INTO concept_sources (cui, source) VALUES (?, ?)",
                [(ci["cui"], src) for src in ci["sources"]],
            )
            con.execute(
                "INSERT INTO evidence (entity_id, cui, matched_text, concept_string, score, start, "
                "length) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entity_id,
                    ci["cui"],
                    ev["matchedtext"],
                    ci["conceptstring"],
                    ev["score"],
                    ev["start"],
                    ev["length"],
                ),
            )
            n_evidence += 1
    return n_entities, n_evidence


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = _resolve_inputs(args.inputs, args.pattern)
    if not files:
        print("error: no input JSON files found", file=sys.stderr)
        return 1

    args.db.parent.mkdir(parents=True, exist_ok=True)
    db_existed = args.db.exists()
    con = sqlite3.connect(args.db)
    con.executescript(SCHEMA)

    total_entities = total_evidence = n_ok = n_failed = 0
    try:
        for path in files:
            try:
                n_entities, n_evidence = load_file(con, path)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                print(f"error: {path}: not a valid annotation JSON file ({e})", file=sys.stderr)
                n_failed += 1
                continue
            print(
                f"{path.name}: {n_entities} entities, {n_evidence} evidence rows", file=sys.stderr
            )
            total_entities += n_entities
            total_evidence += n_evidence
            n_ok += 1
        con.commit()
    finally:
        con.close()
        if not n_ok and not db_existed:
            args.db.unlink(missing_ok=True)  # nothing loaded: don't leave an empty schema behind

    print(
        f"\nloaded {n_ok}/{len(files)} file(s) ({n_failed} failed): {total_entities} entities, "
        f"{total_evidence} evidence rows -> "
        f"{args.db if n_ok or db_existed else '(no database written)'}",
        file=sys.stderr,
    )
    return 1 if n_failed and not n_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
