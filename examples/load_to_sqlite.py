#!/usr/bin/env python
"""Load an exported annotation file (CSV or JSONL) into a queryable SQLite database.

Usage:
    python examples/load_to_sqlite.py examples/annotations.csv   out/annotations.db
    python examples/load_to_sqlite.py examples/annotations.jsonl out/annotations.db --append

Completes the workflow **analyze -> export -> load -> query**: the export is the flat
one-row-per-(span, concept) table defined in `_annotations.py`, and loading it here means the same
annotations can be queried, or loaded more than once, without re-running the pipeline.

Format is detected from the extension:

  * `.csv`   — one row per (span, concept), `semantic_types` comma-joined, `negated` as 0/1
  * `.jsonl` — one record per document (`source`), its rows nested under `annotations`

## Two tables

`documents` carries one row per note **including notes with no annotations**, because "absent"
is a result: a cohort denominator cannot be reconstructed from a list of matches. `annotations`
carries one row per (span, concept). A span with three candidate CUIs is three rows sharing
`doc_id`, `start_offset`, `end_offset` and `text`, so count spans with
`COUNT(DISTINCT doc_id, start_offset, end_offset)` and not `COUNT(*)`.

## Why some columns are nullable

`negated` and the three ConText attributes are **nullable on purpose**: NULL means "not
assessed", which is a different fact from 0 ("assessed, and absent"). mmlite's default
NegEx detector assesses `negated` only and leaves `assertion`, `temporality` and `experiencer`
NULL; `--usecontext` assesses all four; `--no-negation` assesses none. A query that lumps NULL
and 0 together is comparing an absence of evidence with evidence of absence, so use
`IS NULL` / `IS NOT NULL` rather than `= 0`, and note that `SUM(col)` skips NULLs while
`COUNT(*)` does not.

## Why the schema is generated

`ANNOTATION_COLUMNS` is a table rather than one `CREATE TABLE` literal because not every database
wants every column: this script keeps whatever the export it is reading actually has, and
`schema(omit=...)` builds the narrower table for any caller that knows its run never assessed an
attribute. Generating both from one definition is what keeps the shapes compatible: the narrow
table is the wide one minus columns, never a second schema that drifted.

Note the two column renames — `start`/`end` in the flat exports are `start_offset`/`end_offset`
here, because `END` is a reserved word in SQL.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

BATCH = 5_000

DOCUMENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        INTEGER PRIMARY KEY,
    source        TEXT NOT NULL UNIQUE,
    n_annotations INTEGER NOT NULL DEFAULT 0
);
"""

#: The ``annotations`` table, column by column: name -> (declaration, comment).
ANNOTATION_COLUMNS: dict[str, tuple[str, str]] = {
    "cui": ("TEXT NOT NULL", ""),
    "preferred_name": ("TEXT", ""),
    "term": ("TEXT", "the dictionary string that matched - the --excluded-terms key"),
    "semantic_types": ("TEXT", "comma-joined: a concept can carry several"),
    # NULL is the load-bearing value throughout: "not assessed", not "assessed false".
    "negated": ("INTEGER", "0/1, not TRUE/FALSE: SQLite has no bool. NULL = --no-negation"),
    "assertion": ("TEXT", "ConText only: 'Affirmed' | 'Negated' | 'Possible'"),
    "temporality": ("TEXT", "ConText only: 'Recent' | 'Historical' | 'Hypothetical'"),
    "experiencer": ("TEXT", "ConText only: 'Patient' | 'Other'"),
    "start_offset": ("INTEGER", "`start` in the flat exports"),
    "end_offset": ("INTEGER", "`end` in the flat exports; END is a reserved word"),
    "text": ("TEXT", ""),
}

#: Export column name -> SQLite column name, for the two that cannot keep their spelling.
RENAMES: dict[str, str] = {"start": "start_offset", "end": "end_offset"}

#: Indexes, each naming the columns it needs. One that names a column the table does not have is
#: skipped rather than failing the load -- see :func:`indexes`.
_INDEXES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("ix_ann_cui", ("cui",), ""),
    ("ix_ann_doc", ("doc_id",), ""),
    ("ix_ann_negated", ("negated",), ""),
    (
        "ix_ann_cui_neg",
        ("cui", "negated"),
        "the cohort query ('affirmed mentions of CUI X') hits this",
    ),
    # Only these two of the three ConText attributes get an index: they are the ones a cohort
    # query filters on ("the patient's own, not hypothetical"). `assertion` is nearly collinear
    # with `negated`, so an index on it would rarely be chosen.
    ("ix_ann_temporality", ("temporality",), ""),
    ("ix_ann_experiencer", ("experiencer",), ""),
)


def annotation_columns(omit: Iterable[str] = ()) -> tuple[str, ...]:
    """Annotation column names in schema order, minus `omit` (export spelling accepted)."""
    dropped = {RENAMES.get(c, c) for c in omit}
    return tuple(c for c in ANNOTATION_COLUMNS if c not in dropped)


def schema(omit: Iterable[str] = ()) -> str:
    """``CREATE TABLE`` for both tables, without the columns in `omit`."""
    width = max(len(c) for c in ANNOTATION_COLUMNS)
    lines = [
        "    id             INTEGER PRIMARY KEY,",
        "    doc_id         INTEGER NOT NULL REFERENCES documents(doc_id),",
    ]
    columns = annotation_columns(omit)
    for i, name in enumerate(columns):
        declaration, comment = ANNOTATION_COLUMNS[name]
        comma = "," if i < len(columns) - 1 else ""
        line = f"    {name:<{width}} {declaration}{comma}"
        lines.append(f"{line}  -- {comment}" if comment else line)
    return (
        f"{DOCUMENTS_SCHEMA}\nCREATE TABLE IF NOT EXISTS annotations (\n"
        + "\n".join(lines)
        + "\n);\n"
    )


def indexes(omit: Iterable[str] = ()) -> str:
    """``CREATE INDEX`` statements that the columns in this table support."""
    present = set(annotation_columns(omit)) | {"doc_id"}
    out: list[str] = []
    for name, columns, comment in _INDEXES:
        if not present.issuperset(columns):
            continue
        if comment:
            out.append(f"-- {comment}")
        out.append(f"CREATE INDEX IF NOT EXISTS {name} ON annotations({', '.join(columns)});")
    return "\n".join(out) + "\n"


def _as_flag(value: Any) -> int | None:
    """CSV gives ''/'0'/'1'; JSON gives real booleans or null. Normalize all of them.

    An empty string is NULL, not 0: CSV has no null, so "" is how an unassessed attribute is
    spelled there, and turning it into 0 would invent an assessment.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    return int(str(value).strip() not in ("0", "false", "False"))


def _as_text(value: Any) -> str | None:
    """'' becomes NULL for the same reason; a list becomes the flat formats' comma-joined form."""
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return ",".join(value) or None
    return str(value)


class Loader:
    """Batched insert into `documents` + `annotations`, keyed by document label."""

    def __init__(self, conn: sqlite3.Connection, columns: Iterable[str]) -> None:
        self.conn = conn
        #: SQLite column names actually present in this table, in schema order.
        self.columns = tuple(columns)
        self._docs: dict[str, int] = {}
        self._pending: list[tuple[Any, ...]] = []
        self._counts: dict[int, int] = {}
        #: Export columns seen in the input that this table has no column for.
        self.unstorable: set[str] = set()
        for source, doc_id in conn.execute("SELECT source, doc_id FROM documents"):
            self._docs[source] = doc_id

    def doc_id(self, source: str) -> int:
        """Row id for `source`, creating the `documents` row on first sight."""
        if source not in self._docs:
            cur = self.conn.execute("INSERT INTO documents (source) VALUES (?)", (source,))
            self._docs[source] = int(cur.lastrowid or 0)
        return self._docs[source]

    def add(self, source: str, row: dict[str, Any]) -> None:
        doc_id = self.doc_id(source)
        renamed = {RENAMES.get(k, k): v for k, v in row.items() if k != "document"}
        self.unstorable |= set(renamed) - set(self.columns)
        values: list[Any] = [doc_id]
        for name in self.columns:
            raw = renamed.get(name)
            if name == "negated":
                values.append(_as_flag(raw))
            elif name in ("start_offset", "end_offset"):
                values.append(None if raw in (None, "") else int(raw))
            else:
                values.append(_as_text(raw))
        self._pending.append(tuple(values))
        self._counts[doc_id] = self._counts.get(doc_id, 0) + 1
        if len(self._pending) >= BATCH:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            placeholders = ", ".join("?" * (len(self.columns) + 1))
            self.conn.executemany(
                f"INSERT INTO annotations (doc_id, {', '.join(self.columns)}) "
                f"VALUES ({placeholders})",
                self._pending,
            )
            self._pending.clear()
        for doc_id, n in self._counts.items():
            self.conn.execute(
                "UPDATE documents SET n_annotations = n_annotations + ? WHERE doc_id = ?",
                (n, doc_id),
            )
        self._counts.clear()
        self.conn.commit()

    def report_unstorable(self, stream: Any = sys.stderr) -> None:
        if self.unstorable:
            print(
                f"note: {len(self.unstorable)} column(s) in the input have no column in this "
                f"table and were not loaded: {', '.join(sorted(self.unstorable))}",
                file=stream,
            )


def load_csv(path: Path, loader: Loader) -> int:
    n = 0
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            source = row.get("document") or ""
            loader.add(source, row)
            n += 1
    return n


def _jsonl_rows(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source = record.get("source") or record.get("document") or ""
            for ann in record.get("annotations", []):
                yield source, ann


def load_jsonl(path: Path, loader: Loader) -> int:
    n = 0
    for source, ann in _jsonl_rows(path):
        loader.add(source, ann)
        n += 1
    return n


def _input_columns(path: Path) -> list[str]:
    """The export's own column names, so the table is built to fit what is being loaded."""
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as fh:
            return next(csv.reader(fh), [])
    for _, ann in _jsonl_rows(path):
        return list(ann)
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", type=Path, help="Export to load (.csv or .jsonl)")
    ap.add_argument("db", type=Path, help="SQLite database to write")
    ap.add_argument(
        "--append",
        action="store_true",
        help="Add to an existing database instead of refusing to touch it",
    )
    ap.add_argument(
        "--all-columns",
        action="store_true",
        help="Build every column in the schema, even ones this export does not carry "
        "(they are NULL in every row) - for a fixed downstream schema",
    )
    args = ap.parse_args(argv)

    if not args.input.is_file():
        print(f"error: {args.input} is not a file", file=sys.stderr)
        return 1
    if args.input.suffix.lower() not in (".csv", ".jsonl"):
        print(
            f"error: don't know how to read {args.input.suffix!r}; use .csv or .jsonl",
            file=sys.stderr,
        )
        return 1
    if args.db.exists() and not args.append:
        print(f"error: {args.db} exists; pass --append to add to it", file=sys.stderr)
        return 1

    present = _input_columns(args.input)
    if not present:
        print(f"error: {args.input} has no rows to load", file=sys.stderr)
        return 1
    # Build the table to fit the export: a column the producer never assessed would be NULL in
    # every row, which states a fact about that run rather than about any mention.
    omit = (
        ()
        if args.all_columns
        else [c for c in ANNOTATION_COLUMNS if c not in {RENAMES.get(p, p) for p in present}]
    )
    columns = annotation_columns(omit)

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    try:
        conn.executescript(schema(omit))
        conn.executescript(indexes(omit))
        loader = Loader(conn, columns)
        n = (
            load_csv(args.input, loader)
            if args.input.suffix.lower() == ".csv"
            else load_jsonl(args.input, loader)
        )
        loader.flush()
        loader.report_unstorable()
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
    finally:
        conn.close()

    if omit:
        print(f"not in this export, so not built: {', '.join(omit)}", file=sys.stderr)
    print(
        f"{n:,} rows loaded -> {args.db} ({total:,} annotations over {docs} documents)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
