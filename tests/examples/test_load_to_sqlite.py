"""Tests for examples/load_to_sqlite.py — the export -> load route into SQLite.

The exports it reads are written here by :func:`export`, from the same `_annotations.py` rows
and with the same per-format spelling as a CSV or JSONL export, so these tests load real pipeline
output rather than hand-written files. The claim worth defending is that the route is lossless:
every exported row arrives, and CSV and JSONL of one corpus load to the same table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import pytest

from mmlite import MetaMapLite


def _connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn):
    return [
        tuple(r)
        for r in conn.execute(
            "SELECT d.source, a.cui, a.term, a.semantic_types, a.negated, "
            "a.start_offset, a.end_offset, a.text "
            "FROM annotations a JOIN documents d USING(doc_id) "
            "ORDER BY d.source, a.start_offset, a.cui"
        )
    ]


def _csv_value(value):
    """CSV spelling: None is the empty cell ("not assessed"), bools are 0/1, lists comma-joined."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, list):
        return ",".join(value)
    return str(value)


@pytest.fixture
def export(annotations_helper, corpus, base_args):
    """Write `corpus`'s annotations to `path` as CSV or JSONL (by suffix); `extra` are pipeline
    flags such as ``--usecontext``."""
    ah = annotations_helper

    def write(path: Path, *extra: str) -> Path:
        ap = argparse.ArgumentParser()
        ap.add_argument("input_dir", type=Path)
        ah.add_pipeline_args(ap)
        settings = ah.build_settings(ap.parse_args(base_args(corpus, *extra)))
        assessed = ah.assessed_attributes(settings)
        columns = ah.columns_for(assessed)

        docs = []
        with MetaMapLite(settings) as mml:
            for f in ah.iter_text_files(corpus):
                label = ah.document_label(f)
                entities = mml.process_text(f.read_text(encoding="utf-8"), docid=label)
                rows = [
                    ah.row_for(r, columns, assessed) for r in ah.annotation_rows(entities, label)
                ]
                if rows:  # a note with no findings contributes no rows
                    docs.append((label, rows))

        if path.suffix == ".csv":
            with path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=columns)
                writer.writeheader()
                for _, rows in docs:
                    writer.writerows({c: _csv_value(r[c]) for c in columns} for r in rows)
        else:
            # JSONL keys the document on the record (`source`), not on each annotation.
            with path.open("w", encoding="utf-8") as fh:
                for label, rows in docs:
                    annotations = [{k: v for k, v in r.items() if k != "document"} for r in rows]
                    record = {
                        "source": label,
                        "n_annotations": len(rows),
                        "annotations": annotations,
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path

    return write


def test_csv_load_keeps_every_exported_row(export, load_to_sqlite, tmp_path):
    """The headline invariant: nothing is dropped or altered between the export and the table."""
    path = export(tmp_path / "a.csv")
    db = tmp_path / "a.db"
    assert load_to_sqlite.main([str(path), str(db)]) == 0

    with path.open(encoding="utf-8", newline="") as fh:
        exported = sorted(
            (
                r["document"],
                r["cui"],
                r["term"],
                r["semantic_types"],
                int(r["negated"]),
                int(r["start"]),
                int(r["end"]),
                r["text"],
            )
            for r in csv.DictReader(fh)
        )
    loaded = _rows(_connect(db))
    assert exported
    assert sorted(loaded) == exported


def test_jsonl_route_matches_the_csv_route(export, load_to_sqlite, tmp_path):
    from_csv = tmp_path / "csv.db"
    from_jsonl = tmp_path / "jsonl.db"
    load_to_sqlite.main([str(export(tmp_path / "a.csv")), str(from_csv)])
    load_to_sqlite.main([str(export(tmp_path / "a.jsonl")), str(from_jsonl)])

    assert _rows(_connect(from_csv)) == _rows(_connect(from_jsonl))


def test_empty_csv_cell_becomes_null_not_zero(load_to_sqlite, tmp_path):
    """CSV has no null, so "" is how an unassessed attribute is spelled there. Loading it as 0
    would invent an assessment, and `WHERE negated = 0` would then return it."""
    export = tmp_path / "a.csv"
    with export.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["document", "cui", "negated", "start", "end"])
        writer.writeheader()
        writer.writerow({"document": "a.txt", "cui": "C1", "negated": "", "start": 0, "end": 1})
        writer.writerow({"document": "a.txt", "cui": "C2", "negated": "0", "start": 2, "end": 3})

    db = tmp_path / "a.db"
    assert load_to_sqlite.main([str(export), str(db)]) == 0
    conn = _connect(db)
    assert conn.execute("SELECT negated FROM annotations WHERE cui = 'C1'").fetchone()[0] is None
    assert conn.execute("SELECT negated FROM annotations WHERE cui = 'C2'").fetchone()[0] == 0


def test_columns_absent_from_the_export_are_not_built(export, load_to_sqlite, tmp_path, capsys):
    path = export(tmp_path / "a.csv")
    db = tmp_path / "a.db"
    load_to_sqlite.main([str(path), str(db)])
    columns = {r[1] for r in _connect(db).execute("PRAGMA table_info(annotations)")}
    assert "negated" in columns
    assert not ({"assertion", "temporality", "experiencer"} & columns)
    assert "not in this export" in capsys.readouterr().err


def test_all_columns_builds_the_full_schema(export, load_to_sqlite, tmp_path):
    """For a downstream schema that must not change shape between loads."""
    path = export(tmp_path / "a.csv")
    db = tmp_path / "a.db"
    load_to_sqlite.main([str(path), str(db), "--all-columns"])
    columns = {r[1] for r in _connect(db).execute("PRAGMA table_info(annotations)")}
    assert {"assertion", "temporality", "experiencer"} <= columns


def test_usecontext_export_loads_its_extra_columns(export, load_to_sqlite, tmp_path):
    path = export(tmp_path / "a.csv", "--usecontext")
    db = tmp_path / "a.db"
    load_to_sqlite.main([str(path), str(db)])
    conn = _connect(db)
    unassessed = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE assertion IS NULL"
    ).fetchone()[0]
    assert unassessed == 0


def test_refuses_to_touch_an_existing_database_without_append(export, load_to_sqlite, tmp_path):
    path = export(tmp_path / "a.csv")
    db = tmp_path / "a.db"
    assert load_to_sqlite.main([str(path), str(db)]) == 0
    assert load_to_sqlite.main([str(path), str(db)]) == 1


def test_append_accumulates(export, load_to_sqlite, tmp_path):
    path = export(tmp_path / "a.csv")
    db = tmp_path / "a.db"
    load_to_sqlite.main([str(path), str(db)])
    assert load_to_sqlite.main([str(path), str(db), "--append"]) == 0
    assert _connect(db).execute("SELECT COUNT(*) FROM annotations").fetchone()[0] == 8


def test_unknown_extension_is_refused(load_to_sqlite, tmp_path):
    bad = tmp_path / "a.txt"
    bad.write_text("x", encoding="utf-8")
    assert load_to_sqlite.main([str(bad), str(tmp_path / "a.db")]) == 1


def test_missing_input_is_refused(load_to_sqlite, tmp_path):
    assert load_to_sqlite.main([str(tmp_path / "nope.csv"), str(tmp_path / "a.db")]) == 1


def test_empty_export_is_refused(load_to_sqlite, tmp_path):
    """Nothing to infer a schema from, so this is a clear error rather than an empty table."""
    export = tmp_path / "a.csv"
    export.write_text("", encoding="utf-8")
    assert load_to_sqlite.main([str(export), str(tmp_path / "a.db")]) == 1


# --- the schema module itself -----------------------------------------------------------------


def test_schema_omits_requested_columns(load_to_sqlite):
    sql = load_to_sqlite.schema(omit=["assertion", "temporality", "experiencer"])
    assert "negated" in sql
    assert "assertion" not in sql


def test_schema_accepts_export_spelling_of_the_renamed_columns(load_to_sqlite):
    """Omitting accepts the export name (`start`/`end`), not only the SQLite name."""
    assert "start_offset" not in load_to_sqlite.schema(omit=["start"])


@pytest.mark.parametrize("omit", [(), ("negated",)])
def test_indexes_never_name_a_column_that_was_omitted(load_to_sqlite, omit):
    """An index on a dropped column would fail the load; they are skipped instead."""
    sql = load_to_sqlite.indexes(omit=omit)
    for dropped in omit:
        assert f"({dropped})" not in sql
    if not omit:
        assert "ix_ann_cui_neg" in sql
