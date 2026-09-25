"""Tests for examples/json_to_sqlite.py, loading real JSON output (produced via the library
itself, not hand-crafted, so the fixture can't drift from the actual schema) against the
project's shared synthetic index."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mmlite import MetaMapLite


def _annotate_to_json(index_dir: Path, docid: str, text: str) -> str:
    with MetaMapLite(index_directory=index_dir, tagger=_null_tagger()) as mml:
        entities = mml.process_text(text, docid=docid)
        return mml.format(entities, "json")


def _null_tagger():
    from mmlite.pipeline.postag import NullPosTagger

    return NullPosTagger()


@pytest.fixture
def json_file(index_dir: Path, tmp_path: Path) -> Path:
    p = tmp_path / "note1.json"
    p.write_text(_annotate_to_json(index_dir, "note1.txt", "Heart Attack."), encoding="utf-8")
    return p


@pytest.fixture
def second_json_file(index_dir: Path, tmp_path: Path) -> Path:
    p = tmp_path / "note2.json"
    p.write_text(
        _annotate_to_json(index_dir, "note2.txt", "Patient takes aspirin."), encoding="utf-8"
    )
    return p


def test_loads_entities_concepts_and_evidence(json_to_sqlite, json_file, tmp_path):
    db = tmp_path / "annotations.sqlite"
    rc = json_to_sqlite.main([str(json_file), "--db", str(db)])
    assert rc == 0

    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 1
    assert con.execute("SELECT cui, preferred_name FROM concepts").fetchone() == (
        "C0027051",
        "Myocardial Infarction",
    )
    assert con.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == 1
    con.close()


def test_multiple_files_accumulate(json_to_sqlite, json_file, second_json_file, tmp_path):
    db = tmp_path / "annotations.sqlite"
    rc = json_to_sqlite.main([str(json_file), str(second_json_file), "--db", str(db)])
    assert rc == 0
    con = sqlite3.connect(db)
    docids = {row[0] for row in con.execute("SELECT DISTINCT docid FROM entities")}
    con.close()
    assert docids == {"note1.txt", "note2.txt"}


def test_directory_and_pattern_expansion(json_to_sqlite, json_file, second_json_file, tmp_path):
    db = tmp_path / "annotations.sqlite"
    rc = json_to_sqlite.main([str(json_file.parent), "--pattern", "*.json", "--db", str(db)])
    assert rc == 0
    con = sqlite3.connect(db)
    n = con.execute("SELECT COUNT(DISTINCT docid) FROM entities").fetchone()[0]
    con.close()
    assert n == 2


def test_absolute_glob_pattern_is_expanded(json_to_sqlite, json_file, second_json_file, tmp_path):
    """PowerShell hands `C:\\out\\*.json` to the script unexpanded, and tmp_path is absolute, so
    this is the real shape of the argument; it used to crash on Path().glob()."""
    db = tmp_path / "annotations.sqlite"
    pattern = str(json_file.parent / "note*.json")
    assert Path(pattern).is_absolute()
    rc = json_to_sqlite.main([pattern, "--db", str(db)])
    assert rc == 0
    con = sqlite3.connect(db)
    n = con.execute("SELECT COUNT(DISTINCT docid) FROM entities").fetchone()[0]
    con.close()
    assert n == 2


def test_rerun_replaces_docid_not_duplicates(json_to_sqlite, json_file, tmp_path):
    db = tmp_path / "annotations.sqlite"
    json_to_sqlite.main([str(json_file), "--db", str(db)])
    json_to_sqlite.main([str(json_file), "--db", str(db)])
    con = sqlite3.connect(db)
    n = con.execute("SELECT COUNT(*) FROM entities WHERE docid = 'note1.txt'").fetchone()[0]
    con.close()
    assert n == 1


def test_invalid_json_file_reported_and_excluded(json_to_sqlite, json_file, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    db = tmp_path / "annotations.sqlite"
    rc = json_to_sqlite.main([str(json_file), str(bad), "--db", str(db)])
    assert rc == 0  # partial success: one good file loaded
    con = sqlite3.connect(db)
    n = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    con.close()
    assert n == 1


def test_a_file_that_fails_midway_leaves_the_previous_load_intact(
    json_to_sqlite, json_file, tmp_path
):
    """Reloading a docid deletes its rows first — a half-valid file must not destroy them."""
    import json as _json

    db = tmp_path / "annotations.sqlite"
    assert json_to_sqlite.main([str(json_file), "--db", str(db)]) == 0

    # Same docid; the first entity differs (so a partial write is visible) and the second is
    # missing every required field, so it blows up *after* the delete and the first insert.
    entities = _json.loads(json_file.read_text(encoding="utf-8"))
    entities[0]["matchedtext"] = "CLOBBERED"
    broken = tmp_path / "broken.json"
    broken.write_text(_json.dumps(entities + [{"docid": "note1.txt"}]), encoding="utf-8")

    assert json_to_sqlite.main([str(broken), "--db", str(db)]) == 1
    con = sqlite3.connect(db)
    texts = [r[0] for r in con.execute("SELECT matched_text FROM entities WHERE docid='note1.txt'")]
    evidence = con.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    con.close()
    assert texts == ["Heart Attack"], f"a failed reload overwrote the previous load: {texts}"
    assert evidence == 1


def test_all_files_invalid_fails(json_to_sqlite, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    db = tmp_path / "annotations.sqlite"
    rc = json_to_sqlite.main([str(bad), "--db", str(db)])
    assert rc == 1
    assert not db.exists(), "a run that loaded nothing left an empty database behind"


def test_all_files_invalid_keeps_a_preexisting_database(json_to_sqlite, json_file, tmp_path):
    db = tmp_path / "annotations.sqlite"
    assert json_to_sqlite.main([str(json_file), "--db", str(db)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert json_to_sqlite.main([str(bad), "--db", str(db)]) == 1
    con = sqlite3.connect(db)
    n = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    con.close()
    assert n == 1


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16"])
def test_powershell_redirected_json_loads(json_to_sqlite, json_file, tmp_path, encoding):
    """`mmlite annotate ... --outputformat json > note.json` in PowerShell writes UTF-8
    with a BOM (PowerShell 7 / configured 5.1) or UTF-16 (stock Windows PowerShell 5.1); both
    used to fail with 'Unexpected UTF-8 BOM' / a decode error."""
    redirected = tmp_path / "redirected.json"
    redirected.write_text(json_file.read_text(encoding="utf-8"), encoding=encoding)
    assert redirected.read_bytes()[:2] in (b"\xef\xbb", b"\xff\xfe")
    db = tmp_path / "annotations.sqlite"
    assert json_to_sqlite.main([str(redirected), "--db", str(db)]) == 0
    con = sqlite3.connect(db)
    n = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    con.close()
    assert n == 1


def test_no_matching_files_fails(json_to_sqlite, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = json_to_sqlite.main([str(empty), "--db", str(tmp_path / "annotations.sqlite")])
    assert rc == 1
