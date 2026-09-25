"""Tests for examples/analyze_file.py against the project's shared synthetic index (the
`index_dir` fixture from tests/conftest.py, built from the same tiny MRCONSO used by every unit
test - real matches: "Heart Attack" -> C0027051, "aspirin" -> C0004057)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def note(tmp_path: Path) -> Path:
    p = tmp_path / "note.txt"
    p.write_text("Heart Attack and aspirin.", encoding="utf-8")
    return p


def test_mmi_output_written_to_file(analyze_file, index_dir, note, tmp_path):
    out = tmp_path / "note.mmi"
    rc = analyze_file.main(
        ["--index-dir", str(index_dir), "--no-postag", "-o", str(out), str(note)]
    )
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "|MMI|" in text
    assert "C0027051" in text and "C0004057" in text


def test_json_output_structure(analyze_file, index_dir, note, tmp_path):
    out = tmp_path / "note.json"
    rc = analyze_file.main(
        [
            "--index-dir",
            str(index_dir),
            "--no-postag",
            "--outputformat",
            "json",
            "-o",
            str(out),
            str(note),
        ]
    )
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    cuis = {ev["conceptinfo"]["cui"] for e in data for ev in e["evlist"]}
    assert {"C0027051", "C0004057"} <= cuis


def test_restrict_to_sts_filters_output(analyze_file, index_dir, note, tmp_path):
    out = tmp_path / "note.cuis"
    rc = analyze_file.main(
        [
            "--index-dir",
            str(index_dir),
            "--no-postag",
            "--outputformat",
            "cuilist",
            "--restrict-to-sts",
            "phsu",  # pharmacologic substance - matches aspirin, not the heart attack finding
            "-o",
            str(out),
            str(note),
        ]
    )
    assert rc == 0
    cuis = out.read_text(encoding="utf-8").split()
    assert cuis == ["C0004057"]


def test_unknown_semantic_type_is_refused_not_silently_empty(
    analyze_file, index_dir, note, tmp_path, capsys
):
    """`--restrict-to-sts dysn` (a typo for dsyn) used to produce 0 spans and exit 0."""
    out = tmp_path / "note.cuis"
    rc = analyze_file.main(
        [
            "--index-dir",
            str(index_dir),
            "--restrict-to-sts",
            "dysn,T047,zzz",
            "-o",
            str(out),
            str(note),
        ]
    )
    assert rc == 1
    assert "unknown semantic type(s) in --restrict-to-sts: dysn, zzz" in capsys.readouterr().err
    assert not out.exists()


def test_missing_input_file_fails_cleanly(analyze_file, index_dir, tmp_path):
    rc = analyze_file.main(["--index-dir", str(index_dir), str(tmp_path / "does_not_exist.txt")])
    assert rc == 1


def test_bad_index_dir_fails_cleanly(analyze_file, note, tmp_path):
    rc = analyze_file.main(["--index-dir", str(tmp_path / "no_such_index"), str(note)])
    assert rc == 1


def test_output_folder_is_created(analyze_file, index_dir, note, tmp_path):
    """`-o` into a folder that doesn't exist yet used to crash with a FileNotFoundError
    traceback."""
    out = tmp_path / "new" / "nested" / "note.mmi"
    rc = analyze_file.main(
        ["--index-dir", str(index_dir), "--no-postag", "-o", str(out), str(note)]
    )
    assert rc == 0
    assert "C0027051" in out.read_text(encoding="utf-8")


def test_unwritable_output_fails_with_one_line_error(
    analyze_file, index_dir, note, tmp_path, capsys
):
    blocker = tmp_path / "a_file"
    blocker.write_text("", encoding="utf-8")
    rc = analyze_file.main(
        ["--index-dir", str(index_dir), "--no-postag", "-o", str(blocker / "x.mmi"), str(note)]
    )
    err = capsys.readouterr().err
    assert rc == 1
    assert "error: cannot write" in err and "Traceback" not in err


@pytest.fixture
def latin1_note(tmp_path: Path) -> Path:
    p = tmp_path / "latin1.txt"
    p.write_bytes("Heart Attack at the café.".encode("latin-1"))  # 0xE9: invalid UTF-8
    return p


def test_invalid_utf8_warns_but_still_annotates(
    analyze_file, index_dir, latin1_note, tmp_path, capsys
):
    """A cp1252/latin-1 note used to be decoded with silent U+FFFD replacement - accented words
    quietly stopped matching and nothing said why."""
    out = tmp_path / "out.mmi"
    rc = analyze_file.main(
        ["--index-dir", str(index_dir), "--no-postag", "-o", str(out), str(latin1_note)]
    )
    err = capsys.readouterr().err
    assert rc == 0
    assert "warning: latin1.txt: not valid utf-8 (first bad byte at offset 23)" in err
    assert "--encoding" in err
    assert "C0027051" in out.read_text(encoding="utf-8")


def test_encoding_flag_decodes_without_warning(
    analyze_file, index_dir, latin1_note, tmp_path, capsys
):
    out = tmp_path / "out.json"
    rc = analyze_file.main(
        [
            "--index-dir",
            str(index_dir),
            "--no-postag",
            "--encoding",
            "latin-1",
            "--outputformat",
            "json",
            "-o",
            str(out),
            str(latin1_note),
        ]
    )
    assert rc == 0
    assert "warning" not in capsys.readouterr().err
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data[0]["start"] == 0 and data[0]["matchedtext"] == "Heart Attack"


def test_unknown_encoding_fails_cleanly(analyze_file, index_dir, note, capsys):
    rc = analyze_file.main(["--index-dir", str(index_dir), "--encoding", "nope-8", str(note)])
    assert rc == 1
    assert "unknown --encoding 'nope-8'" in capsys.readouterr().err


def test_excluded_terms_suppresses_a_match(analyze_file, index_dir, note, tmp_path):
    """MetaMapLite's own exclusion list is how known-bad matches ("for" -> a gene symbol) are
    suppressed; Java applies one by default, so the examples need a way to pass one."""
    excluded = tmp_path / "specialterms.txt"
    excluded.write_text("C0004057:aspirin\n", encoding="utf-8")
    out = tmp_path / "note.mmi"
    args = ["--index-dir", str(index_dir), "--no-postag", "-o", str(out), str(note)]
    assert analyze_file.main(args) == 0
    assert "C0004057" in out.read_text(encoding="utf-8")

    assert analyze_file.main([*args, "--excluded-terms", str(excluded)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "C0004057" not in text
    assert "C0027051" in text  # the other concept is untouched


def test_missing_excluded_terms_file_fails_cleanly(analyze_file, index_dir, note, tmp_path):
    rc = analyze_file.main(
        [
            "--index-dir",
            str(index_dir),
            "--no-postag",
            "--excluded-terms",
            str(tmp_path / "nope.txt"),
            str(note),
        ]
    )
    assert rc == 1


def test_broken_pipe_on_stdout_exits_quietly(analyze_file, index_dir, note, monkeypatch, capsys):
    """`analyze_file.py note.txt | head` used to dump a BrokenPipeError traceback: the -o path
    was guarded against OSError but the default stdout path wasn't. Closing the reader early is
    normal use of a stdout-by-default tool, so it must exit 0 with nothing on stderr but the
    usual summary."""

    def explode(_text):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(sys.stdout, "write", explode)
    rc = analyze_file.main(["--index-dir", str(index_dir), "--no-postag", str(note)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "Traceback" not in err and "BrokenPipeError" not in err
    assert "note.txt:" in err  # the stderr summary still made it out
