"""Tests for examples/analyze_folder.py against the project's shared synthetic index (the
`index_dir` fixture from tests/conftest.py)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture
def notes_dir(tmp_path: Path) -> Path:
    d = tmp_path / "notes"
    d.mkdir()
    (d / "note1.txt").write_text("Heart Attack.", encoding="utf-8")
    (d / "note2.txt").write_text("Patient takes aspirin.", encoding="utf-8")
    (d / "note3.txt").write_text("   ", encoding="utf-8")  # blank -> skipped, not an error
    return d


def test_sequential_processing_writes_one_output_file_per_input(
    analyze_folder, index_dir, notes_dir, tmp_path
):
    out_dir = tmp_path / "out"
    rc = analyze_folder.main(
        [
            str(notes_dir),
            "--index-dir",
            str(index_dir),
            "--no-postag",
            "--output-dir",
            str(out_dir),
        ]
    )
    # 2/3 annotated; the blank note is skipped and produces no output file at all
    # (_annotate_one returns before ever writing one)
    assert rc == 0
    assert (out_dir / "note1.mmi").exists()
    assert (out_dir / "note2.mmi").exists()
    assert not (out_dir / "note3.mmi").exists()
    assert "C0027051" in (out_dir / "note1.mmi").read_text(encoding="utf-8")
    assert "C0004057" in (out_dir / "note2.mmi").read_text(encoding="utf-8")


def test_blank_and_bom_only_notes_are_skipped_not_failed(
    analyze_folder, index_dir, tmp_path, capsys
):
    """A folder of nothing but empty notes is not a failed batch: nothing went wrong, there was
    just nothing to do - so rc is 0 and the summary says 'skipped', not 'ERROR'.  A BOM-only
    file (what Notepad saves for an 'empty' note) counts as empty too."""
    d = tmp_path / "notes"
    d.mkdir()
    (d / "blank.txt").write_text("   \n", encoding="utf-8")
    (d / "bom.txt").write_bytes(b"\xef\xbb\xbf\r\n")
    rc = analyze_folder.main([str(d), "--index-dir", str(index_dir), "--no-postag"])
    err = capsys.readouterr().err
    assert rc == 0
    assert "ERROR" not in err
    assert "blank.txt: skipped (empty file)" in err
    assert "bom.txt: skipped (empty file)" in err
    assert "0/2 files processed (0 failed, 2 skipped)" in err
    assert not list(d.glob("*.mmi"))


def test_workers_pool_matches_sequential_results(index_dir, notes_dir, tmp_path):
    # Run as a real subprocess (python examples/analyze_folder.py ...), not the dynamically
    # imported module used elsewhere in this file: ProcessPoolExecutor's workers re-import the
    # target module by name to unpickle the worker function, which only resolves for a module
    # imported normally (as this script is when actually run), not one loaded via
    # importlib.util.spec_from_file_location under a synthetic name.
    seq_dir, par_dir = tmp_path / "seq", tmp_path / "par"
    common = [
        sys.executable,
        str(EXAMPLES_DIR / "analyze_folder.py"),
        str(notes_dir),
        "--index-dir",
        str(index_dir),
        "--no-postag",
    ]
    subprocess.run(common + ["--output-dir", str(seq_dir)], check=True, capture_output=True)
    result = subprocess.run(
        common + ["--output-dir", str(par_dir), "--workers", "2"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (par_dir / "note1.mmi").read_text(encoding="utf-8") == (seq_dir / "note1.mmi").read_text(
        encoding="utf-8"
    )
    assert (par_dir / "note2.mmi").read_text(encoding="utf-8") == (seq_dir / "note2.mmi").read_text(
        encoding="utf-8"
    )


def test_summary_lists_preferred_names(analyze_folder, index_dir, notes_dir, capsys):
    rc = analyze_folder.main([str(notes_dir), "--index-dir", str(index_dir), "--no-postag"])
    err = capsys.readouterr().err
    assert rc == 0
    line = next(ln for ln in err.splitlines() if "C0027051" in ln)
    assert line.rstrip().endswith("Myocardial Infarction")


def test_invalid_utf8_file_is_flagged(analyze_folder, index_dir, tmp_path, capsys):
    d = tmp_path / "notes"
    d.mkdir()
    (d / "latin1.txt").write_bytes("Heart Attack at the café.".encode("latin-1"))
    rc = analyze_folder.main([str(d), "--index-dir", str(index_dir), "--no-postag"])
    err = capsys.readouterr().err
    assert rc == 0
    assert re.search(r"latin1\.txt: \d+ spans, \d+ negated - WARNING: not valid utf-8", err)
    assert "1 file(s) had encoding problems" in err

    rc = analyze_folder.main(
        [str(d), "--index-dir", str(index_dir), "--no-postag", "--encoding", "cp1252"]
    )
    assert rc == 0
    assert "WARNING" not in capsys.readouterr().err


def test_empty_folder_fails_cleanly(analyze_folder, index_dir, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = analyze_folder.main([str(empty), "--index-dir", str(index_dir)])
    assert rc == 1


def test_bad_index_dir_fails_cleanly(analyze_folder, notes_dir, tmp_path):
    rc = analyze_folder.main([str(notes_dir), "--index-dir", str(tmp_path / "no_such_index")])
    assert rc == 1


def test_excluded_terms_applies_to_every_file(analyze_folder, index_dir, notes_dir, tmp_path):
    """The exclusion list must reach the worker processes too, not just the serial path."""
    excluded = tmp_path / "specialterms.txt"
    excluded.write_text("C0004057:aspirin\n", encoding="utf-8")
    out = tmp_path / "out"
    rc = analyze_folder.main(
        [
            "--index-dir",
            str(index_dir),
            "--no-postag",
            "--output-dir",
            str(out),
            "--excluded-terms",
            str(excluded),
            str(notes_dir),
        ]
    )
    assert rc == 0
    written = list(out.glob("*.mmi"))
    assert written
    for f in written:
        assert "C0004057" not in f.read_text(encoding="utf-8")


def test_missing_excluded_terms_file_fails_cleanly(analyze_folder, index_dir, notes_dir, tmp_path):
    rc = analyze_folder.main(
        [
            "--index-dir",
            str(index_dir),
            "--no-postag",
            "--excluded-terms",
            str(tmp_path / "nope.txt"),
            str(notes_dir),
        ]
    )
    assert rc == 1


def test_summary_shows_the_exclusion_key_not_the_preferred_name(
    analyze_folder, index_dir, notes_dir, tmp_path, capsys
):
    """--excluded-terms keys on the *normalized matched term*, but the summary used to print
    only the preferred name - which is often a different string entirely (on real notes
    C0332287 shows as "In addition to" yet is matched by the word "with"). Users had no way to
    build an exclusion list from the output they were looking at. The `matched as` column must
    print the term that actually works, so the file can be transcribed straight off the table."""
    rc = analyze_folder.main([str(notes_dir), "--index-dir", str(index_dir), "--no-postag"])
    err = capsys.readouterr().err
    assert rc == 0
    assert "matched as" in err
    line = next(ln for ln in err.splitlines() if "C0027051" in ln)
    key = line.split("C0027051")[1].split()[0]
    assert key == "heart"  # first token of the normalized term, not of "Myocardial Infarction"

    # round-trip: the key read off the table really does suppress that concept
    excluded = tmp_path / "specialterms.txt"
    excluded.write_text("C0027051:heart attack\n", encoding="utf-8")
    rc = analyze_folder.main(
        [
            str(notes_dir),
            "--index-dir",
            str(index_dir),
            "--no-postag",
            "--excluded-terms",
            str(excluded),
        ]
    )
    assert rc == 0
    assert "C0027051" not in capsys.readouterr().err
