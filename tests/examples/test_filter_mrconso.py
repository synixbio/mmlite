"""Tests for examples/filter_mrconso.py against a small synthetic MRCONSO mixing several SABs."""

from __future__ import annotations

from pathlib import Path

import pytest

MRCONSO_ROWS = [
    "C1|ENG|P|L1|PF|S1|Y|A1|||1|SNOMEDCT_US|PT|1|snomed term|9|N||",
    "C2|ENG|P|L2|PF|S2|Y|A2|||2|ICD10CM|PT|I00|icd term|4|N||",
    "C3|ENG|P|L3|PF|S3|Y|A3|||3|RXNORM|IN|3|rxnorm term|0|N||",
    "C4|ENG|P|L4|PF|S4|Y|A4|||4|LNC|PT|4|loinc term|0|N||",
    "C5|ENG|P|L5|PF|S5|Y|A5|||5|MSH|MH|D5|mesh term, not in default --sab|0|N||",
]


@pytest.fixture
def mrconso_file(tmp_path: Path) -> Path:
    p = tmp_path / "MRCONSO.RRF"
    p.write_text("\n".join(MRCONSO_ROWS) + "\n", encoding="utf-8")
    return p


def test_default_sab_list_keeps_the_four_vocabularies(filter_mrconso, mrconso_file, tmp_path):
    out = tmp_path / "filtered.RRF"
    rc = filter_mrconso.main(["--mrconso", str(mrconso_file), "--out", str(out)])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    sabs = {line.split("|")[11] for line in lines}
    assert sabs == {"SNOMEDCT_US", "ICD10CM", "RXNORM", "LNC"}
    assert "MSH" not in "".join(lines).split("|")


def test_custom_sab_list(filter_mrconso, mrconso_file, tmp_path):
    out = tmp_path / "filtered.RRF"
    rc = filter_mrconso.main(["--mrconso", str(mrconso_file), "--sab", "MSH", "--out", str(out)])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].split("|")[11] == "MSH"


def test_rows_are_written_verbatim(filter_mrconso, mrconso_file, tmp_path):
    out = tmp_path / "filtered.RRF"
    filter_mrconso.main(["--mrconso", str(mrconso_file), "--sab", "SNOMEDCT_US", "--out", str(out)])
    assert out.read_text(encoding="utf-8").splitlines() == [MRCONSO_ROWS[0]]


def test_missing_mrconso_fails_cleanly(filter_mrconso, tmp_path):
    rc = filter_mrconso.main(
        ["--mrconso", str(tmp_path / "does_not_exist.RRF"), "--out", str(tmp_path / "out.RRF")]
    )
    assert rc == 1


def test_no_matches_for_sab_still_succeeds_with_empty_output(
    filter_mrconso, mrconso_file, tmp_path
):
    out = tmp_path / "filtered.RRF"
    rc = filter_mrconso.main(
        ["--mrconso", str(mrconso_file), "--sab", "NOSUCHSAB", "--out", str(out)]
    )
    assert rc == 0
    assert out.read_text(encoding="utf-8") == ""
