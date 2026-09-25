"""Tests for examples/filter_mrrel.py against a small synthetic MRREL exercising each of the
three built-in (SAB, REL/RELA) rules - and confirming rows for the SAME SAB but a relationship
type outside the rule are correctly dropped, since (unlike filter_mrconso.py) SAB alone isn't
enough here (see the script's docstring)."""

from __future__ import annotations

from pathlib import Path

import pytest

MRREL_ROWS = [
    # RXNORM: RELA=has_ingredient is one of the three kept RELAs
    "C1|A1|SCUI|RO|C2|A2|SCUI|has_ingredient|R1||RXNORM|RXNORM|||N||",
    # RXNORM: RELA=isa is NOT one of the three - dropped even though SAB matches
    "C1|A1|SCUI|RN|C3|A3|SCUI|isa|R2||RXNORM|RXNORM|||N||",
    # ICD10CM: REL=CHD is kept
    "C4|A4|SCUI|CHD|C5|A5|SCUI||R3||ICD10CM|ICD10CM|||N||",
    # ICD10CM: REL=PAR (the literal inverse) is dropped
    "C4|A4|SCUI|PAR|C5|A5|SCUI||R4||ICD10CM|ICD10CM|||N||",
    # SNOMEDCT_US: REL=CHD is kept
    "C6|A6|SCUI|CHD|C7|A7|SCUI|isa|R5||SNOMEDCT_US|SNOMEDCT_US|||N||",
    # SNOMEDCT_US: REL=RO (a non-hierarchy relationship) is dropped
    "C6|A6|SCUI|RO|C7|A7|SCUI|has_finding_site|R6||SNOMEDCT_US|SNOMEDCT_US|||N||",
    # A SAB with no rule at all is dropped regardless of REL
    "C8|A8|SCUI|CHD|C9|A9|SCUI||R7||MSH|MSH|||N||",
    # RXNORM has_ingredient row that happens to be suppressed - kept anyway (see docstring:
    # suppression is left to the consuming scripts, not filtered out here)
    "C1|A1|SCUI|RO|C10|A10|SCUI|consists_of|R8||RXNORM|RXNORM|||Y||",
]


@pytest.fixture
def mrrel_file(tmp_path: Path) -> Path:
    p = tmp_path / "MRREL.RRF"
    p.write_text("\n".join(MRREL_ROWS) + "\n", encoding="utf-8")
    return p


def test_default_rules_keep_exactly_the_matching_relationship_rows(
    filter_mrrel, mrrel_file, tmp_path
):
    out = tmp_path / "filtered.RRF"
    rc = filter_mrrel.main(["--mrrel", str(mrrel_file), "--out", str(out)])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == [MRREL_ROWS[0], MRREL_ROWS[2], MRREL_ROWS[4], MRREL_ROWS[7]]


def test_suppressed_rows_are_not_dropped(filter_mrrel, mrrel_file, tmp_path):
    out = tmp_path / "filtered.RRF"
    filter_mrrel.main(["--mrrel", str(mrrel_file), "--out", str(out)])
    lines = out.read_text(encoding="utf-8").splitlines()
    assert MRREL_ROWS[7] in lines  # the SUPPRESS=Y row survived


def test_sab_narrows_which_rules_apply(filter_mrrel, mrrel_file, tmp_path):
    out = tmp_path / "filtered.RRF"
    rc = filter_mrrel.main(["--mrrel", str(mrrel_file), "--sab", "ICD10CM", "--out", str(out)])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == [MRREL_ROWS[2]]


def test_unknown_sab_fails_cleanly(filter_mrrel, mrrel_file, tmp_path):
    rc = filter_mrrel.main(
        ["--mrrel", str(mrrel_file), "--sab", "NOSUCHSAB", "--out", str(tmp_path / "out.RRF")]
    )
    assert rc == 1


def test_missing_mrrel_fails_cleanly(filter_mrrel, tmp_path):
    rc = filter_mrrel.main(
        ["--mrrel", str(tmp_path / "does_not_exist.RRF"), "--out", str(tmp_path / "out.RRF")]
    )
    assert rc == 1
