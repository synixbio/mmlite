"""Tests for examples/notes_of_interest.py — the affirmed/negated split that makes this more
than a keyword search.

The corpus fixture is built for exactly this: `b.txt` says "No history of Heart Attack", so a
grep for the term would put it in the cohort and this script must not.
"""

from __future__ import annotations

import sqlite3

import pytest


def test_a_negated_mention_does_not_join_the_cohort(notes_of_interest, corpus, base_args, capsys):
    """The whole point. C0027051 appears once in the corpus, negated."""
    assert notes_of_interest.main(base_args(corpus, "--cui", "C0027051")) == 0
    out = capsys.readouterr().out
    assert "0 affirm the concept" in out
    assert "1 mention it only as negated" in out


def test_an_affirmed_mention_does(notes_of_interest, corpus, base_args, capsys):
    assert notes_of_interest.main(base_args(corpus, "--cui", "C0002395")) == 0
    out = capsys.readouterr().out
    assert "1 affirm the concept" in out
    assert "c.txt" in out


def test_documents_without_the_concept_are_counted_as_absent(
    notes_of_interest, corpus, base_args, capsys
):
    """ "Absent" is a result: the denominator matters as much as the hits."""
    notes_of_interest.main(base_args(corpus, "--cui", "C0002395"))
    out = capsys.readouterr().out
    assert "Scanned 4 documents" in out
    assert "3 do not mention it" in out


def test_show_negated_lists_the_excluded_documents(notes_of_interest, corpus, base_args, capsys):
    notes_of_interest.main(base_args(corpus, "--cui", "C0027051", "--show-negated"))
    out = capsys.readouterr().out
    assert "Negated-only" in out and "b.txt" in out


def test_negated_documents_are_hidden_by_default_but_counted(
    notes_of_interest, corpus, base_args, capsys
):
    notes_of_interest.main(base_args(corpus, "--cui", "C0027051"))
    out = capsys.readouterr().out
    assert "negated-only documents hidden" in out


def test_a_phrase_is_resolved_to_cuis_and_reported(notes_of_interest, corpus, base_args, capsys):
    """The resolved concepts are printed *before* the scan: a phrase resolving to something you
    did not mean is the commonest way a cohort goes wrong."""
    assert notes_of_interest.main(base_args(corpus, "Alzheimer Disease")) == 0
    out = capsys.readouterr().out
    assert "resolves to" in out and "C0002395" in out


def test_an_unresolvable_phrase_is_an_error_not_an_empty_cohort(
    notes_of_interest, corpus, base_args, capsys
):
    """Zero documents would look like a real (and reassuring) answer."""
    assert notes_of_interest.main(base_args(corpus, "zzzz not a concept zzzz")) == 1
    assert "No concept matches" in capsys.readouterr().err


def test_term_and_cui_together_are_rejected(notes_of_interest, corpus, base_args, capsys):
    with pytest.raises(SystemExit):
        notes_of_interest.main(base_args(corpus, "diabetes", "--cui", "C0011860"))
    assert "not both" in capsys.readouterr().err


def test_neither_term_nor_cui_is_rejected(notes_of_interest, corpus, base_args, capsys):
    with pytest.raises(SystemExit):
        notes_of_interest.main(base_args(corpus))
    assert "give a search term or --cui" in capsys.readouterr().err


def test_no_negation_is_rejected(notes_of_interest, corpus, base_args, capsys):
    """Without a negation detector every mention would count as affirmed, quietly turning this
    into the keyword search it exists to replace."""
    with pytest.raises(SystemExit):
        notes_of_interest.main(base_args(corpus, "--cui", "C0027051", "--no-negation"))
    assert "--no-negation" in capsys.readouterr().err


def test_cuis_are_matched_case_insensitively(notes_of_interest, corpus, base_args, capsys):
    notes_of_interest.main(base_args(corpus, "--cui", "c0002395"))
    assert "1 affirm the concept" in capsys.readouterr().out


# --- saving a search --------------------------------------------------------------------------


def test_save_writes_the_four_tables(notes_of_interest, corpus, base_args, tmp_path):
    db = tmp_path / "cohort.sqlite"
    assert notes_of_interest.main(base_args(corpus, "--cui", "C0027051", "--save", str(db))) == 0
    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"search", "concepts", "documents", "console"} <= tables

    search = conn.execute(
        "SELECT n_documents, n_affirmed, n_negated, n_absent FROM search"
    ).fetchone()
    assert search == (4, 0, 1, 3)


def test_saved_documents_include_the_ones_without_the_concept(
    notes_of_interest, corpus, base_args, tmp_path
):
    db = tmp_path / "cohort.sqlite"
    notes_of_interest.main(base_args(corpus, "--cui", "C0027051", "--save", str(db)))
    conn = sqlite3.connect(db)
    statuses = dict(conn.execute("SELECT source, status FROM documents"))
    assert statuses["b.txt"] == "negated_only"
    assert statuses["a.txt"] == "absent"
    assert len(statuses) == 4


def test_saving_twice_appends_rather_than_overwrites(
    notes_of_interest, corpus, base_args, tmp_path
):
    """One file accumulates a searchable history of what was asked."""
    db = tmp_path / "cohort.sqlite"
    notes_of_interest.main(base_args(corpus, "--cui", "C0027051", "--save", str(db)))
    notes_of_interest.main(base_args(corpus, "--cui", "C0002395", "--save", str(db)))
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM search").fetchone()[0] == 2
    # each search keeps its own document rows
    assert conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 8


def test_console_transcript_is_stored_verbatim(notes_of_interest, corpus, base_args, tmp_path):
    """The tables are what you query; the transcript is what you show someone who asks what you
    actually ran."""
    db = tmp_path / "cohort.sqlite"
    notes_of_interest.main(base_args(corpus, "--cui", "C0002395", "--save", str(db)))
    conn = sqlite3.connect(db)
    lines = [r[0] for r in conn.execute("SELECT line FROM console ORDER BY line_no")]
    assert any("affirm the concept" in line for line in lines)


def test_resolved_concepts_are_saved_for_a_phrase_search(
    notes_of_interest, corpus, base_args, tmp_path
):
    db = tmp_path / "cohort.sqlite"
    notes_of_interest.main(base_args(corpus, "Alzheimer Disease", "--save", str(db)))
    conn = sqlite3.connect(db)
    cuis = {r[0] for r in conn.execute("SELECT cui FROM concepts")}
    assert "C0002395" in cuis


def test_nothing_is_written_when_the_search_fails_early(
    notes_of_interest, corpus, base_args, tmp_path
):
    """A `search` row should always mean a search that ran."""
    db = tmp_path / "cohort.sqlite"
    assert notes_of_interest.main(base_args(corpus, "zzzz not a concept", "--save", str(db))) == 1
    assert not db.exists()


def test_no_save_writes_nothing(notes_of_interest, corpus, base_args, tmp_path):
    assert notes_of_interest.main(base_args(corpus, "--cui", "C0002395")) == 0
    assert not list(tmp_path.glob("*.sqlite"))
