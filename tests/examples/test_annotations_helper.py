"""Tests for examples/_annotations.py — the shared row shape of the flat annotation table.

Most of what can go wrong with these exports is not "the pipeline found the wrong thing" but
"the table says something it did not mean": a column that exists when nothing assessed it, an
unassessed attribute written as 0, a document keyed by path instead of file name. Those are what
this file pins down, because they are invisible in a row count and change what a query returns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mmlite.config import Settings


def _settings(**overrides) -> Settings:
    return Settings.load(index_directory=Path("ivf"), **overrides)


# --- which attributes a configuration actually assesses ---------------------------------------


def test_negex_assesses_only_negation(annotations_helper):
    """The default detector sets `negated` and nothing else: Java's NegEx never computes
    temporality or experiencer, so those are NULL rather than false."""
    assessed = annotations_helper.assessed_attributes(_settings())
    assert assessed == frozenset({"negated"})


def test_context_assesses_all_four(annotations_helper):
    assessed = annotations_helper.assessed_attributes(_settings(negation_detector="context"))
    assert assessed == frozenset({"negated", "assertion", "temporality", "experiencer"})


def test_no_negation_assesses_nothing(annotations_helper):
    """`--no-negation` leaves `negated` False on every entity, but that is the *default value*,
    not an assessment - so the column must not claim one."""
    assert annotations_helper.assessed_attributes(_settings(detect_negations=False)) == frozenset()


def test_columns_drop_what_was_not_assessed(annotations_helper):
    columns = annotations_helper.columns_for(frozenset({"negated"}))
    assert "negated" in columns
    assert "assertion" not in columns
    assert "temporality" not in columns
    assert "experiencer" not in columns


def test_all_attributes_keeps_every_column(annotations_helper):
    """For a downstream schema that must not change shape between runs."""
    columns = annotations_helper.columns_for(frozenset({"negated"}), all_attributes=True)
    for name in annotations_helper.ASSERTION_COLUMNS:
        assert name in columns


def test_no_text_drops_only_the_text_column(annotations_helper):
    columns = annotations_helper.columns_for(frozenset({"negated"}), no_text=True)
    assert "text" not in columns
    assert "cui" in columns and "start" in columns


def test_columns_keep_their_declared_order(annotations_helper):
    """Two exports of one corpus have to line up, so the order is the COLUMNS order, not a set's."""
    columns = annotations_helper.columns_for(frozenset(annotations_helper.ASSERTION_COLUMNS))
    assert columns == [c for c in annotations_helper.COLUMNS if c in set(columns)]


# --- blanking an unassessed attribute ---------------------------------------------------------


def test_row_for_blanks_unassessed_attribute_to_none_not_false(annotations_helper):
    """The distinction the whole design rests on: an attribute nothing assessed is None, which
    each format spells as its own null. Writing False would claim an assessment that never
    happened, and `WHERE negated = 0` would then silently include it."""
    row = dict.fromkeys(annotations_helper.COLUMNS, "x")
    row["negated"] = False
    columns = annotations_helper.columns_for(frozenset(), all_attributes=True)
    out = annotations_helper.row_for(row, columns, frozenset())
    assert out["negated"] is None
    assert out["assertion"] is None


def test_row_for_keeps_an_assessed_false(annotations_helper):
    """The other half: "assessed, and not negated" must survive as False, not collapse to None."""
    row = dict.fromkeys(annotations_helper.COLUMNS, "x")
    row["negated"] = False
    columns = annotations_helper.columns_for(frozenset({"negated"}))
    out = annotations_helper.row_for(row, columns, frozenset({"negated"}))
    assert out["negated"] is False


# --- flattening entities to rows --------------------------------------------------------------


def test_one_span_with_several_cuis_becomes_several_rows(annotations_helper, index_dir):
    """The exports are flat, so a span's candidate concepts expand. Callers counting spans need
    COUNT(DISTINCT document, start, end); this is the behaviour that makes that necessary."""
    from mmlite import MetaMapLite

    settings = Settings.load(index_directory=index_dir, enable_postagging=False)
    with MetaMapLite(settings) as mml:
        entities = mml.process_text("Diabetes Mellitus.", docid="d.txt")
    rows = list(annotations_helper.annotation_rows(entities, "d.txt"))
    assert rows, "expected the synthetic index to match 'Diabetes Mellitus'"
    for row in rows:
        assert row["document"] == "d.txt"
    # Every row of one span shares the span's offsets.
    assert len({(r["start"], r["end"]) for r in rows}) == len({(e.start, e.end) for e in entities})


def test_semantic_types_are_sorted(annotations_helper, index_dir):
    """Sorted so two exports of one corpus compare equal - the frozenset's own order is not
    observable, and aspirin carries two types."""
    from mmlite import MetaMapLite

    settings = Settings.load(index_directory=index_dir, enable_postagging=False)
    with MetaMapLite(settings) as mml:
        entities = mml.process_text("Patient takes aspirin.", docid="d.txt")
    rows = [
        r for r in annotations_helper.annotation_rows(entities, "d.txt") if r["cui"] == "C0004057"
    ]
    assert rows, "expected aspirin to match"
    assert rows[0]["semantic_types"] == ["orch", "phsu"]


def test_negation_reaches_the_row(annotations_helper, index_dir):
    from mmlite import MetaMapLite

    settings = Settings.load(index_directory=index_dir, enable_postagging=False)
    with MetaMapLite(settings) as mml:
        entities = mml.process_text("No history of Heart Attack.", docid="d.txt")
    rows = list(annotations_helper.annotation_rows(entities, "d.txt"))
    assert rows and all(r["negated"] for r in rows)


# --- corpus helpers ---------------------------------------------------------------------------


def test_document_label_is_the_file_name_not_the_path(annotations_helper, tmp_path):
    """A path records where the corpus happened to be mounted; two exports of one corpus taken on
    different machines would then fail to join."""
    assert annotations_helper.document_label(tmp_path / "sub" / "note.txt") == "note.txt"


def test_duplicate_file_names_are_warned_about(annotations_helper, tmp_path):
    """Two notes with one name merge into a single `document` key - silently, without this."""
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    paths = [tmp_path / "x" / "note.txt", tmp_path / "y" / "note.txt"]
    warning = annotations_helper.duplicate_label_warning(paths)
    assert warning is not None and "note.txt" in warning


def test_no_warning_when_names_are_unique(annotations_helper, tmp_path):
    paths = [tmp_path / "a.txt", tmp_path / "b.txt"]
    assert annotations_helper.duplicate_label_warning(paths) is None


def test_read_text_reports_a_wrong_encoding_instead_of_silently_mangling(
    annotations_helper, tmp_path
):
    """cp1252 bytes read as UTF-8 become U+FFFD, and a word containing one can never match. The
    read still succeeds - one bad note should not stop a corpus - but it says so."""
    note = tmp_path / "note.txt"
    note.write_bytes("Sjögren".encode("cp1252"))
    text, warning = annotations_helper.read_text(note, "utf-8")
    assert warning is not None and "cp1252" in warning
    assert text  # still annotatable, just lossy


def test_read_text_is_silent_on_clean_utf8(annotations_helper, tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("Sjögren", encoding="utf-8")
    text, warning = annotations_helper.read_text(note, "utf-8")
    assert warning is None and text == "Sjögren"


def test_iter_text_files_is_sorted_and_respects_recursive(annotations_helper, tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "sub" / "c.txt").write_text("x", encoding="utf-8")
    flat = annotations_helper.iter_text_files(tmp_path, "*.txt", recursive=False)
    assert [p.name for p in flat] == ["a.txt", "b.txt"]
    deep = annotations_helper.iter_text_files(tmp_path, "*.txt", recursive=True)
    assert [p.name for p in deep] == ["a.txt", "b.txt", "c.txt"]


def test_split_csv_handles_spacing_and_emptiness(annotations_helper):
    assert annotations_helper.split_csv("dsyn, sosy ,") == {"dsyn", "sosy"}
    assert annotations_helper.split_csv(None) is None
    assert annotations_helper.split_csv("") is None


@pytest.mark.parametrize("detector", ["negex", "context"])
def test_build_settings_selects_the_requested_detector(annotations_helper, detector, tmp_path):
    import argparse

    args = argparse.Namespace(
        index_dir=tmp_path,
        no_postag=True,
        no_negation=False,
        usecontext=(detector == "context"),
        excluded_terms=None,
    )
    assert annotations_helper.build_settings(args).negation_detector == detector
