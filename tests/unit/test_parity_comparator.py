"""The parity comparator, exercised without licensed data.

tests/parity/ checks this port against real MetaMapLite output, but skips wherever the Java
fixtures and full UMLS index are absent -- which includes CI, since both are UMLS-derived and
cannot be committed.  That left the comparison code itself (scripts/compare_with_java.py and
compare_nlp_with_java.py) untested exactly where regressions would go unnoticed.  Here it runs
against hand-written "Java" output in the same formats and against the synthetic index, so a
broken comparator fails CI instead of silently reporting perfect agreement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_nlp_with_java import parse_dump, prf  # noqa: E402
from compare_with_java import (  # noqa: E402
    diff_lines,
    json_field_report,
    mmi_keys,
    records,
    same,
)

DOC = "note.txt"
# Two MMI records in Java's layout; the second's matched text spans a line break, so it is one
# record over two physical lines (Java writes the text verbatim).
JAVA_MMI = (
    f'{DOC}|MMI|3.60|Myocardial Infarction|C0027051|[dsyn]|"Heart Attack"-text-0-"heart attack"'
    "-NN-0|text|0/12||\n"
    f'{DOC}|MMI|3.60|aspirin|C0004057|[orch, phsu]|"Aspirin"-text-0-"as\npirin"-NN-0|text|17/8||\n'
)


def test_a_record_that_spans_a_line_break_stays_one_record():
    recs = records(JAVA_MMI, DOC)
    assert len(recs) == 2
    assert "as\npirin" in recs[1]


def test_mmi_keys_are_cui_and_positions_only():
    assert mmi_keys(JAVA_MMI, DOC) == {("C0027051", "0/12"), ("C0004057", "17/8")}
    # score and POS are ignored: they are what the port is allowed to differ on
    rescored = JAVA_MMI.replace("3.60", "9.99").replace("-NN-", "-JJ-")
    assert mmi_keys(rescored, DOC) == mmi_keys(JAVA_MMI, DOC)


def test_mmi_keys_ignore_other_documents_and_non_mmi_lines():
    text = JAVA_MMI + "other.txt|MMI|1|x|C9999999|[dsyn]|t|text|0/1||\nnot a record\n"
    assert ("C9999999", "0/1") not in mmi_keys(text, DOC)


def test_agreement_counts_against_the_synthetic_index(index_dir):
    """End to end: our real MMI output vs a doctored 'Java' copy with one concept removed and one
    invented.  The comparator must count exactly one false positive and one false negative."""
    from mmlite import MetaMapLite
    from mmlite.config import Settings

    text = "Heart Attack and aspirin. No Alzheimer's disease."
    with MetaMapLite(Settings(index_directory=index_dir, enable_postagging=False)) as mml:
        ours = mml.format(mml.process_text(text, docid=DOC), "mmi")
    ours_keys = mmi_keys(ours, DOC)
    assert ("C0004057", "17/7") in ours_keys and len(ours_keys) >= 3

    java = "".join(r + "\n" for r in records(ours, DOC) if "|C0004057|" not in r)
    java += f'{DOC}|MMI|1.00|Invented|C0000000|[dsyn]|"x"-text-0-"x"-NN-0|text|40/3||\n'
    e, a = mmi_keys(java, DOC), mmi_keys(ours, DOC)
    tp, fp, fn = len(e & a), len(a - e), len(e - a)
    assert (fp, fn) == (1, 1) and tp == len(ours_keys) - 1
    p, r, _ = prf(tp, fp, fn)
    assert p == pytest.approx(tp / (tp + 1)) and r == pytest.approx(tp / (tp + 1))


def test_prf_edge_cases():
    assert prf(0, 0, 0) == (1.0, 1.0, 1.0)  # nothing expected, nothing produced
    assert prf(0, 3, 0)[0] == 0.0
    p, r, f = prf(8, 2, 2)
    assert (p, r) == (0.8, 0.8) and f == pytest.approx(0.8)


def test_json_comparison_is_structural_and_reports_differing_fields():
    java = [{"start": 0, "length": 12, "matchedtext": "Heart Attack", "evlist": []}]
    ours = [{**java[0], "negated": False}]
    reordered = json.dumps([dict(reversed(list(java[0].items())))])
    assert same(json.dumps(java), reordered, "json")  # key order is not a difference
    assert not same("a\n", "b\n", "mmi")

    report = json_field_report(json.dumps(java), json.dumps(ours))
    assert report == ["entity keys only in python: ['negated']"]
    changed = [{**java[0], "matchedtext": "heart attack"}]
    assert "shared key differs: matchedtext (1 entities)" in json_field_report(
        json.dumps(java), json.dumps(changed)
    )
    diff = diff_lines(json.dumps(java), json.dumps(ours), "json")
    assert any(line.startswith("+") for line in diff)


def test_parse_dump_aligns_java_tags_with_our_tokens():
    # Java's --list_sentences_postags layout: "offset|length|sentence" then token(TAG), pairs.
    dump = (
        "0|25|Heart Attack and aspirin.\n"
        "Heart(NN), (WS),Attack(NN), (WS),and(CC), (WS),aspirin(NN),.(.),\n"
        "26|3|No.\n"
        "No(DT),.(.),\n"
    )
    parsed = parse_dump(dump)
    assert [(o, n, s) for o, n, s, _ in parsed] == [
        (0, 25, "Heart Attack and aspirin."),
        (26, 3, "No."),
    ]
    assert parsed[0][3] == ["NN", "WS", "NN", "WS", "CC", "WS", "NN", "."]
    # a token Java does not have shows up as "?", which the parity test treats as a mismatch
    broken = dump.replace("aspirin(NN)", "asprin(NN)")
    assert "?" in parse_dump(broken)[0][3]
