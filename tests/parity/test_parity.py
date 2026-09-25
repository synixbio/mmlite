"""Regression guard against real MetaMapLite 3.6.2rc8 output.

Requires a built index (``ivf/``) and the Java fixtures in ``tests/parity/fixtures/``; both are
skipped when absent, since they derive from licensed UMLS data and are not committed.  Regenerate
with::

    python scripts/compare_with_java.py     --java-dir <public_mm_lite> --refresh
    python scripts/compare_nlp_with_java.py --java-dir <public_mm_lite> --refresh

The thresholds below are the numbers measured on 2026-09-17 (UMLS 2026AA, spaCy en_core_web_sm
3.8); they exist to catch regressions, not to certify quality.  See docs/DEVELOPMENT_PLAN.md.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORPUS = Path(__file__).parent / "corpus"
FIXTURES = Path(__file__).parent / "fixtures"
INDEX = ROOT / "ivf"

sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(
    not (INDEX / "mmlite.sqlite").exists() or not any(FIXTURES.glob("*.mmi")),
    reason="needs a built index and Java fixtures (see module docstring)",
)

# measured 2026-09-18 over the 12-document corpus (P 0.975 / R 0.963); lowered slightly so small
# tagger/model updates do not fail the build.  Five of the 35 disagreements are the known,
# deliberate non-ASCII divergence pinned by test_known_nonascii_divergences_are_exactly_these.
MIN_PRECISION = 0.97
MIN_RECALL = 0.965  # measured 0.968 with the spacy_nlp verb rescue (0.963 without)
MIN_IDENTICAL_DOCS = 1

CRLF = bytes((13, 10))

# Keys this port emits in ``json`` output that Java does not.  Both are deliberate, and the test
# below fails if a *third* one appears:
#   negated  -- Java's resultformats.Json never serialises it, though it computes it.
#   fieldid  -- Java emits it, but org.json drops a key whose value is null, and its free-text
#               file loader sets no "section" infon.  We emit an explicit null instead.
# See DEVELOPMENT_PLAN §13.
JSON_EXTRA_ENTITY_KEYS = {"fieldid", "negated"}
MIN_JSON_SPAN_AGREEMENT = 0.94  # measured 0.963 (285/296)


@pytest.fixture(scope="module")
def mml():
    from mmlite import MetaMapLite
    from mmlite.config import Settings

    special = FIXTURES / "specialterms.txt"
    return MetaMapLite(
        Settings.load(
            index_directory=INDEX,
            excluded_terms_file=special if special.exists() else None,
        )
    )


def _docs():
    return sorted(CORPUS.glob("*.txt"))


def test_corpus_present():
    assert _docs(), "parity corpus is missing"


def test_tokenization_matches_java_exactly():
    """Java's own token dump, re-tokenized by us: every sentence must match token for token."""
    from compare_nlp_with_java import parse_dump

    from mmlite.pipeline.tokenize import tokenize

    checked = 0
    for doc in _docs():
        if CRLF in doc.read_bytes():
            # Java's --list_sentences_postags dump writes each token inline, and the harness
            # normalizes line endings when caching it, so a CRLF document's two whitespace
            # tokens (CR, LF) cannot survive the round trip.  The dump format is the limitation,
            # not the tokenizer; crlf_note.txt is guarded by its own offset test below instead.
            continue
        dump = FIXTURES / f"{doc.stem}.sentences_postags"
        if not dump.exists():
            continue
        # newline="" so the dump is read exactly as written, without translating line endings
        with open(dump, encoding="utf-8", newline="") as fh:
            parsed = parse_dump(fh.read())
        for _, _, sent, tags in parsed:
            toks = tokenize(sent)
            assert len(toks) == len(tags), f"{doc.name}: token count differs for {sent!r}"
            assert "?" not in tags, f"{doc.name}: token text differs from Java in {sent!r}"
            checked += 1
    assert checked >= 50, f"only {checked} sentences checked"


def test_concept_level_agreement(mml):
    """(CUI, positions) agreement with Java's MMI output over the whole corpus."""
    from compare_with_java import mmi_keys

    tp = fp = fn = 0
    identical = 0
    for doc in _docs():
        fixture = FIXTURES / f"{doc.stem}.mmi"
        if not fixture.exists():
            continue
        expected = fixture.read_text(encoding="utf-8")
        # Through the loader, not process_text: it reads the file verbatim, so a CRLF document's
        # offsets line up with Java's (DEVELOPMENT_PLAN §6.6 and §20).
        (document,) = mml.load(doc, docid=doc.name)
        actual = mml.format(mml.process_document(document), "mmi")
        if expected == actual:
            identical += 1
        e, a = mmi_keys(expected, doc.name), mmi_keys(actual, doc.name)
        tp += len(e & a)
        fp += len(a - e)
        fn += len(e - a)
    assert tp, "no concepts compared"
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    assert precision >= MIN_PRECISION, f"precision {precision:.3f} (tp={tp} fp={fp})"
    assert recall >= MIN_RECALL, f"recall {recall:.3f} (tp={tp} fn={fn})"
    assert identical >= MIN_IDENTICAL_DOCS


def _json_pairs(mml):
    """(java, python) parsed ``json`` output per document, via the same path Java used.

    Java loads the file with its ``freetext`` loader, so we go through ours rather than
    ``process_text`` — otherwise the ``fieldid`` comparison is between two different code paths
    (DEVELOPMENT_PLAN §12).
    """
    for doc in _docs():
        fixture = FIXTURES / f"{doc.stem}.json"
        if not fixture.exists():
            continue
        (document,) = mml.load(doc, docid=doc.name)
        actual = mml.format(mml.process_document(document), "json")
        yield doc, json.loads(fixture.read_text(encoding="utf-8")), json.loads(actual)


def test_json_carries_every_field_java_emits_and_only_known_extras(mml):
    """Our ``json`` is a superset of Java's, and the extra keys are the two we chose.

    Byte comparison is meaningless here: ``org.json`` backs ``JSONObject`` with a ``HashMap``,
    so key order is arbitrary, and it drops null values entirely.  The contract that *is*
    meaningful is which keys exist.
    """
    checked = 0
    for doc, expected, actual in _json_pairs(mml):
        if not expected:
            continue
        java_keys = {k for e in expected for k in e}
        our_keys = {k for e in actual for k in e}
        assert java_keys <= our_keys, f"{doc.name}: missing {sorted(java_keys - our_keys)}"
        extra = our_keys - java_keys
        assert extra <= JSON_EXTRA_ENTITY_KEYS, f"{doc.name}: unexpected extra keys {sorted(extra)}"
        checked += 1
    if not checked:
        pytest.skip("no .json fixtures; regenerate with compare_with_java.py --format json")
    assert checked >= 5


def test_json_shared_spans_agree_with_java_field_for_field(mml):
    """Where Java and we found the same span, every field Java emits must match — including
    the order of ``sources`` and ``evlist``, both of which are Java ``HashSet`` iteration order
    rather than anything sorted."""
    total = agree = 0
    for _doc, expected, actual in _json_pairs(mml):
        java_keys = {k for e in expected for k in e}
        by_span = {(e["start"], e["length"]): e for e in expected}
        for entity in actual:
            other = by_span.get((entity["start"], entity["length"]))
            if other is None:
                continue  # a concept-level miss; test_concept_level_agreement covers those
            total += 1
            agree += all(entity.get(k) == other.get(k) for k in java_keys)
    if not total:
        pytest.skip("no .json fixtures; regenerate with compare_with_java.py --format json")
    ratio = agree / total
    assert ratio >= MIN_JSON_SPAN_AGREEMENT, f"json span agreement {ratio:.3f} ({agree}/{total})"


def test_mmi_line_shape_matches_java(mml):
    """Field count, trailing pipe and position syntax must match Java exactly."""
    from compare_with_java import records

    for doc in _docs():
        fixture = FIXTURES / f"{doc.stem}.mmi"
        if not fixture.exists():
            continue
        expected = records(fixture.read_text(encoding="utf-8"), doc.name)
        (document,) = mml.load(doc, docid=doc.name)
        actual = records(mml.format(mml.process_document(document), "mmi"), doc.name)
        if not expected or not actual:
            continue
        assert len(expected[0].split("|")) == len(actual[0].split("|"))
        assert expected[0].endswith("|") and actual[0].endswith("|")
        for rec in actual:
            f = rec.split("|")
            assert f[1] == "MMI"
            assert all("/" in p for p in f[8].split(";") if p)


def test_crlf_document_offsets_agree_with_java_exactly(mml):
    """``crlf_note.txt`` is the corpus's only CRLF document, and it exists for this test.

    Reading a file with universal newlines drops one character per line, shifting every offset
    after the first — the bug of DEVELOPMENT_PLAN §6.6, which no parity document could catch until
    this one was added.  Java reads the bytes verbatim, so **every** concept must agree on its
    position: no false positives, no misses.
    """
    from compare_with_java import mmi_keys

    doc = CORPUS / "crlf_note.txt"
    fixture = FIXTURES / "crlf_note.mmi"
    if not doc.exists() or not fixture.exists():
        pytest.skip("crlf_note fixture missing")
    assert CRLF in doc.read_bytes(), "this document must keep its CRLF line endings"
    expected = mmi_keys(fixture.read_text(encoding="utf-8"), doc.name)
    (document,) = mml.load(doc, docid=doc.name)
    actual = mmi_keys(mml.format(mml.process_document(document), "mmi"), doc.name)
    assert expected and actual == expected


# The only concepts on which this port and Java disagree in unicode_terms.txt, and the reason for
# each: Java's index lookup is unreliable for keys containing non-ASCII characters
# (DEVELOPMENT_PLAN §16, measured in §21), so for these it misses the full term and falls back to
# an ASCII fragment.  Pinned rather than averaged away, so a change in that decision shows up here
# explicitly.
KNOWN_NONASCII_DIVERGENCE = {
    "ours_only": {
        ("C0017040", "113/22"),  # 'γ-glutamyl transferase'
        ("C0221263", "151/18"),  # 'Café au lait spots'
    },
    "java_only": {
        ("C0678107", "115/20"),  # 'glutamyl transferase' -- the ASCII fragment Java falls back to
        ("C0015230", "164/5"),  # 'spots'
        ("C0848332", "164/5"),  # 'spots'
    },
}


def test_known_nonascii_divergences_are_exactly_these(mml):
    """Everything else in ``unicode_terms.txt`` must match, including the Greek-letter spans."""
    from compare_with_java import mmi_keys

    doc = CORPUS / "unicode_terms.txt"
    fixture = FIXTURES / "unicode_terms.mmi"
    if not doc.exists() or not fixture.exists():
        pytest.skip("unicode_terms fixture missing")
    expected = mmi_keys(fixture.read_text(encoding="utf-8"), doc.name)
    (document,) = mml.load(doc, docid=doc.name)
    actual = mmi_keys(mml.format(mml.process_document(document), "mmi"), doc.name)
    assert actual - expected == KNOWN_NONASCII_DIVERGENCE["ours_only"]
    assert expected - actual == KNOWN_NONASCII_DIVERGENCE["java_only"]
    # TNF-α and β-blocker are matched by both: the §16 Greek fix, guarded here.
    assert ("C1456820", "28/5") in actual or any(cui == "C1456820" for cui, _ in actual)


# In an MMI record, field 6 holds one tuple per occurrence and field 8 the matching position
# groups, in the same order.  A tuple is `"concept"-field-sent-"text"-POS-neg`, so the *second*
# quoted string is the matched text -- the only one preceded by a hyphen.
_TUPLE_TEXT = re.compile(r'-"([^"]*)"-')


def _positions_and_texts(record: str):
    f = record.split("|")
    if len(f) < 9 or f[1] != "MMI":
        return
    texts, groups = _TUPLE_TEXT.findall(f[6]), f[8].split(";")
    assert len(texts) == len(groups), f"{len(texts)} tuples but {len(groups)} position groups"
    for text, group in zip(texts, groups, strict=True):
        for pos in (p for p in group.split(",") if p):
            start, length = (int(x) for x in pos.split("/"))
            yield start, length, text


def test_mmi_positions_index_the_corpus_documents_verbatim(mml):
    """Every position in every record, ours *and* Java's, must cut the recorded text out of the
    file as it sits on disk.

    Offsets are the one thing that cannot be checked by comparing the two outputs to each other:
    both sides could agree and both be wrong.  Anchoring them to the raw bytes also pins Java's
    fixtures, so a regenerated fixture that disagrees with its document fails here.  Abbreviation
    entities are included: MMI reports the long form's positions with the long form's text, so the
    invariant holds for them too.
    """
    from compare_with_java import records

    from mmlite import read_document

    checked = 0
    for doc in _docs():
        fixture = FIXTURES / f"{doc.stem}.mmi"
        if not fixture.exists():
            continue
        raw = read_document(doc)
        (document,) = mml.load(doc, docid=doc.name)
        ours = mml.format(mml.process_document(document), "mmi")
        java = fixture.read_text(encoding="utf-8").replace("\r\n", "\n")
        for label, text in (("java", java), ("ours", ours)):
            for record in records(text, doc.name):
                for start, length, matched in _positions_and_texts(record):
                    assert raw[start : start + length] == matched, (
                        f"{doc.name} ({label}): {start}/{length} is "
                        f"{raw[start : start + length]!r}, recorded as {matched!r}"
                    )
                    checked += 1
    assert checked > 1000, f"only {checked} positions checked"


def test_every_entity_offset_indexes_its_own_passage(mml):
    """The same invariant at the API level, for every corpus document.

    ``tests/unit/test_documents.py`` asserts this on synthetic input; here it runs against the
    real index and the real corpus, which is where an offset bug would actually bite.
    """
    checked = 0
    for doc in _docs():
        (document,) = mml.load(doc, docid=doc.name)
        document.with_docids()
        for passage in document.passages:
            for e in mml.process_passage(passage):
                assert passage.text_at(e.start, e.length) == e.text
                checked += 1
    assert checked > 200, f"only {checked} entities checked"
