"""Tests for the Schwartz-Hearst extractor, MarkAbbreviations and user-defined acronyms."""

from pathlib import Path

import pytest

from mmlite.config import Settings
from mmlite.index import IndexLookup
from mmlite.pipeline import TextPipeline
from mmlite.pipeline.abbreviations import (
    AbbrInfo,
    UserDefinedAcronyms,
    extract_abbr_pairs,
    find_best_long_form,
    mark_abbreviations,
)
from mmlite.pipeline.entity_lookup import EntityLookup
from mmlite.pipeline.postag import NullPosTagger
from mmlite.types import ConceptInfo, Entity, Ev


@pytest.mark.parametrize(
    "text, pairs",
    [
        ("type 2 diabetes mellitus (T2DM) is common", [("T2DM", "type 2 diabetes mellitus")]),
        ("the World Health Organization (WHO) said", [("WHO", "World Health Organization")]),
        (
            "tumor necrosis factor alpha (TNF-alpha) and interleukin 6 (IL-6)",
            [("TNF-alpha", "tumor necrosis factor alpha"), ("IL-6", "interleukin 6")],
        ),
        ("a value (5) was seen", []),  # no letters
        ("the ABC (some long form here) text", []),  # too many long-form words for 3 chars
        ("no parens here", []),
    ],
)
def test_extract_pairs(text, pairs):
    got = [(a.short_form, a.long_form) for a in extract_abbr_pairs(text)]
    assert got == pairs


def test_offsets_are_absolute():
    text = "see type 2 diabetes mellitus (T2DM) now"
    (a,) = extract_abbr_pairs(text, base_offset=100)
    assert a.short_form_index == 100 + text.index("T2DM")
    assert a.long_form_index == 100 + text.index("type 2")
    assert text[a.long_form_index - 100 :][: len(a.long_form)] == a.long_form


def test_long_form_in_parentheses():
    text = "WHO (World Health Organization) is here"
    (a,) = extract_abbr_pairs(text)
    assert (a.short_form, a.long_form) == ("WHO", "World Health Organization")
    assert a.short_form_index == 0 and a.long_form_index == text.index("World")


def test_find_best_long_form():
    assert find_best_long_form("T2DM", "diagnosed with type 2 diabetes mellitus") == len(
        "diagnosed with "
    )
    assert find_best_long_form("XYZ", "nothing matches") is None


def _entity(text, phrase, cui="C0011860", start=None):
    start = text.index(phrase) if start is None else start
    ci = ConceptInfo(cui, "Pref", phrase, frozenset({"MSH"}), frozenset({"dsyn"}))
    e = Entity("d", "text", phrase, "NN", 0, start, len(phrase), 0.0)
    e.evs = [Ev(ci, phrase, phrase, start, len(phrase), 0.0, "NN")]
    return e


def test_mark_abbreviations_copies_entity_to_short_forms():
    text = "type 2 diabetes mellitus (T2DM) is common. T2DM is treatable."
    e = _entity(text, "type 2 diabetes mellitus")
    added = mark_abbreviations(text, extract_abbr_pairs(text), [e])
    spans = sorted({(a.start, a.length, a.text) for a in added})
    assert spans == [(26, 4, "T2DM"), (43, 4, "T2DM")]
    # copies share the Ev objects (Java copies the set, not the evidence)
    assert all(a.evs[0] is e.evs[0] for a in added)
    assert all(a.cuis == ["C0011860"] for a in added)


def test_mark_abbreviations_requires_exact_long_form_text():
    text = "type 2 diabetes mellitus (T2DM) is common"
    e = _entity(text, "diabetes mellitus")  # entity text != long form
    assert mark_abbreviations(text, extract_abbr_pairs(text), [e]) == []


def test_mark_abbreviations_tokenizes_the_passage_once(monkeypatch):
    """Java's findMatches re-tokenizes the whole passage for every abbreviation-bearing entity,
    which on a 48 KB note was ~74% of annotation time. The port builds one
    token index per passage; the matches must be exactly what per-call tokenization gives."""
    from mmlite.pipeline import abbreviations as ab

    text = (
        "type 2 diabetes mellitus (T2DM) and World Health Organization (WHO) guidance. "
        "T2DM again; WHO again. type 2 diabetes mellitus and World Health Organization."
    )
    entities = [
        _entity(text, "type 2 diabetes mellitus"),
        _entity(text, "World Health Organization", cui="C0043237"),
        _entity(text, "type 2 diabetes mellitus", start=text.rindex("type 2")),
    ]
    infos = extract_abbr_pairs(text)

    def spans(added):
        return sorted((a.start, a.length, a.text, tuple(a.cuis)) for a in added)

    calls = 0
    real = ab.tokenize

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(ab, "tokenize", counting)
    fast = spans(mark_abbreviations(text, infos, entities))
    assert calls == 1

    # reference: Java's behaviour, tokenizing afresh for every match
    reference = []
    for entity in entities:
        for info in infos:
            if info.long_form == entity.text and info.short_form_index > 0:
                new = ab._copy_at(entity, info.short_form, info.short_form_index)
                reference.append(new)
                reference.extend(ab.find_matches(text, new))
    assert fast == spans(reference)
    assert calls > 2  # the reference really did re-tokenize, so the comparison means something


def test_mark_abbreviations_skips_offset_zero():
    # Java: location.getOffset() > 0
    info = AbbrInfo("WHO", 0, "World Health Organization", 5)
    e = _entity("WHO (World Health Organization)", "World Health Organization")
    assert mark_abbreviations("WHO (World Health Organization)", [info], [e]) == []


# --- user-defined acronyms -----------------------------------------------------------------


@pytest.fixture(scope="module")
def index(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        yield ix


def test_uda_entities(index, tmp_path: Path):
    f = tmp_path / "uda.txt"
    f.write_text("HA|Heart Attack\nZZ|no such concept\n", encoding="utf-8")
    lk = EntityLookup(index, Settings(uda_file=f))
    assert len(lk.udas) == 2
    pipe = TextPipeline(Settings(), tagger=NullPosTagger())
    text = "HA today. ZZ too."
    ents = lk.process_sentences(pipe.process(text), text)
    assert [(e.text, e.start, e.cuis, e.score) for e in ents] == [("HA", 0, ["C0027051"], 100.0)]
    ev = ents[0].evs[0]
    assert (ev.matched_text, ev.start, ev.length, ev.score) == ("HA", 0, 2, 100.0)
    assert (ents[0].fieldid, ents[0].lexical_category, ents[0].sentence_number) == ("TXT", "UNK", 1)


def test_uda_long_form_falls_back_to_normalized_lookup(index):
    lk = EntityLookup(index)
    u = UserDefinedAcronyms.from_file(_write(lk, "AD|Alzheimer's Disease"), lk._lookup)
    assert [c.cui for c in u.map["AD"].concepts] == ["C0002395"]


def _write(lk, content):
    import tempfile

    p = Path(tempfile.mkdtemp()) / "uda.txt"
    p.write_text(content + "\n", encoding="utf-8")
    return p
