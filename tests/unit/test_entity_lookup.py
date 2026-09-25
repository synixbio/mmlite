"""Tests for the EntityLookup4 port, against the synthetic UMLS index in conftest.py."""

from pathlib import Path

import pytest

from mmlite.config import Settings
from mmlite.index import IndexLookup
from mmlite.pipeline import TextPipeline
from mmlite.pipeline.entity_lookup import CustomTerms, EntityLookup, SpecialTerms
from mmlite.pipeline.postag import NullPosTagger
from mmlite.types import Entity


class DictTagger:
    """Tags words from a dict, default NN."""

    def __init__(self, tags=None):
        self.tags = tags or {}

    def tag(self, words):
        return [self.tags.get(w, "NN") for w in words]


@pytest.fixture(scope="module")
def index(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        yield ix


@pytest.fixture
def lookup(index):
    return EntityLookup(index)


def _spans(entities):
    return [(e.text, sorted(e.cuis)) for e in entities]


# --- lookup_term (no tagging, no filtering, no subsumption removal) -------------------------


def test_lookup_term_basic(lookup):
    ents = lookup.lookup_term("Heart Attack")
    assert _spans(ents) == [("Heart Attack", ["C0027051"])]
    e = ents[0]
    assert (e.start, e.length) == (0, 12)
    ev = e.evs[0]
    assert ev.matched_text == "Heart Attack" and ev.norm_term == "heart attack"
    assert ev.concept.preferred_name == "Myocardial Infarction"
    assert ev.concept.concept_string == "Heart Attack"
    assert ev.concept.semantic_types == {"dsyn"} and ev.concept.sources == ("MSH",)


def test_lookup_term_keeps_all_spans_without_subsumption(lookup):
    # "Type 2 Diabetes" (MSH) inside "Type 2 diabetes mellitus" (SNOMED): both survive here
    ents = lookup.lookup_term("Type 2 diabetes mellitus")
    assert _spans(ents) == [
        ("Type 2 diabetes mellitus", ["C0011860"]),
        ("Type 2 diabetes", ["C0011860"]),
        ("diabetes mellitus", ["C0011849"]),
        ("diabetes", ["C0011849"]),
    ]


def test_terms_of_two_chars_or_less_never_match(index):
    lk = EntityLookup(index, custom_terms=CustomTerms([("C0027051", "MI"), ("C0027051", "MIs")]))
    assert lk.lookup_term("MI") == []
    assert _spans(lk.lookup_term("MIs")) == [("MIs", ["C0027051"])]


def test_span_must_start_and_end_alphanumeric(lookup):
    ents = lookup.lookup_term("(aspirin)")
    assert _spans(ents) == [("aspirin", ["C0004057"])]
    assert ents[0].start == 1


def test_possessive_and_greek_go_through_normalize(lookup):
    assert _spans(lookup.lookup_term("Alzheimer's disease")) == [
        ("Alzheimer's disease", ["C0002395"])
    ]
    # CharUtils.isAlphaNumeric counts Greek letters, so a span may end (or start) in one.  This
    # test used to assert the opposite -- that "TNF-α" never matches -- which was this port's bug
    # described as Java's behaviour.  MetaMapLite 3.6.2rc8 matches "TNF-α" in the parity corpus.
    assert _spans(lookup.lookup_term("TNF-α")) == [("TNF-α", ["C1456820"])]
    assert _spans(lookup.lookup_term("TNF-alpha")) == [("TNF-alpha", ["C1456820"])]


def test_a_span_may_start_with_a_greek_letter(lookup):
    """Java matches "β-blocker" in the parity corpus's tricky_tokens.txt; so must we."""
    from mmlite.pipeline.entity_lookup import _is_alnum

    assert _is_alnum("β") and _is_alnum("α") and _is_alnum("Ω")
    assert not _is_alnum("ö") and not _is_alnum("-") and not _is_alnum("͸")  # unassigned
    assert _is_alnum(";")  # GREEK QUESTION MARK: in Java's list, so it counts


def test_capitalized_query_matches_a_possessive_only_index_key(lookup):
    # "Parkinson's Disease" is indexed only in its possessive form (key
    # "parkinson's disease"); normalize() strips the possessive ("parkinson disease"), which
    # does not exist as a key here, so a capitalized query can only match via
    # index_key(original) ("parkinson's disease"), not via the raw verbatim span or via norm.
    assert _spans(lookup.lookup_term("Parkinson's Disease")) == [
        ("Parkinson's Disease", ["C0030567"])
    ]


def test_other_is_never_a_span_start(index):
    lk = EntityLookup(index, custom_terms=CustomTerms([("C0000001", "other aspirin")]))
    assert _spans(lk.lookup_term("other aspirin")) == [("aspirin", ["C0004057"])]


# --- POS gate ----------------------------------------------------------------------------------


def test_pos_gate_uses_first_token_only(index):
    lk = EntityLookup(index)
    pipe = TextPipeline(Settings(), tagger=DictTagger({"Heart": "VB", "attack": "NN"}))
    sents = pipe.process("Heart attack. heart Attack.")
    ents = lk.process_sentences(sents)
    # first "Heart" is VB -> the 2-token span cannot start there; second one is NN -> matches
    assert _spans(ents) == [("heart Attack", ["C0027051"])]


def test_whitespace_tokens_never_start_a_span_when_tagged(index):
    lk = EntityLookup(index)
    pipe = TextPipeline(Settings(), tagger=DictTagger())
    ents = lk.process_sentences(pipe.process("  aspirin"))
    assert [(e.start, e.text) for e in ents] == [(2, "aspirin")]


def test_untagged_tokens_are_accepted(index):
    lk = EntityLookup(index)
    pipe = TextPipeline(Settings(enable_postagging=False), tagger=NullPosTagger())
    assert _spans(lk.process_sentences(pipe.process("aspirin"))) == [("aspirin", ["C0004057"])]


# --- document-level post-processing ----------------------------------------------------------


def _doc(index, text, **kw):
    lk = EntityLookup(index, **kw)
    pipe = TextPipeline(Settings(), tagger=DictTagger({"The": "DT"}))
    return lk.process_sentences(pipe.process(text))


def test_subsumed_entities_removed_and_sorted(index):
    ents = _doc(index, "Type 2 diabetes mellitus. Heart Attack and aspirin.")
    assert _spans(ents) == [
        ("Type 2 diabetes mellitus", ["C0011860"]),
        ("Heart Attack", ["C0027051"]),
        ("aspirin", ["C0004057"]),
    ]
    assert [e.location_position for e in ents] == [0, 1, 1]
    assert [e.start for e in ents] == sorted(e.start for e in ents)


def test_keep_subsumed(index):
    ents = _doc(index, "Type 2 diabetes mellitus", remove_subsumed=False)
    assert len(ents) == 4
    assert [e.start for e in ents] == [0, 0, 7, 7]  # start asc, longer first on ties


def test_semantic_type_restriction_drops_evs_then_entities(index):
    lk = EntityLookup(index)
    pipe = TextPipeline(Settings(), tagger=DictTagger())
    sents = pipe.process("Heart Attack and aspirin.")
    assert _spans(lk.process_sentences(sents, semantic_types={"phsu"})) == [
        ("aspirin", ["C0004057"])
    ]
    assert _spans(lk.process_sentences(sents, semantic_types={"all"})) == [
        ("Heart Attack", ["C0027051"]),
        ("aspirin", ["C0004057"]),
    ]


def test_remove_subsumed_matches_the_pairwise_definition():
    """The sweep replaces Java's O(n^2) pairwise scan; pin it to that definition."""
    import random

    def pairwise(ents):
        return [
            e
            for e in ents
            if not any(o is not e and o.start <= e.start and e.end <= o.end for o in ents)
        ]

    rng = random.Random(20260918)
    for _ in range(200):
        spans = rng.sample([(s, ln) for s in range(20) for ln in range(1, 8)], k=rng.randint(1, 25))
        ents = [Entity("d", "text", "x" * ln, "NN", 0, s, ln) for s, ln in spans]
        assert sorted(
            (e.start, e.end) for e in EntityLookup.remove_subsumed_entities(ents)
        ) == sorted((e.start, e.end) for e in pairwise(ents))


@pytest.mark.parametrize("sts", [{"phsu"}, {"PHSU"}, {"T121"}, {"t121"}, {"phsu", "T121"}])
def test_semantic_type_restriction_accepts_abbreviations_tuis_and_any_case(index, sts):
    lk = EntityLookup(index)
    pipe = TextPipeline(Settings(), tagger=DictTagger())
    sents = pipe.process("Heart Attack and aspirin.")
    assert _spans(lk.process_sentences(sents, semantic_types=sts)) == [("aspirin", ["C0004057"])]


@pytest.mark.parametrize("sts", [{"all"}, {"ALL"}, set(), None])
def test_semantic_type_restriction_all_means_unrestricted(index, sts):
    lk = EntityLookup(index)
    pipe = TextPipeline(Settings(), tagger=DictTagger())
    sents = pipe.process("Heart Attack and aspirin.")
    assert _spans(lk.process_sentences(sents, semantic_types=sts)) == [
        ("Heart Attack", ["C0027051"]),
        ("aspirin", ["C0004057"]),
    ]


def test_source_restriction_is_case_insensitive(index):
    lk = EntityLookup(index)
    pipe = TextPipeline(Settings(), tagger=DictTagger())
    sents = pipe.process("Heart Attack. Diabetes.")
    assert _spans(lk.process_sentences(sents, sources={"snomedct_us"})) == [
        ("Diabetes", ["C0011849"])
    ]


def test_source_restriction_uses_full_cui_source_set(index):
    lk = EntityLookup(index)
    pipe = TextPipeline(Settings(), tagger=DictTagger())
    # "Heart Attack" is an MSH-only string, and C0027051 is MSH-only -> dropped
    # "Diabetes" is an MSH-only string, but C0011849 also has SNOMED -> kept
    sents = pipe.process("Heart Attack. Diabetes.")
    assert _spans(lk.process_sentences(sents, sources={"SNOMEDCT_US"})) == [
        ("Diabetes", ["C0011849"])
    ]


def test_restriction_precedes_subsumption(index):
    # Use ST restriction to strip the long span's concept so the short span can survive:
    lk = EntityLookup(index, custom_terms=CustomTerms([("C0004057", "aspirin heart attack")]))
    pipe = TextPipeline(Settings(), tagger=DictTagger())
    sents = pipe.process("aspirin heart attack")
    # dsyn: the 3-token custom span is phsu/orch -> dropped, so "heart attack" (dsyn) survives
    assert _spans(lk.process_sentences(sents, semantic_types={"dsyn"})) == [
        ("heart attack", ["C0027051"])
    ]
    # no restriction: the longer span subsumes it
    assert _spans(lk.process_sentences(sents)) == [("aspirin heart attack", ["C0004057"])]


# --- excluded and custom terms ---------------------------------------------------------------


def test_excluded_terms_by_cui_and_wildcard(index, tmp_path: Path):
    f = tmp_path / "specialterms.txt"
    f.write_text("C0004057:aspirin\n*:heart attack\n", encoding="utf-8")
    lk = EntityLookup(index, excluded_terms=SpecialTerms.from_file(f))
    assert lk.lookup_term("aspirin") == []
    assert lk.lookup_term("Heart Attack") == []  # wildcard matches on the normalized term
    assert _spans(lk.lookup_term("diabetes")) == [("diabetes", ["C0011849"])]


def test_excluded_terms_via_settings_and_missing_file(index, tmp_path: Path):
    lk = EntityLookup(index, Settings(excluded_terms_file=tmp_path / "nope.txt"))
    assert len(lk.excluded_terms) == 0


def test_custom_terms_and_udas_tolerate_a_missing_file(index, tmp_path: Path):
    # Same contract as the excluded-terms file above: warn and carry on rather than crash
    # mid-annotation, since these paths can also arrive from a properties file or env var.
    lk = EntityLookup(
        index,
        Settings(cui_term_list_file=tmp_path / "nope.txt", uda_file=tmp_path / "nope2.txt"),
    )
    assert len(lk.custom_terms) == 0 and len(lk.udas) == 0
    assert _spans(lk.lookup_term("aspirin")) == [("aspirin", ["C0004057"])]


def test_custom_terms_from_file(index, tmp_path: Path):
    f = tmp_path / "custom.txt"
    f.write_text("C9999999|Frobnitz Syndrome\nC0027051|Ticker Trouble\n", encoding="utf-8")
    lk = EntityLookup(index, Settings(cui_term_list_file=f))
    ents = lk.lookup_term("frobnitz syndrome and ticker trouble")
    assert _spans(ents) == [
        ("frobnitz syndrome", ["C9999999"]),
        ("ticker trouble", ["C0027051"]),
    ]
    frob = ents[0].evs[0].concept
    assert frob.preferred_name == "C9999999"  # unknown CUI: falls back to the CUI itself
    assert frob.sources == ("USERDEFINED",) and frob.semantic_types == {"unknown"}
    assert frob.concept_string == "frobnitz syndrome"
    tick = ents[1].evs[0].concept
    # USERDEFINED is appended, so the index's own sources keep their order ahead of it.
    assert tick.sources == ("MSH", "USERDEFINED") and tick.semantic_types == {"dsyn"}


def test_custom_term_does_not_duplicate_index_cui(index):
    lk = EntityLookup(index, custom_terms=CustomTerms([("C0004057", "aspirin")]))
    ents = lk.lookup_term("aspirin")
    assert len(ents) == 1 and len(ents[0].evs) == 1
    assert ents[0].evs[0].concept.sources == ("MSH", "SNOMEDCT_US")  # index concept wins


# --- identity semantics ----------------------------------------------------------------------


def test_entity_identity_is_span():
    a = Entity("d", "f", "x", "", 0, 5, 3)
    b = Entity("d", "f", "y", "", 1, 5, 3)
    assert a == b and len({a, b}) == 1
    assert a != Entity("d", "f", "x", "", 0, 5, 4)


# --- negation is handed one sentence's entities at a time --------------------------------------


class Recorder:
    """Wraps a detector and records how many entities each sentence was given."""

    def __init__(self, inner, sentence_local):
        self.inner, self.sentence_local, self.sizes = inner, sentence_local, []

    def detect(self, tokens, entities, sentence=""):
        entities = list(entities)
        self.sizes.append(len(entities))
        self.inner.detect(tokens, entities, sentence)


NOTE = "No fever today. Patient has Type 2 diabetes mellitus (T2DM). No T2DM complications."


def _annotate(index, detector):
    lk = EntityLookup(index, negex=detector)
    return lk.process_text(NOTE, TextPipeline(tagger=NullPosTagger()))


def test_sentence_local_detector_gets_only_its_sentence_and_the_same_answer(index):
    from mmlite.pipeline.negation import NegEx

    local, whole = Recorder(NegEx(), True), Recorder(NegEx(), False)
    fast, slow = _annotate(index, local), _annotate(index, whole)

    def view(ents):
        return [(e.start, e.text, e.negated) for e in ents]

    assert view(fast) == view(slow)
    # Java: every sentence gets the whole passage (before subsumed spans are pruned, so more
    # than the final count)
    assert len(set(whole.sizes)) == 1 and whole.sizes[0] >= len(slow)
    assert sum(local.sizes) < sum(whole.sizes)

    # The trap the bucketing must avoid: abbreviation-propagated entities have their sentence
    # index reset to 0, so bucketing on location_position would analyse the third sentence's
    # "T2DM" against "No fever today." and lose its negation.
    late = next(e for e in fast if e.text == "T2DM" and e.start > NOTE.index("No T2DM"))
    assert late.location_position == 0
    assert late.negated is True


def test_detectors_are_sentence_local_only_where_that_preserves_java_output():
    from mmlite.pipeline import get_negation_detector

    assert get_negation_detector("negex").sentence_local is True
    assert get_negation_detector("context").sentence_local is False  # Java needs every entity
    assert get_negation_detector("context", strict_parity=False).sentence_local is True
