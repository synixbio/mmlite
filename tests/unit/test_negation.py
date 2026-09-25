"""Tests for the NegEx port."""

from mmlite.pipeline.negation import NegEx
from mmlite.pipeline.tokenize import tokenize
from mmlite.types import ConceptInfo, Entity, Ev

DSYN = ConceptInfo("C1", "X", "x", frozenset({"MSH"}), frozenset({"dsyn"}))
TMCO = ConceptInfo("C2", "Y", "y", frozenset({"MSH"}), frozenset({"tmco"}))


def _entity(text: str, phrase: str, concept=DSYN) -> Entity:
    start = text.index(phrase)
    e = Entity("d", "text", phrase, "NN", 0, start, len(phrase), 0.0)
    e.evs = [Ev(concept, phrase, phrase.lower(), start, len(phrase), 0.0, "NN")]
    return e


def _negated(text: str, phrase: str, concept=DSYN, **kw) -> bool:
    e = _entity(text, phrase, concept)
    NegEx(**kw).detect(tokenize(text), [e])
    return e.negated


def test_pre_negation():
    assert _negated("There was no sign of pneumonia.", "pneumonia")
    assert _negated("She denies chest pain.", "chest pain")
    assert _negated("No history of myocardial infarction.", "myocardial infarction")


def test_post_negation():
    assert _negated("Pneumonia was ruled out.", "Pneumonia")


def test_not_negated_without_trigger():
    assert not _negated("She has chest pain.", "chest pain")


def test_pseudo_negation():
    # "rule out" is only a pseudo-negation (pnega); "rules out" is a real trigger
    assert not _negated("Please rule out pneumonia.", "pneumonia")
    assert _negated("This rules out pneumonia.", "pneumonia")
    # "not ruled out" (pnegb) out-lengthens "not" (nega) at the same position; "ruled out" is a
    # *pre*-negation trigger so it cannot reach an entity before it -> not negated
    assert not _negated("Pneumonia was not ruled out.", "Pneumonia")


def test_conjunction_terminates_scope():
    assert not _negated("No fever but pneumonia persists.", "pneumonia")


def test_window():
    text = "No evidence of a b c d e f pneumonia."
    assert not _negated(text, "pneumonia")  # 8 tokens away
    assert _negated(text, "pneumonia", token_window=10)


def test_semantic_type_gate():
    assert not _negated("No sign of tomorrow.", "tomorrow", concept=TMCO)


def test_meta_tokens_merge_runs_of_non_whitespace():
    toks = tokenize("No b.i.d. dosing (T2DM).")
    merged = [t.text for t in NegEx().add_meta_tokens(toks)]
    assert "b.i" in merged and "(T2DM)" in merged  # runs stop at a period token


def test_filter_tokens_drops_whitespace_and_periods():
    toks = tokenize("no. sign")
    assert [t.text for t in NegEx().filter_tokens(toks)] == ["no", "sign"]


def test_phrase_list_matches_the_naive_per_phrase_scan():
    """The bucketed scan is an optimization of ``find_phrase`` over every trigger; pin them."""
    negex = NegEx()
    sentences = [
        "No history of myocardial infarction but chest pain is not ruled out.",
        "Patient denies fever, denies chills, and denies no shortness of breath.",
        "No no no evidence of evidence of disease.",  # overlapping and repeated triggers
        "Nothing here matches at all.",
        "",
    ]
    for text in sentences:
        words = [t.text for t in negex.filter_tokens(negex.add_meta_tokens(tokenize(text)))]
        naive = [
            (phrase, kind, positions)
            for phrase, kind in negex.phrase_types.items()
            if (positions := negex.find_phrase(words, phrase))
        ]
        fast = [(p.phrase, p.kind, p.positions) for p in negex.phrase_list(words)]
        assert fast == naive, text


def test_entity_outside_sentence_is_ignored():
    text = "No pneumonia."
    e = _entity(text, "pneumonia")
    e.start += 100
    NegEx().detect(tokenize(text), [e])
    assert not e.negated
