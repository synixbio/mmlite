"""Tests for the ConText port (``--usecontext``).

Every expectation here is **MetaMapLite 3.6.2rc8's own output**, taken from a harness that calls
``context.implementation.ConText.applyContext`` in ``context-2012.jar`` directly.  The port was
developed against that harness over 17,748 generated (concept, sentence) pairs, which it matches
exactly; the cases below are the interesting ones frozen so the jar is not needed to run the
suite.  Several of them look wrong — they are what Java does.  See DEVELOPMENT_PLAN §18.
"""

import pytest

from mmlite.pipeline import NegationDetector, get_negation_detector
from mmlite.pipeline.context import (
    AFFIRMED,
    HISTORICAL,
    HYPOTHETICAL,
    NEGATED,
    OTHER,
    PATIENT,
    POSSIBLE,
    RECENT,
    ConText,
)
from mmlite.pipeline.context_triggers import CONTEXT_TRIGGERS
from mmlite.types import ConceptInfo, Entity, Ev

CONCEPT = "chest pain"

# (sentence, negation, temporality, experiencer) exactly as the jar returns them.
JAVA_CASES = [
    ("No chest pain.", NEGATED, RECENT, PATIENT),
    ("absence of chest pain.", NEGATED, RECENT, PATIENT),
    ("The patient denies chest pain.", NEGATED, RECENT, PATIENT),
    ("no chest pain but fever", NEGATED, RECENT, PATIENT),
    ("no  chest   pain", NEGATED, RECENT, PATIENT),
    ("History of chest pain.", AFFIRMED, HISTORICAL, PATIENT),
    ("if chest pain recurs, return.", AFFIRMED, HYPOTHETICAL, PATIENT),
    ("Her mother had chest pain.", AFFIRMED, RECENT, OTHER),
    # Java misses these, for the reasons documented in pipeline/context.py.
    ("chest pain is ruled out.", AFFIRMED, RECENT, PATIENT),
    ("possible chest pain.", AFFIRMED, RECENT, PATIENT),
    ("chest pain is unlikely.", AFFIRMED, RECENT, PATIENT),
    ("gram negative chest pain", AFFIRMED, RECENT, PATIENT),
    ("no increase in chest pain", AFFIRMED, RECENT, PATIENT),
    ("CHEST PAIN denied", AFFIRMED, RECENT, PATIENT),
    ("for the past 3 months of chest pain", AFFIRMED, RECENT, PATIENT),
    ("since last june chest pain", AFFIRMED, RECENT, PATIENT),
    ("previous chest pain", AFFIRMED, RECENT, PATIENT),
    ("chest pain, no fever", AFFIRMED, RECENT, PATIENT),
]


@pytest.fixture(scope="module")
def ctx():
    return ConText()


@pytest.mark.parametrize(
    ("sentence", "negation", "temporality", "experiencer"),
    JAVA_CASES,
    ids=[c[0] for c in JAVA_CASES],
)
def test_matches_the_java_jar(ctx, sentence, negation, temporality, experiencer):
    result = ctx.apply(CONCEPT, sentence)
    assert result is not None
    assert (result.negation, result.temporality, result.experiencer) == (
        negation,
        temporality,
        experiencer,
    )


@pytest.mark.parametrize("concept,sentence", [("chest pain", "no fever."), ("chest pain", "")])
def test_returns_none_when_the_concept_is_absent_or_either_side_is_empty(ctx, concept, sentence):
    assert ctx.apply(concept, sentence) is None


def test_concept_is_marked_and_triggers_are_tagged(ctx):
    """The tags carry a space on both sides, so they survive the word split."""
    assert ctx.preprocess("No chest pain.", CONCEPT) == " <NEG_PRE> [0] ."


def test_pseudo_triggers_are_tagged_before_negation_triggers(ctx):
    """``gram negative`` must not leave a usable ``negative`` behind."""
    assert "<NEG_PSEUDO>" in ctx.preprocess("gram negative chest pain", CONCEPT)


def test_the_first_pseudo_phrase_loses_its_first_character(ctx):
    """(Java quirk) the pseudo alternation is built with a leading ``|`` and then ``substring(2)``
    is taken, which eats one character of the first phrase as well as the separator."""
    first_pseudo = next(e for e in CONTEXT_TRIGGERS if ",pseudo," in e)
    phrase = first_pseudo[: first_pseudo.index(",")]
    assert phrase == "gram negative "
    assert ctx.pseudo.pattern.startswith(phrase[1:].replace(" ", r"[ \t\n\x0b\f\r-]"))


def test_the_lexicon_entry_with_a_comma_in_its_phrase_is_dropped(ctx):
    """Java splits on the first and last comma, so ``"history, physical",pseudo,hist`` yields the
    position ``physical",pseudo`` and matches no branch at all."""
    assert '"history, physical",pseudo,hist' in CONTEXT_TRIGGERS
    # Its phrase would have been `"history` -- the quote proves the entry never reached a bucket.
    # (``history and physical`` is a separate, well-formed entry and is in the pattern.)
    assert '"history' not in ctx.pseudo.pattern
    for pattern in (ctx.hist_pre, ctx.hist_end, ctx.neg_pre):
        assert pattern is None or '"history' not in pattern.pattern


def test_hist_1w_is_never_produced(ctx):
    """(Java quirk) its bucket is shadowed by an identical earlier test, so it stays empty."""
    assert ctx.hist_1w is None


def test_detect_sets_negated_temporality_and_experiencer_on_entities(ctx):
    ci = ConceptInfo("C0008031", "Chest Pain", "chest pain", ("MSH",), frozenset({"sosy"}))
    entity = Entity(
        "d", "text", "chest pain", "NN", 0, 3, 10, evs=[Ev(ci, "chest pain", "", 3, 10)]
    )
    ctx.detect([], [entity], "No chest pain.")
    assert (entity.negated, entity.temporality, entity.experiencer) == (True, RECENT, PATIENT)


def test_detect_never_clears_a_negation_set_by_an_earlier_sentence(ctx):
    """Java only ever calls setNegated(true), so the flag is sticky."""
    entity = Entity("d", "text", "chest pain", "NN", 0, 0, 10, negated=True)
    ctx.detect([], [entity], "chest pain today.")
    assert entity.negated is True


def test_detect_ignores_entities_whose_text_is_not_in_the_sentence(ctx):
    ci = ConceptInfo("C0004057", "aspirin", "aspirin", ("MSH",), frozenset({"phsu"}))
    entity = Entity("d", "text", "aspirin", "NN", 0, 0, 7, evs=[Ev(ci, "aspirin", "", 0, 7)])
    ctx.detect([], [entity], "No chest pain.")
    assert (entity.negated, entity.temporality) == (False, None)


def test_context_is_selectable_by_its_java_property_value():
    assert isinstance(get_negation_detector("context"), ConText)
    assert isinstance(
        get_negation_detector("gov.nih.nlm.nls.metamap.lite.context.ContextWrapper"), ConText
    )
    assert isinstance(ConText(), NegationDetector)


def test_an_unknown_detector_name_is_rejected():
    with pytest.raises(ValueError, match="unknown negation detector"):
        get_negation_detector("nope")


# --- strict_parity=False ----------------------------------------------------------------------
# Unlike everything above, these are *not* Java's output: they pin the fixes that
# Settings.strict_parity=False applies to the jar's bugs.


@pytest.fixture(scope="module")
def fixed():
    return ConText(strict_parity=False)


def _sentences(*sentences):
    """(tokens, text) per sentence, with document-wide offsets, as EntityLookup passes them."""
    from mmlite.pipeline.tokenize import tokenize

    out, offset = [], 0
    for s in sentences:
        out.append((tokenize(s, offset), s))
        offset += len(s) + 1
    return out, " ".join(sentences)


def _mention(doc, phrase, nth=0):
    start = -1
    for _ in range(nth + 1):
        start = doc.index(phrase, start + 1)
    ci = ConceptInfo("C0020538", phrase, phrase, ("MSH",), frozenset({"dsyn"}))
    return Entity(
        "d",
        "text",
        phrase,
        "NN",
        0,
        start,
        len(phrase),
        evs=[Ev(ci, phrase, "", start, len(phrase))],
    )


def _run(detector, *sentences):
    sents, doc = _sentences(*sentences)
    own = _mention(doc, "hypertension", 0)
    family = _mention(doc, "hypertension", 1)
    for tokens, text in sents:
        detector.detect(tokens, [own, family], text)
    return own, family


def test_java_lets_a_later_sentence_overwrite_an_earlier_mention(ctx):
    """The bug being fixed, pinned so the strict mode keeps reproducing it: the patient's own
    diagnosis ends up as someone else's history."""
    own, _family = _run(ctx, "Patient has hypertension today.", "Family history of hypertension.")
    assert (own.experiencer, own.temporality) == (OTHER, HISTORICAL)


def test_fixed_mode_keeps_each_sentence_to_its_own_mentions(fixed):
    own, family = _run(fixed, "Patient has hypertension today.", "Family history of hypertension.")
    assert (own.experiencer, own.temporality) == (PATIENT, RECENT)
    assert (family.experiencer, family.temporality) == (OTHER, HISTORICAL)


def test_fixed_mode_negation_does_not_leak_across_sentences(fixed):
    own, family = _run(fixed, "No hypertension.", "Treated for hypertension.")
    assert (own.negated, family.negated) == (True, False)


@pytest.mark.parametrize(
    "sentence, concept, temporality",
    [
        # [day|days] was one character, so none of these matched in Java
        ("Patient had 3 months of fatigue.", "fatigue", HISTORICAL),
        ("Present for the past 20 days of cough.", "cough", HISTORICAL),
        # "since" was consumed as a trigger before the since-patterns ran
        ("Patient has had fatigue since last march.", "fatigue", HISTORICAL),
        ("Fatigue since 3 weeks ago.", "fatigue", HISTORICAL),
        # the lexicon's one "one,hist" entry
        ("Patient with previous stroke.", "stroke", HISTORICAL),
        # ...which covers only the next word
        ("previous surgery, now stroke.", "stroke", RECENT),
        # \b guards: no partial-number or partial-word matches
        ("Seen after 114 days of rest; cough today.", "cough", RECENT),
    ],
)
def test_fixed_mode_temporality_patterns(fixed, ctx, sentence, concept, temporality):
    assert fixed.apply(concept, sentence).temporality == temporality
    if temporality == HISTORICAL:
        assert ctx.apply(concept, sentence).temporality != HISTORICAL  # Java misses it


def test_fixed_mode_keeps_the_first_pseudo_trigger_whole(fixed, ctx):
    """Java drops the first character of the first pseudo phrase ("gram negative "); the stray
    "g" left behind then stops "no" from being recognised."""
    assert ctx.pseudo.pattern.startswith("ram")
    assert fixed.pseudo.pattern.startswith("gram")
    assert ctx.apply("infection", "no gram negative infection.").negation == AFFIRMED
    assert fixed.apply("infection", "no gram negative infection.").negation == NEGATED


def test_strict_parity_reaches_context_through_the_factory():
    assert get_negation_detector("context").strict_parity is True
    assert get_negation_detector("context", strict_parity=False).strict_parity is False


LIST = "- Denies focal swelling, joint locking, instability, or acute trauma"


@pytest.mark.parametrize("concept", ["joint locking", "instability", "acute trauma"])
def test_fixed_mode_trigger_with_a_trailing_space_reaches_the_whole_list(fixed, ctx, concept):
    """(Java quirk) "denies " carries a trailing space, so its regex needs two separators and only
    fires right before the concept -- the first item of a denied list, never the rest."""
    assert ctx.apply(concept, LIST).negation == AFFIRMED
    assert fixed.apply(concept, LIST).negation == NEGATED


def test_fixed_mode_pseudo_triggers_keep_their_trailing_space(fixed):
    """Pseudo-triggers match without separators; their trailing space is their word boundary."""
    import re

    # "gram negative " ends in a space; stripping it would let it fire inside "negativeness"
    assert re.search(fixed.pseudo, " gram negative rods ") is not None
    assert re.search(fixed.pseudo, " gram negativeness ") is None


def test_fixed_mode_poss_pre_starts_and_stops_a_scan(fixed, ctx):
    # (Java quirk) <POSS_PRE> never starts a forward scan
    assert ctx.apply("pneumonia", "Rule out pneumonia.").negation == AFFIRMED
    assert fixed.apply("pneumonia", "Rule out pneumonia.").negation == POSSIBLE
    assert fixed.apply("pneumonia", "Plan: r/o pneumonia, recheck.").negation == POSSIBLE
    # ...nor stops one, so a leading "no" would negate what is only to be ruled out
    sentence = "No rash, rule out pneumonia."
    assert fixed.apply("pneumonia", sentence).negation == POSSIBLE
    assert fixed.apply("rash", sentence).negation == NEGATED
    java_stops = ConText(strict_parity=False)
    java_stops._fwd_stops = tuple(s for s in java_stops._fwd_stops if s != "<POSS_PRE>")
    assert java_stops.apply("pneumonia", sentence).negation == NEGATED


@pytest.mark.parametrize(
    "concept, sentence, temporality, experiencer",
    [
        (
            "family history of hypertension",
            "Family history of hypertension in mother.",
            HISTORICAL,
            OTHER,
        ),
        ("history of stroke", "History of stroke.", HISTORICAL, PATIENT),
        ("previous stroke", "previous stroke noted", HISTORICAL, PATIENT),
    ],
)
def test_fixed_mode_reads_triggers_inside_the_concept_name(
    fixed, ctx, concept, sentence, temporality, experiencer
):
    """(Java quirk) the concept is replaced by [0] before triggers are tagged, so a UMLS concept
    whose own name says "family history" or "history of" came out Recent / Patient."""
    assert (ctx.apply(concept, sentence).temporality, ctx.apply(concept, sentence).experiencer) == (
        RECENT,
        PATIENT,
    )
    result = fixed.apply(concept, sentence)
    assert (result.temporality, result.experiencer) == (temporality, experiencer)


def test_fixed_mode_never_negates_a_concept_by_its_own_name(fixed):
    """A concept named "No known allergies" *is* the negative finding; negating it would turn it
    into its opposite, so negation triggers inside a concept's name are ignored."""
    assert fixed.apply("no known allergies", "No known allergies.").negation == AFFIRMED
    # the same words as *context* for a narrower concept still negate it
    assert fixed.apply("allergies", "No known allergies.").negation == NEGATED


def test_fixed_mode_keeps_each_mention_to_its_bullet(fixed):
    """Unpunctuated bullets are one "sentence" to the splitter; without this, the 15-word window of
    one bullet's "Denies" negated the next bullet's findings."""
    text = "Plan:\n   - Denies swelling, locking\n   - Symptoms managed with OTC acetaminophen"
    sents, doc = _sentences(text)
    tokens = sents[0][0]

    def mention(phrase):
        start = doc.index(phrase)
        ci = ConceptInfo("C0000001", phrase, phrase, ("MSH",), frozenset({"sosy"}))
        return Entity(
            "d",
            "text",
            phrase,
            "NN",
            0,
            start,
            len(phrase),
            evs=[Ev(ci, phrase, "", start, len(phrase))],
        )

    locking, symptoms = mention("locking"), mention("Symptoms")
    fixed.detect(tokens, [locking, symptoms], text)
    assert (locking.negated, symptoms.negated) == (True, False)

    strict_locking, strict_symptoms = mention("locking"), mention("Symptoms")
    ConText().detect(tokens, [strict_locking, strict_symptoms], text)
    assert strict_locking.negated is False  # Java: "denies " reaches only its first item


@pytest.mark.parametrize(
    "sentence",
    [
        "Pneumonia was ruled out.",
        "Pneumonia is ruled out.",
        "Pneumonia has been ruled out.",
        "Pneumonia have been ruled out.",
        "Pneumonia are ruled out.",
    ],
)
def test_fixed_mode_prefers_the_longest_trigger(fixed, ctx, sentence):
    """(Java quirk) buckets are substituted one after another, so "ruled out ,pre,neg" claims the
    phrase before "was ruled out ,post,neg" is tried -- leaving a *pre* trigger at the end of the
    sentence with nothing in front of it to negate.  Fixed mode takes the longest trigger."""
    assert ctx.apply("pneumonia", sentence).negation == AFFIRMED
    assert fixed.apply("pneumonia", sentence).negation == NEGATED


def test_fixed_mode_longest_match_keeps_the_shorter_trigger_where_it_is_the_whole_phrase(fixed):
    """The pre-trigger still fires when no longer trigger covers it."""
    assert fixed.apply("pneumonia", "Ruled out pneumonia.").negation == NEGATED
    assert fixed.apply("pneumonia", "Rules out pneumonia.").negation == NEGATED


def test_fixed_mode_single_pass_leaves_neighbouring_triggers_intact(fixed):
    """Tagging every bucket in one pass must not let a trigger eat the separator the next one
    needs: the trailing separator is a lookahead, so adjacent triggers both fire."""
    tagged = fixed.preprocess("no evidence of pneumonia", "pneumonia")
    assert "<NEG_PRE>" in tagged and "[0]" in tagged


@pytest.mark.parametrize(
    "sentence, negation",
    [
        ("Patient has pneumonia.", AFFIRMED),
        ("Admitted with pneumonia.", AFFIRMED),
        ("Treated for pneumonia.", AFFIRMED),
        ("No evidence of pneumonia.", NEGATED),
        ("Without evidence of pneumonia.", NEGATED),
        ("Negative for pneumonia.", NEGATED),
        ("Denies pneumonia.", NEGATED),
    ],
)
def test_fixed_mode_negation_table(fixed, sentence, negation):
    """The phrasings from docs/Negation detection.md, pinned."""
    assert fixed.apply("pneumonia", sentence).negation == negation


# --- the assertion verdict and the phrases the jar's lexicon lacks --------------------------------


@pytest.mark.parametrize(
    "sentence, assertion, negated",
    [
        # "cannot ,pre,neg" made these negations, the opposite of what they say
        ("Cannot exclude pneumonia.", POSSIBLE, False),
        ("Unable to exclude pneumonia.", POSSIBLE, False),
        ("Pneumonia cannot be excluded.", POSSIBLE, False),
        ("Cannot rule out pneumonia.", POSSIBLE, False),
        # hedging the lexicon has no entry for at all
        ("Findings concerning for pneumonia.", POSSIBLE, False),
        ("Suspicious for pneumonia.", POSSIBLE, False),
        ("Pneumonia is suspected.", POSSIBLE, False),
        ("Differential includes pneumonia.", POSSIBLE, False),
        # the adjective, where the lexicon has only "absence of"
        ("Absent pneumonia.", NEGATED, True),
        # unchanged
        ("Patient has pneumonia.", AFFIRMED, False),
        ("No evidence of pneumonia.", NEGATED, True),
    ],
)
def test_extra_triggers(fixed, sentence, assertion, negated):
    assert fixed.apply("pneumonia", sentence).negation == assertion


def test_extra_triggers_are_fixed_mode_only(ctx):
    """Strict mode sees the jar's lexicon and nothing else, so Java parity is untouched.

    (Here "cannot " keeps its trailing space, so it needs two separators and never fires before
    "exclude" -- Java reaches Affirmed by a different route than fixed mode's Possible.)"""
    assert ctx.apply("pneumonia", "Cannot exclude pneumonia.").negation == AFFIRMED
    assert ctx.apply("pneumonia", "Findings concerning for pneumonia.").negation == AFFIRMED
    assert ctx.apply("pneumonia", "Absent pneumonia.").negation == AFFIRMED


def test_fixed_mode_can_run_without_the_extra_triggers():
    bare = ConText(strict_parity=False, extra_triggers=())
    assert bare.apply("pneumonia", "Cannot exclude pneumonia.").negation == NEGATED


def test_extra_trigger_phrases_do_not_collide_with_the_lexicon():
    """The longest-match tagger keys its tag lookup on the phrase, so phrases must be unique."""
    from mmlite.pipeline.context_triggers import EXTRA_TRIGGERS

    base = [e[: e.index(",")].strip() for e in CONTEXT_TRIGGERS]
    extra = [e[: e.index(",")].strip() for e in EXTRA_TRIGGERS]
    assert not set(base) & set(extra)
    assert len(set(extra)) == len(extra)


@pytest.mark.parametrize(
    "sentence, assertion, negated",
    [
        ("No evidence of hypertension.", NEGATED, True),
        ("Rule out hypertension.", POSSIBLE, False),
        ("Patient has hypertension.", AFFIRMED, False),
    ],
)
def test_detect_records_the_assertion_on_the_entity(fixed, sentence, assertion, negated):
    """Java keeps only the boolean, so a hedged mention is indistinguishable from an affirmed
    one; the verdict it computes and discards is kept on its own field."""
    sents, _doc = _sentences(sentence)
    tokens, text = sents[0]
    entity = _mention(sentence, "hypertension")
    fixed.detect(tokens, [entity], text)
    assert (entity.assertion, entity.negated) == (assertion, negated)


def test_negex_leaves_the_assertion_unset():
    """The default detector has no verdict to report, as it has no temporality or experiencer."""
    from mmlite.pipeline.negation import NegEx
    from mmlite.pipeline.tokenize import tokenize

    sentence = "No hypertension."
    entity = _mention(sentence, "hypertension")
    NegEx().detect(tokenize(sentence, 0), [entity], sentence)
    assert entity.negated is True
    assert entity.assertion is None
