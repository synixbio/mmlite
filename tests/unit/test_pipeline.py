import pytest

from mmlite.config import Settings
from mmlite.pipeline import ALLOWED_PART_OF_SPEECH, TextPipeline
from mmlite.pipeline.postag import NullPosTagger, add_part_of_speech
from mmlite.pipeline.tokenize import tokenize


class FakeTagger:
    """Tags every word NN, so we can check the whitespace/word interleaving without spaCy."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def tag(self, words):
        self.calls.append(list(words))
        return ["NN"] * len(words)


def test_add_part_of_speech_skips_whitespace():
    toks = tokenize("heart attack, now")
    tagger = FakeTagger()
    add_part_of_speech(toks, tagger)
    assert tagger.calls == [["heart", "attack", ",", "now"]]
    assert [(t.text, t.pos) for t in toks] == [
        ("heart", "NN"),
        (" ", "WS"),
        ("attack", "NN"),
        (",", "NN"),
        (" ", "WS"),
        ("now", "NN"),
    ]


def test_add_part_of_speech_length_mismatch():
    class Bad:
        def tag(self, words):
            return ["NN"]

    with pytest.raises(ValueError):
        add_part_of_speech(tokenize("a b"), Bad())


def test_null_tagger_leaves_empty_pos():
    toks = add_part_of_speech(tokenize("a b"), NullPosTagger())
    assert [t.pos for t in toks] == ["", "WS", ""]
    assert "" in ALLOWED_PART_OF_SPEECH  # untagged spans are accepted by EntityLookup4


def test_pipeline_with_fake_tagger():
    pipe = TextPipeline(Settings(segmentation_method="LINES"), tagger=FakeTagger())
    text = "Heart attack.\nAspirin 81 mg."
    out = pipe.process(text, base_offset=7)
    assert [ts.index for ts in out] == [0, 1]
    assert out[1].sentence.offset == 7 + text.index("Aspirin")
    for ts in out:
        for t in ts.tokens:
            assert text[t.start - 7 : t.end - 7] == t.text
            assert t.pos == ("WS" if t.is_whitespace else "NN")


def test_pipeline_postagging_disabled_uses_null_tagger():
    pipe = TextPipeline(Settings(enable_postagging=False))
    assert isinstance(pipe.tagger, NullPosTagger)
    toks = pipe.process("Chest pain.")[0].tokens
    assert all(t.pos in ("", "WS") for t in toks)


def test_pipeline_builds_the_spacy_tagger_from_the_configured_model(monkeypatch):
    """``postag_model`` is the only way a properties file or env var can pick the tagger."""
    from mmlite.pipeline import spacy_nlp

    built: list[tuple[str, float]] = []

    class FakeSpacyPosTagger(NullPosTagger):
        def __init__(self, model="unset", verb_rescue=-1.0):
            built.append((model, verb_rescue))

    monkeypatch.setattr(spacy_nlp, "SpacyPosTagger", FakeSpacyPosTagger)
    TextPipeline(Settings(postag_model="en_core_sci_sm", postag_verb_rescue=0))
    TextPipeline(Settings())
    assert built == [
        ("en_core_sci_sm", 0),
        (spacy_nlp.DEFAULT_MODEL, spacy_nlp.VERB_RESCUE_THRESHOLD),
    ]


def test_pipeline_rejects_an_unknown_segmentation_method():
    # The CLI validates its own flag, but the method can also come from a properties file or
    # MMLITE_SEGMENTATION_METHOD - fail on construction, not on the first document.
    with pytest.raises(ValueError, match="unknown segmentation method"):
        TextPipeline(Settings(segmentation_method="PARAGRAPHS"), tagger=NullPosTagger())


def test_pipeline_segmentation_method_is_case_insensitive():
    pipe = TextPipeline(Settings(segmentation_method="lines"), tagger=NullPosTagger())
    assert [ts.sentence.text for ts in pipe.process("one\ntwo")] == ["one", "two"]


# --- spaCy-backed components (skipped when the model is not installed) ----------------------

spacy = pytest.importorskip("spacy")


@pytest.fixture(scope="module")
def spacy_available():
    from mmlite.pipeline.spacy_nlp import DEFAULT_MODEL

    if not spacy.util.is_package(DEFAULT_MODEL):
        pytest.skip(f"{DEFAULT_MODEL} not installed")


def test_spacy_tagger_tags_metamaplite_tokens(spacy_available):
    from mmlite.pipeline.spacy_nlp import SpacyPosTagger

    toks = tokenize("The patient has type 2 diabetes mellitus.")
    add_part_of_speech(toks, SpacyPosTagger())
    tagged = {t.text: t.pos for t in toks if not t.is_whitespace}
    assert tagged["The"] == "DT"
    assert tagged["patient"].startswith("NN")
    assert tagged["2"] == "CD"
    assert tagged["."] == "."
    assert [t.pos for t in toks if t.is_whitespace] == ["WS"] * 6


def test_spacy_sentence_detector(spacy_available):
    from mmlite.pipeline.spacy_nlp import SpacySentenceDetector

    text = "She denies chest pain. Dr. Smith prescribed metformin."
    spans = SpacySentenceDetector().spans(text)
    assert [text[a:b] for a, b in spans] == [
        "She denies chest pain.",
        "Dr. Smith prescribed metformin.",
    ]


def test_missing_postag_model_fails_at_construction_with_an_install_hint(spacy_available):
    # A typo'd or uninstalled model must fail when the pipeline is built, naming the model,
    # not on the first document.  Non-spaCy models (scispaCy's) are not on `spacy download`.
    with pytest.raises(OSError, match=r"'en_core_sci_nope'.*pip install"):
        TextPipeline(Settings(postag_model="en_core_sci_nope"))
    with pytest.raises(OSError, match=r"'en_core_web_nope'.*spacy download"):
        TextPipeline(Settings(postag_model="en_core_web_nope"))


def test_default_pipeline_uses_spacy(spacy_available):
    from mmlite.pipeline.spacy_nlp import DEFAULT_MODEL, SpacyPosTagger

    pipe = TextPipeline()
    assert isinstance(pipe.tagger, SpacyPosTagger)
    assert pipe.tagger.model == DEFAULT_MODEL
    out = pipe.process("Aspirin 81 mg daily. No chest pain.")
    assert len(out) == 2
    assert any(t.pos == "CD" for t in out[0].tokens)


def test_tag_batch_matches_tagging_one_sentence_at_a_time():
    """Batching is a throughput change only: every sentence is still tagged independently."""
    from mmlite.pipeline.postag import NullPosTagger, tag_batch

    class Reversing:
        """A tagger whose output depends only on its own input, like a real one."""

        def tag(self, words):
            return [w[::-1].upper() for w in words]

    sentences = [["heart", "attack"], [], ["aspirin"]]
    assert tag_batch(Reversing(), sentences) == [Reversing().tag(s) for s in sentences]
    assert tag_batch(NullPosTagger(), sentences) == [[""] * len(s) for s in sentences]


def test_add_part_of_speech_batch_assigns_the_same_tags_as_the_single_sentence_path():
    from mmlite.pipeline.postag import add_part_of_speech, add_part_of_speech_batch
    from mmlite.pipeline.tokenize import tokenize

    class Fixed:
        def tag(self, words):
            return ["NN"] * len(words)

    texts = ["heart attack today", "aspirin was given"]
    one = [add_part_of_speech(tokenize(t), Fixed()) for t in texts]
    many = add_part_of_speech_batch([tokenize(t) for t in texts], Fixed())
    assert [[tok.pos for tok in toks] for toks in many] == [
        [tok.pos for tok in toks] for toks in one
    ]
    assert any(tok.pos == "WS" for toks in many for tok in toks)


def test_spacy_tagger_rescues_drug_names_tagged_as_verbs(spacy_available):
    """en_core_web_sm reads rare clinical words as verbs, which stops a span dead at the lookup's
    POS gate; the rescue re-tags them when the noun reading is not ruled out (spacy_nlp)."""
    from mmlite.pipeline.spacy_nlp import SpacyPosTagger

    plain, rescued = SpacyPosTagger(verb_rescue=0), SpacyPosTagger()
    exercised = 0
    for words, word in (
        (["Discontinue", "PRN", "famotidine", "."], "famotidine"),
        (["Famotidine", "20", "mg", "PO", ";", "take", "1", "tablet"], "Famotidine"),
        (['"', "Quarterly", "diabetes", "follow", "-", "up"], "diabetes"),
    ):
        i = words.index(word)
        assert rescued.tag(words)[i].startswith("NN")
        # Which of these a model gets wrong varies by version: en_core_web_sm 3.8 mis-tags all
        # three, 3.7.1 (the one requirements-scispacy-py312.txt pairs with) tags "famotidine"
        # correctly on its own.  Count the cases that really needed the rescue.
        exercised += plain.tag(words)[i].startswith("VB")
    assert exercised >= 2, "the installed model no longer makes these mistakes; pick new cases"


def test_spacy_tagger_rescue_leaves_real_verbs_alone(spacy_available):
    """Only the uncertain cases move: a confident verb keeps its tag, and participles are never
    rescued at all (they are routinely adjectival, and Java's tagger keeps them verbs too)."""
    from mmlite.pipeline.spacy_nlp import SpacyPosTagger

    rescued = SpacyPosTagger()
    assert rescued.tag(["Take", "famotidine", "daily", "."])[0] == "VB"
    assert rescued.tag(["Patient", "denies", "chest", "pain", "."])[1] == "VBZ"
    assert rescued.tag(["Aspirin", "81", "mg", "was", "continued", "."])[4] == "VBN"


def test_spacy_tagger_rescue_disabled_matches_spacy_exactly(spacy_available):
    """verb_rescue=0 must be the plain tagger, so the setting can be turned off cleanly."""
    import spacy as _spacy

    from mmlite.pipeline.spacy_nlp import SpacyPosTagger

    words = ["Discontinue", "PRN", "famotidine", "and", "start", "omeprazole", "."]
    nlp = _spacy.load(SpacyPosTagger().model, exclude=["ner", "lemmatizer"])
    expected = [t.tag_ for t in nlp(_spacy.tokens.Doc(nlp.vocab, words=words))]
    assert SpacyPosTagger(verb_rescue=0, ptb_fixups=False).tag(words) == expected
