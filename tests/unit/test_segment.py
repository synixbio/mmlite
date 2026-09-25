import pytest

from mmlite.pipeline.segment import RegexSentenceDetector, Sentence, segment

TEXT = (
    "The patient was diagnosed with type 2 diabetes mellitus in 2019. She denies chest pain. "
    "Dr. Smith prescribed metformin 500 mg b.i.d.\nHbA1c was 7.5%. "
    "(See Fig. 2.) Follow-up in 3 mos."
)


def _check_offsets(text, sents):
    for s in sents:
        assert text[s.offset : s.end] == s.text


def test_sentences_default_detector():
    sents = segment(TEXT)
    _check_offsets(TEXT, sents)
    assert [s.text for s in sents] == [
        "The patient was diagnosed with type 2 diabetes mellitus in 2019.",
        "She denies chest pain.",
        "Dr. Smith prescribed metformin 500 mg b.i.d.",
        "HbA1c was 7.5%.",
        "(See Fig. 2.)",
        "Follow-up in 3 mos.",
    ]


def test_sentences_base_offset():
    sents = segment("One. Two.", base_offset=50)
    assert sents == [Sentence("One.", 50), Sentence("Two.", 55)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("A. B. Smith went home. He slept.", ["A. B. Smith went home.", "He slept."]),  # initials
        (
            "Take 5 mg. Then rest.",
            ["Take 5 mg.", "Then rest."],
        ),  # units are only guarded when abbreviated
        ("Is it? Yes! Fine.", ["Is it?", "Yes!", "Fine."]),
        ('He said "Stop." Then left.', ['He said "Stop."', "Then left."]),
        ("no boundary here. lowercase follows", ["no boundary here. lowercase follows"]),
        ("", []),
        ("   ", []),
    ],
)
def test_regex_detector_cases(text, expected):
    sents = segment(text)
    _check_offsets(text, sents)
    assert [s.text for s in sents] == expected


def test_blanklines():
    text = "para one\nstill one\n\npara two\n\n\n\npara three\n\n"
    sents = segment(text, "BLANKLINES", base_offset=10)
    _check_offsets(text, [Sentence(s.text, s.offset - 10) for s in sents])
    assert [s.text for s in sents] == ["para one\nstill one", "para two", "para three"]
    assert sents[1].offset == 10 + text.index("para two")


def test_lines_skips_blank_lines_but_keeps_offsets():
    text = "line one\n\n   \nline four"
    sents = segment(text, "LINES")
    _check_offsets(text, sents)
    assert [s.text for s in sents] == ["line one", "line four"]
    assert sents[1].offset == text.index("line four")


@pytest.mark.parametrize("nl", ["\n", "\r\n", "\r"])
def test_lines_handles_every_newline_convention(nl):
    text = f"line one{nl}{nl}   {nl}line four"
    sents = segment(text, "LINES")
    _check_offsets(text, sents)
    assert [s.text for s in sents] == ["line one", "line four"]
    assert sents[1].offset == text.index("line four")


@pytest.mark.parametrize("nl", ["\n", "\r\n", "\r"])
def test_blanklines_handles_every_newline_convention(nl):
    text = f"para one{nl}still one{nl}{nl}para two{nl}{nl}{nl}para three{nl}{nl}"
    sents = segment(text, "BLANKLINES")
    _check_offsets(text, sents)
    assert [s.text for s in sents] == [f"para one{nl}still one", "para two", "para three"]


def test_unknown_method():
    with pytest.raises(ValueError):
        segment("x", "WORDS")


def test_custom_detector_is_used():
    class Halves:
        def spans(self, text):
            h = len(text) // 2
            return [(0, h), (h, len(text))]

    sents = segment("abcdef", detector=Halves())
    assert [s.text for s in sents] == ["abc", "def"]


def test_detector_abbreviation_list_is_configurable():
    det = RegexSentenceDetector(abbreviations=frozenset())
    assert [s.text for s in segment("Dr. Smith came.", detector=det)] == ["Dr.", "Smith came."]
