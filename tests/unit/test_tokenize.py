"""Tests for the port of Tokenize.mmPosTokenize / Scanner (MetaMapLite tokenization)."""

import pytest

from mmlite.pipeline.tokenize import (
    JAVA_WHITESPACE,
    PUNCT,
    classify,
    remove_whitespace_tokens,
    tokenize,
    tokens_text,
)


def test_words_and_single_char_separators():
    toks = tokenize("Type 2 diabetes (T2DM), pt.'s")
    assert [t.text for t in toks] == [
        "Type",
        " ",
        "2",
        " ",
        "diabetes",
        " ",
        "(",
        "T2DM",
        ")",
        ",",
        " ",
        "pt",
        ".",
        "'",
        "s",
    ]


def test_every_separator_is_its_own_token():
    toks = tokenize("a  b--c")
    assert [t.text for t in toks] == ["a", " ", " ", "b", "-", "-", "c"]


def test_offsets_are_cumulative_and_reproduce_text():
    text = "Heart  attack, 2019.\tOk"
    toks = tokenize(text, base_offset=100)
    assert tokens_text(toks) == text
    for t in toks:
        assert text[t.start - 100 : t.end - 100] == t.text
    assert toks[0].start == 100 and toks[-1].start == 100 + len(text) - 2


@pytest.mark.parametrize(
    "text, expected",
    [
        (" ", "ws"),
        ("\t", "ws"),
        (chr(0xA0), "unknown"),  # Java \s is ASCII-only: NBSP is not "ws"
        ("HbA1c", "unknown"),  # letters+digits+letters matches no pattern
        ("T2DM", "uc"),
        ("IL6", "an"),
        ("il6", "an"),
        ("ABC", "uc"),
        ("A", "ic"),  # ucpattern needs 2+ chars; falls through to icpattern
        ("abc", "lc"),
        ("Abc", "ic"),
        ("123", "nu"),
        ("α", "gr"),
        ("αβ", "gr"),
        ("(", "op"),
        (")", "cp"),
        ("[", "ob"),
        ("]", "cb"),
        (",", "cm"),
        (".", "pd"),
        ("-", "pn"),
        ("%", "pn"),
        ("'", "pn"),
        ("&", "pn"),
        ("~", "unknown"),  # punct per CharUtils but not in pnpattern
        ("", "unknown"),
    ],
)
def test_classify(text, expected):
    assert classify(text) == expected


def test_java_whitespace_set():
    assert (
        " " in JAVA_WHITESPACE and chr(0xA0) in JAVA_WHITESPACE and chr(0x2003) in JAVA_WHITESPACE
    )
    assert "\x85" not in JAVA_WHITESPACE  # NEL is a control char in Java


def test_nel_is_a_word_character_like_java():
    assert [t.text for t in tokenize("a\x85b c")] == ["a\x85b", " ", "c"]


def test_punct_set_matches_charutils():
    assert set("~!@#$%^&*()_+-=|\\<>?/,.`';:[]{}\"") == PUNCT
    assert "€" not in PUNCT  # non-ASCII symbols are word characters


def test_empty_text():
    assert tokenize("") == []


def test_remove_whitespace_tokens():
    toks = tokenize("a b")
    assert [t.text for t in remove_whitespace_tokens(toks)] == ["a", "b"]
