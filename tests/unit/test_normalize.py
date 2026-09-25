"""Tests for the port of MetaMapLite's Normalization.normalizeUtf8AsciiString.

Expected values follow the Java code paths in NLSStrings / MetamapTokenization; entries marked
``# java`` were confirmed against the Java implementation's logic line by line.
"""

import pytest

from mmlite.greek import greek_to_ascii
from mmlite.normalize import (
    index_key,
    lookup_keys,
    normalize,
    remove_left_parentheticals,
    remove_possessives,
    strip_possessives,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Diabetes Mellitus", "diabetes mellitus"),
        ("Diabetes Mellitus, Type 2", "diabetes mellitus, type 2"),  # punctuation kept, order kept
        ("Type 2 Diabetes", "type 2 diabetes"),
        ("Alzheimer's disease", "alzheimer disease"),  # java: possessive stripped
        ("Crohns' disease", "crohns disease"),  # java: trailing ' after s stripped
        (
            "non-insulin dependent",
            "non-insulin dependent",
        ),  # hyphen kept (not normalizeLiteMetaString)
        ("  extra   spaces  ", "extra spaces"),
        ("Aspirin (product)", "aspirin (product)"),
        ("[X]Other disorders", "other disorders"),  # java: left parenthetical stripped
        ("[V]Vaccination", "vaccination"),
        ("TNF-α", "tnf-alpha"),  # java: greek expanded before lowercasing
        ("IL-1β levels", "il-1beta levels"),
        ("cost $5", "cost 5"),  # $ is a TOKEN_DELIMITER
        ("a|b~c", "a b c"),
        ("", ""),
    ],
)
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_normalize_is_order_sensitive():
    # MetaMapLite does not sort words; both forms must be present in MRCONSO to match.
    assert normalize("Type 2 Diabetes Mellitus") != normalize("Diabetes Mellitus, Type 2")


@pytest.mark.parametrize(
    "token, expected",
    [
        ("alzheimer's", "alzheimer"),
        ("crohns'", "crohns"),
        ("'s", "'s"),  # nothing alphanumeric before 's -> unchanged
        ("it's", "it"),
        ("'", "'"),
        ("s'", "s"),  # pos != 0 guard only protects a lone apostrophe
        ("boys'", "boys"),
        ("plain", "plain"),
        ("5's", "5"),
    ],
)
def test_remove_possessives(token, expected):
    assert remove_possessives(token) == expected


def test_strip_possessives_tokenizes_on_java_delimiters():
    assert strip_possessives("alzheimer's\tdisease$x|y~z") == "alzheimer disease x y z"


def test_remove_left_parentheticals_only_at_start():
    assert remove_left_parentheticals("[X]Foo") == "Foo"
    assert remove_left_parentheticals("Foo [X]") == "Foo [X]"
    assert remove_left_parentheticals("a  b") == "a b"  # else-branch collapses blanks
    assert remove_left_parentheticals("[X]a  b") == "a  b"  # tag branch does not


def test_greek_to_ascii():
    assert greek_to_ascii("Ω α β") == "Omega alpha beta"
    assert greek_to_ascii("plain") == "plain"


def test_index_key_is_plain_lowercase():
    assert index_key("Alzheimer's Disease") == "alzheimer's disease"


def test_lookup_keys():
    assert lookup_keys("heart attack") == ("heart attack",)
    assert lookup_keys("Alzheimer's Disease") == ("alzheimer's disease", "alzheimer disease")
    assert lookup_keys("Heart Attack") == ("heart attack",)  # lower-cased by the IVF lookup
