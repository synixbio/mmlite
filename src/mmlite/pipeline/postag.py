"""Part-of-speech tagging — port of ``OpenNLPPoSTagger.addPartOfSpeech(List<ERToken>)``.

Java tags the non-whitespace tokens with the OpenNLP Penn Treebank tagger and marks whitespace
tokens ``"WS"``.  When tagging is disabled every token keeps ``""``, which ``EntityLookup4``
treats as "accept everything".  The tagger itself is pluggable; see
:class:`mmlite.pipeline.spacy_nlp.SpacyPosTagger` for the default implementation.
"""

from __future__ import annotations

from typing import Protocol

from .tokenize import Token

# EntityLookup4.allowedPartOfSpeechSet: a candidate span may only *start* with one of these.
ALLOWED_PART_OF_SPEECH = frozenset(
    {
        "ADJ",
        "CD",  # cardinal number (chemicals)
        "FW",  # foreign word
        "RB",
        "IN",  # preposition / subordinating conjunction
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "NOUN",
        "JJ",
        "JJR",
        "JJS",
        "LS",  # list item marker (chemicals)
        "PROPN",
        "",  # not tagged: accept everything
    }
)


class PosTagger(Protocol):
    def tag(self, words: list[str]) -> list[str]:
        """Return one Penn Treebank tag per word."""
        ...


class NullPosTagger:
    """Tagging disabled (``metamaplite.enable.postagging=false``): every tag is ``""``."""

    def tag(self, words: list[str]) -> list[str]:
        return [""] * len(words)

    def tag_batch(self, sentences: list[list[str]]) -> list[list[str]]:
        return [[""] * len(words) for words in sentences]


def tag_batch(tagger: PosTagger, sentences: list[list[str]]) -> list[list[str]]:
    """Tag several sentences at once, using ``tagger.tag_batch`` when it has one.

    Sentences are tagged independently either way — the tagger never sees across a boundary — so
    batching changes throughput, not tags.  A tagger that only implements ``tag`` still works.
    """
    batch = getattr(tagger, "tag_batch", None)
    if batch is not None:
        tags: list[list[str]] = batch(sentences)
        return tags
    return [tagger.tag(words) for words in sentences]


def _assign(tokens: list[Token], tags: list[str]) -> list[Token]:
    words = [t for t in tokens if not t.is_whitespace]
    if len(tags) != len(words):
        raise ValueError(f"tagger returned {len(tags)} tags for {len(words)} words")
    it = iter(tags)
    for t in tokens:
        t.pos = "WS" if t.is_whitespace else next(it)
    return tokens


def add_part_of_speech(tokens: list[Token], tagger: PosTagger) -> list[Token]:
    """Tag non-whitespace tokens in place; whitespace tokens get ``"WS"``. Returns ``tokens``."""
    words = [t.text for t in tokens if not t.is_whitespace]
    return _assign(tokens, tagger.tag(words) if words else [])


def add_part_of_speech_batch(
    token_lists: list[list[Token]], tagger: PosTagger
) -> list[list[Token]]:
    """:func:`add_part_of_speech` over several sentences, with one call into the tagger."""
    words = [[t.text for t in tokens if not t.is_whitespace] for tokens in token_lists]
    tags = tag_batch(tagger, words) if any(words) else [[] for _ in token_lists]
    for tokens, sentence_tags in zip(token_lists, tags, strict=True):
        _assign(tokens, sentence_tags)
    return token_lists
