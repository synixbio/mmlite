"""Text pipeline: segmentation -> MetaMapLite tokenization -> POS tagging.

Mirrors the front half of ``MetaMapLite.processPassage``: the output is a list of
:class:`TokenizedSentence`, each with absolute offsets, ready for
:mod:`mmlite.pipeline.entity_lookup`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..config import Settings
from ..types import Entity
from .assertion import Assertion, AssertionClassifier
from .postag import (
    ALLOWED_PART_OF_SPEECH,
    NullPosTagger,
    PosTagger,
    add_part_of_speech,
    add_part_of_speech_batch,
)
from .segment import (
    SEGMENTATION_METHODS,
    RegexSentenceDetector,
    Sentence,
    SentenceDetector,
    segment,
)
from .tokenize import Token, tokenize, tokens_text

__all__ = [
    "ALLOWED_PART_OF_SPEECH",
    "ENTRY_POINT_GROUP",
    "SEGMENTATION_METHODS",
    "Assertion",
    "AssertionClassifier",
    "NegationDetector",
    "PosTagger",
    "Sentence",
    "SentenceDetector",
    "TextPipeline",
    "Token",
    "TokenizedSentence",
    "add_part_of_speech",
    "add_part_of_speech_batch",
    "get_negation_detector",
    "negation_detector_names",
    "register_negation_detector",
    "segment",
    "tokenize",
    "tokens_text",
]


@runtime_checkable
class NegationDetector(Protocol):
    """``metamap.lite.NegationDetector``: NegEx (the default) or ConText.

    Java's interface is ``detectNegations(entitySet, sentence, tokenList)`` plus an
    ``initProperties``; configuration happens in ``__init__`` here, so one method is enough.
    Implementations mutate the entities in place.

    A detector may set ``sentence_local = True`` to promise it ignores entities outside the
    sentence it is given; :class:`~.entity_lookup.EntityLookup` then passes it only that
    sentence's entities.  Without it, every sentence gets the whole passage's list, as in Java.
    """

    def detect(
        self, tokens: list[Token], entities: Iterable[Entity], sentence: str = ""
    ) -> None: ...


@dataclass
class TokenizedSentence:
    sentence: Sentence
    tokens: list[Token]
    index: int  # sentence number within the document


class TextPipeline:
    """Configured segmenter + tokenizer + tagger.

    ``tagger=None`` with ``settings.enable_postagging`` selects the spaCy tagger lazily;
    pass :class:`~mmlite.pipeline.postag.NullPosTagger` to skip tagging explicitly.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        tagger: PosTagger | None = None,
        sentence_detector: SentenceDetector | None = None,
    ):
        self.settings = settings or Settings()
        self.method = self.settings.segmentation_method.upper()
        # Fail here rather than on the first document: the method can also arrive from a
        # properties file or MMLITE_SEGMENTATION_METHOD, which nothing else validates.
        if self.method not in SEGMENTATION_METHODS:
            raise ValueError(
                f"unknown segmentation method {self.settings.segmentation_method!r}; "
                f"expected one of {SEGMENTATION_METHODS}"
            )
        self.sentence_detector = sentence_detector or RegexSentenceDetector()
        if tagger is not None:
            self.tagger = tagger
        elif self.settings.enable_postagging:
            from .spacy_nlp import SpacyPosTagger

            self.tagger = SpacyPosTagger(
                self.settings.postag_model, verb_rescue=self.settings.postag_verb_rescue
            )
        else:
            self.tagger = NullPosTagger()

    def process(self, text: str, base_offset: int = 0) -> list[TokenizedSentence]:
        """Segment, tokenize, then tag **every sentence in one call** into the tagger.

        Tagging is per-sentence either way, so the tags are unchanged; batching just avoids
        paying the tagger's fixed per-call cost once per sentence.
        """
        sentences = segment(text, self.method, base_offset, self.sentence_detector)
        token_lists = [tokenize(sent.text, sent.offset) for sent in sentences]
        add_part_of_speech_batch(token_lists, self.tagger)
        return [
            TokenizedSentence(sent, tokens, i)
            for i, (sent, tokens) in enumerate(zip(sentences, token_lists, strict=True))
        ]


# MetaMapLite's own values for metamaplite.negation.detector, plus short aliases.  Detectors
# added with register_negation_detector() (or found through ENTRY_POINT_GROUP) join this map.
NEGATION_DETECTORS = {
    "negex": "negex",
    "gov.nih.nlm.nls.metamap.lite.negex": "negex",
    "context": "context",
    "gov.nih.nlm.nls.metamap.lite.context.contextwrapper": "context",
}

# A detector factory is called as ``factory(strict_parity=...)``; a detector with no Java
# original to be faithful to may ignore the flag.
DetectorFactory = Callable[..., NegationDetector]

# Installed packages can contribute detectors without being imported by hand: an entry point
# named e.g. ``medspacy`` in this group, pointing at a factory, makes ``--negation-detector``,
# ``MMLITE_NEGATION_DETECTOR=medspacy`` and the REST ``detector`` field accept it.
ENTRY_POINT_GROUP = "mmlite.negation_detectors"


def _negex(strict_parity: bool = True) -> NegationDetector:
    from .negation import NegEx

    return NegEx()


def _context(strict_parity: bool = True) -> NegationDetector:
    from .context import ConText

    return ConText(strict_parity=strict_parity)


_FACTORIES: dict[str, DetectorFactory] = {"negex": _negex, "context": _context}
_BUILT_IN = frozenset(_FACTORIES)
_entry_points_loaded = False


def register_negation_detector(
    name: str, factory: DetectorFactory, aliases: Iterable[str] = ()
) -> None:
    """Make ``factory`` selectable by ``name`` (and ``aliases``) wherever a detector is named.

    Registering a name again replaces its factory, but the built-in ``negex`` and ``context``,
    and a name already used as another detector's alias, cannot be taken over.
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("a negation detector needs a name")
    if key in _BUILT_IN:
        raise ValueError(f"{key!r} is a built-in negation detector and cannot be replaced")
    names = [key, *(alias.strip().lower() for alias in aliases)]
    for alias in names:
        taken = NEGATION_DETECTORS.get(alias)
        if taken is not None and taken != key:
            raise ValueError(f"{alias!r} already names the {taken!r} negation detector")
    _FACTORIES[key] = factory
    for alias in names:
        NEGATION_DETECTORS[alias] = key


def _load_entry_points() -> None:
    """Register the detectors installed packages declare, once per process."""
    global _entry_points_loaded
    if _entry_points_loaded:
        return
    _entry_points_loaded = True
    from importlib.metadata import entry_points

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name.strip().lower() in NEGATION_DETECTORS:
            continue  # registered in code already, or clashes with a built-in
        register_negation_detector(ep.name, ep.load())


def negation_detector_names() -> list[str]:
    """The canonical name of every available detector, plugins included."""
    _load_entry_points()
    return sorted(_FACTORIES)


def get_negation_detector(name: str = "negex", strict_parity: bool = True) -> NegationDetector:
    """A detector by ``metamaplite.negation.detector`` value (``negex``, ``context``, a Java
    class name, or any registered detector).

    ``strict_parity=False`` only changes ConText among the built-ins, whose Java original has
    bugs worth fixing (see :class:`~.context.ConText`); NegEx has none that this flag addresses.
    """
    key = NEGATION_DETECTORS.get(name.strip().lower())
    if key is None:
        _load_entry_points()
        key = NEGATION_DETECTORS.get(name.strip().lower())
    if key is None:
        raise ValueError(
            f"unknown negation detector {name!r}; expected one of {sorted(set(NEGATION_DETECTORS))}"
        )
    return _FACTORIES[key](strict_parity=strict_parity)
