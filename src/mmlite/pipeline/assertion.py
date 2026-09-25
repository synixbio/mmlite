"""Adapter for plugging another assertion model in beside NegEx and ConText.

MetaMapLite's two detectors are fixed rule sets.  medspaCy's ConText, a transformer assertion
classifier or a site's own rules can take their place without touching the pipeline:

1. Subclass :class:`AssertionClassifier` and implement :meth:`~AssertionClassifier.classify`,
   which sees one sentence and the character spans of the mentions found in it -- no tokens,
   no :class:`~mmlite.types.Entity` objects.
2. Register it, in code with :func:`~mmlite.pipeline.register_negation_detector` or
   from an installed package with an entry point in the
   :data:`~mmlite.pipeline.ENTRY_POINT_GROUP` group.  Its name then works anywhere a
   detector is named: ``Settings.negation_detector``, ``MMLITE_NEGATION_DETECTOR``,
   ``metamaplite.negation.detector`` and the REST ``detector`` field.

A plugged-in detector is outside the parity path: nothing here is compared with Java.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..types import Entity
from .tokenize import Token


@dataclass(frozen=True)
class Assertion:
    """What a classifier says about one mention.

    ``temporality`` and ``experiencer`` use ConText's values where they apply (``Recent`` /
    ``Historical`` / ``Hypothetical``, ``Patient`` / ``Other``); ``None`` leaves the entity's
    field unset, as NegEx does.
    """

    negated: bool = False
    temporality: str | None = None
    experiencer: str | None = None


class AssertionClassifier(ABC):
    """A :class:`~mmlite.pipeline.NegationDetector` built from a per-sentence classifier.

    The pipeline hands it one sentence at a time, with only the entities inside that sentence
    (``sentence_local``), and it writes each :class:`Assertion` back onto its entity.
    """

    sentence_local = True

    @abstractmethod
    def classify(
        self, sentence: str, spans: Sequence[tuple[int, int]]
    ) -> Sequence[Assertion | None]:
        """One result per span, in order.

        ``spans`` are ``(start, end)`` character offsets into ``sentence``, so
        ``sentence[start:end]`` is the mention's text.  ``None`` leaves that entity untouched.
        """

    def detect(self, tokens: list[Token], entities: Iterable[Entity], sentence: str = "") -> None:
        if not sentence or not tokens:
            return
        base, end = tokens[0].start, tokens[-1].end  # tokens are cut from the sentence at base
        mentions = [e for e in entities if base <= e.start and e.end <= end]
        if not mentions:
            return
        results = self.classify(sentence, [(e.start - base, e.end - base) for e in mentions])
        if len(results) != len(mentions):
            raise ValueError(
                f"{type(self).__name__}.classify returned {len(results)} results "
                f"for {len(mentions)} spans"
            )
        for entity, result in zip(mentions, results, strict=True):
            if result is None:
                continue
            entity.negated = result.negated
            entity.temporality = result.temporality
            entity.experiencer = result.experiencer


__all__ = ["Assertion", "AssertionClassifier"]
