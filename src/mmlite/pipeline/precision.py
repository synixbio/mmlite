"""An optional precision filter for acronym collisions and function words.

Enabled by ``Settings.precision_filter``.

Dictionary lookup matches everyday words against unrelated concepts that share their spelling:
"Plan" against OMIM's ``PLAN`` (Infantile Neuroaxonal Dystrophy), "eGFR" against the gene
``EGFR``, "for" against the gene symbol ``FOR``, "with" against a qualifier.  Restricting semantic
types does not catch these when the wrong concept has a clinical type (``PLAN`` is ``dsyn``).
Java MetaMapLite keeps them all; this filter is the port's own and is off by default, so parity
is unaffected.

Two rules, each aimed at one kind of collision:

1. **Case-mismatched acronyms** (per concept): a concept whose only dictionary strings for the
   matched key are written as acronyms (all capitals, at least two letters, one word) does not
   match text that is not itself in capitals.  "Plan" loses ``PLAN`` but keeps the concepts named
   "Plan"; "CHF" keeps ``CHF``.  A concept that also has a lower-case string for the key keeps
   its match: some sources store whole words in capitals (``ARTHRALGIAS``, ``INSULIN``), and
   those are ordinary words, not acronyms.
2. **Function words** (per mention): a one-word mention that is a preposition, conjunction,
   determiner, pronoun or auxiliary verb (:data:`FUNCTION_WORDS`) is dropped, unless it is
   written in capitals: "us" goes, "US" (ultrasound) stays.

Words of one or two characters need no rule: the lookup never tries a span that short.

The filter runs after subsumption removal, so dropping a mention never exposes a shorter match
inside it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence

from ..lru import LRUCache
from ..normalize import lookup_keys
from ..types import Entity, Ev

FUNCTION_WORDS = frozenset(
    """
    a about above across after against along among an and any are as at be been before being
    below beneath beside between beyond both but by can could did do does down during each
    either for from had has have he her hers him his how i if in into is it its may me might
    must my neither nor not of off on onto or our out over per shall she should since so some
    than that the their them then there these they this those though through throughout to
    toward towards under unless until up upon us via was we were what when where whether which
    while who whom whose why will with within without would yet you your
    """.split()  # noqa: SIM905 - a word list reads better as prose
)

_ONE_WORD = re.compile(r"[^\W_]+")

# (index key, CUI) -> every dictionary string the index holds for that CUI under that key
StringsFor = Callable[[str, str], Sequence[str]]


def _is_acronym(term: str) -> bool:
    return term.isupper() and " " not in term and sum(c.isalpha() for c in term) >= 2


def keep_mention(entity: Entity) -> bool:
    """Rule 2: whether a mention survives on its text alone.

    Capitals exempt a word, because clinical abbreviations collide with function words: "US"
    (ultrasound), "AS" (aortic stenosis), "OR".
    """
    text = entity.text
    if not _ONE_WORD.fullmatch(text) or text.isupper():
        return True
    return text.lower() not in FUNCTION_WORDS


class PrecisionFilter:
    """Both rules.  ``strings`` looks up a concept's other dictionary strings for a key; without
    it, rule 1 judges a concept by the one string recorded on it."""

    def __init__(self, strings: StringsFor | None = None):
        self._strings = strings
        self._acronym_only: LRUCache[tuple[str, str], bool] = LRUCache()

    def acronym_only(self, ev: Ev) -> bool:
        """Is every dictionary string that could have produced this match an acronym?"""
        if not _is_acronym(ev.concept.concept_string):
            return False
        if self._strings is None:
            return True
        ck = (ev.matched_text, ev.cui)
        verdict = self._acronym_only.get(ck)
        if verdict is None:
            strings = [
                s for key in lookup_keys(ev.matched_text) for s in self._strings(key, ev.cui)
            ]
            verdict = all(_is_acronym(s) for s in strings)
            self._acronym_only[ck] = verdict
        return verdict

    def __call__(self, entities: Iterable[Entity]) -> list[Entity]:
        """Apply both rules; a mention left with no concepts is dropped."""
        out = []
        for entity in entities:
            if not keep_mention(entity):
                continue
            if not entity.text.isupper():
                entity.evs = [ev for ev in entity.evs if not self.acronym_only(ev)]
                if not entity.evs:
                    continue
            out.append(entity)
        return out


__all__ = ["FUNCTION_WORDS", "PrecisionFilter", "keep_mention"]
