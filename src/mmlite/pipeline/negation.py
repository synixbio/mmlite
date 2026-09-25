"""Negation detection — port of ``metamap.lite.NegEx`` (MetaMapLite's default detector).

Per sentence (``tokenNegex``):
1. ``addMetaTokens``: runs of 3+ consecutive non-whitespace tokens are merged into one token
   (stopping at a period or a single-word trigger), so ``b.i.d`` or ``non-smoker`` count as one
   token for distance purposes.
2. ``filterTokenList``: drop whitespace and period tokens (unless the period is itself a trigger).
3. Find every trigger phrase (case-insensitive, token-exact) in the remaining token strings;
   at each position keep only the longest phrase, so a pseudo-negation such as ``no increase``
   beats ``no``.  ``conj`` phrases terminate scope.
4. An entity with at least one concept in :data:`NEGATION_SEMANTIC_TYPES` is negated when a
   ``nega`` trigger precedes it (or a ``negb`` trigger follows it) within ``token_window``
   filtered tokens and no ``conj`` phrase lies between them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..types import Entity
from .negex_triggers import NEGATION_PHRASE_TYPES
from .tokenize import Token

DEFAULT_TOKEN_WINDOW = 6  # metamaplite.negex.tokenwindowsize
NEGATION_SEMANTIC_TYPES = frozenset(
    [
        "acab",
        "anab",
        "biof",
        "cgab",
        "comd",
        "dsyn",
        "emod",
        "fndg",
        "inpo",
        "lbtr",
        "menp",
        "mobd",
        "neop",
        "patf",
        "phsf",
        "sosy",
    ]
)  # metamaplite.negex.semantic.type.set


@dataclass
class NegPhraseInfo:
    phrase: tuple[str, ...]
    kind: str
    positions: list[int]


class NegEx:
    # entity_token_position() returns -1 for any entity outside the sentence's tokens, so
    # EntityLookup may pass only that sentence's entities: same result, far less work.
    sentence_local = True

    def __init__(
        self,
        token_window: int = DEFAULT_TOKEN_WINDOW,
        semantic_types: frozenset[str] = NEGATION_SEMANTIC_TYPES,
        phrase_types: dict[tuple[str, ...], str] = NEGATION_PHRASE_TYPES,
    ):
        self.token_window = token_window
        self.semantic_types = semantic_types
        self.phrase_types = phrase_types
        self._single = {p[0] for p in phrase_types if len(p) == 1}
        # Trigger phrases bucketed by their first word, so a sentence is scanned once and each
        # position only tests the handful of phrases that could start there (262 phrases share
        # 74 first words).  ``_order`` restores the phrase_types iteration order afterwards.
        by_first: dict[str, list[tuple[str, ...]]] = {}
        for phrase in phrase_types:
            by_first.setdefault(phrase[0], []).append(phrase)
        self._by_first_word = {word: tuple(ps) for word, ps in by_first.items()}
        self._order = {phrase: i for i, phrase in enumerate(phrase_types)}

    # -- token preparation -----------------------------------------------------------------------

    def _range_of_meta_token(self, sub: list[Token]) -> int:
        r = 0
        while r < len(sub) - 1 and sub[r + 1].token_class not in ("ws", "pd"):
            r += 1
        return r

    def add_meta_tokens(self, tokens: list[Token]) -> list[Token]:
        out: list[Token] = []
        i, n = 0, len(tokens)
        while i + 3 < n:
            t0, t1, t2 = tokens[i], tokens[i + 1], tokens[i + 2]
            if (
                t0.token_class != "ws"
                and t1.token_class != "ws"
                and t2.token_class != "ws"
                and t2.token_class != "pd"
                and t2.text not in self._single
            ):
                rng = self._range_of_meta_token(tokens[i + 2 :]) + 3
                rend = i + rng
                text = "".join(t.text for t in tokens[i : min(rend, n)])
                out.append(Token(text, t0.start, t0.token_class, t0.pos))
                i = min(rend, n)
            else:
                out.append(t0)
                i += 1
        out.extend(tokens[i : i + 3])
        return out

    def filter_tokens(self, tokens: list[Token]) -> list[Token]:
        return [
            t
            for t in tokens
            if (t.token_class != "ws" and t.token_class != "pd") or t.text in self._single
        ]

    # -- phrase detection -----------------------------------------------------------------------

    def find_phrase(self, words: list[str], phrase: tuple[str, ...]) -> list[int]:
        """Token positions where ``phrase`` occurs in ``words`` (case-insensitive, token-exact)."""
        plen = len(phrase)
        lowered = [w.lower() for w in words]
        return [n for n in range(len(words) - plen + 1) if tuple(lowered[n : n + plen]) == phrase]

    def phrase_list(self, words: list[str]) -> list[NegPhraseInfo]:
        """Every trigger phrase present in ``words``, in ``phrase_types`` order.

        Equivalent to calling :meth:`find_phrase` for each of the 262 trigger phrases, but scans
        the sentence once instead of once per phrase (and lower-cases it once, not 262 times) —
        this is the pipeline's hottest loop.
        """
        lowered = [w.lower() for w in words]
        n = len(lowered)
        found: dict[tuple[str, ...], list[int]] = {}
        for i, word in enumerate(lowered):
            for phrase in self._by_first_word.get(word, ()):
                end = i + len(phrase)
                if end <= n and tuple(lowered[i:end]) == phrase:
                    found.setdefault(phrase, []).append(i)  # positions stay ascending
        return [
            NegPhraseInfo(phrase, self.phrase_types[phrase], positions)
            for phrase, positions in sorted(found.items(), key=lambda kv: self._order[kv[0]])
        ]

    @staticmethod
    def keep_longest(phrases: list[NegPhraseInfo]) -> list[NegPhraseInfo]:
        by_pos: dict[int, NegPhraseInfo] = {}
        for info in phrases:
            for p in info.positions:
                if p not in by_pos or len(info.phrase) > len(by_pos[p].phrase):
                    by_pos[p] = info
        # dedupe by identity while preserving first-seen order
        seen: set[int] = set()
        out = []
        for info in by_pos.values():
            if id(info) not in seen:
                seen.add(id(info))
                out.append(info)
        return out

    # -- entity marking -------------------------------------------------------------------------

    @staticmethod
    def entity_token_position(entity: Entity, tokens: list[Token]) -> int:
        if not tokens:
            return -1
        if entity.start < tokens[0].start or entity.start > tokens[-1].end:
            return -1
        pos = -1
        for i, t in enumerate(tokens):
            if entity.start == t.start or (entity.start > t.start and entity.end <= t.end):
                pos = i
        return pos

    @staticmethod
    def _no_conj_between(a: int, b: int, conj: list[NegPhraseInfo]) -> bool:
        lo, hi = min(a, b), max(a, b)
        return not any(lo < p < hi for info in conj for p in info.positions)

    def mark_negated(
        self,
        tokens: list[Token],
        phrases: list[NegPhraseInfo],
        conj: list[NegPhraseInfo],
        entities: Iterable[Entity],
    ) -> None:
        for entity in entities:
            for info in phrases:
                if info.kind not in ("nega", "negb"):
                    continue
                for tpos in info.positions:
                    trigger_offset = tokens[tpos].start
                    if info.kind == "nega" and entity.start < trigger_offset:
                        continue
                    if info.kind == "negb" and entity.start >= trigger_offset:
                        continue
                    epos = self.entity_token_position(entity, tokens)
                    if epos < 0 or abs(epos - tpos) > self.token_window:
                        continue
                    if self._no_conj_between(epos, tpos, conj):
                        entity.negated = True

    def in_scope(self, entity: Entity) -> bool:
        return any(ev.concept.semantic_types & self.semantic_types for ev in entity.evs)

    def detect(self, tokens: list[Token], entities: Iterable[Entity], sentence: str = "") -> None:
        """NegEx.tokenNegex: set ``entity.negated`` in place for entities of this sentence.

        ``sentence`` is part of the ``NegationDetector`` contract (ConText needs the raw
        text) and is unused here, exactly as Java's NegEx ignores its own sentence argument.
        """
        filtered = self.filter_tokens(self.add_meta_tokens(tokens))
        candidates = [e for e in entities if self.in_scope(e)]
        if not candidates:
            return
        words = [t.text for t in filtered]
        phrases0 = self.phrase_list(words)
        if not phrases0:
            return
        conj = [p for p in phrases0 if p.kind == "conj"]
        self.mark_negated(filtered, self.keep_longest(phrases0), conj, candidates)
