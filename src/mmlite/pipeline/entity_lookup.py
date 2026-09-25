"""Entity lookup — port of ``metamap.lite.EntityLookup4`` (plus ``SpecialTerms`` and
``dictionary.AugmentedDictionary``).

Algorithm (Java ``processSentenceTokenList`` + ``findLongestMatch``), for one sentence:

1. Candidate spans are every contiguous run of 1..``max_token_size`` tokens — whitespace
   tokens included.  (Java builds them as all sub-lists of each 15-token window starting at
   each index, which is the same set with duplicates; ``Entity`` identity is the span, so the
   duplicates collapse.)
2. A candidate is skipped when its first token is ``other`` (any case) or its first token's
   POS is not in ``allowed_pos``; when the joined text has length <= 2; or when the joined
   text does not start *and* end with a ``CharUtils.isAlphaNumeric`` character — an ASCII
   letter or digit, or a Greek letter (see :func:`_is_alnum`).
3. The dictionary is queried with the joined text verbatim and with ``normalize(text)``
   (see :mod:`mmlite.normalize`); concepts on the excluded-terms list are dropped.
4. Matches are grouped by span into :class:`~mmlite.types.Entity` objects, one
   :class:`~mmlite.types.Ev` per CUI.

Then, over the whole document (``processText`` / ``processPassage``): Evs are filtered by
semantic-type and source restriction (source = every SAB of the CUI), entities left with no
Evs are dropped, entities whose span lies inside another entity's span are removed
(``removeSubsumedEntities``), and the result is sorted by start offset.

Document level also applies user-defined acronyms, abbreviation propagation and negation
(``processPassage`` order).  The negation detector is pluggable
(:class:`~mmlite.pipeline.NegationDetector`): NegEx by default, ConText with
``metamaplite.negation.detector``.  MetaMap-style scoring (``EntityLookup5``) is not ported.
"""

from __future__ import annotations

import bisect
import logging
from collections.abc import Iterable
from pathlib import Path

from ..config import Settings
from ..index.lookup import IndexLookup
from ..lru import LRUCache
from ..normalize import index_key, normalize
from ..semtypes import to_source_set, to_tui, to_tui_set
from ..types import ConceptInfo, Entity, Ev
from . import NegationDetector, TextPipeline, TokenizedSentence, get_negation_detector
from .abbreviations import UserDefinedAcronyms, extract_abbr_pairs, mark_abbreviations
from .postag import ALLOWED_PART_OF_SPEECH
from .precision import PrecisionFilter
from .tokenize import Token, tokenize

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKEN_SIZE = 15  # metamaplite.entitylookup4.maxtokensize
CUSTOM_SOURCE = "USERDEFINED"
CUSTOM_SEMANTIC_TYPE = "unknown"


# CharUtils.isAlphaNumeric's Greek cases: the Greek and Coptic block, U+0370-U+03FF, less the code
# points its switch omits -- U+0378-U+0379, U+037F-U+0383, U+038B, U+038D and U+03A2, i.e. the
# ones unassigned when it was written.  It includes a few non-letters (U+037E GREEK QUESTION MARK,
# U+0387 ANO TELEIA, the tonos marks); they are in the list, so they count.
_JAVA_GREEK_ALNUM = frozenset(
    chr(c)
    for c in range(0x0370, 0x0400)
    if not (0x0378 <= c <= 0x0379 or 0x037F <= c <= 0x0383 or c in (0x038B, 0x038D, 0x03A2))
)


def _is_alnum(ch: str) -> bool:
    """``CharUtils.isAlphaNumeric``: ASCII letters and digits, **and Greek**.

    Not ASCII-only, as this port long assumed and documented: Java's switch continues past
    ``'z'`` into ~130 Greek code points ("greek characters, not exhaustive?"), so a span may start
    or end with a Greek letter.  Verified against MetaMapLite 3.6.2rc8, which matches
    ``β-blocker`` (via the normalized key ``beta-blocker``) where the ASCII-only test rejected it.
    """
    return ("0" <= ch <= "9") or ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ch in _JAVA_GREEK_ALNUM


class SpecialTerms:
    """Excluded terms (``metamaplite.excluded.termsfile``, e.g. ``data/specialterms.txt``).

    Each line is ``CUI:normalized term`` or ``*:normalized term``; a match excludes that CUI
    (or every CUI) for spans whose normalized text equals the term.
    """

    def __init__(self, terms: Iterable[str] = ()):
        self.terms: set[str] = {t.strip() for t in terms if t.strip()}

    @classmethod
    def from_file(cls, path: Path | str) -> SpecialTerms:
        path = Path(path)
        if not path.exists():
            log.warning("special terms file %s does not exist", path)
            return cls()
        return cls(path.read_text(encoding="utf-8").splitlines())

    def is_excluded(self, cui: str, norm_term: str) -> bool:
        return f"{cui}:{norm_term}" in self.terms or f"*:{norm_term}" in self.terms

    def __len__(self) -> int:
        return len(self.terms)


class CustomTerms:
    """User concepts (``metamaplite.cuitermlistfile.filename``): lines of ``CUI|term``.

    Keyed by ``term.lower()`` (NameIdListMap) and consulted with the same keys as the index.
    """

    def __init__(self, pairs: Iterable[tuple[str, str]] = ()):
        self.map: dict[str, list[str]] = {}
        for cui, term in pairs:
            self.map.setdefault(index_key(term), []).append(cui)

    @classmethod
    def from_file(cls, path: Path | str) -> CustomTerms:
        path = Path(path)
        if not path.exists():
            log.warning("custom concept file %s does not exist", path)
            return cls()
        pairs = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if "|" in line:
                cui, term = line.split("|", 1)
                if cui.strip() and term.strip():
                    pairs.append((cui.strip(), term.strip()))
        return cls(pairs)

    def cuis(self, key: str) -> list[str]:
        return self.map.get(key, [])

    def __len__(self) -> int:
        return len(self.map)


class EntityLookup:
    def __init__(
        self,
        index: IndexLookup,
        settings: Settings | None = None,
        *,
        excluded_terms: SpecialTerms | None = None,
        custom_terms: CustomTerms | None = None,
        max_token_size: int = DEFAULT_MAX_TOKEN_SIZE,
        allowed_pos: frozenset[str] = ALLOWED_PART_OF_SPEECH,
        remove_subsumed: bool = True,
        negex: NegationDetector | None = None,
        udas: UserDefinedAcronyms | None = None,
    ):
        self.index = index
        self.settings = settings or Settings()
        self.max_token_size = max_token_size
        self.allowed_pos = (
            frozenset(self.settings.postag_list) | {""}
            if self.settings.postag_list
            else allowed_pos
        )
        self.remove_subsumed = remove_subsumed
        if excluded_terms is None and self.settings.excluded_terms_file:
            excluded_terms = SpecialTerms.from_file(self.settings.excluded_terms_file)
        self.excluded_terms = excluded_terms or SpecialTerms()
        if custom_terms is None and self.settings.cui_term_list_file:
            custom_terms = CustomTerms.from_file(self.settings.cui_term_list_file)
        self.custom_terms = custom_terms
        self._concept_cache: LRUCache[tuple[str, str], ConceptInfo] = LRUCache()
        self.negex = negex or get_negation_detector(
            self.settings.negation_detector, strict_parity=self.settings.strict_parity
        )
        if udas is None and self.settings.uda_file:
            udas = UserDefinedAcronyms.from_file(self.settings.uda_file, self._lookup)
        self.udas = udas
        # The port's own, off by default: Java MetaMapLite has no such filter.
        self.precision_filter = (
            PrecisionFilter(index.concept_strings) if self.settings.precision_filter else None
        )

    # -- dictionary access -----------------------------------------------------------------------

    def _concept(self, key: str, cui: str, norm_term: str, from_index: bool) -> ConceptInfo:
        ck = (key, cui)
        ci = self._concept_cache.get(ck)
        if ci is None:
            sources = self.index.sources(cui)  # ordered; see IndexLookup.sources
            sts = frozenset(self.index.semantic_types(cui))
            if from_index:
                concept_string = self.index.concept_string(key, cui)
            else:  # custom concept (AugmentedDictionary.createCustomConceptInfoSet)
                concept_string = norm_term
                if CUSTOM_SOURCE not in sources:
                    sources = (*sources, CUSTOM_SOURCE)
                sts = sts or frozenset({CUSTOM_SEMANTIC_TYPE})
            ci = ConceptInfo(cui, self.index.preferred_name(cui), concept_string, sources, sts)
            self._concept_cache[ck] = ci
        return ci

    def _lookup(
        self, key: str, norm_term: str, cuis: tuple[str, ...] | None = None
    ) -> list[ConceptInfo]:
        """IVFLookup.lookup + AugmentedDictionary: concepts stored under one exact key.

        ``cuis`` lets the caller supply what the index holds for ``key`` when it has already
        fetched it in bulk; ``None`` looks it up here.
        """
        if cuis is None:
            cuis = self.index.cuis_for_key(key)
        out = [self._concept(key, cui, norm_term, True) for cui in cuis]
        if self.custom_terms:
            seen = set(cuis)
            out.extend(
                self._concept(key, cui, norm_term, False)
                for cui in self.custom_terms.cuis(key)
                if cui not in seen
            )
        return out

    # -- per-sentence matching --------------------------------------------------------------------

    def find_matches(
        self, tokens: list[Token], docid: str = "000000", fieldid: str | None = "text"
    ) -> dict[tuple[int, int], Entity]:
        """All matching spans in one sentence's token list, keyed by (start, length)."""
        span_map: dict[tuple[int, int], Entity] = {}
        n = len(tokens)
        # Two passes over the sentence: enumerate every candidate span and its lookup keys,
        # resolve them all together, then build the entities in the original order.  On ordinary
        # text roughly four in five of these keys are not in the index, and asking one at a time
        # made a separate SQLite round trip out of every miss.
        windows: list[tuple[Token, list[tuple[int, Token, str, str, tuple[str, ...]]]]] = []
        for i in range(n):
            first = tokens[i]
            if first.text.lower() == "other" or first.pos not in self.allowed_pos:
                continue
            # Every span starting here shares this first character, so the "must start with an
            # alphanumeric" rule decides the whole window at once.
            if not _is_alnum(first.text[0]):
                continue
            candidates: list[tuple[int, Token, str, str, tuple[str, ...]]] = []
            original = ""
            for j in range(i + 1, min(i + self.max_token_size, n) + 1):
                last = tokens[j - 1]
                original += last.text  # each span extends the previous one by one token
                if len(original) <= 2:
                    continue
                if not _is_alnum(original[-1]):
                    continue
                norm = normalize(original)
                low = index_key(original)
                candidates.append((j, last, original, norm, (low, norm) if norm != low else (low,)))
            if candidates:
                windows.append((first, candidates))
        if not windows:
            return span_map
        resolved = self.index.cuis_for_keys(
            list(dict.fromkeys(key for _, cands in windows for c in cands for key in c[4]))
        )
        for first, candidates in windows:
            for j, last, original, norm, keys in candidates:
                concepts: dict[str, ConceptInfo] = {}
                for key in keys:
                    for ci in self._lookup(key, norm, resolved.get(key, ())):
                        concepts.setdefault(ci.cui, ci)
                if not concepts:
                    continue
                length = last.end - first.start
                evs = [
                    Ev(ci, original, norm, first.start, length, 0.0, first.pos)
                    for ci in concepts.values()
                    if not self.excluded_terms.is_excluded(ci.cui, norm)
                ]
                if not evs:
                    continue
                span = (first.start, length)
                if span in span_map:
                    span_map[span].add_evs(evs)
                else:
                    # Java passes the index of the first 15-token window containing the span as
                    # "sentenceNumber"; it surfaces in MMI output, so reproduce it.
                    window = max(0, j - self.max_token_size)
                    span_map[span] = Entity(
                        docid, fieldid, original, first.pos, window, first.start, length, 0.0, evs
                    )
        return span_map

    # -- document-level post-processing ---------------------------------------------------------

    @staticmethod
    def filter_by_restrictions(
        entities: Iterable[Entity],
        semantic_types: set[str] | None,
        sources: set[str] | None,
    ) -> list[Entity]:
        """ConceptInfoUtils.filterEntityEvListBy{SemanticType,Source}; drops emptied entities.

        ``semantic_types`` accepts abbreviations (``dsyn``) or TUIs (``T047``) in any case, and
        ``sources`` accepts SABs in any case — the same reading the dictionary lookup uses.
        """
        st = to_tui_set(semantic_types)
        src = to_source_set(sources)
        out = []
        for e in entities:
            evs = e.evs
            if st is not None:
                evs = [ev for ev in evs if {to_tui(a) for a in ev.concept.semantic_types} & st]
            if src is not None:
                evs = [ev for ev in evs if {s.upper() for s in ev.concept.sources} & src]
            if evs:
                e.evs = evs
                out.append(e)
        return out

    @staticmethod
    def remove_subsumed_entities(entities: Iterable[Entity]) -> list[Entity]:
        """EntityLookup4.removeSubsumedEntities: drop spans contained in another entity's span.

        Java compares every pair; this sweeps instead.  Sorting by start ascending and end
        descending means every entity already seen starts no later than this one, so the widest
        end so far is enough to decide containment — O(n log n) instead of O(n^2), which matters
        on a long document where the pairwise scan dominates.

        The result is returned in that sorted order (``finish`` sorts identically anyway).  Callers
        upstream of this method key entities by span, so spans are unique; given two *distinct*
        objects with the same span the pairwise version drops both, while this keeps the first.
        """
        keep = []
        widest_end = -1
        for e in sorted(entities, key=lambda e: (e.start, -e.end)):
            if widest_end < e.end:
                keep.append(e)
                widest_end = e.end
        return keep

    def finish(
        self,
        entities: Iterable[Entity],
        semantic_types: set[str] | None = None,
        sources: set[str] | None = None,
    ) -> list[Entity]:
        ents = self.filter_by_restrictions(entities, semantic_types, sources)
        if self.remove_subsumed:
            ents = self.remove_subsumed_entities(ents)
        if self.precision_filter is not None:  # after subsumption on purpose; see precision.py
            ents = self.precision_filter(ents)
        ents.sort(key=lambda e: (e.start, -e.length))
        return ents

    # -- entry points ----------------------------------------------------------------------------

    def process_sentences(
        self,
        sentences: Iterable[TokenizedSentence],
        text: str = "",
        docid: str = "000000",
        fieldid: str | None = "text",
        semantic_types: set[str] | None = None,
        sources: set[str] | None = None,
        base_offset: int = 0,
        detect_negations: bool | None = None,
    ) -> list[Entity]:
        """EntityLookup4.processPassage over already-segmented sentences of ``text``.

        ``text`` (starting at ``base_offset``) is needed for abbreviation propagation; when it
        is empty that step is skipped.
        """
        sentences = list(sentences)
        if detect_negations is None:
            detect_negations = self.settings.detect_negations
        # Java: HashSet<Entity> keyed by span; the first entity for a span wins.
        found: dict[tuple[int, int], Entity] = {}

        def add_all(ents: Iterable[Entity]) -> None:
            for e in ents:
                found.setdefault((e.start, e.length), e)

        for ts in sentences:
            sent_ents = list(self.find_matches(ts.tokens, docid, fieldid).values())
            if self.udas:
                sent_ents.extend(self.udas.generate_entities(docid, ts.tokens))
            for e in sent_ents:
                e.location_position = ts.index
            add_all(sent_ents)

        abbr_infos = []
        if text:
            for ts in sentences:
                abbr_infos.extend(extract_abbr_pairs(ts.sentence.text, ts.sentence.offset))
        if abbr_infos:
            # Passage-level, not per-sentence: ``mark_abbreviations`` already scans the whole
            # passage for every short form, so repeating it once per sentence only re-derived
            # the same entities at O(sentences x entities) cost.
            add_all(mark_abbreviations(text, abbr_infos, list(found.values()), base_offset))
        if detect_negations:
            self.detect_negations(sentences, list(found.values()))
        return self.finish(found.values(), semantic_types, sources)

    def detect_negations(self, sentences: list[TokenizedSentence], entities: list[Entity]) -> None:
        """Run the negation detector over each sentence.

        Java hands every sentence the whole passage's entity list.  A detector that declares
        ``sentence_local = True`` ignores entities outside the sentence anyway, so it is given
        only the ones that start within the sentence's token span -- identical output, without
        rescanning the passage per sentence.  Bucketed on character
        offsets, not ``Entity.location_position``: entities added by abbreviation propagation
        have that reset to 0.
        """
        if not getattr(self.negex, "sentence_local", False):
            for ts in sentences:
                self.negex.detect(ts.tokens, entities, ts.sentence.text)
            return
        ordered = sorted(entities, key=lambda e: e.start)
        starts = [e.start for e in ordered]
        for ts in sentences:
            local: list[Entity] = []
            if ts.tokens:
                lo = bisect.bisect_left(starts, ts.tokens[0].start)
                hi = bisect.bisect_right(starts, ts.tokens[-1].end)
                local = ordered[lo:hi]
            self.negex.detect(ts.tokens, local, ts.sentence.text)

    def process_text(
        self,
        text: str,
        pipeline: TextPipeline,
        docid: str = "000000",
        fieldid: str | None = "text",
        semantic_types: set[str] | None = None,
        sources: set[str] | None = None,
        detect_negations: bool | None = None,
    ) -> list[Entity]:
        """EntityLookup4.processText: segment + tag, then :meth:`process_sentences`."""
        return self.process_sentences(
            pipeline.process(text),
            text,
            docid,
            fieldid,
            semantic_types,
            sources,
            detect_negations=detect_negations,
        )

    def lookup_term(self, term: str) -> list[Entity]:
        """EntityLookup4.lookupTerm.

        Match a bare term, untagged, with no filtering or subsumption removal.
        """
        return sorted(
            self.find_matches(tokenize(term)).values(), key=lambda e: (e.start, -e.length)
        )
