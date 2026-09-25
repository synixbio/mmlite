"""ConText — port of ``context.implementation.ConText`` (``context-2012.jar``), the algorithm
behind MetaMapLite's ``--usecontext``.

Ported from bytecode: the jar ships no source and no lexicon file, and the trigger table lives in
the class as string constants (``scripts/gen_context_triggers.py`` extracts it into
:mod:`.context_triggers`).  Every behaviour below was then **verified against the jar itself**,
through a harness calling ``ConText.applyContext`` over a few thousand generated
(concept, sentence) pairs -- see DEVELOPMENT_PLAN §18.  Where the Java is wrong, the port is
wrong the same way; the notable cases are marked *(Java quirk)*.

The algorithm, for one concept in one sentence (``applyContext``):

1. Collapse whitespace and lower-case both; prefix the sentence with a space; replace the
   **first** occurrence of the concept with `` [0] `` (spaces on both sides, like the trigger
   tags).  A concept not present returns ``None``.
2. Replace each trigger phrase, in a fixed order, with a tag such as `` <NEG_PRE> `` -- spaces
   on **both** sides, so the tag survives the word split that follows.  A trigger
   regex is ``[\\s.]+PHRASE[\\s.:;,]+`` with each space in the phrase widened to ``[\\s-]``, so a
   trigger must be surrounded by separators.  *(Java quirk)*: a third of the phrases carry a
   trailing space, so their regex needs **two** separators after the phrase; in single-spaced
   text they only match right before the concept (where `` [0] `` supplied a second space) or
   before punctuation.  Pseudo-triggers are replaced first and match with no separators at all.
3. Split on ``[,;\\s]+`` and scan the words for tags near ``[0]``:
   negation forward from ``<NEG_PRE>`` and backward from ``<NEG_POST>``/``<POSS_POST>``, each
   over a 15-word window that stops at another tag; temporality forward from ``<HYPO_PRE>`` and
   ``<HIST_PRE>``/``<TIME_PRE>``, backward from ``<TIME_POST>``; experiencer forward from
   ``<EXP_PRE>``.  Those last scans run to the end of the sentence.

Results are the enum names Java returns: ``Negated``/``Possible``/``Affirmed``,
``Historical``/``Hypothetical``/``Recent``, ``Other``/``Patient``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..types import Entity
from .context_triggers import CONTEXT_TRIGGERS, EXTRA_TRIGGERS
from .tokenize import Token

# Java's Pattern \s and \d are ASCII unless UNICODE_CHARACTER_CLASS is set (it is not here).
_S = r"[ \t\n\x0b\f\r]"
_SEP_BEFORE = rf"[{_S[1:-1]}.]+"  # [\s\.]+
_SEP_AFTER = rf"[{_S[1:-1]}.:;,]+"  # [\s\.\:;\,]+
# Fixed mode tags every bucket in one pass (see ConText._tag), so a trigger must not eat the
# separator its neighbour needs on the left: the trailing one becomes a lookahead, and a phrase
# may also end the string, which the consuming form cannot express.
_SEP_AFTER_AHEAD = rf"(?=[{_S[1:-1]}.:;,]|$)"
_WS_RUN = re.compile(rf"{_S}+")
_SPLIT = re.compile(rf"[,;{_S[1:-1]}]+")
_CONCEPT = re.compile(r"\[[0-9]+\]")

# The Java constructor compiles these from literals with character-class brackets where the
# author meant groups -- ``[ |-]`` is "space, pipe or hyphen", ``[day|days]`` is one character
# from that set.  Reproduced verbatim; they match what Java's match, which is little.
_TIME = (
    r"((1[4-9]|[1-9]?[2-9][0-9])[ |-][day|days] of)|(([2-9]|[1-9][0-9])[ |-][week|weeks] of)"
    r"|(([1-9]?[0-9])[ |-][month|months|year|years] of)"
)
_TIME_FOR = rf"[for|over] the [last|past] ({_TIME})"
_TIME_SINCE = (
    r"since [last|the last]? ((([2-9]|[1-9][0-9]) weeks ago)|(([1-9]?[0-9])? "
    r"[month|months|year|years] ago)|([january|february|march|april|may|june|july|august"
    r"|september|october|november|december|spring|summer|fall|winter]))"
)

# What the three patterns above were written to say, used when ``strict_parity`` is off: real
# groups instead of character classes, "space or hyphen" as ``[ -]``, and optional words that do
# not leave a doubled space behind when absent.  ``\b`` keeps "114 days" from matching as "14 days"
# and "since marching" from matching "since march".
_TIME_FIXED = (
    r"(\b(1[4-9]|[1-9]?[2-9][0-9])[ -]days? of)|(\b([2-9]|[1-9][0-9])[ -]weeks? of)"
    r"|(\b([1-9]?[0-9])[ -](months?|years?) of)"
)
_TIME_FOR_FIXED = rf"(for|over) the (last|past) ({_TIME_FIXED})"
_TIME_SINCE_FIXED = (
    r"since (?:(?:the )?last )?((\b([2-9]|[1-9][0-9]) weeks ago)|(\b([1-9]?[0-9] )?(months?|years?)"
    r" ago)|((january|february|march|april|may|june|july|august|september|october|november"
    r"|december|spring|summer|fall|winter)\b))"
)

# Triggers that, found inside a concept's own name, describe that concept (fixed mode only; see
# ConText._concept_marker).  Negation, pseudo and terminator tags are excluded on purpose.
_CONCEPT_OWN_TAGS = frozenset(
    {"<EXP_PRE>", "<HYPO_PRE>", "<HIST_PRE>", "<HIST_1W>", "<TIME_PRE>", "<TIME_POST>"}
)

# A line that starts with a list marker ("- ", "* ", "• ", "2. ", "3) ").  Notes write review of
# systems and plans as unpunctuated bullets, which the sentence splitter joins into one "sentence";
# ConText's 15-word windows then carried "Denies ..." from one bullet into the next.
_BULLET = re.compile(r"\n[ \t]*(?:[-*•]|\d{1,2}[.)])[ \t]")


# A trigger match, back to the lexicon phrase that produced it: the leading separators the regex
# consumed are dropped, and the phrase's own separators (a space or a hyphen in the text) become
# the single spaces the lexicon writes them with.  Nothing trailing is consumed, so nothing
# trailing is stripped.  No lexicon phrase contains a hyphen or a dot, so this cannot collide.
_MATCH_LEAD = re.compile(rf"^[{_S[1:-1]}.]+")
_MATCH_INNER = re.compile(rf"[{_S[1:-1]}-]+")


def _phrase_key(matched: str) -> str:
    return _MATCH_INNER.sub(" ", _MATCH_LEAD.sub("", matched))


def _clause_of(sentence: str, pos: int) -> str:
    """The bullet item of ``sentence`` containing character ``pos`` (fixed mode only)."""
    start, end = 0, len(sentence)
    for m in _BULLET.finditer(sentence):
        if m.start() < pos:
            start = m.start()
        else:
            end = m.start()
            break
    return sentence[start:end]


# Bucket -> the tag its triggers are replaced with, used by fixed mode's single-pass tagger.
_BUCKET_TAGS = {
    "neg_pre": "NEG_PRE",
    "poss_pre": "POSS_PRE",
    "neg_post": "NEG_POST",
    "poss_post": "POSS_POST",
    "neg_end": "NEG_END",
    "exp_pre": "EXP_PRE",
    "exp_end": "EXP_END",
    "hypo_pre": "HYPO_PRE",
    "hypo_end": "HYPO_END",
    "hist_pre": "HIST_PRE",
    "hist_1w": "HIST_1W",
    "hist_end": "HIST_END",
    "hypo_exp_end": "HYPO_EXP_END",
    "hist_exp_end": "HIST_EXP_END",
}

NEGATED, POSSIBLE, AFFIRMED = "Negated", "Possible", "Affirmed"
HISTORICAL, HYPOTHETICAL, RECENT = "Historical", "Hypothetical", "Recent"
OTHER, PATIENT = "Other", "Patient"

WINDOW = 15  # words scanned for negation on either side of a trigger
_NEG_BACK_STOP = ("<NEG_PRE>", "<POSS_PRE>", "<NEG_POST>", "<POSS_POST>", "<NEG_END>")


def _java_split(pattern: re.Pattern[str], text: str) -> list[str]:
    """``String.split``: trailing empty strings dropped, a leading one kept."""
    parts = pattern.split(text)
    while parts and parts[-1] == "":
        parts.pop()
    return parts


@dataclass(frozen=True)
class ContextResult:
    negation: str
    temporality: str
    experiencer: str


class ConText:
    """One instance builds the trigger regexes once, as the Java constructor does.

    ``strict_parity=True`` (the default) reproduces the jar exactly.  ``False`` fixes its bugs
    (``Settings.strict_parity``; USER_GUIDE "Strict parity"):

    * :meth:`detect` only considers entities inside the sentence being analysed, each within its
      own bullet item.  Java applies every sentence to every entity in the document, so a later
      "family history of X" overwrote the experiencer and temporality of an earlier, unrelated
      mention of X -- and paid a regex pass per (entity, sentence) pair to do it.
    * The time patterns use groups, not character classes (``[day|days]`` was one character), and
      run before the lexicon triggers, which otherwise consume their "since".
    * The first pseudo-trigger keeps its first character.
    * ``previous`` (the lexicon's one ``one,hist`` entry) is tagged ``<HIST_1W>`` and marks the
      single following word as historical, as its position name and tag say; Java routes it
      nowhere.
    * Triggers stored with a trailing space ("denies ", "no ") no longer need two separators.
    * The longest trigger at a position wins, instead of whichever bucket is substituted first
      (:meth:`_compile_longest`): "Pneumonia was ruled out." is negated by the *post* trigger
      "was ruled out" rather than tagged with the *pre* trigger "ruled out", which sits at the
      end of the sentence with nothing after it to negate.
    * ``<POSS_PRE>`` starts a forward scan (verdict Possible) and ends a negation scan.
    * Experiencer and temporality triggers inside the concept's own name count
      (:meth:`_concept_marker`); negation triggers there deliberately do not.
    """

    def __init__(
        self,
        triggers: Iterable[str] = CONTEXT_TRIGGERS,
        strict_parity: bool = True,
        extra_triggers: Iterable[str] | None = None,
    ):
        self.strict_parity = strict_parity
        # Fixed mode ignores out-of-sentence entities (detect()), so EntityLookup may pre-filter;
        # strict mode must see the whole passage, as Java does.
        self.sentence_local = not strict_parity
        # Phrases the jar's lexicon lacks (EXTRA_TRIGGERS).  Fixed mode only: strict mode must
        # see exactly the lexicon Java sees.  Pass () to run fixed mode on the jar's phrases
        # alone, or a sequence of "phrase,position,type" entries to supply your own.
        if not strict_parity:
            extras = EXTRA_TRIGGERS if extra_triggers is None else extra_triggers
            triggers = (*triggers, *extras)
        alternations: dict[str, list[str]] = {
            "pseudo": [],
            "neg_pre": [],
            "neg_post": [],
            "poss_pre": [],
            "poss_post": [],
            "neg_end": [],
            "exp_pre": [],
            "exp_end": [],
            "hypo_pre": [],
            "hist_pre": [],
            "hist_1w": [],
            "hypo_end": [],
            "hist_end": [],
            "hist_exp_end": [],
            "hypo_exp_end": [],
        }
        # (phrase length, bucket, phrase, regex) for every non-pseudo trigger; fixed mode only.
        longest: list[tuple[int, str, str, str]] = []
        for entry in triggers:
            # indexOf(',') / lastIndexOf(','): a comma inside the phrase corrupts the position
            # field and the entry falls through every branch below -- one lexicon entry does.
            first, last = entry.index(","), entry.rindex(",")
            raw, position, kind = entry[:first], entry[first + 1 : last], entry[last + 1 :]
            if position == "pseudo":
                # Pseudo-triggers match with no separators around them, so where the lexicon gives
                # one a trailing space ("gram negative ") that space is its only word boundary:
                # kept in both modes.  (Not every pseudo entry has one; "history and" doesn't.)
                alternations["pseudo"].append(raw.replace(" ", rf"[{_S[1:-1]}-]"))
                continue
            # (Java quirk) a third of the phrases carry a trailing space ("denies ", "no "), which
            # the regex turns into a *second* required separator, so they only fire right before
            # the concept or before punctuation.  The regex already demands a separator on each
            # side, so fixed mode drops it: "Denies swelling, joint locking" then reaches both.
            if not strict_parity:
                raw = raw.strip()
            phrase = raw.replace(" ", rf"[{_S[1:-1]}-]")
            bucket = {
                ("termin", "neg"): "neg_end",
                ("termin", "hypo"): "hypo_end",
                ("termin", "hist"): "hist_end",
                ("termin", "histexp"): "hist_exp_end",
                ("termin", "hypoexp"): "hypo_exp_end",
                ("termin", "exp"): "exp_end",
                ("pre", "neg"): "neg_pre",
                ("pre", "poss"): "poss_pre",
                ("pre", "hypo"): "hypo_pre",
                ("pre", "exp"): "exp_pre",
                ("pre", "hist"): "hist_pre",
                ("post", "neg"): "neg_post",
                ("post", "poss"): "poss_post",
            }.get((position, kind))
            # (Java quirk) "hist_1w" is reachable only via a second, identical test for
            # ("pre", "hist") that the first one shadows, and the one "one,hist" entry matches no
            # position at all -- so <HIST_1W> is never produced.
            if bucket is None and not strict_parity and (position, kind) == ("one", "hist"):
                bucket = "hist_1w"
            if bucket is not None:
                alternations[bucket].append(rf"{_SEP_BEFORE}{phrase}{_SEP_AFTER}")
                # Fixed mode keeps the phrase's length, to prefer the longest trigger later.
                longest.append((len(raw), bucket, raw, rf"{_SEP_BEFORE}{phrase}{_SEP_AFTER_AHEAD}"))

        def compile_(name: str) -> re.Pattern[str] | None:
            alts = alternations[name]
            return re.compile("|".join(alts)) if alts else None

        # (Java quirk) the pseudo alternation is built as "|p1|p2|..." and then substring(2) is
        # taken, which drops the "|" *and the first character of the first pseudo phrase*.
        pseudo = "|".join(alternations["pseudo"])
        if strict_parity:
            pseudo = pseudo[1:]
        self.pseudo = re.compile(pseudo) if pseudo else None
        self.neg_pre = compile_("neg_pre")
        self.neg_post = compile_("neg_post")
        self.poss_pre = compile_("poss_pre")
        self.poss_post = compile_("poss_post")
        self.neg_end = compile_("neg_end")
        self.exp_pre = compile_("exp_pre")
        self.exp_end = compile_("exp_end")
        self.hypo_pre = compile_("hypo_pre")
        self.hypo_end = compile_("hypo_end")
        self.hist_pre = compile_("hist_pre")
        self.hist_1w = compile_("hist_1w")
        self.hist_end = compile_("hist_end")
        self.hypo_exp_end = compile_("hypo_exp_end")
        self.hist_exp_end = compile_("hist_exp_end")
        if strict_parity:
            self.time = re.compile(_TIME)
            self.time_for = re.compile(_TIME_FOR)
            self.time_since = re.compile(_TIME_SINCE)
        else:
            self.time = re.compile(_TIME_FIXED)
            self.time_for = re.compile(_TIME_FOR_FIXED)
            self.time_since = re.compile(_TIME_SINCE_FIXED)
        # (Java quirk) Java's replacement has no trailing space, which glues the tag to the next
        # word; moot in Java, where the tag is never produced, but not once it is.
        self._hist_1w_tag = " <HIST_1W>" if strict_parity else " <HIST_1W> "
        java_starts = ("<NEG_PRE>", "<PREP>")
        java_stops = ("<NEG_PRE>", "<PREP>", "<NEG_POST>", "<POSS_POST>", "<NEG_END>")
        self._fwd_starts = java_starts if strict_parity else (*java_starts, "<POSS_PRE>")
        self._fwd_stops = java_stops if strict_parity else (*java_stops, "<POSS_PRE>")
        # Strict mode tags bucket by bucket, as Java does, and never builds this.
        self.combined: re.Pattern[str] | None = None
        self._phrase_tag: dict[str, str] = {}
        if not strict_parity:
            self.combined, self._phrase_tag = self._compile_longest(longest)

    @staticmethod
    def _compile_longest(
        entries: list[tuple[int, str, str, str]],
    ) -> tuple[re.Pattern[str] | None, dict[str, str]]:
        """One alternation over every bucket, longest phrase first (fixed mode).

        Java substitutes bucket by bucket, so whichever bucket runs first wins a phrase that two
        of them share a prefix of: ``"ruled out ,pre,neg"`` is tagged before
        ``"was ruled out ,post,neg"`` can be, and "Pneumonia was ruled out." comes out with a
        *pre* trigger at the end of the sentence, negating nothing.  Python's alternation is
        leftmost-first, so ordering the alternatives by phrase length makes the longest trigger
        at any position win, which is what the lexicon means.

        The alternation carries no capture groups: 326 of them cost more than the match itself
        (2.9 ms a sentence against 0.4 ms), so the tag is looked up from the matched text, which
        :func:`_phrase_key` normalises back to its lexicon phrase.
        """
        if not entries:
            return None, {}
        phrase_tag: dict[str, str] = {}
        alts = []
        for _, bucket, phrase, regex in sorted(entries, key=lambda e: -e[0]):
            # stable sort: phrases of equal length keep Java's bucket order
            phrase_tag.setdefault(phrase, f" <{_BUCKET_TAGS[bucket]}> ")
            alts.append(regex)
        return re.compile("|".join(alts)), phrase_tag

    # -- preprocessing -----------------------------------------------------------------------------

    def preprocess(self, sentence: str, concept: str) -> str | None:
        """``preProcessSentence``: mark the concept, then tag every trigger."""
        text = " " + _WS_RUN.sub(" ", sentence).lower()
        needle = _WS_RUN.sub(" ", concept).lower()
        at = text.find(needle)
        if at == -1:
            return None
        marker = " [0] " if self.strict_parity else self._concept_marker(needle)
        return self._tag(text[:at] + marker + text[at + len(needle) :])

    def _concept_marker(self, needle: str) -> str:
        """The concept's placeholder in fixed mode: ``[0]`` plus the concept's *own* triggers.

        (Java quirk) the concept is replaced before triggers are tagged, so words inside it are
        never seen: the UMLS concept "Family history of hypertension" comes out Recent / Patient.
        Fixed mode tags the concept text on its own and keeps the experiencer and temporality
        triggers it contains, in place, around ``[0]``.  Negation triggers are deliberately
        dropped: a concept named "No known allergies" *is* the negative finding, and marking it
        negated would invert it.
        """
        words = _java_split(_SPLIT, self._tag(f" {needle} "))
        out: list[str] = []
        for w in words:
            if w in _CONCEPT_OWN_TAGS:
                out.append(w)
            elif w and not w.startswith("<") and (not out or out[-1] != "[0]"):
                out.append("[0]")
        if "[0]" not in out:  # the concept is nothing but a trigger
            out.append("[0]")
        return " " + " ".join(out) + " "

    def _tag(self, text: str) -> str:
        """Replace every trigger phrase in ``text`` with its tag, in Java's order."""
        time_patterns = (
            (self.time_for, " <TIME_PRE> "),
            (self.time, " <TIME_PRE> "),
            (self.time_since, " <TIME_POST> "),
        )
        # (Java quirk) the time patterns run last, after "since" (a lexicon trigger) has already
        # become <HYPO_END> -- so "since last march" can never match.  Fixed mode tags the
        # multi-word time phrases first, as the more specific match.
        if not self.strict_parity:
            for time_pattern, time_tag in time_patterns:
                text = time_pattern.sub(time_tag, text)
            # Pseudo-triggers still go first: they exist to mask text a negation trigger would
            # otherwise claim.  Every other bucket is then tagged in one longest-match pass.
            if self.pseudo is not None:
                text = self.pseudo.sub(" <NEG_PSEUDO> ", text)
            if self.combined is not None:
                text = self.combined.sub(lambda m: self._phrase_tag[_phrase_key(m.group())], text)
            return text
        for pattern, tag in (
            (self.pseudo, " <NEG_PSEUDO> "),
            (self.neg_pre, " <NEG_PRE> "),
            (self.poss_pre, " <POSS_PRE> "),
            (self.neg_post, " <NEG_POST> "),
            (self.poss_post, " <POSS_POST> "),
            (self.neg_end, " <NEG_END> "),
            (self.exp_pre, " <EXP_PRE> "),
            (self.exp_end, " <EXP_END> "),
            (self.hypo_pre, " <HYPO_PRE> "),
            (self.hypo_end, " <HYPO_END> "),
            (self.hist_pre, " <HIST_PRE> "),
            (self.hist_1w, self._hist_1w_tag),
            (self.hist_end, " <HIST_END> "),
            (self.hypo_exp_end, " <HYPO_EXP_END> "),
            (self.hist_exp_end, " <HIST_EXP_END> "),
            *(time_patterns if self.strict_parity else ()),
        ):
            if pattern is not None:
                text = pattern.sub(tag, text)
        return text

    # -- the three scanners ------------------------------------------------------------------------

    @staticmethod
    def _found(collected: list[str]) -> bool:
        return any(_CONCEPT.fullmatch(w) for w in collected)

    def apply_negex(self, words: list[str]) -> str:
        n = len(words)
        i = 0
        while i < n:
            w = words[i]
            if w == "<NEG_PSEUDO>":
                i += 1
                continue
            # (Java quirk) String.matches is whole-string, so only "<NEG_PRE>" (or the never
            # produced "<PREP>") starts a forward scan; "<POSS_PRE>" does not, which makes the
            # Possible branch below unreachable from the front.  Fixed mode lets it start a scan,
            # and stop one, as the backward scan's stop list (_NEG_BACK_STOP) already does.
            if w in self._fwd_starts:
                window = min(WINDOW, n - i)
                collected = []
                for j in range(1, window):
                    nxt = words[i + j]
                    if nxt in self._fwd_stops:
                        break
                    collected.append(nxt)
                if w == "<NEG_PRE>":
                    verdict = NEGATED
                elif w == "<POSS_PRE>":
                    verdict = POSSIBLE
                else:
                    verdict = AFFIRMED
                if self._found(collected):
                    return verdict
                i += 1
                continue
            if w in ("<NEG_POST>", "<POSS_POST>"):
                window = min(WINDOW, i)
                collected = []
                for j in range(1, window):
                    prev = words[i - j]
                    if prev in _NEG_BACK_STOP:
                        break
                    collected.append(prev)
                verdict = NEGATED if w == "<NEG_POST>" else POSSIBLE
                if self._found(collected):
                    return verdict
                i += 1
                continue
            i += 1
        return AFFIRMED

    def apply_temporality(self, words: list[str]) -> str:
        n = len(words)
        i = 0
        while i < n:
            w = words[i]
            if w == "<NEG_PSEUDO>":
                i += 1
                continue
            if w == "<HYPO_PRE>":
                # (Java quirk) the terminator test is equals() against the literal string
                # "<HYPO_END>|<HYPO_EXP_END>|<HYPO_PRE>", which no word ever is: the scan runs to
                # the end of the sentence.
                collected = [words[i + j] for j in range(1, n - i)]
                if self._found(collected):
                    return HYPOTHETICAL
                i += 1
                continue
            if w in ("<HIST_PRE>", "<TIME_PRE>"):
                collected = []
                for j in range(1, n - i):
                    nxt = words[i + j]
                    if nxt in ("<HIST_END>", "<HIST_EXP_END>", "<HIST_PRE>", "<HIST_1W>"):
                        break
                    collected.append(nxt)
                if self._found(collected):
                    return HISTORICAL
                i += 1
                continue
            if w == "<HIST_1W>":
                # Only produced when strict_parity is off: the one word after the trigger.
                if i + 1 < n and self._found([words[i + 1]]):
                    return HISTORICAL
                i += 1
                continue
            if w == "<TIME_POST>":
                collected = []
                for j in range(1, i + 1):
                    prev = words[i - j]
                    if prev in ("<HIST_END>", "<HIST_EXP_END>", "<HIST_PRE>", "<HIST_1W>"):
                        break
                    collected.append(prev)
                if self._found(collected):
                    return HISTORICAL
                i += 1
                continue
            i += 1
        return RECENT

    def apply_experiencer(self, words: list[str]) -> str:
        n = len(words)
        i = 0
        while i < n:
            w = words[i]
            if w == "<NEG_PSEUDO>":
                i += 1
                continue
            if w == "<EXP_PRE>":
                # (Java quirk) same equals()-against-a-literal terminator as the hypothetical
                # scan: runs to the end of the sentence.
                collected = [words[i + j] for j in range(1, n - i)]
                if self._found(collected):
                    return OTHER
                i += 1
                continue
            i += 1
        return PATIENT

    # -- entry point -------------------------------------------------------------------------------

    def apply(self, concept: str, sentence: str) -> ContextResult | None:
        """``applyContext(concept, sentence)``; ``None`` when either is empty or the concept is
        not found in the sentence."""
        if concept == "" or sentence == "":
            return None
        tagged = self.preprocess(sentence, concept)
        if tagged is None:
            return None
        words = _java_split(_SPLIT, tagged)
        return ContextResult(
            self.apply_negex(words), self.apply_temporality(words), self.apply_experiencer(words)
        )

    # -- NegationDetector protocol ----------------------------------------------------------------

    def detect(self, tokens: list[Token], entities: Iterable[Entity], sentence: str = "") -> None:
        """``ContextWrapper.detectNegations``: ``applyContext`` for every entity against this
        sentence's text.  Java sets ``negated`` only to true (so it is sticky across sentences)
        and overwrites ``temporality`` on every non-null result; ``experiencer`` is kept too,
        which Java's ``Entity`` has no field for.  ``tokens`` is unused, as in Java.

        (Java quirk) ``entities`` is the whole document's list, so any later sentence that
        contains the same text overwrites an earlier entity's results.  With ``strict_parity``
        off, ``tokens`` bounds the sentence and only entities inside it are analysed, each within
        its own bullet item (:func:`_clause_of`).
        """
        if not sentence:
            return
        base = 0
        if not self.strict_parity:
            if not tokens:
                return
            lo, hi = tokens[0].start, tokens[-1].end
            entities = [e for e in entities if lo <= e.start and e.start + e.length <= hi]
            base = lo  # tokens are cut from the sentence at its own offset
        for entity in entities:
            clause = sentence if self.strict_parity else _clause_of(sentence, entity.start - base)
            result = self.apply(entity.text, clause)
            if result is None and clause is not sentence:  # mention straddles a bullet
                result = self.apply(entity.text, sentence)
            if result is None:
                continue
            if result.negation == NEGATED:
                entity.negated = True
            entity.temporality = result.temporality
            entity.experiencer = result.experiencer
            # Java drops the verdict here, keeping only the boolean above, so "rule out
            # pneumonia" reads exactly like "pneumonia".  Kept, on its own field.
            entity.assertion = result.negation
