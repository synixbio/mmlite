"""MMI output — port of ``resultformats.mmi.MMI`` + ``mmi.Ranking`` / ``TermFrequency`` / ``AATF``.

One line per CUI per document::

    docid|MMI|score|preferred name|CUI|[semtypes]
        |"concept"-field-sent-"text"-POS-neg,...|fields|start/len;...|treecodes|

(one physical line; wrapped here only to fit)

* ``score`` = ``-10000 * negNRank`` from ``Ranking.processTF(tfList, 1000)``, 2 decimals.
* semantic types render as Java ``List.toString()`` over a ``HashSet`` iteration:
  ``[bacs, aapp]`` — see :mod:`mmlite.javautil`.
* ``sent`` is ``Entity.sentenceNumber``, which in Java is the index of the 15-token window
  that first produced the span, i.e. ``max(0, last_token_index - 14)`` — not a sentence number.
* one tuple per ``Ev``: Java's ``LinkedHashSet<Tuple>`` never dedupes because ``Tuple7.hashCode``
  is inconsistent with ``equals``.
* tree codes: MeSH table looked up by the preferred name (case-insensitive).
Verified line-for-line against MetaMapLite 3.6.2rc8 output; see ``tests/parity/README.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..javautil import hash_set_order
from ..types import Entity

# Ranking constants
NC, NF, NM, NMM, NW, NZ = 0.0, -5.0, 0.0, -10.0, 0.0, 0.0
WC, WD, WM, WMM, WW = 0.0, 1.0, 14.0, 1.0, 0.0
MMI_TREE_DEPTH_SPECIFICITY_DIVISOR = 8
MMI_WORD_SPECIFICITY_DIVISOR = 26
MMI_CHARACTER_SPECIFICITY_DIVISOR = 102
MAX_FREQ = 1000.0


@dataclass
class Tuple7:
    term: str  # concept string
    field: str
    nsent: int
    text: str  # matched text
    lexcat: str | None
    neg: int
    pos: list[tuple[int, int]]  # (start, end)


@dataclass
class TermFrequency:
    concept: str
    semantic_types: list[str]
    tuples: list[Tuple7]
    title_flag: bool
    cui: str
    frequency: int
    average_value: float
    treecodes: list[str]

    def add(self, t: Tuple7) -> None:
        self.tuples.append(t)
        self.frequency += 1


def _set_value1(v: float) -> float:
    return 1.0 if v > 1.0 else max(v, 0.0)


def normalize_value(n: float, value: float) -> float:
    if n == 0.0:
        return value
    v = _set_value1(value)
    if n > 0.0:
        en = math.exp(n)
        a, b = en + 1, en - 1
        ec = math.exp(-n * v)
        return (a / b) * ((1 - ec) / (1 + ec))
    m = -n
    em = math.exp(m)
    a, b = em + 1, em - 1
    return math.log((a + b * v) / (a - b * v)) / m


def _mm_tokenize_count(text: str) -> int:
    """Tokenize.mmTokenize(text, STRIP_WHITE_SPACE) token count: words and punct chars."""
    from ..pipeline.tokenize import tokenize

    return sum(1 for t in tokenize(text) if not t.is_whitespace)


def compute_specificities(concept: str, mm_value: float, treecodes: list[str]) -> list[float]:
    nmm_spec = normalize_value(NMM, mm_value / 1000)
    m_value = max(WD, sum(len(tc.split(".")) for tc in treecodes))
    nm_spec = normalize_value(NM, m_value / MMI_TREE_DEPTH_SPECIFICITY_DIVISOR)
    nw_spec = normalize_value(NW, _mm_tokenize_count(concept) / MMI_WORD_SPECIFICITY_DIVISOR)
    c_spec = len(concept) // MMI_CHARACTER_SPECIFICITY_DIVISOR  # Java int division
    nc_spec = normalize_value(NC, c_spec)
    return [nmm_spec, nm_spec, nw_spec, nc_spec]


def neg_n_rank(tf: TermFrequency) -> float:
    n_freq = normalize_value(NF, tf.frequency / MAX_FREQ)
    specs = compute_specificities(tf.concept, tf.average_value, tf.treecodes)
    weights = [WMM, WM, WW, WC]
    spec = sum(w * s for w, s in zip(weights, specs, strict=True)) / sum(weights)
    rank = spec if tf.title_flag else n_freq * spec
    return -1 * normalize_value(NZ, rank)


class MMIFormatter:
    def __init__(
        self, treecode_lookup: Callable[[str], Sequence[str]] | None = None, java_order: bool = True
    ):
        """``treecode_lookup(preferred_name) -> list[str]`` (verbatim key, see module doc).

        ``java_order=False`` (``Settings.strict_parity`` off) lists semantic types alphabetically
        and documents in input order, instead of Java ``HashSet`` order.
        """
        self.treecode_lookup: Callable[[str], Sequence[str]] = treecode_lookup or (lambda name: [])
        self.java_order = java_order

    def term_frequencies(self, entities: list[Entity]) -> list[TermFrequency]:
        tfs: dict[str, TermFrequency] = {}
        for e in entities:
            for ev in e.evs:
                cui = ev.cui
                t = Tuple7(
                    ev.concept.concept_string,
                    e.fieldid if e.fieldid is not None else "text",
                    e.sentence_number,
                    ev.matched_text,
                    e.lexical_category,
                    1 if e.negated else 0,
                    [(ev.start, ev.end)],
                )
                if cui in tfs:
                    tfs[cui].add(t)
                else:
                    name = ev.concept.preferred_name
                    tf = TermFrequency(
                        name,
                        (
                            hash_set_order(sorted(ev.concept.semantic_types))
                            if self.java_order
                            else sorted(ev.concept.semantic_types)
                        ),
                        [],
                        e.fieldid in ("title", "TI"),
                        cui,
                        0,
                        ev.score,
                        list(self.treecode_lookup(name)),
                    )
                    tf.add(t)
                    tfs[cui] = tf
        return list(tfs.values())

    @staticmethod
    def render_tuple(t: Tuple7) -> str:
        lexcat = "null" if t.lexcat is None else t.lexcat
        return f'"{t.term}"-{t.field}-{t.nsent}-"{t.text}"-{lexcat}-{t.neg}'

    @staticmethod
    def render_positions(t: Tuple7) -> str:
        return ",".join(f"{s}/{e - s}" for s, e in t.pos)  # PositionImpl.toStringStartLength

    def format_document(self, docid: str, entities: list[Entity]) -> str:
        rows = []
        for tf in self.term_frequencies(entities):
            rows.append((neg_n_rank(tf), tf.concept, tf))
        rows.sort(key=lambda r: (r[0], r[1]))
        lines = []
        for rank, _, tf in rows:
            fields = []
            for t in tf.tuples:
                if t.field not in fields:
                    fields.append(t.field)
            lines.append(
                "|".join(
                    [
                        docid,
                        "MMI",
                        f"{-10000 * rank:.2f}",
                        tf.concept,
                        tf.cui,
                        "[" + ", ".join(tf.semantic_types) + "]",
                        ",".join(self.render_tuple(t) for t in tf.tuples),
                        ";".join(fields),
                        ";".join(self.render_positions(t) for t in tf.tuples),
                        ";".join(tf.treecodes),
                        "",  # the PrintWriter variant (used by the CLI) ends each line with "|"
                    ]
                )
                + "\n"
            )
        return "".join(lines)

    def format(self, entities: list[Entity]) -> str:
        """One block of records per document, in Java's document order.

        ``MMI.genDocidEntityMap`` is a ``HashMap<String, List<Entity>>``, so with several
        documents in one run the blocks come out in hash-table order of the docids, not in the
        order the documents were read.  Single-document runs cannot tell the difference; a BioC
        collection whose docids are ``D1``, ``D2`` and ``OWN`` prints ``OWN`` first, as Java does.
        With ``java_order=False`` the blocks follow the order the documents were read.
        """
        by_doc: dict[str, list[Entity]] = {}
        for e in entities:
            by_doc.setdefault(e.docid, []).append(e)
        order = hash_set_order(by_doc) if self.java_order else list(by_doc)
        return "".join(self.format_document(d, by_doc[d]) for d in order)
