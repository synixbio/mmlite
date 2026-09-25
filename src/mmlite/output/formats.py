"""JSON, BRAT, CuiList and Full formatters — ports of ``resultformats.Json``, ``Brat``,
``CuiList`` and ``Full``.

Semantic types and sources are emitted in Java ``HashSet`` iteration order (see
:mod:`mmlite.javautil`) so the output matches MetaMapLite field for field; ``Full`` is
our own layout and sorts them for readability."""

from __future__ import annotations

import json
from typing import Any

from ..javautil import ev_hash, hash_set_order, hash_set_order_by
from ..types import ConceptInfo, Entity, Ev


def _evs_in_java_order(e: Entity, java_order: bool = True) -> list[Ev]:
    """``Entity.getEvList()`` = ``new ArrayList<>(evSet)`` — a ``HashSet``, so hash-table order.

    Applied at format time rather than to ``Entity.evs`` itself, so nothing upstream (MMI in
    particular, which is under byte-level parity test) sees a reordered evidence list.
    ``java_order=False`` (``Settings.strict_parity`` off) sorts by position and CUI instead.
    """
    if not java_order:
        return sorted(e.evs, key=lambda ev: (ev.start, ev.length, ev.cui))
    return hash_set_order_by(e.evs, lambda ev: ev_hash(ev.start, ev.length, ev.cui))


def _sts(ci: ConceptInfo, java_order: bool = True) -> list[str]:
    ordered = sorted(ci.semantic_types)
    return hash_set_order(ordered) if java_order else ordered


def _srcs(ci: ConceptInfo, java_order: bool = True) -> list[str]:
    """Sources in Java ``HashSet`` order — built from *insertion* order, not sorted.

    Unlike :func:`_sts`, this must not sort first: source sets run to 20-30 entries, so bucket
    collisions are common and within-bucket order (which is insertion order) is observable.
    Semantic-type sets are small enough that collisions are rare and sorting is harmless there.
    ``java_order=False`` sorts them alphabetically.
    """
    return hash_set_order(ci.sources) if java_order else sorted(set(ci.sources))


def concept_to_json(ev: Ev, java_order: bool = True) -> dict[str, Any]:
    ci = ev.concept
    return {
        "cui": ci.cui,
        "preferredname": ci.preferred_name,
        "conceptstring": ci.concept_string,
        "semantictypes": _sts(ci, java_order),
        "sources": _srcs(ci, java_order),
    }


def ev_to_json(ev: Ev, java_order: bool = True) -> dict[str, Any]:
    return {
        "id": "ev0",
        "matchedtext": ev.matched_text,
        "score": ev.score,
        "start": ev.start,
        "length": ev.length,
        "conceptinfo": concept_to_json(ev, java_order),
    }


def entity_to_json(e: Entity, java_order: bool = True) -> dict[str, Any]:
    return {
        "docid": e.docid,
        "id": "en0",
        "matchedtext": e.text,
        "fieldid": e.fieldid,
        "start": e.start,
        "length": e.length,
        "negated": e.negated,
        "evlist": [ev_to_json(ev, java_order) for ev in _evs_in_java_order(e, java_order)],
    }


class JsonFormatter:
    def __init__(self, indent: int | None = None, java_order: bool = True):
        self.indent = indent
        self.java_order = java_order

    def format(self, entities: list[Entity]) -> str:
        return (
            json.dumps([entity_to_json(e, self.java_order) for e in entities], indent=self.indent)
            + "\n"
        )


class BratFormatter:
    """BRAT standoff annotation.

    One ``T<n>\\t<type> start end\\ttext`` line per span, each followed by its
    ``N<n>\\tReference T<n> <rid>:<eid>\\t<text>`` note lines.
    """

    def __init__(self, type_name: str = "MMLite", java_order: bool = True):
        self.type_name = type_name  # metamaplite.brat.typename
        self.java_order = java_order

    def references(self, e: Entity) -> list[tuple[str, str, str]]:
        refs: list[tuple[str, str, str]] = []
        if e.score > 0.0:
            refs.append(("Score", str(e.score), str(e.score)))
        for ev in e.evs:
            ci = ev.concept
            refs.append(("ConceptId", ci.cui, ci.preferred_name))
            for st in _sts(ci, self.java_order):
                refs.append(("SemanticType", f"{ci.cui}:{st}", st))
            for src in _srcs(ci, self.java_order):
                refs.append(("Source", f"{ci.cui}:{src}", src))
            if e.negated:
                refs.append(("Negated", ev.matched_text, ev.matched_text))
        # Java collects these in a HashSet: drop duplicates, keep first occurrence
        seen: set[tuple[str, str, str]] = set()
        out = []
        for r in refs:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    def format(self, entities: list[Entity]) -> str:
        by_span: dict[tuple[int, int, str], Entity] = {}
        for e in entities:
            by_span.setdefault((e.start, e.end, e.text), e)
        lines = []
        n_index = 0
        for t_index, ((start, end, text), e) in enumerate(sorted(by_span.items()), 1):
            tid = f"T{t_index}"
            lines.append(f"{tid}\t{self.type_name} {start} {end}\t{text}")
            for rid, eid, rtext in self.references(e):
                n_index += 1
                lines.append(f"N{n_index}\tReference {tid} {rid}:{eid}\t{rtext}")
        return "".join(line + "\n" for line in lines)


class CuiListFormatter:
    """One CUI per line (``resultformats.CuiList``), first occurrence order."""

    def format(self, entities: list[Entity]) -> str:
        seen: set[str] = set()
        out = []
        for e in entities:
            for ev in e.evs:
                if ev.cui not in seen:
                    seen.add(ev.cui)
                    out.append(ev.cui + "\n")
        return "".join(out)


def java_number_format(value: float) -> str:
    """``NumberFormat.getInstance()`` with ``setMaximumFractionDigits(2)``.

    Grouping separators on, no minimum fraction digits (so ``0.0`` prints ``0``), and half-even
    rounding — which is what Python's own formatting does.  Java's instance is locale-sensitive;
    this fixes the English convention, the only one the fixtures exercise.
    """
    text = f"{value:,.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


class BcEvaluateFormatter:
    """BioCreative evaluation format (``resultformats.BcEvaluate``).

    ``docid<TAB>matched text<TAB>rank<TAB>score``, one line per entity, rank counting from 1
    across the whole run.  Registered by Java under four names — ``bc``, ``bc-evaluate``, ``cdi``
    and, confusingly, ``bioc``: it is a four-column TSV, **not** BioC XML, and MetaMapLite has no
    BioC XML writer at all.  The score is ``Entity.score``, which is 0 unless MetaMap-style
    scoring is on (not ported; DEVELOPMENT_PLAN §10.1), so in practice every line ends in ``0``.
    """

    def format(self, entities: list[Entity]) -> str:
        return "".join(
            f"{e.docid}\t{e.text}\t{rank}\t{java_number_format(e.score)}\n"
            for rank, e in enumerate(entities, 1)
        )


class FullFormatter:
    """Human-readable dump (``resultformats.Full``).

    This is the port's own layout, so unlike ``json`` it also shows ConText's temporality,
    experiencer and assertion when a ConText run has set them.
    """

    def format(self, entities: list[Entity]) -> str:
        out = []
        for e in entities:
            flags = [" [negated]"] if e.negated else []
            # ConText only; NegEx leaves these unset, so nothing changes for the default
            # detector.  "Affirmed" and "Negated" are already carried by [negated], so only
            # "Possible" is worth a flag of its own.
            if e.assertion == "Possible":
                flags.append(" [possible]")
            flags += [f" [{v}]" for v in (e.temporality, e.experiencer) if v]
            marks = "".join(flags)
            out.append(f"{e.start}:{e.length} {e.text!r} ({e.lexical_category}){marks}\n")
            for ev in e.evs:
                ci = ev.concept
                out.append(
                    f"    {ci.cui} {ci.preferred_name} [{','.join(sorted(ci.semantic_types))}] "
                    f"{{{','.join(sorted(ci.sources))}}} <- {ci.concept_string!r}\n"
                )
        return "".join(out)
