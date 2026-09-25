"""Tests for the MMI / JSON / BRAT formatters."""

import json
import math

import pytest

from mmlite.output import get_formatter
from mmlite.output.mmi import (
    MMIFormatter,
    TermFrequency,
    Tuple7,
    compute_specificities,
    neg_n_rank,
    normalize_value,
)
from mmlite.types import ConceptInfo, Entity, Ev


def _entity(
    text, phrase, cui, name, sts, docid="doc.txt", sentence=0, negated=False, pos="NN", after=0
):
    start = text.index(phrase, after)
    ci = ConceptInfo(cui, name, phrase.title(), frozenset({"MSH"}), frozenset(sts))
    e = Entity(docid, "text", phrase, pos, sentence, start, len(phrase), 0.0, negated=negated)
    e.evs = [Ev(ci, phrase, phrase.lower(), start, len(phrase), 0.0, pos)]
    return e


TEXT = "heart attack and aspirin then heart attack again"


@pytest.fixture
def entities():
    return [
        _entity(TEXT, "heart attack", "C0027051", "Myocardial Infarction", ["dsyn"], negated=True),
        _entity(TEXT, "aspirin", "C0004057", "aspirin", ["orch", "phsu"]),
        _entity(
            TEXT,
            "heart attack",
            "C0027051",
            "Myocardial Infarction",
            ["dsyn"],
            sentence=1,
            after=20,
        ),
    ]


def test_normalize_value_matches_java_formulas():
    assert normalize_value(0.0, 0.42) == 0.42
    # n < 0 branch: value in [0, 1]
    v = normalize_value(-5.0, 0.002)
    assert 0 < v < 0.01 and math.isclose(
        v,
        math.log(
            (math.e**5 + 1 + (math.e**5 - 1) * 0.002) / (math.e**5 + 1 - (math.e**5 - 1) * 0.002)
        )
        / 5,
    )
    # n > 0 branch and clamping
    assert 0 < normalize_value(2.0, 5.0) <= 1.0000001
    assert normalize_value(-5.0, -3.0) == 0.0


def test_specificities_and_rank():
    specs = compute_specificities("aspirin", 0.0, ["D02.755.410.700.110"])
    assert specs[0] == 0.0  # mmValue 0 -> nmm spec 0
    assert specs[1] == 5 / 8  # tree depth 5 / divisor
    assert specs[2] == 1 / 26  # one word
    assert specs[3] == 0  # 7 chars // 102
    tf = TermFrequency("aspirin", ["orch"], [], False, "C0004057", 2, 0.0, ["D02.755.410.700.110"])
    r = neg_n_rank(tf)
    assert r < 0
    # title flag: rank == spec, independent of frequency
    tf.title_flag = True
    assert neg_n_rank(tf) < r


def test_mmi_lines(entities):
    out = MMIFormatter(lambda name: ["C14.280.647.500"] if name == "aspirin" else []).format(
        entities
    )
    lines = out.splitlines()
    assert len(lines) == 2  # one line per CUI
    mi = next(line for line in lines if "|C0027051|" in line)
    asp = next(line for line in lines if "|C0004057|" in line)
    f = mi.split("|")
    assert f[0] == "doc.txt" and f[1] == "MMI"
    assert float(f[2]) > 0
    assert f[3] == "Myocardial Infarction" and f[5] == "[dsyn]"
    # two tuples (one per occurrence), negation flag 1 on the first only, sentence numbers 0 and 1
    assert (
        f[6]
        == '"Heart Attack"-text-0-"heart attack"-NN-1,"Heart Attack"-text-1-"heart attack"-NN-0'
    )
    assert f[7] == "text"
    assert f[8] == "0/12;30/12"  # PositionImpl: start/length
    assert f[9] == "" and f[10] == "" and mi.endswith("|")
    assert asp.split("|")[5] == "[orch, phsu]"  # Java List.toString rendering
    assert asp.split("|")[9] == "C14.280.647.500"
    # more frequent concept ranks first (lower negNRank -> higher score)
    assert lines[0] is mi or float(lines[0].split("|")[2]) >= float(lines[1].split("|")[2])


def test_mmi_tuples_are_never_deduped():
    # Java's Tuple7.hashCode is inconsistent with equals, so its LinkedHashSet keeps everything
    tf = TermFrequency("x", [], [], False, "C1", 0, 0.0, [])
    tf.add(Tuple7("x", "text", 0, "x", "NN", 0, [(0, 1)]))
    tf.add(Tuple7("x", "text", 3, "x", "NN", 1, [(0, 1)]))
    assert len(tf.tuples) == 2 and tf.frequency == 2


def test_mmi_groups_by_docid(entities):
    entities[1].docid = "other.txt"
    out = MMIFormatter().format(entities)
    assert sorted(line.split("|")[0] for line in out.splitlines()) == ["doc.txt", "other.txt"]


def test_json_formatter(entities):
    data = json.loads(get_formatter("json").format(entities))
    assert [d["matchedtext"] for d in data] == ["heart attack", "aspirin", "heart attack"]
    assert data[0]["negated"] is True and data[1]["negated"] is False
    ev = data[1]["evlist"][0]
    assert ev["conceptinfo"] == {
        "cui": "C0004057",
        "preferredname": "aspirin",
        "conceptstring": "Aspirin",
        "semantictypes": ["orch", "phsu"],
        "sources": ["MSH"],
    }
    assert (ev["start"], ev["length"]) == (17, 7)


def test_brat_formatter(entities):
    out = get_formatter("brat", brat_type="Med").format(entities[:2])
    lines = out.splitlines()
    assert lines[0] == "T1\tMed 0 12\theart attack"
    assert "N1\tReference T1 ConceptId:C0027051\tMyocardial Infarction" in lines
    assert "Reference T1 SemanticType:C0027051:dsyn\tdsyn" in out
    assert "Reference T1 Source:C0027051:MSH\tMSH" in out
    assert "Reference T1 Negated:heart attack\theart attack" in out
    assert "T2\tMed 17 24\taspirin" in lines
    assert "Negated" not in out.split("T2\t")[1]


def test_cuilist_and_full(entities):
    assert get_formatter("cuilist").format(entities) == "C0027051\nC0004057\n"
    full = get_formatter("full").format(entities)
    assert "[negated]" in full and "C0004057 aspirin [orch,phsu]" in full


def test_mmi_emits_documents_in_java_hashmap_order_of_their_docids():
    """``MMI.genDocidEntityMap`` is a ``HashMap``; verified on a BioC collection where Java
    prints ``OWN`` before ``D1`` and ``D2`` although it was read last."""
    from mmlite.javautil import hash_set_order
    from mmlite.output.mmi import MMIFormatter
    from mmlite.types import ConceptInfo, Entity, Ev

    ci = ConceptInfo("C0004057", "aspirin", "aspirin", ("MSH",), frozenset({"phsu"}))
    ents = [
        Entity(d, "text", "aspirin", "NN", 0, 0, 7, evs=[Ev(ci, "aspirin", "aspirin", 0, 7)])
        for d in ("D1", "D2", "OWN")
    ]
    out = MMIFormatter().format(ents)
    order = [line.split("|")[0] for line in out.splitlines()]
    assert order == hash_set_order(["D1", "D2", "OWN"]) == ["OWN", "D1", "D2"]


def test_java_order_off_uses_input_and_alphabetical_order():
    """``Settings.strict_parity=False`` swaps every Java ``HashSet`` order for a readable one:
    documents in input order, semantic types and sources alphabetical, evidence by position."""
    from mmlite.javautil import hash_set_order

    sources = ("MTH", "LNC", "HL7V2.5", "MSH")
    sts = frozenset({"dsyn", "bacs", "aapp"})
    assert hash_set_order(sorted(sts)) != sorted(sts)  # else this test proves nothing
    assert hash_set_order(sources) != sorted(sources)
    ci = ConceptInfo("C0004057", "aspirin", "aspirin", sources, sts)
    ents = [
        Entity(d, "text", "aspirin", "NN", 0, 0, 7, evs=[Ev(ci, "aspirin", "aspirin", 0, 7)])
        for d in ("D1", "D2", "OWN")
    ]

    mmi = get_formatter("mmi", java_order=False).format(ents)
    assert [line.split("|")[0] for line in mmi.splitlines()] == ["D1", "D2", "OWN"]
    assert "[aapp, bacs, dsyn]" in mmi

    concept = json.loads(get_formatter("json", java_order=False).format(ents[:1]))[0]["evlist"][0]
    assert concept["conceptinfo"]["semantictypes"] == ["aapp", "bacs", "dsyn"]
    assert concept["conceptinfo"]["sources"] == sorted(sources)

    brat = get_formatter("brat", java_order=False).format(ents[:1])
    assert [ln.rsplit("\t", 1)[1] for ln in brat.splitlines() if "Source" in ln] == sorted(sources)

    # the default is still Java's order
    java = json.loads(get_formatter("json").format(ents[:1]))[0]["evlist"][0]["conceptinfo"]
    assert java["sources"] == hash_set_order(sources)


def _bc_entity(docid, text, start, score=0.0):
    from mmlite.types import ConceptInfo, Entity, Ev

    ci = ConceptInfo("C0004057", "aspirin", "aspirin", ("MSH",), frozenset({"phsu"}))
    return Entity(
        docid,
        "text",
        text,
        "NN",
        0,
        start,
        len(text),
        score,
        evs=[Ev(ci, text, text.lower(), start, len(text))],
    )


def test_bc_evaluate_format_is_docid_text_rank_score():
    """Verified byte-identical against Java's --outputformat=bc on two corpus documents."""
    from mmlite.output import get_formatter

    out = get_formatter("bc").format(
        [_bc_entity("d.txt", "aspirin", 0), _bc_entity("d.txt", "fever", 9)]
    )
    assert out == "d.txt\taspirin\t1\t0\nd.txt\tfever\t2\t0\n"


def test_bc_evaluate_is_registered_under_all_four_java_names():
    from mmlite.output import BcEvaluateFormatter, get_formatter

    for name in ("bc", "bc-evaluate", "cdi", "bioc"):
        assert isinstance(get_formatter(name), BcEvaluateFormatter)


def test_fulljson_is_an_alias_for_json():
    """Java registers both names to the same FullJson class."""
    from mmlite.output import JsonFormatter, get_formatter

    assert isinstance(get_formatter("fulljson"), JsonFormatter)


def test_java_number_format_matches_numberformat_getinstance():
    from mmlite.output.formats import java_number_format

    assert java_number_format(0.0) == "0"  # no minimum fraction digits
    assert java_number_format(666.6666) == "666.67"
    assert java_number_format(21000.0) == "21,000"  # grouping separators are on
    assert java_number_format(1234.567) == "1,234.57"


def test_bc_evaluate_ranks_continue_across_documents():
    from mmlite.output import get_formatter

    out = get_formatter("bc").format(
        [_bc_entity("a.txt", "aspirin", 0), _bc_entity("b.txt", "fever", 0)]
    )
    assert [line.split("\t")[2] for line in out.splitlines()] == ["1", "2"]
