import os
from pathlib import Path

import pytest

from mmlite.index import IndexLookup, build_index
from mmlite.index.schema import DB_FILENAME
from mmlite.normalize import NORM_VERSION


def test_build_writes_db_and_meta(index_dir: Path):
    assert (index_dir / DB_FILENAME).exists()
    with IndexLookup(index_dir) as ix:
        assert ix.meta["norm_version"] == NORM_VERSION
        stats = ix.stats()
        assert stats["rows_cuiconcept"] == 7
        # 18 MRCONSO rows - 1 FRE - 1 suppressed(O) = 16
        assert stats["rows_cuisourceinfo"] == 16
        assert stats["rows_cuist"] == 8
        # every MSH string x its CUI's MN codes: DM (2 strings x 2 codes) + MI (2 x 1)
        # + Aspirin (1 x 1)
        # + "Diabetes Mellitus, Type 2"/"Type 2 Diabetes"/"Alzheimer Disease"/"Parkinson's Disease"
        # (4 x x.x.x.x, no MRSAT MN code) = 11
        assert stats["rows_meshtcrelaxed"] == 11


def test_preferred_name_prefers_pf_row(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        assert ix.preferred_name("C0011860") == "Diabetes Mellitus, Type 2"
        assert ix.preferred_name("C0004057") == "Aspirin"  # only one P/PF row
        assert ix.preferred_name("C0027051") == "Myocardial Infarction"
        assert ix.preferred_name("C9999999") == "C9999999"  # unknown -> echo cui


def test_lookup_is_case_insensitive_but_order_sensitive(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        hits = ix.lookup("type 2 DIABETES mellitus")
        assert {h.cui for h in hits} == {"C0011860"}
        assert all(h.semantic_types == ("dsyn",) for h in hits)
        assert {h.sab for h in hits} == {"SNOMEDCT_US"}  # only SNOMED has this word order
        assert {h.sab for h in ix.lookup("Type 2 Diabetes")} == {"MSH"}
        assert ix.lookup("mellitus diabetes") == []


def test_lookup_strips_possessives_and_greek(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        # MRCONSO has "Alzheimer Disease"; query with possessive matches via normalize()
        assert {h.cui for h in ix.lookup("Alzheimer's Disease")} == {"C0002395"}
        # Greek text matches only via the ASCII spelling (Java keys the Greek string as "tnf-α")
        assert {h.matched_string for h in ix.lookup("TNF-α")} == {"TNF-alpha"}
        assert ix.cuis_for_key("tnf-α") == ("C1456820",)  # the verbatim lowercase key exists too
        assert ix.cuis_for_term("alzheimer's disease") == ("C0002395",)


def test_lookup_suppressed_rows_excluded_by_default(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        assert ix.lookup("NIDDM (obsolete)") == []


def test_lookup_non_english_excluded(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        assert ix.lookup("Infarctus du myocarde") == []


def test_lookup_restrictions(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        assert {h.sab for h in ix.lookup("aspirin", sources={"MSH"})} == {"MSH"}
        assert ix.lookup("aspirin", semantic_types={"dsyn"}) == []
        assert {h.cui for h in ix.lookup("aspirin", semantic_types={"phsu"})} == {"C0004057"}
        assert {h.cui for h in ix.lookup("aspirin", semantic_types={"T121"})} == {"C0004057"}


def test_lookup_restrictions_are_case_insensitive_and_accept_all(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        unrestricted = ix.lookup("aspirin")
        for sts in ({"PHSU"}, {"t121"}):
            assert {h.cui for h in ix.lookup("aspirin", semantic_types=sts)} == {"C0004057"}
        assert {h.sab for h in ix.lookup("aspirin", sources={"msh"})} == {"MSH"}
        assert ix.lookup("aspirin", semantic_types={"all"}) == unrestricted
        assert ix.lookup("aspirin", sources={"ALL"}) == unrestricted


def test_semantic_types_and_sources(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        assert ix.semantic_types("C0004057") == ("orch", "phsu")
        assert ix.sources("C0011849") == ("MSH", "SNOMEDCT_US")


def test_sources_are_in_mrconso_order_not_alphabetical(tmp_path: Path):
    """Java fills a ``HashSet`` from the postings, so their order decides the printed order.

    ``MTH`` and ``LNC`` collide in hash bucket 0, so alphabetising here would silently swap them
    in ``json`` and ``brat`` output (see ``IndexLookup.sources`` and ``output.formats._srcs``).
    """
    row = "C9000001|ENG|{ts}|L1|{stt}|S{n}|Y|A{n}|||X|{sab}|PT|C1|{s}|0|N||"
    mrconso = "\n".join(
        [
            row.format(ts="P", stt="PF", n=1, sab="MTH", s="Widget"),
            row.format(ts="S", stt="VC", n=2, sab="LNC", s="Widget thing"),
            row.format(ts="S", stt="VC", n=3, sab="AOD", s="Widget device"),
            # A repeat of an earlier SAB must not move it to the back.
            row.format(ts="S", stt="VC", n=4, sab="MTH", s="Widget item"),
        ]
    )
    (tmp_path / "MRCONSO.RRF").write_text(mrconso + "\n", encoding="utf-8")
    (tmp_path / "MRSTY.RRF").write_text("C9000001|T073|A1|Manufactured Object|AT1||\n", "utf-8")
    out = tmp_path / "ivf"
    build_index(mrconso=tmp_path / "MRCONSO.RRF", mrsty=tmp_path / "MRSTY.RRF", out_dir=out)
    with IndexLookup(out) as ix:
        assert ix.sources("C9000001") == ("MTH", "LNC", "AOD")


def test_mesh_treecodes(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        assert set(ix.mesh_treecodes("Diabetes Mellitus")) == {"C18.452.394.750", "C19.246"}
        assert set(ix.mesh_treecodes("diabetes")) == {
            "C18.452.394.750",
            "C19.246",
        }  # every MSH string
        assert ix.mesh_treecodes("Heart Attack") == ("C14.280.647.500",)
        assert ix.mesh_treecodes("Type 2 Diabetes") == ("x.x.x.x",)  # MSH string, CUI has no MN
        assert ix.mesh_treecodes("Aspirin (product)") == ()  # SNOMED string


def test_in_memory_map_matches_sql(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        sql_cuis = ix.cuis_for_key("heart attack")
        ix.load_in_memory()
        assert ix.cuis_for_key("heart attack") == sql_cuis == ("C0027051",)
        assert ix.cuis_for_key("Heart Attack") == ()  # keys are lowercase; use cuis_for_term
        assert ix.has_term("Heart attack")
        assert not ix.has_term("no such term")


def test_term_map_cache_survives_losing_the_write_race(umls_dir: Path, tmp_path: Path):
    """Two preloading worker processes race to write termmap.pkl.  Simulate the loser: the cache
    already exists and (on Windows) is held open by another process, so the rename onto it
    fails.  That must not raise, must leave the in-memory map usable, and must not litter the
    index directory with temp files."""
    from mmlite.index.schema import TERMMAP_FILENAME

    build_index(umls_dir / "MRCONSO.RRF", umls_dir / "MRSTY.RRF", tmp_path)
    pkl = tmp_path / TERMMAP_FILENAME
    with IndexLookup(tmp_path) as ix:
        ix.load_in_memory()  # first process: writes the cache
        assert pkl.exists()
        # Make the cache look stale so the next load rebuilds and tries to replace it, while a
        # handle on the file stands in for the process that is still reading it.
        os.utime(pkl, (0, 0))
        with open(pkl, "rb"):
            ix.load_in_memory()
        assert ix.in_memory
        assert ix.cuis_for_key("heart attack") == ("C0027051",)
    assert not list(tmp_path.glob("*.tmp"))


def test_build_refuses_overwrite_without_flag(index_dir: Path, umls_dir: Path):
    with pytest.raises(FileExistsError):
        build_index(umls_dir / "MRCONSO.RRF", umls_dir / "MRSTY.RRF", index_dir)


def test_build_with_source_filter_and_suppressed(umls_dir: Path, tmp_path: Path):
    build_index(
        umls_dir / "MRCONSO.RRF",
        umls_dir / "MRSTY.RRF",
        tmp_path,
        sources={"SNOMEDCT_US"},
        include_suppressed=True,
    )
    with IndexLookup(tmp_path) as ix:
        assert ix.stats()["rows_cuisourceinfo"] == 8  # 8 SNOMEDCT_US rows incl. the obsolete one
        assert {h.cui for h in ix.lookup("NIDDM (obsolete)")} == {"C0011860"}
        assert ix.lookup("Heart Attack") == []  # MSH-only string


def test_mesh_treecodes_survive_a_source_filter_that_matches_nothing(
    umls_dir: Path, tmp_path: Path
):
    # MSH strings are collected from rows the source filter rejects, so they must be flushed
    # even when no cuisourceinfo batch is ever written.
    build_index(
        umls_dir / "MRCONSO.RRF",
        umls_dir / "MRSTY.RRF",
        tmp_path,
        mrsat=umls_dir / "MRSAT.RRF",
        sources={"NOSUCHSAB"},
    )
    with IndexLookup(tmp_path) as ix:
        assert ix.stats()["rows_cuisourceinfo"] == 0
        assert ix.stats()["rows_meshtcrelaxed"] == 11
        assert set(ix.mesh_treecodes("Diabetes Mellitus")) == {"C18.452.394.750", "C19.246"}


def test_cuis_for_keys_matches_per_key_lookups(index_dir: Path):
    """The batched path must agree with the per-key one, including on misses."""
    keys = ["diabetes", "aspirin", "not a term at all", "heart attack", ""]
    with IndexLookup(index_dir) as ix:
        one_at_a_time = {k: ix.cuis_for_key(k) for k in keys}
        batched = ix.cuis_for_keys(keys)
        # keys with no concepts are simply absent from the batched result
        assert batched == {k: v for k, v in one_at_a_time.items() if v}
        assert ix.cuis_for_keys([]) == {}


def test_cuis_for_keys_chunks_past_the_sqlite_parameter_limit(index_dir: Path):
    """A long sentence can ask about more keys than SQLite allows in one statement."""
    from mmlite.index.lookup import _MAX_KEYS_PER_QUERY

    with IndexLookup(index_dir) as ix:
        filler = [f"nonexistent term {i}" for i in range(_MAX_KEYS_PER_QUERY * 2 + 3)]
        result = ix.cuis_for_keys([*filler, "aspirin"])
        assert result == {"aspirin": ix.cuis_for_key("aspirin")}


def test_cuis_for_keys_agrees_between_sqlite_and_the_preloaded_term_map(index_dir: Path):
    keys = ["diabetes", "aspirin", "missing", "alzheimer disease"]
    with IndexLookup(index_dir) as ix:
        from_sqlite = ix.cuis_for_keys(keys)
        ix.load_in_memory(cache=False)
        assert ix.in_memory
        assert ix.cuis_for_keys(keys) == from_sqlite


def test_concept_strings_lists_every_string_for_a_key_and_cui(index_dir: Path):
    with IndexLookup(index_dir) as ix:
        assert ix.concept_strings("diabetes mellitus", "C0011849") == ["Diabetes Mellitus"]
        assert ix.concept_strings("diabetes mellitus", "C0027051") == []
