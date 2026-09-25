"""Tests for the document model, the loader registry and the passage annotation path.

The contract these pin is the one every remaining loader (DEVELOPMENT_PLAN §10.3) will be built
against, so they are deliberately about *shape and offsets* rather than about which concepts a
particular text yields.
"""

from pathlib import Path

import pytest

from mmlite import documents
from mmlite.api import MetaMapLite
from mmlite.documents import (
    DEFAULT_DOCUMENT_ID,
    DEFAULT_PASSAGE_DOCID,
    FREETEXT_DOCID,
    Document,
    DocumentLoader,
    FreeTextLoader,
    Passage,
    get_loader,
    loader_names,
    register,
)
from mmlite.documents.bioc import BioCLoader
from mmlite.documents.corpora import (
    ChemDNERLoader,
    ChemDNERSLDILoader,
    NCBICorpusLoader,
    PubTatorLoader,
    remove_category_tags,
)
from mmlite.documents.freetext import java_basename
from mmlite.documents.pubmed import MedlineLoader, PubMedXMLLoader, java_trim
from mmlite.documents.singleline import (
    SLI_DOCID,
    SingleLineDelimitedWithIdLoader,
    SingleLineInputLoader,
    java_read_lines,
    java_split,
)

TEXT = "The patient has type 2 diabetes mellitus. Aspirin was prescribed."


@pytest.fixture(scope="module")
def mml(index_dir: Path):
    with MetaMapLite(index_directory=index_dir, tagger=None) as m:
        yield m


# --- model -----------------------------------------------------------------------------------


def test_passage_reports_section_and_docid_from_infons():
    p = Passage("abc", 10, {"docid": "d1", "section": "title"})
    assert (p.docid, p.section, p.end) == ("d1", "title", 13)


def test_passage_without_infons_has_no_section_or_docid():
    p = Passage("abc")
    assert p.section is None and p.docid is None


def test_text_at_resolves_an_absolute_span_against_a_shifted_passage():
    p = Passage("hello world", offset=100)
    assert p.text_at(106, 5) == "world"


def test_blank_or_missing_document_id_falls_back_to_the_java_default():
    assert Document().resolved_id() == DEFAULT_DOCUMENT_ID
    assert Document(id="   ").resolved_id() == DEFAULT_DOCUMENT_ID
    assert Document(id="real").resolved_id() == "real"


def test_with_docids_fills_passages_but_never_overwrites_one_that_is_set():
    doc = Document(id="D", passages=[Passage("a"), Passage("b", infons={"docid": "own"})])
    doc.with_docids()
    assert doc.infons["docid"] == "D"
    assert [p.docid for p in doc.passages] == ["D", "own"]


# --- registry --------------------------------------------------------------------------------


def test_freetext_is_registered_and_the_loader_satisfies_the_protocol():
    loader = get_loader("freetext")
    assert isinstance(loader, FreeTextLoader)
    assert isinstance(loader, DocumentLoader)
    assert "freetext" in loader_names()


def test_loader_names_are_matched_case_insensitively():
    assert get_loader("FreeText") is get_loader("freetext")


def test_an_unknown_input_format_names_the_registered_ones():
    with pytest.raises(ValueError, match="unknown input format 'nope'"):
        get_loader("nope")


def test_one_loader_may_hold_several_names_as_sli_and_sldi_do_in_java():
    loader = FreeTextLoader()
    register("probe-a", loader)
    register("probe-b", loader)
    try:
        assert get_loader("probe-a") is get_loader("probe-b") is loader
    finally:
        # The registry is process-global, as Java's static one is; there is no public
        # unregister, so reach into it rather than leak names into other tests.
        for name in ("probe-a", "probe-b"):
            documents._LOADERS.pop(name)


# --- FreeText loader, including the Java quirks it reproduces ---------------------------------


def test_read_sets_section_text_and_the_eight_zero_docid():
    (doc,) = FreeTextLoader().read(TEXT)
    assert doc.id == FREETEXT_DOCID
    (p,) = doc.passages
    assert (p.offset, p.section, p.docid) == (0, "text", FREETEXT_DOCID)
    assert p.infons["inputformat"] == "freetext"


def test_load_file_sets_no_section_matching_javas_filename_document_builder(tmp_path: Path):
    """Java's two-argument instantiateBioCDocument omits ``section``; ours must too.

    Confirmed against MetaMapLite 3.6.2rc8, whose ``--outputformat=json`` on a file emits no
    ``fieldid`` key at all.
    """
    f = tmp_path / "note.txt"
    f.write_text(TEXT, encoding="utf-8")
    (doc,) = FreeTextLoader().load_file(f)
    (p,) = doc.passages
    assert p.section is None
    # Java splits the filename on "/" only, so on Windows the docid is the whole path.
    assert p.docid == java_basename(str(f))
    assert doc.id == str(f)


def test_load_file_docid_is_the_basename_when_the_path_uses_forward_slashes(tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text(TEXT, encoding="utf-8")
    (doc,) = FreeTextLoader().load_file(f.as_posix())
    assert doc.passages[0].docid == "note.txt"


def test_load_file_docid_can_be_overridden(tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text(TEXT, encoding="utf-8")
    (doc,) = FreeTextLoader().load_file(f, docid="chosen")
    assert doc.passages[0].docid == "chosen"


def test_java_basename_splits_on_forward_slash_only():
    """Java's ``filename.split("/")`` leaves a Windows path whole — the quirk, reproduced."""
    assert java_basename("dir/sub/note.txt") == "note.txt"
    assert java_basename(r"C:\dir\note.txt") == r"C:\dir\note.txt"


def test_load_file_preserves_crlf_so_offsets_index_the_file_verbatim(tmp_path: Path):
    f = tmp_path / "crlf.txt"
    f.write_bytes(b"Line one.\r\nThe patient has type 2 diabetes mellitus.\r\n")
    (doc,) = FreeTextLoader().load_file(f)
    assert "\r\n" in doc.passages[0].text


def test_read_document_drops_a_utf8_bom_so_offsets_match_the_editor(tmp_path: Path):
    from mmlite import read_document

    f = tmp_path / "bom.txt"
    # what Notepad and PowerShell's Set-Content -Encoding utf8 both write
    f.write_bytes(b"\xef\xbb\xbfHeart attack.\r\n")
    text = read_document(f)
    assert text == "Heart attack.\r\n"  # BOM gone, CRLF kept
    assert read_document(f, encoding="utf-8-sig") == text  # and the same either way
    (tmp_path / "bom_only.txt").write_bytes(b"\xef\xbb\xbf\r\n")
    assert not read_document(tmp_path / "bom_only.txt").strip()  # an "empty" file reads as empty


# --- annotation through the document path -----------------------------------------------------


def test_process_document_finds_the_same_entities_as_process_text(mml):
    (doc,) = FreeTextLoader().read(TEXT)
    from_doc = mml.process_document(doc)
    from_text = mml.process_text(TEXT)
    assert [(e.start, e.length, e.text) for e in from_doc] == [
        (e.start, e.length, e.text) for e in from_text
    ]


def test_process_document_takes_docid_from_the_passage_infon(mml):
    (doc,) = FreeTextLoader().read(TEXT, docid="my-doc")
    entities = mml.process_document(doc)
    assert entities and {e.docid for e in entities} == {"my-doc"}


def test_a_document_without_ids_falls_back_through_javas_defaults(mml):
    doc = Document(passages=[Passage(TEXT, infons={"section": "text"})])
    entities = mml.process_document(doc)
    assert entities and {e.docid for e in entities} == {DEFAULT_DOCUMENT_ID}


def test_a_bare_passage_falls_back_to_the_passage_default_docid(mml):
    entities = mml.process_passage(Passage(TEXT))
    assert entities and {e.docid for e in entities} == {DEFAULT_PASSAGE_DOCID}


def test_fieldid_is_the_section_infon_and_may_be_none(mml):
    (with_section,) = FreeTextLoader().read(TEXT)
    assert {e.fieldid for e in mml.process_document(with_section)} == {"text"}

    doc = Document(id="d", passages=[Passage(TEXT)])
    assert {e.fieldid for e in mml.process_document(doc)} == {None}


def test_a_shifted_passage_reports_offsets_relative_to_its_source(mml):
    plain = mml.process_passage(Passage(TEXT))
    shifted = mml.process_passage(Passage(TEXT, offset=1000))
    assert [e.start for e in shifted] == [e.start + 1000 for e in plain]
    assert [e.text for e in shifted] == [e.text for e in plain]


def test_process_documents_concatenates_over_several_documents(mml):
    docs = FreeTextLoader().read(TEXT) + FreeTextLoader().read(TEXT, docid="second")
    entities = mml.process_documents(docs)
    assert {e.docid for e in entities} == {FREETEXT_DOCID, "second"}
    assert len(entities) == 2 * len(mml.process_text(TEXT))


def test_load_dispatches_through_the_registry(tmp_path: Path, mml):
    f = tmp_path / "note.txt"
    f.write_text(TEXT, encoding="utf-8")
    (doc,) = mml.load(f)
    assert doc.passages[0].docid == java_basename(str(f))
    with pytest.raises(ValueError, match="unknown input format"):
        mml.load(f, inputformat="semeval14")


# --- single-line loaders (sli / sldi / sldiwi) ------------------------------------------------

SLI_TEXT = "The patient has type 2 diabetes mellitus.\nAspirin was prescribed.\nNo chest pain."
SLDIWI_TEXT = "D1|The patient has type 2 diabetes mellitus.\nD2|Aspirin was prescribed."


def test_java_read_lines_splits_on_the_three_java_line_terminators():
    assert java_read_lines("a\nb\r\nc\rd") == ["a", "b", "c", "d"]


def test_java_read_lines_ignores_a_terminator_at_the_very_end():
    assert java_read_lines("a\nb\n") == ["a", "b"]
    assert java_read_lines("a\nb") == ["a", "b"]
    assert java_read_lines("\n") == [""]
    assert java_read_lines("") == []


def test_java_read_lines_does_not_break_where_str_splitlines_would():
    """``BufferedReader.readLine`` knows three terminators; ``str.splitlines`` knows many more."""
    text = "a\x0bb\x0cc d"
    assert java_read_lines(text) == [text]
    assert len("a\x0bb\x0cc d".splitlines()) == 4


def test_java_split_drops_trailing_empty_fields_as_java_does():
    assert java_split(r"\|", "id|text") == ["id", "text"]
    assert java_split(r"\|", "id|") == ["id"]  # re.split would give ["id", ""]
    assert java_split(r"\|", "|text") == ["", "text"]  # a leading empty is kept


def test_sli_and_sldi_are_the_same_loader_instance():
    assert get_loader("sli") is get_loader("sldi")


def test_sli_makes_one_document_per_line_each_at_offset_zero():
    docs = SingleLineInputLoader().read(SLI_TEXT)
    assert len(docs) == 3
    assert [d.passages[0].text for d in docs] == SLI_TEXT.split("\n")
    assert all(d.passages[0].offset == 0 for d in docs)


def test_sli_read_sets_no_passage_infons_unlike_its_file_path(tmp_path: Path):
    """The opposite asymmetry to FreeText, whose *read* path is the one that sets ``section``."""
    (doc,) = SingleLineInputLoader().read("Aspirin was prescribed.")
    assert doc.passages[0].infons == {}
    assert doc.id == SLI_DOCID

    f = tmp_path / "lines.txt"
    f.write_text("Aspirin was prescribed.", encoding="utf-8")
    (from_file,) = SingleLineInputLoader().load_file(f, docid="lines.txt")
    assert from_file.passages[0].section == "text"
    assert from_file.passages[0].infons["inputformat"] == "sli"


def test_sli_per_line_sequence_id_is_computed_then_discarded_exactly_as_java_does():
    """Java writes ``%08d.TX`` into the document infons; ``processDocument`` overwrites it."""
    docs = SingleLineInputLoader().read(SLI_TEXT)
    assert [d.infons["docid"] for d in docs] == ["00000000.TX", "00000001.TX", "00000002.TX"]
    for d in docs:
        d.with_docids()
    assert {d.infons["docid"] for d in docs} == {SLI_DOCID}


def test_sldiwi_splits_id_from_text():
    docs = SingleLineDelimitedWithIdLoader().read(SLDIWI_TEXT)
    assert [d.id for d in docs] == ["D1", "D2"]
    assert docs[0].passages[0].text == "The patient has type 2 diabetes mellitus."
    assert docs[0].passages[0].docid == "D1"
    assert docs[0].passages[0].section == "text"


def test_sldiwi_keeps_only_the_second_field_dropping_anything_after_it():
    """Confirmed against Java: ``D3|extra|more`` annotates ``extra`` and loses the rest."""
    (doc,) = SingleLineDelimitedWithIdLoader().read("D3|extra|delimiters|here")
    assert doc.passages[0].text == "extra"


def test_sldiwi_yields_an_empty_document_when_a_line_has_too_few_fields(caplog):
    docs = SingleLineDelimitedWithIdLoader().read("no delimiter here\nD5|")
    assert [d.passages for d in docs] == [[], []]
    assert "too few fields" in caplog.text


def test_sldiwi_delimiter_is_configurable():
    loader = SingleLineDelimitedWithIdLoader(delimiter=r"\t")
    (doc,) = loader.read("D1\tsome text")
    assert (doc.id, doc.passages[0].text) == ("D1", "some text")


def test_sldiwi_an_empty_document_contributes_no_entities(mml):
    docs = SingleLineDelimitedWithIdLoader().read("no delimiter here")
    assert mml.process_documents(docs) == []


def test_single_line_documents_report_line_relative_offsets(mml):
    """Every line sits at offset 0, so offsets restart per document — as Java's do."""
    docs = SingleLineInputLoader().read("Aspirin was prescribed.\nAspirin was prescribed.")
    first, second = (mml.process_document(d) for d in docs)
    assert [(e.start, e.text) for e in first] == [(e.start, e.text) for e in second]


# --- title/abstract corpus formats ------------------------------------------------------------

CHEM_TITLE = "Aspirin and heart attack."
CHEM_ABSTRACT = "The patient has type 2 diabetes mellitus."
CHEM_TEXT = f"D1\t{CHEM_TITLE}\t{CHEM_ABSTRACT}\nD2\t{CHEM_TITLE}\t{CHEM_ABSTRACT}\n"
SLDI_CHEM_TEXT = f"D1|{CHEM_TITLE}\t{CHEM_ABSTRACT}\n"
NCBI_TEXT = (
    f"D1\t{CHEM_TITLE}\tThe patient has "
    '<category="SpecificDisease">type 2 diabetes mellitus</category>.\n'
)
PUBTATOR_TEXT = (
    f"D1|t|{CHEM_TITLE}\nD1|a|{CHEM_ABSTRACT}\n"
    "D1\t0\t7\tAspirin\tChemical\tMESH:D001241\n"
    f"D2|t|{CHEM_TITLE}\nD2|a|{CHEM_ABSTRACT}\n"
)


def test_chemdner_builds_a_title_and_an_abstract_passage_both_at_offset_zero():
    docs = ChemDNERLoader().read(CHEM_TEXT)
    assert [d.id for d in docs] == ["D1", "D2"]
    title, abstract = docs[0].passages
    assert (title.text, title.section, title.offset) == (CHEM_TITLE, "title", 0)
    assert (abstract.text, abstract.section, abstract.offset) == (CHEM_ABSTRACT, "abstract", 0)
    assert title.docid == abstract.docid == "D1"


def test_chemdner_skips_a_line_with_too_few_fields(caplog):
    """Java indexes docFields[2] unchecked and dies mid-file; we warn and carry on."""
    docs = ChemDNERLoader().read(f"D1\tonly a title\nD2\t{CHEM_TITLE}\t{CHEM_ABSTRACT}\n")
    assert [d.id for d in docs] == ["D2"]
    assert "too few fields" in caplog.text


def test_ncbicorpus_strips_the_corpus_annotation_tags():
    (doc,) = NCBICorpusLoader().read(NCBI_TEXT)
    assert doc.passages[1].text == "The patient has type 2 diabetes mellitus."


def test_remove_category_tags_only_matches_javas_alphabetic_category_names():
    assert remove_category_tags('<category="Disease">x</category>') == "x"
    # Java's regex is [A-Za-z]+, so a digit or underscore in the name leaves the tag in place.
    assert remove_category_tags('<category="Modifier_2">x</category>') == '<category="Modifier_2">x'


def test_chemdnersldi_splits_the_id_on_a_pipe_and_the_body_on_a_tab():
    (doc,) = ChemDNERSLDILoader().read(SLDI_CHEM_TEXT)
    assert doc.id == "D1"
    assert [p.text for p in doc.passages] == [CHEM_TITLE, CHEM_ABSTRACT]
    assert [p.section for p in doc.passages] == ["title", "abstract"]


def test_chemdnersldi_skips_a_line_missing_either_delimiter(caplog):
    docs = ChemDNERSLDILoader().read("no pipe here\nD1|title but no tab\n")
    assert docs == []
    assert caplog.text.count("too few fields") == 2


def test_pubtator_groups_lines_into_one_document_per_record_id():
    docs = PubTatorLoader().read(PUBTATOR_TEXT)
    assert [d.id for d in docs] == ["D1", "D2"]
    assert [p.text for p in docs[0].passages] == [CHEM_TITLE, CHEM_ABSTRACT]


def test_pubtator_ignores_tab_delimited_annotation_lines():
    docs = PubTatorLoader().read(f"D1|t|{CHEM_TITLE}\nD1\t0\t7\tAspirin\tChemical\tMESH:D001241\n")
    assert len(docs) == 1 and docs[0].passages[0].text == CHEM_TITLE


def test_pubtator_puts_the_abstract_at_the_titles_length():
    """The one passage in any loader with a non-zero offset — Java's own formula."""
    (doc,) = PubTatorLoader().read(f"D1|t|{CHEM_TITLE}\nD1|a|{CHEM_ABSTRACT}\n")
    assert doc.passages[1].offset == len(CHEM_TITLE)


def test_pubtator_tolerates_a_record_with_only_a_title_or_only_an_abstract():
    docs = PubTatorLoader().read(f"D1|t|{CHEM_TITLE}\nD2|a|{CHEM_ABSTRACT}\n")
    assert [(d.id, d.passages[0].text, d.passages[1].text) for d in docs] == [
        ("D1", CHEM_TITLE, ""),
        ("D2", "", CHEM_ABSTRACT),
    ]


def test_title_passages_make_entities_that_mmi_ranks_as_titles(mml):
    """``section="title"`` reaches ``Entity.fieldid``, which is MMI's title flag.

    Verified against Java: a title concept scores 21000.00 where a body concept scores single
    digits, because ``Ranking`` drops the frequency factor for titles.
    """
    (doc,) = ChemDNERLoader().read(f"D1\t{CHEM_TITLE}\t{CHEM_ABSTRACT}\n")
    entities = mml.process_document(doc)
    assert {e.fieldid for e in entities} == {"title", "abstract"}
    scores = [
        float(line.split("|")[2])
        for line in mml.format(entities, "mmi").splitlines()
        if "|MMI|" in line
    ]
    assert max(scores) > 1000  # the title branch fired


# --- PubMed XML and MEDLINE ------------------------------------------------------------------


def _article(pmid, title, *abstracts, cited=None):
    body = "".join(
        f'<AbstractText Label="{label}">{text}</AbstractText>'
        if label
        else f"<AbstractText>{text}</AbstractText>"
        for label, text in abstracts
    )
    refs = (
        f"<CommentsCorrectionsList><CommentsCorrections><PMID>{cited}</PMID>"
        "</CommentsCorrections></CommentsCorrectionsList>"
        if cited
        else ""
    )
    return (
        f"<PubmedArticle><MedlineCitation><PMID Version='1'>{pmid}</PMID><Article>"
        f"<ArticleTitle>{title}</ArticleTitle><Abstract>{body}</Abstract></Article>{refs}"
        "</MedlineCitation></PubmedArticle>"
    )


def _set(*articles):
    return (
        '<?xml version="1.0" encoding="UTF-8"?><PubmedArticleSet>'
        + "".join(articles)
        + ("</PubmedArticleSet>")
    )


PUBMED_XML = _set(
    _article("111", CHEM_TITLE, (None, CHEM_ABSTRACT)),
    _article("222", CHEM_TITLE, ("BACKGROUND", "Aspirin was given."), ("RESULTS", CHEM_ABSTRACT)),
)
MEDLINE_TEXT = (
    "PMID- 111\nOWN - NLM\nTI  - Aspirin and heart\n      attack.\nAU  - Smith J\n"
    f"AB  - {CHEM_ABSTRACT}\n\nPMID- 222\nTI  - {CHEM_TITLE}\nAB  - Aspirin was given.\n"
)


def test_pubmed_makes_one_document_per_article_with_title_and_abstract_passages():
    docs = PubMedXMLLoader().read(PUBMED_XML)
    assert [d.id for d in docs] == ["111", "222"]
    title, abstract = docs[0].passages
    assert (title.section, title.text, title.offset) == ("title", CHEM_TITLE, 0)
    assert (abstract.section, abstract.text, abstract.offset) == ("abstract", CHEM_ABSTRACT, 0)


def test_pubmed_structured_abstract_is_one_passage_per_abstracttext():
    docs = PubMedXMLLoader().read(PUBMED_XML)
    assert [p.section for p in docs[1].passages] == ["title", "abstract", "abstract"]


def test_pubmed_passages_keep_the_pmid_current_when_they_were_created():
    """A cited PMID after the abstract re-ids the document but not its passages — as in Java."""
    (doc,) = PubMedXMLLoader().read(_set(_article("111", CHEM_TITLE, (None, "x"), cited="999")))
    assert doc.id == "999"
    assert {p.docid for p in doc.passages} == {"111"}


def test_pubmed_extracts_text_through_inline_markup():
    """Java's getElementText() throws on child elements and aborts the whole run."""
    (doc,) = PubMedXMLLoader().read(
        _set(_article("1", "T", (None, "Patients with <i>type 2</i> diabetes<sup>1</sup>.")))
    )
    assert doc.passages[1].text == "Patients with type 2 diabetes1."


def test_pubmed_decodes_character_entities():
    (doc,) = PubMedXMLLoader().read(_set(_article("1", "Aspirin &amp; heart attack", (None, "a"))))
    assert doc.passages[0].text == "Aspirin & heart attack"


def test_pubmed_drops_content_before_the_first_pubmedarticle():
    """Java's initial document is never added to its list."""
    xml = "<PubmedArticleSet><PMID>0</PMID><ArticleTitle>lost</ArticleTitle></PubmedArticleSet>"
    assert PubMedXMLLoader().read(xml) == []


def test_pubmed_malformed_xml_names_the_source(tmp_path: Path):
    f = tmp_path / "bad.xml"
    f.write_text("<PubmedArticleSet><PubmedArticle>", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.xml: not well-formed XML"):
        PubMedXMLLoader().load_file(f)


def test_java_trim_strips_control_characters_but_not_unicode_spaces():
    assert java_trim("\x00\t abc \x1f") == "abc"
    assert java_trim(" abc ") == " abc "  # NBSP, EM SPACE are above U+0020


def test_medline_joins_continuation_lines_and_keeps_only_pmid_ti_ab():
    docs = MedlineLoader().read(MEDLINE_TEXT)
    assert [d.id for d in docs] == ["111", "222"]
    title, abstract = docs[0].passages
    assert title.text == "Aspirin and heart attack. "  # each line gets a trailing space
    assert abstract.text == CHEM_ABSTRACT + " "
    assert [p.section for p in docs[0].passages] == ["TI", "AB"]
    assert all(p.offset == 0 for d in docs for p in d.passages)


def test_medline_a_trailing_blank_line_adds_an_empty_document_repeating_the_last_pmid():
    """Java resets title and abstract at a blank line but not the PMID, then always emits once
    more at end of input."""
    docs = MedlineLoader().read(MEDLINE_TEXT + "\n")
    assert [d.id for d in docs] == ["111", "222", "222"]
    assert [p.text for p in docs[2].passages] == ["", ""]


def test_medline_record_without_a_pmid_reports_an_empty_docid_not_the_default(mml):
    """Java tests the docid infon for null, so "" survives rather than becoming "00000000"."""
    (doc,) = MedlineLoader().read(f"TI  - {CHEM_TITLE}\n")
    entities = mml.process_document(doc)
    assert entities and {e.docid for e in entities} == {""}


def test_medline_tolerates_lines_too_short_for_javas_substring_calls():
    """Java's substring(6) throws on "TI"; here it is a TI line with an empty value, which gets
    the same trailing space as any other."""
    docs = MedlineLoader().read("PMID- 1\nTI\nTI  - Aspirin.\n")
    assert docs[0].passages[0].text == " Aspirin. "


def test_medline_ti_is_ranked_as_a_title_by_mmi(mml):
    (doc,) = MedlineLoader().read(f"PMID- 1\nTI  - {CHEM_TITLE}\nAB  - {CHEM_ABSTRACT}\n")
    rows = [line.split("|") for line in mml.format(mml.process_document(doc), "mmi").splitlines()]
    assert max(float(r[2]) for r in rows if len(r) > 2) > 1000


# --- BioC XML --------------------------------------------------------------------------------

BIOC_XML = (
    '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE collection SYSTEM "BioC.dtd">'
    "<collection><source>t</source><date>d</date><key>k</key>"
    "<document><id>D1</id><infon key='note'>n</infon>"
    f"<passage><infon key='type'>title</infon><offset>0</offset><text>{CHEM_TITLE}</text></passage>"
    "<passage><infon key='section'>abstract</infon><offset>34</offset>"
    f"<text>{CHEM_ABSTRACT}</text></passage></document>"
    "<document><id>D2</id><passage><infon key='docid'>OWN</infon><offset>100</offset>"
    "<text>Aspirin &amp; heart attack.</text></passage></document>"
    "</collection>"
)


def test_bioc_maps_documents_passages_and_infons_onto_the_model():
    docs = BioCLoader().read(BIOC_XML)
    assert [d.id for d in docs] == ["D1", "D2"]
    assert docs[0].infons == {"note": "n"}
    title, abstract = docs[0].passages
    assert title.infons == {"type": "title"} and title.section is None  # "type" is not "section"
    assert abstract.section == "abstract" and abstract.text == CHEM_ABSTRACT


def test_bioc_passage_offsets_are_ignored_as_java_ignores_them():
    """A passage at <offset>100</offset> reports its concepts from 0 in Java (§16)."""
    docs = BioCLoader().read(BIOC_XML)
    assert [p.offset for d in docs for p in d.passages] == [0, 0, 0]


def test_bioc_docid_infon_in_the_file_reaches_the_entity(mml):
    docs = BioCLoader().read(BIOC_XML)
    assert {e.docid for e in mml.process_document(docs[1])} == {"OWN"}


def test_bioc_decodes_entities_and_tolerates_dtd_and_missing_header():
    docs = BioCLoader().read(BIOC_XML)
    assert docs[1].passages[0].text == "Aspirin & heart attack."
    bare = (
        "<collection><document><id>X</id><passage><text>a</text></passage></document></collection>"
    )
    assert BioCLoader().read(bare)[0].id == "X"


def test_bioc_sentence_elements_become_passages_at_their_own_offsets():
    """Verified against Java: sentence offsets in the file are used, passage offsets are not."""
    xml = (
        "<collection><document><id>S</id><passage><offset>50</offset>"
        "<sentence><offset>60</offset><text>Aspirin was given.</text></sentence>"
        "<sentence><offset>80</offset><text>No chest pain.</text></sentence>"
        "</passage></document></collection>"
    )
    (doc,) = BioCLoader().read(xml)
    assert [(p.offset, p.text) for p in doc.passages] == [
        (60, "Aspirin was given."),
        (80, "No chest pain."),
    ]


def test_bioc_passage_with_text_and_sentences_yields_both():
    xml = (
        "<collection><document><id>B</id><passage><infon key='section'>s</infon><offset>0</offset>"
        "<text>Aspirin was given.</text>"
        "<sentence><offset>200</offset><text>No chest pain.</text></sentence>"
        "</passage></document></collection>"
    )
    (doc,) = BioCLoader().read(xml)
    assert [(p.offset, p.text, p.section) for p in doc.passages] == [
        (0, "Aspirin was given.", "s"),
        (200, "No chest pain.", "s"),
    ]


def test_bioc_accepts_a_lone_document_root():
    xml = "<document><id>L</id><passage><text>x</text></passage></document>"
    assert [d.id for d in BioCLoader().read(xml)] == ["L"]


def test_bioc_malformed_xml_names_the_source(tmp_path: Path):
    f = tmp_path / "bad.xml"
    f.write_text("<collection><document>", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.xml: not well-formed XML"):
        BioCLoader().load_file(f)


# --- the offset invariant every loader must satisfy -------------------------------------------


@pytest.mark.parametrize(
    "passages",
    [
        pytest.param([Passage(TEXT, 0, {"section": "text"})], id="single"),
        pytest.param([Passage(TEXT, 1000, {"section": "text"})], id="shifted"),
        pytest.param(
            [Passage("Aspirin was prescribed.", 0, {"section": "title"}), Passage(TEXT, 23)],
            id="two-passages-contiguous",
        ),
        pytest.param(
            # Java's PubMedXMLDocument gives every passage offset 0; entity offsets are then
            # passage-relative and no single source string indexes them all.
            [Passage("Aspirin was prescribed.", 0), Passage(TEXT, 0)],
            id="two-passages-both-at-zero",
        ),
        pytest.param([Passage("Line one.\r\n" + TEXT, 0)], id="crlf"),
    ],
)
def test_every_entity_offset_indexes_its_own_passage_text(mml, passages):
    """``passage.text_at(e.start, e.length) == e.text`` — for every passage, every loader.

    This is the check that would have caught both offset bugs in DEVELOPMENT_PLAN §6, and it is
    the one contract a new loader cannot be allowed to break.
    """
    doc = Document(id="d", passages=passages)
    doc.with_docids()
    for passage in doc.passages:
        entities = mml.process_passage(passage)
        assert entities, "expected this fixture to produce entities worth checking"
        for e in entities:
            assert passage.text_at(e.start, e.length) == e.text


# A sample per registered input format, used to hold every loader to the offset invariant.
LOADER_SAMPLES = {
    "freetext": TEXT,
    "sli": SLI_TEXT,
    "sldi": SLI_TEXT,
    "sldiwi": SLDIWI_TEXT,
    "chemdner": CHEM_TEXT,
    "chemdnersldi": SLDI_CHEM_TEXT,
    "ncbicorpus": NCBI_TEXT,
    "pubtator": PUBTATOR_TEXT,
    "pubmed": PUBMED_XML,
    "medline": MEDLINE_TEXT,
    "bioc": BIOC_XML,
}


def test_every_registered_loader_has_an_offset_invariant_sample():
    """Adding a loader without adding a sample below fails here rather than going unchecked."""
    assert set(loader_names()) == set(LOADER_SAMPLES), (
        f"no offset-invariant sample for {sorted(set(loader_names()) - set(LOADER_SAMPLES))}"
    )


@pytest.mark.parametrize("name", sorted(LOADER_SAMPLES))
def test_each_loader_produces_offsets_that_index_their_own_passage(mml, tmp_path, name):
    f = tmp_path / "input.txt"
    f.write_text(LOADER_SAMPLES[name], encoding="utf-8")
    documents = get_loader(name).load_file(f, docid="input.txt")
    checked = 0
    for document in documents:
        document.with_docids()
        for passage in document.passages:
            for e in mml.process_passage(passage):
                assert passage.text_at(e.start, e.length) == e.text
                checked += 1
    assert checked, f"{name}: sample produced no entities, so nothing was actually checked"


def test_process_file_field_id_default_and_java_behaviour(tmp_path: Path, mml):
    """``process_file`` defaults to ``fieldid="text"``; ``None`` reproduces Java and the CLI.

    Java's free-text *file* loader sets no ``section`` infon, so its entities carry no field id
    (DEVELOPMENT_PLAN §12).  The default here differs deliberately, to keep existing callers'
    output stable; §21 records the decision.
    """
    f = tmp_path / "note.txt"
    f.write_text(TEXT, encoding="utf-8")

    assert {e.fieldid for e in mml.process_file(f)} == {"text"}
    assert {e.fieldid for e in mml.process_file(f, fieldid=None)} == {None}

    # the document path -- what `annotate --input` uses -- agrees with Java without being asked
    (doc,) = mml.load(f, docid=f.name)
    assert {e.fieldid for e in mml.process_document(doc)} == {None}
