import json
from pathlib import Path

from typer.testing import CliRunner

from mmlite.cli import app

runner = CliRunner()


def test_build_index_cli(umls_dir: Path, tmp_path: Path):
    out = tmp_path / "ivf"
    r = runner.invoke(
        app,
        [
            "build-index",
            "--mrconso",
            str(umls_dir / "MRCONSO.RRF"),
            "--mrsty",
            str(umls_dir / "MRSTY.RRF"),
            "--mrsat",
            str(umls_dir / "MRSAT.RRF"),
            "--out",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "index written to" in r.output


def test_lookup_cli_table_and_json(index_dir: Path):
    r = runner.invoke(app, ["lookup", "--indexdir", str(index_dir), "heart attack"])
    assert r.exit_code == 0, r.output
    assert "C0027051" in r.output and "Myocardial Infarction" in r.output

    r = runner.invoke(
        app,
        ["lookup", "--indexdir", str(index_dir), "--json", "aspirin", "--restrict-to-sts", "phsu"],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert {d["cui"] for d in data} == {"C0004057"}


def test_lookup_cli_no_match_exits_1(index_dir: Path):
    r = runner.invoke(app, ["lookup", "--indexdir", str(index_dir), "zzz not a term"])
    assert r.exit_code == 1
    assert "no matches" in r.output


def test_index_stats_cli(index_dir: Path):
    r = runner.invoke(app, ["index-stats", "--indexdir", str(index_dir)])
    assert r.exit_code == 0, r.output
    assert "rows_cuisourceinfo: 16" in r.output


def test_annotate_cli_formats(index_dir: Path):
    base = ["annotate", "--indexdir", str(index_dir), "--no-postag"]
    text = "Heart Attack and aspirin."
    r = runner.invoke(app, base + [text])
    assert r.exit_code == 0, r.output
    lines = r.output.splitlines()
    assert all(line.startswith("00000000.tx|MMI|") and line.endswith("|") for line in lines)
    assert any("|C0027051|[dsyn]|" in line for line in lines)

    r = runner.invoke(app, base + ["--outputformat", "json", "--restrict-to-sts", "dsyn", text])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert [d["matchedtext"] for d in data] == ["Heart Attack"]
    assert data[0]["evlist"][0]["conceptinfo"]["cui"] == "C0027051"

    r = runner.invoke(app, base + ["--outputformat", "brat", text])
    assert r.exit_code == 0, r.output
    assert r.output.startswith("T1\tMMLite 0 12\tHeart Attack\n")

    r = runner.invoke(app, base + ["--outputformat", "cuilist", text])
    assert r.output.splitlines() == ["C0027051", "C0004057"]

    r = runner.invoke(app, base + ["--outputformat", "nope", text])
    assert r.exit_code != 0

    # a typo'd semantic type is refused, not silently matched against nothing
    r = runner.invoke(app, base + ["--restrict-to-sts", "dysn,all,T047", text])
    assert r.exit_code == 1
    assert "unknown semantic type(s) in --restrict-to-sts: dysn" in r.output
    assert "T047" not in r.output.split("--restrict-to-sts:")[1]  # only the bad one is named


def test_annotate_cli_pipe_and_file(index_dir: Path, tmp_path: Path):
    base = ["annotate", "--indexdir", str(index_dir), "--no-postag"]
    r = runner.invoke(app, base + ["--pipe"], input="aspirin\n")
    assert r.exit_code == 0, r.output
    assert r.output.startswith("00000000.tx|MMI|") and "C0004057" in r.output

    f = tmp_path / "note1.txt"
    f.write_text("No heart attack.", encoding="utf-8")
    r = runner.invoke(app, base + ["--input", str(f)])
    assert r.exit_code == 0, r.output
    assert r.output.startswith("note1.txt|MMI|")
    assert '"Heart Attack"-text-0-"heart attack"--1|' in r.output  # negated, untagged POS ""


def test_annotate_cli_offsets_index_a_crlf_file_verbatim(index_dir: Path, tmp_path: Path):
    """Universal newlines would drop one char per line and shift every later offset."""
    f = tmp_path / "crlf.txt"
    raw = "No heart attack.\r\nAspirin was given.\r\n"
    f.write_bytes(raw.encode("utf-8"))
    r = runner.invoke(
        app,
        [
            "annotate",
            "--indexdir",
            str(index_dir),
            "--no-postag",
            "--outputformat",
            "brat",
            "--segmentation",
            "LINES",
            "--input",
            str(f),
        ],
    )
    assert r.exit_code == 0, r.output
    spans = [line.split("\t") for line in r.output.splitlines() if line.startswith("T")]
    assert spans, r.output
    for _tid, type_and_offsets, text in spans:
        _type, start, end = type_and_offsets.split()
        assert raw[int(start) : int(end)] == text
    assert any(text == "Aspirin" for *_rest, text in spans)


def test_annotate_cli_inputformat_sli_makes_one_document_per_line(index_dir: Path, tmp_path: Path):
    """Every line is its own document, and all of them share the file's docid — as Java's do."""
    f = tmp_path / "lines.txt"
    f.write_text("No heart attack.\naspirin\n", encoding="utf-8")
    r = runner.invoke(
        app,
        [
            "annotate",
            "--indexdir",
            str(index_dir),
            "--no-postag",
            "--inputformat",
            "sli",
            "--outputformat",
            "cuilist",
            "--input",
            str(f),
        ],
    )
    assert r.exit_code == 0, r.output
    assert set(r.output.split()) == {"C0027051", "C0004057"}


def test_annotate_cli_inputformat_sldiwi_takes_the_docid_from_each_line(
    index_dir: Path, tmp_path: Path
):
    f = tmp_path / "withid.txt"
    f.write_text("DOC-A|No heart attack.\nDOC-B|aspirin\n", encoding="utf-8")
    r = runner.invoke(
        app,
        [
            "annotate",
            "--indexdir",
            str(index_dir),
            "--no-postag",
            "--inputformat",
            "sldiwi",
            "--input",
            str(f),
        ],
    )
    assert r.exit_code == 0, r.output
    docids = {line.split("|")[0] for line in r.output.splitlines() if "|MMI|" in line}
    assert docids == {"DOC-A", "DOC-B"}


def test_annotate_cli_rejects_an_unknown_inputformat(index_dir: Path, tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text("aspirin", encoding="utf-8")
    r = runner.invoke(
        app,
        ["annotate", "--indexdir", str(index_dir), "--inputformat", "semeval14", "--input", str(f)],
    )
    assert r.exit_code != 0
    assert "unknown input format 'semeval14'" in r.output


def test_annotate_cli_sldiwi_with_no_usable_line_reports_no_text(index_dir: Path, tmp_path: Path):
    """Every line lacking a delimiter yields empty documents, so there is nothing to annotate."""
    f = tmp_path / "bad.txt"
    f.write_text("no delimiter here\nnor here\n", encoding="utf-8")
    r = runner.invoke(
        app,
        ["annotate", "--indexdir", str(index_dir), "--inputformat", "sldiwi", "--input", str(f)],
    )
    assert r.exit_code != 0
    assert "no text given" in r.output


def test_annotate_cli_inputformat_chemdner_ranks_title_concepts_as_titles(
    index_dir: Path, tmp_path: Path
):
    f = tmp_path / "chem.txt"
    f.write_text("D1\tAspirin and heart attack.\tThe patient has diabetes.\n", encoding="utf-8")
    r = runner.invoke(
        app,
        [
            "annotate",
            "--indexdir",
            str(index_dir),
            "--no-postag",
            "--inputformat",
            "chemdner",
            "--input",
            str(f),
        ],
    )
    assert r.exit_code == 0, r.output
    rows = [line.split("|") for line in r.output.splitlines() if "|MMI|" in line]
    assert {row[0] for row in rows} == {"D1"}
    fields = {row[4]: row[7] for row in rows}
    assert fields["C0004057"] == "title" and fields["C0011849"] == "abstract"
    assert float(next(row[2] for row in rows if row[4] == "C0004057")) > 1000


def test_annotate_cli_inputformat_pubtator_groups_by_record(index_dir: Path, tmp_path: Path):
    f = tmp_path / "pub.txt"
    f.write_text(
        "P1|t|Aspirin trial.\nP1|a|No heart attack.\nP1\t0\t7\tAspirin\tChemical\tX\n"
        "P2|t|Diabetes study.\nP2|a|Aspirin given.\n",
        encoding="utf-8",
    )
    r = runner.invoke(
        app,
        [
            "annotate",
            "--indexdir",
            str(index_dir),
            "--no-postag",
            "--inputformat",
            "pubtator",
            "--input",
            str(f),
        ],
    )
    assert r.exit_code == 0, r.output
    assert {line.split("|")[0] for line in r.output.splitlines() if "|MMI|" in line} == {
        "P1",
        "P2",
    }


def test_annotate_cli_malformed_pubmed_xml_is_a_one_line_error(index_dir: Path, tmp_path: Path):
    f = tmp_path / "bad.xml"
    f.write_text("<PubmedArticleSet><PubmedArticle>", encoding="utf-8")
    r = runner.invoke(
        app,
        ["annotate", "--indexdir", str(index_dir), "--inputformat", "pubmed", "--input", str(f)],
    )
    assert r.exit_code != 0
    assert "not well-formed XML" in r.output
    assert "Traceback" not in r.output


# --- start-up failures suggest --no-postag only when that would actually help ------------------
#
# A missing model and a missing index are both OSError, so the hint cannot be chosen by broad
# exception type; TaggerUnavailable is what separates them.


def test_missing_index_does_not_blame_the_pos_tagger(tmp_path):
    """The first thing a new user hits. 0.1.1 appended the tagging hint to every start-up error,
    so a missing index was answered with advice about POS tagging -- and the flag it suggested
    was already in effect, making it impossible to act on."""
    result = runner.invoke(
        app,
        ["annotate", "Heart Attack.", "--indexdir", str(tmp_path / "absent"), "--no-postag"],
    )
    assert result.exit_code == 1
    assert "no index at" in result.output
    assert "--no-postag" not in result.output


def test_missing_index_with_tagging_on_still_does_not_blame_the_tagger(tmp_path):
    """Same error with tagging left enabled: the index is still the problem."""
    result = runner.invoke(
        app, ["annotate", "Heart Attack.", "--indexdir", str(tmp_path / "absent")]
    )
    assert result.exit_code == 1
    assert "no index at" in result.output
    assert "--no-postag" not in result.output


def test_missing_spacy_model_does_suggest_no_postag(index_dir):
    """The case the hint exists for, exercised through the real loader rather than a stub: the
    index is fine and the *model* is absent, so skipping tagging genuinely gets you moving."""
    result = runner.invoke(
        app,
        [
            "annotate",
            "Heart Attack.",
            "--indexdir",
            str(index_dir),
            "--postag-model",
            "en_core_web_nope",
        ],
    )
    assert result.exit_code == 1
    assert "en_core_web_nope" in result.output
    assert "--no-postag" in result.output


def test_tagger_errors_keep_their_original_base_classes():
    """Existing handlers must keep working: a missing package was an ImportError and a missing
    model an OSError. One class cannot be both -- their C layouts conflict -- so the pair is
    named by a tuple instead."""
    from mmlite.pipeline.spacy_nlp import (
        TAGGER_UNAVAILABLE,
        TaggerImportError,
        TaggerModelNotFound,
    )

    assert issubclass(TaggerImportError, ImportError)
    assert issubclass(TaggerModelNotFound, OSError)
    assert set(TAGGER_UNAVAILABLE) == {TaggerImportError, TaggerModelNotFound}
