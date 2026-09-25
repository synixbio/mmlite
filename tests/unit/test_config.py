from pathlib import Path

import pytest

from mmlite.config import Settings, read_properties


def test_read_properties_equals_and_colon(tmp_path: Path):
    p = tmp_path / "mm.properties"
    p.write_text(
        "# a comment\n"
        "! a bang comment\n"
        "\n"
        "metamaplite.sourceset=MSH,SNOMEDCT_US\n"
        "metamaplite.segmentation.method: LINES\n",
        encoding="utf-8",
    )
    out = read_properties(p)
    assert out == {
        "metamaplite.sourceset": "MSH,SNOMEDCT_US",
        "metamaplite.segmentation.method": "LINES",
    }


def test_read_properties_splits_on_first_separator(tmp_path: Path):
    # A colon-delimited key whose value happens to contain '=' must still split on the
    # colon (the separator that occurs first), not unconditionally on '='.
    p = tmp_path / "mm.properties"
    p.write_text("some.key: value=extra\n", encoding="utf-8")
    assert read_properties(p) == {"some.key": "value=extra"}


def test_settings_load_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    p = tmp_path / "mm.properties"
    p.write_text(
        "metamaplite.segmentation.method=BLANKLINES\nmetamaplite.detect.negations=false\n",
        encoding="utf-8",
    )
    # defaults only
    assert Settings.load().segmentation_method == "SENTENCES"
    # properties file overrides defaults
    s = Settings.load(properties_file=p)
    assert s.segmentation_method == "BLANKLINES"
    assert s.detect_negations is False
    # env overrides properties file
    monkeypatch.setenv("MMLITE_SEGMENTATION_METHOD", "LINES")
    assert Settings.load(properties_file=p).segmentation_method == "LINES"
    # explicit kwarg overrides env
    assert Settings.load(
        properties_file=p, segmentation_method="SENTENCES"
    ).segmentation_method == ("SENTENCES")


def test_settings_load_none_override_does_not_clobber_properties_file(tmp_path: Path):
    # CLI commands always pass every Settings.load kwarg; an unset CLI flag surfaces as
    # None here and must not erase a value the caller configured via a properties file.
    p = tmp_path / "mm.properties"
    p.write_text("metamaplite.excluded.termsfile=/some/path.txt\n", encoding="utf-8")
    s = Settings.load(properties_file=p, excluded_terms_file=None)
    assert s.excluded_terms_file == Path("/some/path.txt")


def test_coerce_types(tmp_path: Path):
    p = tmp_path / "mm.properties"
    p.write_text(
        "metamaplite.enable.postagging=false\n"
        "metamaplite.normalized.string.cache.size=42\n"
        "metamaplite.sourceset=MSH, SNOMEDCT_US ,\n"
        "metamaplite.index.directory=/some/ivf\n",
        encoding="utf-8",
    )
    s = Settings.load(properties_file=p)
    assert s.enable_postagging is False
    assert s.normalized_string_cache_size == 42
    assert s.sourceset == ["MSH", "SNOMEDCT_US"]
    assert s.index_directory == Path("/some/ivf")


def test_postag_model_from_properties_env_and_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The tagger model follows the same precedence as every other setting.

    ``metamaplite.postag.model`` is this port's own key (Java's tagger is not selectable), and
    ``Settings`` must agree with ``spacy_nlp.DEFAULT_MODEL`` about what the default is.
    """
    from mmlite.pipeline.spacy_nlp import DEFAULT_MODEL

    assert Settings().postag_model == DEFAULT_MODEL
    p = tmp_path / "mm.properties"
    p.write_text("metamaplite.postag.model=en_core_sci_sm\n", encoding="utf-8")
    assert Settings.load(properties_file=p).postag_model == "en_core_sci_sm"
    monkeypatch.setenv("MMLITE_POSTAG_MODEL", "en_core_web_md")
    assert Settings.load(properties_file=p).postag_model == "en_core_web_md"
    assert Settings.load(properties_file=p, postag_model="en_core_web_lg").postag_model == (
        "en_core_web_lg"
    )
    # None is "not given", not "reset to default"
    assert Settings.load(properties_file=p, postag_model=None).postag_model == "en_core_web_md"
