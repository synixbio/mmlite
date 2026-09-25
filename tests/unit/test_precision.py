"""The optional acronym and function-word precision filter (Settings.precision_filter)."""

from pathlib import Path

from typer.testing import CliRunner

from mmlite import MetaMapLite
from mmlite.cli import app
from mmlite.config import Settings
from mmlite.pipeline.precision import PrecisionFilter, keep_mention
from mmlite.types import ConceptInfo, Entity, Ev


def _entity(text: str, *concept_strings: str) -> Entity:
    evs = [
        Ev(
            ConceptInfo(f"C{i:07d}", s, s, frozenset(), frozenset({"dsyn"})),
            text,
            text.lower(),
            0,
            len(text),
        )
        for i, s in enumerate(concept_strings)
    ]
    return Entity("doc", "text", text, "NN", 0, 0, len(text), evs=evs)


def _strings(entity: Entity) -> list[str]:
    return [ev.concept.concept_string for ev in entity.evs]


def test_acronym_concepts_need_the_text_in_capitals():
    """On UMLS, "Plan" matches OMIM's PLAN (a dsyn) as well as "Plan"."""
    keep = PrecisionFilter()
    [plan] = keep([_entity("Plan", "PLAN", "Plan")])
    assert _strings(plan) == ["Plan"]
    assert keep([_entity("Plan", "PLAN")]) == []  # nothing left, so the mention goes
    [chf] = keep([_entity("CHF", "CHF")])  # written as an acronym: kept
    assert _strings(chf) == ["CHF"]
    # Not acronyms: one capital letter, several words, no letters at all
    for term in ("Aspirin", "HEART ATTACK", "120", "Vitamin A"):
        assert keep([_entity("thing", term)]), term


def test_a_capitalised_word_is_not_an_acronym_if_the_concept_also_has_it_in_lower_case():
    """Some sources store whole words in capitals; "arthralgias" must keep ARTHRALGIAS when the
    concept is also stored as "Arthralgias" under the same key."""
    index = {
        ("arthralgias", "C0000000"): ["ARTHRALGIAS", "Arthralgias"],
        ("plan", "C0000000"): ["PLAN"],
    }
    asked = []

    def strings(key, cui):
        asked.append((key, cui))
        return index.get((key, cui), [])

    keep = PrecisionFilter(strings)
    assert _strings(keep([_entity("arthralgias", "ARTHRALGIAS")])[0]) == ["ARTHRALGIAS"]
    assert keep([_entity("Plan", "PLAN")]) == []
    keep([_entity("Plan", "PLAN")])
    assert asked.count(("plan", "C0000000")) == 1  # the verdict is cached


def test_function_words_go_unless_in_capitals():
    for text in ("for", "with", "The", "us", "Without"):
        assert not keep_mention(_entity(text, "x")), text
    for text in ("US", "AS", "OR", "today", "male", "heart attack", "ICD-10", "per cent"):
        assert keep_mention(_entity(text, "x")), text


def _process(index_dir: Path, text: str, **settings) -> list[str]:
    s = Settings(index_directory=index_dir, enable_postagging=False, **settings)
    with MetaMapLite(s) as mml:
        return [e.text for e in mml.process_text(text)]


def test_off_by_default_and_applied_by_the_pipeline_when_on(index_dir: Path, monkeypatch):
    text = "Diabetes and aspirin."
    assert _process(index_dir, text) == ["Diabetes", "aspirin"]
    monkeypatch.setattr("mmlite.pipeline.precision.FUNCTION_WORDS", frozenset({"aspirin"}))
    assert _process(index_dir, text, precision_filter=True) == ["Diabetes"]


def test_runs_after_subsumption_so_no_shorter_match_is_exposed(index_dir: Path, monkeypatch):
    """Dropping a mention must not let a match inside it survive."""
    seen = []
    real = PrecisionFilter.__call__

    def spy(self, entities):
        entities = list(entities)
        seen.append([e.text for e in entities])
        return real(self, entities)

    monkeypatch.setattr(PrecisionFilter, "__call__", spy)
    _process(index_dir, "Type 2 diabetes mellitus.", precision_filter=True)
    assert seen == [["Type 2 diabetes mellitus"]]  # "diabetes" was already subsumed


def test_cli_flag(index_dir: Path):
    args = ["annotate", "Aspirin.", "--indexdir", str(index_dir), "--no-postag"]
    r = CliRunner().invoke(app, [*args, "--precision-filter"])
    assert r.exit_code == 0 and "C0004057" in r.output, r.output
