"""Plugging a third-party assertion model in through the detector registry."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mmlite import MetaMapLite, pipeline
from mmlite.cli import app
from mmlite.config import Settings
from mmlite.pipeline import (
    NEGATION_DETECTORS,
    Assertion,
    AssertionClassifier,
    NegationDetector,
    get_negation_detector,
    negation_detector_names,
    register_negation_detector,
)


class Hedges(AssertionClassifier):
    """A toy model: 'maybe' in the sentence negates nothing but makes every mention
    hypothetical; 'mother' makes it the mother's.  Records what it was shown."""

    def __init__(self, strict_parity: bool = True):
        self.seen: list[tuple[str, list[str]]] = []

    def classify(self, sentence: str, spans: Sequence[tuple[int, int]]):
        self.seen.append((sentence, [sentence[s:e] for s, e in spans]))
        lowered = sentence.lower()
        return [
            None
            if "skip" in lowered
            else Assertion(
                negated=lowered.startswith("no "),
                temporality="Hypothetical" if "maybe" in lowered else "Recent",
                experiencer="Other" if "mother" in lowered else "Patient",
            )
            for _ in spans
        ]


@pytest.fixture
def registry():
    """Undo registrations so tests stay independent."""
    aliases, factories = dict(NEGATION_DETECTORS), dict(pipeline._FACTORIES)
    yield
    NEGATION_DETECTORS.clear()
    NEGATION_DETECTORS.update(aliases)
    pipeline._FACTORIES.clear()
    pipeline._FACTORIES.update(factories)


def _mml(index_dir: Path, detector: str) -> MetaMapLite:
    return MetaMapLite(
        Settings(index_directory=index_dir, enable_postagging=False, negation_detector=detector)
    )


def test_a_classifier_sees_sentence_relative_spans_and_sets_each_entity(index_dir, registry):
    model = Hedges()
    register_negation_detector("hedges", lambda strict_parity: model, aliases=["org.example.H"])
    assert isinstance(get_negation_detector("org.example.h"), NegationDetector)
    text = "Maybe diabetes. No aspirin. Her mother had a heart attack."
    with _mml(index_dir, "Hedges") as mml:
        entities = {e.text: e for e in mml.process_text(text)}
    assert model.seen == [
        ("Maybe diabetes.", ["diabetes"]),
        ("No aspirin.", ["aspirin"]),
        ("Her mother had a heart attack.", ["heart attack"]),
    ]
    assert (entities["diabetes"].negated, entities["diabetes"].temporality) == (
        False,
        "Hypothetical",
    )
    assert entities["aspirin"].negated is True
    assert entities["heart attack"].experiencer == "Other"


def test_none_leaves_an_entity_as_it_was(index_dir, registry):
    register_negation_detector("hedges", Hedges)
    with _mml(index_dir, "hedges") as mml:
        [entity] = mml.process_text("Skip aspirin.")
    assert (entity.negated, entity.temporality, entity.experiencer) == (False, None, None)


def test_a_wrong_number_of_results_is_an_error(index_dir, registry):
    class Short(AssertionClassifier):
        def classify(self, sentence, spans):
            return []

    register_negation_detector("short", lambda strict_parity: Short())
    with _mml(index_dir, "short") as mml, pytest.raises(ValueError, match="0 results for 1"):
        mml.process_text("Aspirin.")


def test_registration_guards_the_built_ins_and_other_detectors_aliases(registry):
    with pytest.raises(ValueError, match="built-in"):
        register_negation_detector("NegEx", Hedges)
    with pytest.raises(ValueError, match="'context' negation detector"):
        register_negation_detector(
            "mine", Hedges, aliases=["gov.nih.nlm.nls.metamap.lite.context.ContextWrapper"]
        )
    register_negation_detector("mine", Hedges)
    register_negation_detector("mine", Hedges)  # re-registering one's own name replaces it
    assert "mine" in negation_detector_names()


def test_entry_points_are_discovered_on_first_unknown_name(monkeypatch, registry):
    class FakeEntryPoint:
        name = "plugged"

        def load(self):
            return Hedges

    def fake_entry_points(group):
        assert group == pipeline.ENTRY_POINT_GROUP
        return [FakeEntryPoint()]

    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)
    monkeypatch.setattr(pipeline, "_entry_points_loaded", False)
    assert isinstance(get_negation_detector("plugged"), Hedges)
    assert "plugged" in negation_detector_names()


def test_cli_selects_a_registered_detector_by_name(index_dir, registry):
    register_negation_detector("hedges", Hedges)
    runner = CliRunner()
    args = ["annotate", "Maybe diabetes.", "--indexdir", str(index_dir), "--no-postag"]
    r = runner.invoke(app, [*args, "--outputformat", "json", "--negation-detector", "hedges"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, [*args, "--usecontext", "--negation-detector", "hedges"])
    assert r.exit_code == 1 and "conflicts" in r.output
    r = runner.invoke(app, [*args, "--negation-detector", "nope"])
    assert r.exit_code == 1 and "unknown negation detector" in r.output


def test_json_round_trip_is_unaffected_by_the_plugin(index_dir, registry):
    """The plugin changes assertion fields only; the Java-shaped JSON stays the same shape."""
    register_negation_detector("hedges", Hedges)
    with _mml(index_dir, "hedges") as mml:
        out = json.loads(mml.format(mml.process_text("No aspirin."), "json"))
    assert out[0]["negated"] is True
