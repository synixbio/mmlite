"""Fixtures for testing the examples/ scripts. They live outside src/mmlite (they are
usage examples, not library code) and are not installed as a package, so they're loaded directly
from their file paths rather than imported by name.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _load_script(name: str):
    path = EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"examples_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def analyze_file():
    return _load_script("analyze_file")


@pytest.fixture(scope="session")
def analyze_folder():
    return _load_script("analyze_folder")


@pytest.fixture(scope="session")
def json_to_sqlite():
    return _load_script("json_to_sqlite")


@pytest.fixture(scope="session")
def filter_mrconso():
    return _load_script("filter_mrconso")


@pytest.fixture(scope="session")
def filter_mrrel():
    return _load_script("filter_mrrel")


# --- the flat annotation table ----------------------------------------------------------------
# These share a row shape defined in examples/_annotations.py, so most of their tests are about
# what that shape can get wrong: which columns exist for a given pipeline configuration, and
# whether a format still round-trips.


@pytest.fixture(scope="session")
def annotations_helper():
    return _load_script("_annotations")


@pytest.fixture(scope="session")
def load_to_sqlite():
    return _load_script("load_to_sqlite")


@pytest.fixture(scope="session")
def notes_of_interest():
    return _load_script("notes_of_interest")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A small corpus written against the synthetic index in tests/conftest.py.

    Deliberately includes a negated mention and a note with nothing in it: those are the two
    cases the table treats specially (negation is what `notes_of_interest.py` filters on, and
    a note with no findings contributes no rows but is still counted).
    """
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "a.txt").write_text("Patient has Type 2 Diabetes and takes aspirin.", encoding="utf-8")
    (d / "b.txt").write_text("No history of Heart Attack.", encoding="utf-8")
    (d / "c.txt").write_text("Alzheimer Disease was noted.", encoding="utf-8")
    (d / "empty.txt").write_text("   \n", encoding="utf-8")
    return d


@pytest.fixture
def base_args(index_dir: Path):
    """The flags every exporter needs in tests: the synthetic index, and no spaCy dependency."""

    def build(corpus: Path, *extra: str) -> list[str]:
        return [str(corpus), "--index-dir", str(index_dir), "--no-postag", *extra]

    return build
