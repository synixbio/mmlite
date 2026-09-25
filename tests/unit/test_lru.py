"""Tests for the bounded per-concept caches."""

from pathlib import Path

import pytest

from mmlite.index import IndexLookup
from mmlite.lru import LRUCache
from mmlite.pipeline import TextPipeline
from mmlite.pipeline.entity_lookup import EntityLookup
from mmlite.pipeline.postag import NullPosTagger


def test_evicts_the_least_recently_used_entry():
    cache = LRUCache(2)
    cache["a"], cache["b"] = 1, 2
    assert cache.get("a") == 1  # "a" is now the most recent
    cache["c"] = 3
    assert "b" not in cache and cache.get("a") == 1 and cache.get("c") == 3
    assert len(cache) == 2


def test_get_distinguishes_a_missing_key_from_a_falsy_value():
    cache = LRUCache(4)
    cache["empty"] = ()  # a CUI with no semantic types caches an empty tuple
    assert cache.get("empty") == () and cache.get("absent") is None


def test_rejects_a_size_below_one():
    with pytest.raises(ValueError):
        LRUCache(0)


def _fields(ci):
    # ConceptInfo compares by CUI alone; check everything the caches feed into it
    return (ci.cui, ci.preferred_name, ci.concept_string, ci.sources, sorted(ci.semantic_types))


def test_caches_are_bounded_and_eviction_never_changes_the_result(index_dir: Path):
    """The four caches used to be unbounded dicts.  Starved to two entries each, so that nearly
    every lookup is a miss and entries are evicted mid-document, annotation must be unchanged."""
    text = (
        "Patient with Type 2 diabetes mellitus and a prior heart attack takes aspirin. "
        "No Alzheimer's disease. Diabetes and Parkinson's Disease noted; aspirin continued."
    )
    pipeline = TextPipeline(tagger=NullPosTagger())

    def run(maxsize):
        with IndexLookup(index_dir) as index:
            lookup = EntityLookup(index)
            if maxsize is not None:
                lookup._concept_cache = LRUCache(maxsize)
                index._name_cache = LRUCache(maxsize)
                index._st_cache = LRUCache(maxsize)
                index._src_cache = LRUCache(maxsize)
            ents = lookup.process_text(text, pipeline)
            caches = (lookup._concept_cache, index._name_cache, index._st_cache, index._src_cache)
            return (
                [(e.start, e.length, e.negated, [_fields(v.concept) for v in e.evs]) for e in ents],
                [len(c) for c in caches],
            )

    roomy, _ = run(None)
    starved, sizes = run(2)
    assert starved == roomy
    assert len({c[0] for e in roomy for c in e[3]}) > 2  # more concepts than fit
    assert max(sizes) <= 2
