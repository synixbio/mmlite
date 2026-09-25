"""Configuration compatible with MetaMapLite's ``metamaplite.properties`` keys.

Resolution order (highest wins): explicit kwargs > environment > properties file > defaults.
:data:`_PROPERTY_MAP` lists the properties this port honours; MetaMapLite keys outside it are
read from the file but ignored, since they configure behaviour that is not ported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

ENV_PREFIX = "MMLITE_"

# property key -> Settings attribute
_PROPERTY_MAP = {
    "metamaplite.index.directory": "index_directory",
    "metamaplite.sourceset": "sourceset",
    "metamaplite.semanticgroup": "semantic_types",
    "metamaplite.excluded.termsfile": "excluded_terms_file",
    "metamaplite.uda.filename": "uda_file",
    "metamaplite.cuitermlistfile.filename": "cui_term_list_file",
    "metamaplite.segmentation.method": "segmentation_method",
    "metamaplite.enable.postagging": "enable_postagging",
    "metamaplite.postaglist": "postag_list",
    # This port's own key, not Java's: MetaMapLite's tagger is fixed (OpenNLP), ours is a spaCy
    # model chosen by name.  See USER_GUIDE "Choosing the POS tagger".
    "metamaplite.postag.model": "postag_model",
    # Also this port's own: see spacy_nlp.VERB_RESCUE_THRESHOLD.
    "metamaplite.postag.verbrescue": "postag_verb_rescue",
    "metamaplite.detect.negations": "detect_negations",
    "metamaplite.negation.detector": "negation_detector",
    # This port's own: false fixes upstream bugs the port otherwise reproduces (see strict_parity).
    "metamaplite.strictparity": "strict_parity",
    # Also the port's own: see pipeline/precision.py.
    "metamaplite.precisionfilter": "precision_filter",
    "metamaplite.brat.typename": "brat_typename",
    "metamaplite.normalized.string.cache.size": "normalized_string_cache_size",
}


@dataclass
class Settings:
    index_directory: Path = Path("ivf")
    sourceset: list[str] = field(default_factory=list)  # empty = all sources
    semantic_types: list[str] = field(default_factory=list)  # empty = all STs
    excluded_terms_file: Path | None = None
    uda_file: Path | None = None
    cui_term_list_file: Path | None = None
    segmentation_method: str = "SENTENCES"  # SENTENCES | BLANKLINES | LINES
    enable_postagging: bool = True
    postag_list: list[str] = field(default_factory=list)  # empty = the built-in allowed set
    # spaCy pipeline whose tagger supplies the Penn Treebank tags.  "en_core_sci_sm" (scispaCy)
    # measures closer to Java and finds more line-initial drug names; see the user guide.
    postag_model: str = "en_core_web_sm"
    # Re-tag a spaCy verb as a noun/adjective when that reading has at least this probability, so
    # drug and disease names the tagger mistakes for verbs still start a span.  0 disables it.
    postag_verb_rescue: float = 0.03
    detect_negations: bool = True
    # "negex" (the default) or "context"; Java's own class names are accepted too.
    negation_detector: str = "negex"
    # True (the default) reproduces MetaMapLite 3.6.2rc8 exactly, bugs included. False fixes the
    # upstream bugs listed in USER_GUIDE "Strict parity": ConText scoped to its own sentence (Java
    # lets a later sentence overwrite an earlier entity's experiencer/temporality), ConText's
    # malformed trigger regexes, and Java hash-set ordering in JSON/BRAT output.
    strict_parity: bool = True
    # Drop case-mismatched acronym concepts ("Plan" -> PLAN) and one-word function words ("for",
    # "with").  Off by default: Java has no such filter (pipeline/precision.py).
    precision_filter: bool = False
    brat_typename: str = "MMLite"
    normalized_string_cache_size: int = 100_000

    @classmethod
    def load(cls, properties_file: Path | None = None, **overrides: Any) -> Settings:
        values: dict[str, str] = {}
        if properties_file is not None:
            values.update(read_properties(properties_file))
        for key, attr in _PROPERTY_MAP.items():
            env = os.environ.get(ENV_PREFIX + attr.upper())
            if env is not None:
                values[key] = env
        kwargs = {
            attr: _coerce(cls, attr, values[key])
            for key, attr in _PROPERTY_MAP.items()
            if key in values
        }
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kwargs)


def read_properties(path: Path) -> dict[str, str]:
    """Minimal Java .properties reader (key=value / key: value, # and ! comments).

    Splits on whichever of ``=``/``:`` occurs first in the line, matching Java's
    ``Properties`` parsing (a value may itself contain the other separator).
    """
    out: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line[0] in "#!":
            continue
        positions = [i for i in (line.find("="), line.find(":")) if i >= 0]
        if not positions:
            continue
        i = min(positions)
        k, v = line[:i], line[i + 1 :]
        out[k.strip()] = v.strip()
    return out


def _coerce(cls: type[Settings], attr: str, value: str) -> Any:
    ftype = {f.name: f.type for f in fields(cls)}[attr]
    if ftype in ("bool", bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if ftype in ("int", int):
        return int(value)
    if ftype in ("float", float):
        return float(value)
    if ftype.startswith("list"):
        return [s.strip() for s in value.split(",") if s.strip()]
    if ftype.startswith("Path"):
        return Path(value)
    return value
