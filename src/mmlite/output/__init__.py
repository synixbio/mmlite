"""Result formatters (``resultformats.ResultFormatterRegistry``).

Names mirror ``--outputformat``: ``mmi`` (default), ``json``, ``brat``, ``cuilist``, ``full``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from ..types import Entity
from .formats import (
    BcEvaluateFormatter,
    BratFormatter,
    CuiListFormatter,
    FullFormatter,
    JsonFormatter,
)
from .mmi import MMIFormatter

__all__ = [
    "FORMATS",
    "BcEvaluateFormatter",
    "BratFormatter",
    "CuiListFormatter",
    "FullFormatter",
    "JsonFormatter",
    "MMIFormatter",
    "ResultFormatter",
    "get_formatter",
]


class ResultFormatter(Protocol):
    def format(self, entities: list[Entity]) -> str: ...


# Java's registry names, plus "full" (the port's own readable dump).  Several are aliases for one
# formatter, exactly as MetaMapLite registers them: "fulljson" is "json" (both are FullJson), and
# "bc" / "bc-evaluate" / "cdi" / "bioc" are all BcEvaluate -- "bioc" included, which despite the
# name is a four-column TSV rather than BioC XML.
FORMATS = (
    "mmi",
    "json",
    "fulljson",
    "brat",
    "cuilist",
    "full",
    "bc",
    "bc-evaluate",
    "cdi",
    "bioc",
)


def get_formatter(
    name: str,
    *,
    treecode_lookup: Callable[[str], Sequence[str]] | None = None,
    brat_type: str = "MMLite",
    java_order: bool = True,
) -> ResultFormatter:
    """A formatter by ``--outputformat`` name.  ``java_order=False`` replaces Java ``HashSet``
    iteration order with sorted order where MMI, JSON and BRAT would otherwise use it."""
    name = name.lower()
    if name == "mmi":
        return MMIFormatter(treecode_lookup, java_order=java_order)
    if name in ("json", "fulljson"):
        return JsonFormatter(java_order=java_order)
    if name in ("bc", "bc-evaluate", "cdi", "bioc"):
        return BcEvaluateFormatter()
    if name == "brat":
        return BratFormatter(brat_type, java_order=java_order)
    if name == "cuilist":
        return CuiListFormatter()
    if name == "full":
        return FullFormatter()
    raise ValueError(f"unknown output format {name!r}; expected one of {FORMATS}")
