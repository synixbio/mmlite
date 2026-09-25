"""Java collection semantics that leak into MetaMapLite's output.

``ConceptInfo`` holds its semantic types and sources in ``HashSet<String>``s, and the MMI, JSON
and BRAT formatters simply iterate them, so the *order* of those fields in the output is Java's
hash-table order.  :func:`hash_set_order` reproduces it exactly.

``Entity`` holds its evidence the same way — ``Set<Ev> evSet``, with
``Ev.hashCode() = length + start + conceptInfo.hashCode()`` and
``ConceptInfo.hashCode() = cui.hashCode()`` — and ``getEvList()`` is just
``new ArrayList<>(evSet)``, so the order of ``evlist`` in JSON output is hash-table order too.
:func:`ev_hash` and :func:`hash_set_order_by` reproduce that.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")

_INT_MASK = 0xFFFFFFFF
DEFAULT_CAPACITY = 16
LOAD_FACTOR = 0.75


def string_hash(s: str) -> int:
    """``java.lang.String.hashCode()`` (32-bit signed, as an unsigned int here)."""
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & _INT_MASK
    return h


def _spread(h: int) -> int:
    """``java.util.HashMap.hash``: ``h ^ (h >>> 16)``."""
    return (h ^ (h >> 16)) & _INT_MASK


def table_capacity(size: int, initial: int = DEFAULT_CAPACITY) -> int:
    """Capacity of a HashMap built by repeated ``put`` (doubles when size exceeds the threshold)."""
    cap = initial
    while size > cap * LOAD_FACTOR:
        cap *= 2
    return cap


def hash_set_order_by(
    items: Iterable[T],
    java_hash: Callable[[T], int],
    initial_capacity: int = DEFAULT_CAPACITY,
) -> list[T]:
    """Iteration order of a ``HashSet`` whose elements hash as ``java_hash`` says.

    Buckets are visited in table order; within a bucket, entries keep insertion order (true for
    the short chains these sets produce — MetaMapLite never has 8+ colliding elements).  The sort
    is stable, which is what preserves that within-bucket order.

    ``items`` is assumed already deduplicated by the caller where that matters; unlike
    :func:`hash_set_order` there is no general way to dedupe arbitrary elements by Java's
    ``equals`` from here.
    """
    seq = list(items)
    cap = table_capacity(len(seq), initial_capacity)
    return sorted(seq, key=lambda item: _spread(java_hash(item) & _INT_MASK) & (cap - 1))


def hash_set_order(items: Iterable[str], initial_capacity: int = DEFAULT_CAPACITY) -> list[str]:
    """Iteration order of a ``HashSet<String>`` filled with ``items`` in that order."""
    return hash_set_order_by(dict.fromkeys(items), string_hash, initial_capacity)


def ev_hash(start: int, length: int, cui: str) -> int:
    """``Ev.hashCode()``: ``length + start + conceptInfo.hashCode()``, and the latter is the CUI's.

    Java adds these as 32-bit ints and lets them overflow; the mask in
    :func:`hash_set_order_by` reproduces that.
    """
    return (length + start + string_hash(cui)) & _INT_MASK
