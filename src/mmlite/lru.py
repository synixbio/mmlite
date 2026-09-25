"""A small least-recently-used cache for the per-concept lookups.

The index and entity-lookup caches used to be plain dicts that only ever grew, so a long-lived
server's RSS crept up with every new concept it met.  This bounds them.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")

# Entries per cache.  Measured at ~620 bytes an entry (an upper bound), so four caches at this
# size cost at most ~125 MB; the parity corpus plus both example notes touch ~2,100 CUIs, so a
# realistic working set fits many times over and the hit rate is unaffected.
DEFAULT_MAXSIZE = 50_000

_MISSING = object()


class LRUCache(Generic[K, V]):
    """Mapping that evicts the least recently used entry once it holds ``maxsize``."""

    def __init__(self, maxsize: int = DEFAULT_MAXSIZE):
        if maxsize < 1:
            raise ValueError(f"maxsize must be at least 1, got {maxsize}")
        self.maxsize = maxsize
        self._data: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K, default: V | None = None) -> V | None:
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def __setitem__(self, key: K, value: V) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)
