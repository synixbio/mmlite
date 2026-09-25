"""Query the mmlite index (equivalent of Java ``IVFLookup`` + ``MappedMultiKeyIndexLookup``).

A lookup for a text span tries the keys from :func:`mmlite.normalize.lookup_keys`
(the span verbatim and its normalized form), exactly as ``EntityLookup4`` does.

Runtime pattern: open the SQLite database, and optionally call ``load_in_memory()`` to pull the
key -> CUI map into a dict, trading memory for speed on the entity-lookup hot path.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from ..lru import LRUCache
from ..normalize import NORM_VERSION, index_key, lookup_keys
from ..semtypes import to_source_set, to_tui, to_tui_set
from .schema import DB_FILENAME, TERMMAP_FILENAME

# Bound on the parameters in one `norm IN (...)` statement; see cuis_for_keys.
_MAX_KEYS_PER_QUERY = 500


@dataclass(frozen=True)
class ConceptHit:
    cui: str
    preferred_name: str
    matched_string: str
    sab: str
    tty: str
    semantic_types: tuple[str, ...]  # abbreviations, e.g. ("dsyn",)


class IndexLookup:
    def __init__(self, index_dir: Path | str, check_norm_version: bool = True):
        self.index_dir = Path(index_dir)
        db = self.index_dir / DB_FILENAME
        if not db.exists():
            raise FileNotFoundError(f"no index at {db}; run `mmlite build-index` first")
        self.con = sqlite3.connect(
            f"file:{db.as_posix()}?mode=ro", uri=True, check_same_thread=False
        )
        try:
            self.meta: dict[str, str] = dict(self.con.execute("SELECT key, value FROM meta"))
            if check_norm_version and self.meta.get("norm_version") != NORM_VERSION:
                raise RuntimeError(
                    f"index was built with normalizer {self.meta.get('norm_version')!r}, "
                    f"code is {NORM_VERSION!r}; rebuild the index"
                )
        except BaseException:
            self.con.close()
            raise
        self._term_map: dict[str, tuple[str, ...]] | None = None
        # Bounded: a long-running server meets new concepts indefinitely (see lru.py).
        self._st_cache: LRUCache[str, tuple[str, ...]] = LRUCache()
        self._name_cache: LRUCache[str, str] = LRUCache()
        self._src_cache: LRUCache[str, tuple[str, ...]] = LRUCache()

    # -- bulk loading ----------------------------------------------------------------------------

    def load_in_memory(self, cache: bool = True) -> None:
        """Materialize key -> (cui, ...) for the entity-lookup hot path.

        Building the map from SQLite takes ~45 s on a full UMLS; with ``cache=True`` it is
        pickled next to the database (``termmap.pkl``) and reloaded from there afterwards.
        """
        pkl = self.index_dir / TERMMAP_FILENAME
        db = self.index_dir / DB_FILENAME
        if cache and pkl.exists() and pkl.stat().st_mtime >= db.stat().st_mtime:
            with open(pkl, "rb") as fh:
                self._term_map = pickle.load(fh)
            return
        m: dict[str, set[str]] = defaultdict(set)
        for norm, cui in self.con.execute("SELECT DISTINCT norm, cui FROM cuisourceinfo"):
            m[norm].add(cui)
        self._term_map = {k: tuple(sorted(v)) for k, v in m.items()}
        if cache:
            # Several processes may build the cache at once (a ProcessPoolExecutor whose workers
            # all preload on a cold index).  Each writes its own temp file, and whichever rename
            # lands first wins; the others' copies are identical, so losing the race is not an
            # error - on Windows the rename fails outright while another process still has the
            # target open, on POSIX it silently overwrites with the same bytes.
            tmp = pkl.with_name(f"{pkl.stem}.{os.getpid()}.tmp")
            try:
                with open(tmp, "wb") as fh:
                    pickle.dump(self._term_map, fh, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.replace(pkl)
            except OSError:
                if not pkl.exists():
                    raise
            finally:
                tmp.unlink(missing_ok=True)

    @property
    def in_memory(self) -> bool:
        return self._term_map is not None

    def concept_string(self, key: str, cui: str) -> str:
        """The dictionary string stored under ``key`` for ``cui``.

        The first row, matching the order of Java's postings.
        """
        row = self.con.execute(
            "SELECT str FROM cuisourceinfo WHERE norm = ? AND cui = ? LIMIT 1", (key, cui)
        ).fetchone()
        return row[0] if row else key

    def concept_strings(self, key: str, cui: str) -> list[str]:
        """Every dictionary string stored under ``key`` for ``cui`` (distinct, any order)."""
        rows = self.con.execute(
            "SELECT DISTINCT str FROM cuisourceinfo WHERE norm = ? AND cui = ?", (key, cui)
        )
        return [row[0] for row in rows]

    def has_key(self, key: str) -> bool:
        """Is this exact index key present? (no normalization applied)"""
        if self._term_map is not None:
            return key in self._term_map
        row = self.con.execute(
            "SELECT 1 FROM cuisourceinfo WHERE norm = ? LIMIT 1", (key,)
        ).fetchone()
        return row is not None

    def has_term(self, text: str) -> bool:
        return any(self.has_key(k) for k in lookup_keys(text))

    def cuis_for_key(self, key: str) -> tuple[str, ...]:
        """CUIs stored under one exact index key."""
        if self._term_map is not None:
            return self._term_map.get(key, ())
        rows = self.con.execute("SELECT DISTINCT cui FROM cuisourceinfo WHERE norm = ?", (key,))
        return tuple(sorted(r[0] for r in rows))

    def cuis_for_keys(self, keys: Sequence[str]) -> dict[str, tuple[str, ...]]:
        """:meth:`cuis_for_key` for several keys, in one statement.

        Candidate generation asks about every span of a 15-token window, and on ordinary text
        around four in five of those keys are not in the index at all — each one otherwise a
        separate round trip into SQLite.  Keys absent from the index are absent from the result,
        exactly as a per-key miss returns ``()``.
        """
        if not keys:
            return {}
        if self._term_map is not None:
            return {k: v for k in keys if (v := self._term_map.get(k))}
        out: dict[str, list[str]] = {}
        # Chunked so the statement never approaches SQLITE_MAX_VARIABLE_NUMBER, which is 999 on
        # SQLite before 3.32 and 32766 after; a long sentence can ask about more keys than that.
        for start in range(0, len(keys), _MAX_KEYS_PER_QUERY):
            chunk = tuple(keys[start : start + _MAX_KEYS_PER_QUERY])
            placeholders = ",".join("?" * len(chunk))
            rows = self.con.execute(
                f"SELECT DISTINCT norm, cui FROM cuisourceinfo WHERE norm IN ({placeholders})",
                chunk,
            )
            for norm, cui in rows:
                out.setdefault(norm, []).append(cui)
        return {k: tuple(sorted(v)) for k, v in out.items()}

    def cuis_for_term(self, text: str) -> tuple[str, ...]:
        """Union of CUIs over all lookup keys of the span (EntityLookup4 behaviour)."""
        out: set[str] = set()
        for k in lookup_keys(text):
            out.update(self.cuis_for_key(k))
        return tuple(sorted(out))

    # -- per-concept accessors -------------------------------------------------------------------

    def preferred_name(self, cui: str) -> str:
        name = self._name_cache.get(cui)
        if name is None:
            row = self.con.execute(
                "SELECT preferred_name FROM cuiconcept WHERE cui = ?", (cui,)
            ).fetchone()
            name = row[0] if row else cui
            self._name_cache[cui] = name
        return name

    def semantic_types(self, cui: str) -> tuple[str, ...]:
        sts = self._st_cache.get(cui)
        if sts is None:
            rows = self.con.execute(
                "SELECT abbrev FROM cuist WHERE cui = ? ORDER BY abbrev", (cui,)
            )
            sts = tuple(r[0] for r in rows)
            self._st_cache[cui] = sts
        return sts

    def sources(self, cui: str) -> tuple[str, ...]:
        """Every SAB the CUI appears in (``CuiSourceSetIndex``), in first-appearance order.

        The order is load-bearing, not cosmetic.  Java fills a ``HashSet<String>`` from this
        CUI's postings — which are in MRCONSO row order — and the JSON and BRAT formatters print
        that set's iteration order.  Within a hash bucket a ``HashSet`` keeps *insertion* order,
        so reproducing Java's output needs the postings order, not an alphabetical one:
        ``MTH`` and ``LNC`` collide in bucket 0, and sorting first swaps them.

        Rows are inserted in MRCONSO order, so ``rowid`` order is posting order.  Deduplication
        happens here rather than via ``SELECT DISTINCT``, which would impose its own ordering.
        """
        srcs = self._src_cache.get(cui)
        if srcs is None:
            rows = self.con.execute(
                "SELECT sab FROM cuisourceinfo WHERE cui = ? ORDER BY rowid", (cui,)
            )
            srcs = tuple(dict.fromkeys(r[0] for r in rows))
            self._src_cache[cui] = srcs
        return srcs

    def mesh_treecodes(self, term: str) -> tuple[str, ...]:
        """MeSH tree codes for a term (case-insensitive, as the IVF lookup lower-cases queries)."""
        rows = self.con.execute(
            "SELECT treecode FROM meshtcrelaxed WHERE term = ?", (index_key(term),)
        )
        return tuple(r[0] for r in rows)

    # -- main query ------------------------------------------------------------------------------

    def lookup(
        self,
        text: str,
        sources: set[str] | None = None,
        semantic_types: set[str] | None = None,
    ) -> list[ConceptHit]:
        """All concepts stored under any of the span's lookup keys, optionally restricted.

        ``semantic_types`` accepts abbreviations (``dsyn``) or TUIs (``T047``) and ``sources``
        accepts SABs, both in any case; ``"all"`` means no restriction.
        """
        keys = [k for k in lookup_keys(text) if k]
        if not keys:
            return []
        placeholders = ",".join("?" * len(keys))
        sql = f"SELECT cui, str, sab, tty FROM cuisourceinfo WHERE norm IN ({placeholders})"
        params: list[str] = list(keys)
        src_filter = to_source_set(sources)
        if src_filter:
            sql += f" AND sab IN ({','.join('?' * len(src_filter))})"
            params.extend(sorted(src_filter))
        st_filter = to_tui_set(semantic_types)

        hits: list[ConceptHit] = []
        seen: set[tuple[str, str, str]] = set()
        for cui, s, sab, tty in self.con.execute(sql, params):
            key = (cui, sab, tty)
            if key in seen:
                continue
            seen.add(key)
            sts = self.semantic_types(cui)
            if st_filter and not ({to_tui(a) for a in sts} & st_filter):
                continue
            hits.append(ConceptHit(cui, self.preferred_name(cui), s, sab, tty, sts))
        hits.sort(key=lambda h: (h.cui, h.sab, h.tty))
        return hits

    def stats(self) -> dict[str, int | str]:
        out: dict[str, int | str] = dict(self.meta)
        for table in ("cuiconcept", "cuisourceinfo", "cuist", "meshtcrelaxed"):
            out[f"rows_{table}"] = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        out["distinct_norms"] = self.con.execute(
            "SELECT COUNT(DISTINCT norm) FROM cuisourceinfo"
        ).fetchone()[0]
        return out

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
