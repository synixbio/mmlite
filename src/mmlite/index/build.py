"""Build the mmlite index from UMLS RRF files (equivalent of Java ``CreateIndexes``).

Inputs
------
MRCONSO.RRF  CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF|
MRSTY.RRF    CUI|TUI|STN|STY|ATUI|CVF|
MRSAT.RRF    CUI|LUI|SUI|METAUI|STYPE|CODE|ATUI|SATUI|ATN|SAB|ATV|SUPPRESS|CVF|

Parity notes (Java ``CreateIndexes`` + ``Extract*`` classes, metamaplite master):
* Java consumes a pre-filtered ``mrconso.eng``; we filter ``LAT=ENG`` ourselves.
* Suppressible rows (SUPPRESS in O/E/Y) are dropped by default (``include_suppressed=True`` keeps
  them) — whether Java's ``mrconso.eng`` contains them depends on how the user produced it.
* Index key = ``STR.toLowerCase()`` (``MappedMultiKeyIndexGeneration``), stored in ``norm``.
* Preferred name (``ExtractMrconsoPreferredNames``): the *last* English ``TS=P, STT=PF`` row for
  the CUI, else the last row of any language.  ISPREF and SUPPRESS are not consulted, so every
  CUI in MRCONSO gets a name (verified against the NLM-built 2026AA tables: identical).
* MeSH tree codes (``ExtractTreecodes``): every ``SAB=MSH`` string (suppressed ones too) maps to
  each ``ATN=MN`` value of its CUI, or to ``x.x.x.x`` when the CUI has none.  Rows are **not**
  deduplicated: two MSH strings differing only in case share one key, and Java's postings (and
  hence the MMI tree-code field and the tree-depth part of the score) contain both.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from ..normalize import NORM_VERSION, index_key
from ..semtypes import to_abbrev
from .schema import DB_FILENAME, INDEXES, SCHEMA

log = logging.getLogger(__name__)

# MRCONSO column positions
_CUI, _LAT, _TS, _LUI, _STT, _SUI, _ISPREF = 0, 1, 2, 3, 4, 5, 6
_SAB, _TTY, _STR, _SUPPRESS = 11, 12, 14, 16
# MRSTY
_STY_CUI, _STY_TUI = 0, 1
# MRSAT
_SAT_CUI, _SAT_ATN, _SAT_ATV = 0, 8, 10

BATCH = 50_000


def _rrf_rows(path: Path, progress_every: int = 2_000_000) -> Iterator[list[str]]:
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for n, line in enumerate(fh, 1):
            if n % progress_every == 0:
                log.info("%s: %d lines", path.name, n)
            yield line.rstrip("\r\n").split("|")


T = TypeVar("T")


def _batched(rows: Iterable[T], size: int = BATCH) -> Iterator[list[T]]:
    buf: list[T] = []
    for r in rows:
        buf.append(r)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def build_index(
    mrconso: Path,
    mrsty: Path,
    out_dir: Path,
    mrsat: Path | None = None,
    sources: set[str] | None = None,
    include_suppressed: bool = False,
    overwrite: bool = False,
) -> Path:
    """Build ``out_dir/mmlite.sqlite``. Returns the database path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / DB_FILENAME
    if db_path.exists():
        if not overwrite:
            raise FileExistsError(f"{db_path} exists; pass overwrite=True to rebuild")
        db_path.unlink()

    t0 = time.time()
    con = sqlite3.connect(db_path)
    try:
        con.executescript(
            "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; PRAGMA cache_size=-524288;"
        )
        con.executescript(SCHEMA)
        con.execute("CREATE TEMP TABLE msh_strings (cui TEXT NOT NULL, key TEXT NOT NULL)")

        n_terms = _load_mrconso(con, Path(mrconso), sources, include_suppressed)
        n_sty = _load_mrsty(con, Path(mrsty))
        n_tc = _load_mrsat(con, Path(mrsat)) if mrsat else 0
        con.execute("DROP TABLE msh_strings")

        log.info("creating indexes")
        con.executescript(INDEXES)

        meta = {
            "norm_version": NORM_VERSION,
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "mrconso": str(mrconso),
            "sources": ",".join(sorted(sources)) if sources else "ALL",
            "include_suppressed": str(include_suppressed),
            "n_cuisourceinfo": str(n_terms),
            "n_cuist": str(n_sty),
            "n_meshtc": str(n_tc),
        }
        con.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", list(meta.items()))
        con.commit()
        con.execute("VACUUM")
    except BaseException:
        con.close()
        db_path.unlink(missing_ok=True)
        raise
    con.close()
    log.info("index built in %.1fs -> %s", time.time() - t0, db_path)
    return db_path


def _load_mrconso(
    con: sqlite3.Connection, path: Path, sources: set[str] | None, include_suppressed: bool
) -> int:
    log.info("loading %s", path)
    preferred: dict[str, str] = {}
    fallback: dict[str, str] = {}
    n = 0

    def rows() -> Iterator[tuple[str, ...]]:
        nonlocal n
        for f in _rrf_rows(path):
            if len(f) <= _SUPPRESS:
                continue
            cui, s, sab, eng = f[_CUI], f[_STR], f[_SAB], f[_LAT] == "ENG"
            # Preferred names and MeSH strings come from *every* row (any language, suppressed
            # included), as the NLM-built tables show; only cuisourceinfo is filtered.
            if eng and f[_TS] == "P" and f[_STT] == "PF":
                preferred[cui] = s  # last one wins, as in Java
            else:
                fallback[cui] = s
            key = index_key(s)
            if sab == "MSH" and key:
                msh.append((cui, key))
            if not eng or not key:
                continue
            if sources and sab not in sources:
                continue
            if not include_suppressed and f[_SUPPRESS] in ("O", "E", "Y"):
                continue
            n += 1
            yield (cui, f[_SUI], s, key, sab, f[_TTY])

    msh: list[tuple[str, str]] = []

    def flush_msh() -> None:
        if msh:
            con.executemany("INSERT INTO msh_strings VALUES (?,?)", msh)
            msh.clear()

    for batch in _batched(rows()):
        con.executemany("INSERT INTO cuisourceinfo VALUES (?,?,?,?,?,?)", batch)
        flush_msh()
    # MSH strings are collected from rows the cuisourceinfo filter may reject, so the generator
    # can finish with pending ones and no final batch to piggy-back on (e.g. --sources excludes
    # every remaining row, or the row count is an exact multiple of BATCH).
    flush_msh()
    con.commit()

    for cui, s in fallback.items():
        preferred.setdefault(cui, s)
    con.executemany("INSERT INTO cuiconcept VALUES (?,?)", list(preferred.items()))
    con.commit()
    log.info("cuisourceinfo: %d rows, cuiconcept: %d CUIs", n, len(preferred))
    return n


def _load_mrsty(con: sqlite3.Connection, path: Path) -> int:
    log.info("loading %s", path)
    n = 0

    def rows() -> Iterator[tuple[str, ...]]:
        nonlocal n
        for f in _rrf_rows(path):
            if len(f) <= _STY_TUI:
                continue
            n += 1
            yield (f[_STY_CUI], f[_STY_TUI], to_abbrev(f[_STY_TUI]))

    for batch in _batched(rows()):
        con.executemany("INSERT INTO cuist VALUES (?,?,?)", batch)
    con.commit()
    log.info("cuist: %d rows", n)
    return n


def _load_mrsat(con: sqlite3.Connection, path: Path) -> int:
    """MeSH tree codes: every MSH string -> each ATN=MN value of its CUI (or x.x.x.x if none)."""
    log.info("loading %s (MeSH tree codes)", path)
    con.execute("CREATE TEMP TABLE cui_treecode (cui TEXT NOT NULL, treecode TEXT NOT NULL)")

    def rows() -> Iterator[tuple[str, ...]]:
        for f in _rrf_rows(path):
            if len(f) > _SAT_ATV and f[_SAT_ATN] == "MN":
                yield (f[_SAT_CUI], f[_SAT_ATV])

    for batch in _batched(rows()):
        con.executemany("INSERT INTO cui_treecode VALUES (?,?)", batch)
    con.execute("CREATE INDEX ix_tmp_tc ON cui_treecode(cui)")
    con.execute(
        """INSERT INTO meshtcrelaxed(term, treecode)
           SELECT m.key, COALESCE(t.treecode, 'x.x.x.x')
             FROM msh_strings m LEFT JOIN cui_treecode t ON t.cui = m.cui
            ORDER BY m.rowid, t.rowid"""
    )
    con.commit()
    n: int = con.execute("SELECT COUNT(*) FROM meshtcrelaxed").fetchone()[0]
    con.execute("DROP TABLE cui_treecode")
    log.info("meshtcrelaxed: %d rows", n)
    return n
