#!/usr/bin/env python
"""Pre-filter MRCONSO.RRF down to just the source vocabularies you map codes from.

Usage:
    python examples/filter_mrconso.py --mrconso <path>\\MRCONSO.RRF
    python examples/filter_mrconso.py --mrconso <path>\\MRCONSO.RRF `
        --sab SNOMEDCT_US,ICD10CM,RXNORM,LNC,MSH

A code-mapping step that reads MRCONSO.RRF streams the *entire* file (18M+ rows, multiple GB)
every time it runs, even though it usually keeps rows for one SAB - four such steps against the
full file cost roughly 4x a single full scan (measured: ~13s each against UMLS 2026AA, ~50s
total). This script does that one full scan once: it keeps every row whose SAB is in --sab
(default: SNOMED CT, ICD-10-CM, RxNorm and LOINC) and writes them verbatim to --out - still a
valid MRCONSO.RRF, just much smaller, so anything that reads MRCONSO.RRF can read the result
unchanged (default: data/mrconso_filtered.RRF).

Re-run this script (e.g. with a wider --sab list, for a vocabulary not yet covered here) any
time; it always overwrites --out from scratch rather than appending to it.

The output file contains real UMLS-derived strings, exactly like MRCONSO.RRF itself, and must
not be committed or redistributed - same license restriction as the source file. It's gitignored.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

DEFAULT_SABS = ("SNOMEDCT_US", "ICD10CM", "RXNORM", "LNC")

# MRCONSO.RRF: CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
_SAB = 11


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mrconso", type=Path, required=True, help="Path to the full MRCONSO.RRF")
    ap.add_argument(
        "--sab",
        default=",".join(DEFAULT_SABS),
        help=f"Comma-separated SABs to keep (default: {','.join(DEFAULT_SABS)})",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "mrconso_filtered.RRF",
        help="Output path (default: data/mrconso_filtered.RRF)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.mrconso.is_file():
        print(f"error: {args.mrconso} is not a file", file=sys.stderr)
        return 1
    sabs = {s.strip() for s in args.sab.split(",") if s.strip()}
    if not sabs:
        print("error: --sab produced an empty set", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_lines = 0
    kept: Counter = Counter()
    with (
        open(args.mrconso, encoding="utf-8", errors="replace") as fin,
        open(args.out, "w", encoding="utf-8", newline="\n") as fout,
    ):
        for line in fin:
            n_lines += 1
            # cheap pre-check before the split: every kept row's SAB field appears in the line
            if not any(sab in line for sab in sabs):
                continue
            f = line.split("|", _SAB + 2)  # only split as far as we need
            if len(f) <= _SAB or f[_SAB] not in sabs:
                continue
            sab = f[_SAB]
            fout.write(line if line.endswith("\n") else line + "\n")
            kept[sab] += 1

    elapsed = time.time() - t0
    total_kept = sum(kept.values())
    breakdown = ", ".join(f"{sab}={kept[sab]:,}" for sab in sorted(sabs))
    print(
        f"scanned {n_lines:,} rows in {elapsed:.1f}s; kept {total_kept:,} rows ({breakdown}) -> "
        f"{args.out}",
        file=sys.stderr,
    )
    for sab in sorted(sabs - kept.keys()):
        print(
            f"warning: no rows found for SAB={sab!r} - check spelling / that it's in your UMLS "
            f"release",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
