#!/usr/bin/env python
"""Pre-filter MRREL.RRF down to the drug-ingredient and code-hierarchy relationship rows.

Usage:
    python examples/filter_mrrel.py --mrrel <path>\\MRREL.RRF
    python examples/filter_mrrel.py --mrrel <path>\\MRREL.RRF --sab RXNORM,ICD10CM

Resolving drug ingredients or rolling a code up its hierarchy needs only a tiny slice of the
*entire* MRREL.RRF (66M+ rows, several GB) - measured on UMLS 2026AA: 1,741,968 rows survive out
of 66,241,184, a ~38x reduction (~181MB from ~6.4GB, one 40s pass). Each step that streams the
full file costs ~35-40s; this script does that one full scan once and writes the surviving rows
verbatim to --out (default: data/mrrel_filtered.RRF) - still a valid MRREL.RRF, so anything that
reads MRREL.RRF can read the result unchanged.

*** Unlike filter_mrconso.py, filtering by SAB alone would barely shrink the file. *** RXNORM's
own relationships alone span dozens of RELA types across 15M+ rows (SY, isa, has_dose_form, ...)
- only three (has_ingredient/consists_of/tradename_of) link a drug to its ingredients. So each
SAB here is paired with the exact REL or RELA values that task needs - see DEFAULT_RULES below.
If a consumer of the filtered file starts reading other relationships, widen the matching rule
here too, or the filtered file will silently stop carrying the rows it now wants.

Rows with SUPPRESS in O/E/Y are *not* dropped here - suppression is a per-consumer decision, so
the filtered file carries them exactly as the original does.

--sab narrows which of the three built-in rules run (default: all three - RXNORM, ICD10CM,
SNOMEDCT_US). Re-run this script any time; it always overwrites --out from scratch rather than
appending to it.

The output file contains real UMLS-derived strings, exactly like MRREL.RRF itself, and must not
be committed or redistributed - same license restriction as the source file. It's gitignored.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

# (SAB, field to check, allowed values): RxNorm's drug-to-ingredient relationships, and the
# parent-to-child (REL='CHD') links that make up the ICD-10-CM and SNOMED CT hierarchies.
DEFAULT_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("RXNORM", "RELA", ("has_ingredient", "consists_of", "tradename_of")),
    ("ICD10CM", "REL", ("CHD",)),
    ("SNOMEDCT_US", "REL", ("CHD",)),
)

# MRREL.RRF: CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|RUI|SRUI|SAB|SL|RG|DIR|SUPPRESS|CVF
_REL, _RELA, _SAB = 3, 7, 10
_FIELD_INDEX = {"REL": _REL, "RELA": _RELA}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mrrel", type=Path, required=True, help="Path to the full MRREL.RRF")
    ap.add_argument(
        "--sab",
        help="Comma-separated SABs to keep, narrowing the built-in rules (default: all three - "
        + ",".join(sab for sab, _, _ in DEFAULT_RULES)
        + ")",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "mrrel_filtered.RRF",
        help="Output path (default: data/mrrel_filtered.RRF)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.mrrel.is_file():
        print(f"error: {args.mrrel} is not a file", file=sys.stderr)
        return 1

    wanted = {s.strip() for s in args.sab.split(",") if s.strip()} if args.sab else None
    rules = [r for r in DEFAULT_RULES if wanted is None or r[0] in wanted]
    if not rules:
        print("error: --sab matched none of the built-in rules", file=sys.stderr)
        return 1
    by_sab = {sab: (_FIELD_INDEX[field], set(values)) for sab, field, values in rules}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_lines = 0
    kept: Counter = Counter()
    with (
        open(args.mrrel, encoding="utf-8", errors="replace") as fin,
        open(args.out, "w", encoding="utf-8", newline="\n") as fout,
    ):
        for line in fin:
            n_lines += 1
            # cheap pre-check before the split: every kept row's SAB appears in the line
            if not any(sab in line for sab in by_sab):
                continue
            f = line.split("|", _SAB + 1)  # only split as far as we need
            if len(f) <= _SAB:
                continue
            rule = by_sab.get(f[_SAB])
            if rule is None:
                continue
            field_index, values = rule
            if f[field_index] not in values:
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
            kept[f[_SAB]] += 1

    elapsed = time.time() - t0
    total_kept = sum(kept.values())
    breakdown = ", ".join(f"{sab}={kept[sab]:,}" for sab in sorted(by_sab))
    print(
        f"scanned {n_lines:,} rows in {elapsed:.1f}s; kept {total_kept:,} rows ({breakdown}) -> "
        f"{args.out}",
        file=sys.stderr,
    )
    for sab in sorted(set(by_sab) - kept.keys()):
        print(
            f"warning: no rows found for SAB={sab!r} - check spelling / that it's in your UMLS "
            f"release",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
