# Examples

Runnable scripts against the Python API — a starting point to adapt, not a library. Both need a
built index first (`mmlite build-index ...` — see `mmlite build-index --help`).

Every script here has automated tests in [tests/examples/](../tests/examples/) — one file per
script, against small synthetic MRCONSO/MRREL fixtures (or the project's shared synthetic index
for the ones that need one), so running them with `pytest tests/examples/` needs no UMLS
download.

- **`analyze_file.py`** — annotate one text file, print a one-line summary to stderr, write the
  formatted output to stdout or `-o FILE`.

  ```powershell
  python examples/analyze_file.py note.txt --index-dir ivf --outputformat json -o note.json
  ```

- **`analyze_folder.py`** — annotate every file matching a glob under a folder, reusing one
  loaded index across all of them; writes one output file per input plus a corpus-wide summary
  (files processed/failed, most frequent concepts with the term each actually matched on).
  `--workers N` switches to a process pool for large batches — each worker loads its own index
  copy, so consider the memory cost before
  turning it on for a modest folder.

  ```powershell
  python examples/analyze_folder.py notes/ --output-dir out/ --outputformat mmi --restrict-to-sts dsyn,sosy
  ```

  **When `--workers` pays off** (measured, 16-core Windows box, warm cache, these two notes
  duplicated to make the batch):

  | files | serial | `--workers 4` |         |
  |------:|-------:|--------------:|---------|
  |     2 |   9.2s |          9.6s | slower  |
  |    10 |  19.0s |         13.2s | 1.4x    |
  |    40 |  55.6s |         23.7s | 2.3x    |

  Break-even is somewhere under ten files; below that the pool costs more than it saves.

  **Don't reach for `--preload` to speed this up** — it is the opposite of a free win. Loading
  the whole term index costs ~15s *per process*, so it makes both paths slower on any batch this
  size: 40 files took 40.6s with `--workers 4 --preload` against 23.7s for `--workers 4` alone,
  and 24.9s serial-with-preload against 9.2s serial on two files. Nor does it amortize over a
  bigger batch: per-note time is the same with or without it (33 vs 36 ms for a 2 KB note,
  measured A/B), so the
  load is pure overhead.

- **`json_to_sqlite.py`** — load one or more `--outputformat json` files (from either script
  above) into a normalized SQLite database for querying: `entities`, `concepts`,
  `concept_semantic_types`, `concept_sources`, `evidence`. Re-running for a docid already in the
  database replaces that docid's rows, so it's safe to re-run after re-annotating a file.

  ```powershell
  python examples/json_to_sqlite.py out/*.json --db examples/annotations.sqlite
  sqlite3 examples/annotations.sqlite "select preferred_name, count(*) n from evidence join concepts using(cui) group by cui order by n desc limit 10"
  ```
  Takes file paths, directories (with `--pattern`, default `*.json`), or glob patterns — and
  expands the glob itself, so `*.json` works the same in PowerShell (which doesn't expand
  wildcards before handing them to a program) as in a POSIX shell. Also accepts JSON written by
  a PowerShell `>` redirect (`mmlite annotate ... --outputformat json > note.json`), which
  is UTF-8 with a byte-order mark or UTF-16 depending on the PowerShell version.

## The flat annotation table

**One row per (span, concept)**: the shape to query for cohorts. The row shape is defined once, in
**`_annotations.py`** (a helper module, not a script), and the scripts below share it.

Three conventions run through it:

- `document` (or `source` in JSONL) is the note's **file name**, never its path — a path records
  where the corpus happened to be mounted and stops two exports joining.
- A note with no findings contributes **no rows**.
- An attribute the run never assessed gets **no column**, and where a column is kept it is
  empty/NULL — *not* 0. NegEx (the default) assesses `negated` only; `--usecontext` also assesses
  `assertion`, `temporality` and `experiencer`; `--no-negation` assesses none. NULL means "not
  assessed" and 0 means "assessed, and absent", so query with `IS NULL`, not `= 0`.

Remember that one span can carry several candidate CUIs, so count spans with
`COUNT(DISTINCT document, start, end)` rather than `COUNT(*)`.

- **`load_to_sqlite.py`** — load a CSV or JSONL export in that shape into a queryable SQLite
  database. Columns absent from the export are not built (`--all-columns` builds them anyway, for
  a schema that must not change shape between loads), and `--append` adds to an existing database.

  ```powershell
  python examples/load_to_sqlite.py out/annotations.csv out/annotations.db
  ```

  **This is not a duplicate of `json_to_sqlite.py`.** That one loads JSON files mmlite has
  already written, into a *normalized* schema (`entities`, `concepts`, `evidence`, …) — the shape
  to query when you care about concepts as first-class things. This one writes the *flat*
  one-row-per-(span, concept) table.

- **`notes_of_interest.py`** — simple phenotyping: which notes *affirm* a concept. This is the one
  that shows why the pipeline beats `grep` — a keyword search for "chest pain" also matches
  "patient denies chest pain", and on the sample corpus 2 of the 6 notes mentioning it mention it
  only as negated. Matching is by CUI, so "T2DM", "DM2" and "diabetes mellitus type 2" collapse
  to one concept. `--save PATH` appends the search to SQLite (`search`, `concepts`, `documents`,
  `console`), keeping a history rather than overwriting.

  ```powershell
  python examples/notes_of_interest.py free_texts/synthetic "chest pain" --show-negated
  python examples/notes_of_interest.py free_texts/synthetic --cui C0011860 --save cohort.sqlite
  ```

  A bare phrase is resolved against the index first and **every** CUI it resolves to becomes a
  target — the resolved list is printed before the scan because a phrase resolving to something
  you did not mean is the commonest way a cohort goes wrong. `"chest pain"` resolves to seven
  concepts, including bare `Pain` and `Chest`. Use `--cui` when you already know what you want.

- **`filter_mrconso.py`** — shrink MRCONSO.RRF (18M+ rows / multiple GB) to the source
  vocabularies you actually map codes from. Anything that streams MRCONSO once per vocabulary pays
  for a full scan every time; this does that scan once, keeps only the SABs you name (default
  `SNOMEDCT_US,ICD10CM,RXNORM,LNC` — override with `--sab`), and writes them to a small
  standalone file (default `data/mrconso_filtered.RRF`) — still valid MRCONSO.RRF, so anything
  that reads MRCONSO reads it unchanged.

  ```powershell
  python examples/filter_mrconso.py --mrconso <path>\MRCONSO.RRF
  ```
  Measured on UMLS 2026AA: one pass (16.7s warm, 28.2s cold) shrinks 18,064,970 rows / ~2.3GB
  down to 3,070,523 rows / ~430MB. These are wall-clock on one Windows machine and the work is
  I/O-bound, so the absolute numbers swing roughly 2x with page-cache state and any concurrent
  disk load; the reduction is the part that holds.

  Contains real UMLS-derived strings just like MRCONSO.RRF itself — gitignored (`data/`), never
  commit or redistribute it.

- **`filter_mrrel.py`** — the same idea for `MRREL.RRF`, keeping the relationships that link a
  drug to its ingredients and that make up the ICD-10-CM and SNOMED CT hierarchies. **Unlike
  `filter_mrconso.py`, filtering by source vocabulary alone barely helps here** — `RXNORM`'s own
  relationships span dozens of relationship types across 15M+ rows, and only three
  (`has_ingredient`/`consists_of`/`tradename_of`) link a drug to its ingredients. So this filters
  by exact `(SAB, REL or RELA)` combinations; see the script's docstring before widening one.
  Suppressed rows are deliberately *not* dropped — suppression is left to whatever reads the
  result.

  ```powershell
  python examples/filter_mrrel.py --mrrel <path>\MRREL.RRF
  ```
  Measured on UMLS 2026AA: one pass (37.8s warm, 63.2s cold) shrinks 66,241,184 rows / ~6.4GB
  down to 1,741,968 rows / ~181MB (default `data/mrrel_filtered.RRF`), a ~38x reduction counting
  suppressed rows. As above, the reduction is the reliable figure, not the seconds.

Run any of these with `--help` for the full flag list. All exit 1 with a one-line error (no
traceback) on a missing/corrupt index, bad input path, unwritable output path, or (for
`json_to_sqlite.py`) input files that all fail to parse; `analyze_folder.py` and
`json_to_sqlite.py` additionally report per-file errors without aborting the rest of the batch.

`analyze_file.py` and `analyze_folder.py` read notes as UTF-8. A note that isn't valid UTF-8
(e.g. a Windows-1252 export containing "Sjögren") is still annotated but gets a **warning**,
because its undecodable bytes become U+FFFD and the words containing them can no longer match —
rerun with `--encoding cp1252` (or whatever the files really are).

## Reading the matches critically

Free-text matching is ambiguous, and MetaMapLite (Java included — these were all confirmed
against it on this project's own notes) resolves some of it in ways that surprise people:

- **Common words match concepts.** "for" → *WWOX gene*, "Plan" → *Infantile Neuroaxonal
  Dystrophy*, "HPI" → *Proline dehydrogenase deficiency*. These dominate the "top concepts"
  summary `analyze_folder.py` prints, which is a frequency count, not a clinical ranking.
- **Ordinary words match drugs.** "Today" resolves to the veterinary brand *ToDAY* (cephapirin);
  "date" matches *date allergenic extract*. Nothing
  downstream can tell these from a real prescription, so check `evidence.matched_text` before
  trusting a drug row.
- **`--restrict-to-sts` can *add* a false positive.** Semantic-type filtering runs before
  overlapping spans are pruned, so removing a longer match can expose a shorter one hiding
  inside it: with `--restrict-to-sts dsyn,sosy`, "ICD-10" loses its own (non-dsyn) match and the
  bare "ICD" then matches *Type II Mucolipidosis*. Java behaves identically.

Two levers help, and both are worth setting before you trust a corpus-wide count:

- `--restrict-to-sts` to the semantic types you actually want (`dsyn,sosy` for problems,
  `phsu,orch,clnd` for drugs), keeping the caveat above in mind.
- `--excluded-terms specialterms.txt` — a `CUI:term` exclusion list, the same mechanism Java
  applies **by default** and this port does not. Java's own list (43 entries, in its
  `data/specialterms.txt`) is a starting point at best: it is aimed at general English
  (`C0340978:may`, `C1420522:fact`), and on this project's notes it removed exactly **one** of
  1,281 concepts. None of the matches listed above are in it — so expect to write your own
  lines, not to inherit a fix.

  **The `term` is the normalized matched text, never the preferred name.** This is the easiest
  thing to get wrong, because the preferred name is what the summary used to show you and what
  looks like the obvious key. `C0332287` is displayed as *"In addition to"* but is matched by
  the word **"with"**, so `C0332287:in addition to` does nothing and `C0332287:with` is the line
  that works. `analyze_folder.py` now prints the correct key in its `matched as` column:

  ```
    mentions  files  cui       matched as              preferred name
          13      2  C0332287  with                    In addition to
          12      2  C1421525  for                     WWOX gene
  ```

  so the exclusion file can be transcribed straight off that table. A leading `*` wildcards the
  CUI — `*:for` drops *every* concept matched by "for", which on these notes is four of them
  (`C0521125`, `C1421525`, `C1562169`, `C4321252`) in one line.
