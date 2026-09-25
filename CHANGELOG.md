# Changelog

All notable changes to mmlite are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Rationale for the parity-driven decisions behind these changes is recorded in the project's
development plan, an internal working document that is not distributed.

## [Unreleased]

## [0.1.4] — 2026-09-25

**No library change** — `src/mmlite` is identical to 0.1.3, so the wheel behaves exactly as
before. This release exists for the sdist, which no longer bundles the example scripts below.

### Removed

- The corpus exporters `examples/parse_to_csv.py`, `parse_to_jsonl.py`, `parse_to_jsonl_batch.py`,
  `parse_to_parquet.py` and `parse_to_sqlite.py`, the comparator `compare_exports.py` and their
  run-directory helper `_runs.py` are no longer distributed, in the repository or the sdist.
  `load_to_sqlite.py` still loads a CSV or JSONL export in the shared `_annotations.py` row shape;
  its tests now write those exports themselves.
- The UMLS code-mapping scripts `examples/add_snomed_codes.py`, `add_icd10cm_codes.py`,
  `add_rxnorm_codes.py`, `add_loinc_codes.py`, `add_rxnorm_ingredients.py`,
  `add_icd10cm_hierarchy.py`, `add_snomedct_hierarchy.py` and `create_combined_codes_view.py` are
  no longer distributed either. `filter_mrconso.py` and `filter_mrrel.py` remain, documented by
  what they produce rather than by those scripts.

### Fixed

- On Python 3.11 `examples/notes_of_interest.py` rejected its optional `term` phrase whenever a
  flag came before it (argparse's pre-3.12 behaviour), failing with "unrecognized arguments". It
  now uses `parse_intermixed_args`. The same bug in the exporters above was the cause of the
  failing 3.11 CI job.

## [0.1.3] — 2026-09-22

**No library change** — `src/mmlite` is identical to 0.1.1, so the wheel behaves exactly as
before. This release exists for the sdist, whose bundled `examples/` otherwise documents paths
that no longer exist.

### Changed

- The filtered MRCONSO/MRREL extracts the `examples/add_*.py` scripts read now live in **`data/`**
  rather than beside the scripts, leaving `examples/` holding source only. The five `--mrconso` /
  `--mrrel` defaults resolve there; both filter scripts already create the directory on demand,
  so nothing needs creating by hand.
- The sample clinical notes moved to `free_texts/synthetic/`, joining the rest of the corpus.
  They are no longer distributed: `free_texts/` is gitignored, being notes rather than source.
- `json_to_sqlite.py`'s first usage line referenced an annotation-output file that is no longer
  kept; it now shows how that JSON is produced before loading it.

## [0.1.2] — 2026-09-22

### Fixed

- **A start-up failure no longer blames the POS tagger for something else.** Every error while
  opening an index or building a pipeline had `(or pass --no-postag to skip part-of-speech
  tagging)` appended, so the commonest first-run failure —

      $ mmlite annotate "..." --no-postag
      no index at ivf/mmlite.sqlite; run `mmlite build-index` first
      (or pass --no-postag to skip part-of-speech tagging)

  answered a missing *index* with advice about *tagging*, and suggested a flag that was already
  in effect. The hint now appears only when the tagger really is what failed.

  Distinguishing the two needed a type: a missing model and a missing index are both `OSError`.
  `spacy_nlp` now raises `TaggerImportError` (spaCy absent) or `TaggerModelNotFound` (model
  absent), named together by `spacy_nlp.TAGGER_UNAVAILABLE`. Both keep their original base
  classes — `ImportError` and `OSError` respectively — so existing handlers are unaffected; one
  class cannot subclass both, as their C layouts conflict.

- **README links now resolve on the PyPI project page.** They were repository-relative, which
  works on GitHub and 404s from PyPI, where there is no repository context. They are absolute
  now, written reference-style so the prose stays readable.

## [0.1.1] — 2026-09-22

**Renamed from `pymetamaplite` to `mmlite`.** NLM's own Lister Hill Center publishes a Python
MetaMapLite of its own at [LHNCBC/pymetamaplite](https://github.com/LHNCBC/pymetamaplite), whose
distribution is also named `pymetamaplite` and whose author is MetaMapLite's own maintainer. That
project predates this one by four years and is still maintained. It has never been published to
PyPI, so the name was *available* — but it was not unclaimed, and anyone following its install
instructions with `pip install pymetamaplite` would have silently received this package instead.

The old name has been released back to PyPI. `mmlite` was already this project's internal short
form (it is the `brat` type name), and does not collide with NLM's distribution name or with the
`metamaplite` module their package installs.

This is a breaking rename with no deprecation shim: `pymetamaplite` 0.1.0 was published the same
day and had no users.

- `pip install pymetamaplite` → `pip install mmlite`
- `import pymetamaplite` → `import mmlite`
- `pymetamaplite …` on the command line → `mmlite …`
- `PYMETAMAPLITE_*` environment variables → `MMLITE_*`
- the index file is now `<indexdir>/mmlite.sqlite`. An existing index needs only a file rename,
  not a rebuild.

No functional change: the pipeline, output formats and parity figures are exactly as in 0.1.0.

## [0.1.0] — 2026-09-19

First release, as `pymetamaplite`. A Python port of MetaMapLite 3.6.2rc8 — the whole
`EntityLookup4` path — as a library, a CLI and a REST API, with no JVM.

Measured against Java MetaMapLite 3.6.2rc8 on UMLS 2026AA: the index tables are identical row
for row, tokenization is exact, and concepts found `(CUI, positions)` agree at P 0.975 / R 0.963
/ F1 0.969. Most of the residual gap is the two NLP models MetaMapLite uses (OpenNLP's sentence
detector and POS tagger), which have no Python equivalent. See
[tests/parity/README.md](tests/parity/README.md) for the method.

### Annotation

- **Index building** from UMLS RRF (`MRCONSO`, `MRSTY`, optionally `MRSAT` for MeSH tree codes),
  producing the `cuisourceinfo`, `cuiconcept`, `cuist` and `meshtcrelaxed` tables in SQLite.
  `--sources` restricts the build; `--include-suppressed` keeps `SUPPRESS=O/E/Y` rows.
- **The lookup pipeline**: normalization, sentence segmentation, tokenization, POS gating,
  longest-match candidate generation, semantic-type and source restriction (abbreviations or
  TUIs, any case), subsumption removal, Schwartz–Hearst abbreviation propagation, user-defined
  acronyms, custom concept lists and excluded terms.
- **Negation**, two detectors: NegEx (the default, as in Java) and ConText (`--usecontext`),
  which also reports temporality and experiencer.
- **Greek-letter spans.** A span may start or end with a Greek letter, as Java's `CharUtils`
  allows, so `β-blocker` matches.
- **Non-ASCII dictionary keys** resolve reliably. Java's binary search compares UTF-8 as signed
  bytes against a file sorted in `String` order and finds only about 44% of them, missing terms
  such as `Sjögren` and `Ménière disease`; this is a deliberate divergence, pinned by name in the
  test suite rather than averaged into the parity figures.
- **Precision filter** (`Settings.precision_filter`, default off, not in Java): drops a concept
  whose only dictionary strings for the matched text are acronyms when the text is not in
  capitals ("Plan" loses OMIM's `PLAN`), and one-word function-word mentions ("for", "with")
  unless capitalised. Over the parity corpus it removed 267 of 2,907 matches, 261 of them (97.8%)
  judged wrong or useless by hand.

### Input and output

- **Every input format MetaMapLite registers**: `freetext`, `sli`, `sldi`, `sldiwi`, `chemdner`,
  `chemdnersldi`, `ncbicorpus`, `pubtator`, `pubmed`, `medline`, `bioc`. `pubtator` is a
  correction rather than a port — Java's loader closes its reader inside the read loop and throws
  on every input. A document model and loader registry (`mmlite.documents`) takes a new
  format without changes elsewhere.
- **Output formats**: `mmi` (the default), `json`, `brat`, `cuilist`, `full`, and `bc` with
  Java's aliases `bc-evaluate`, `cdi` and `bioc`.
- **Segmentation**: `SENTENCES`, `BLANKLINES`, `LINES`.

### Interfaces

- **CLI** (`mmlite`): `annotate`, `lookup`, `normalize`, `tokenize`, `build-index`,
  `index-stats`, `serve`. `annotate` mirrors `metamaplite.sh`.
- **Python API**: `MetaMapLite(index_directory=...)`, `process_text`, `process_file`, `format`.
  Configuration comes from a `Settings` object, MetaMapLite's own `metamaplite.properties` keys,
  or `MMLITE_<NAME>` environment variables.
- **REST API** (`serve`): `POST /annotate`, `/annotate/batch`, `/annotate/text`,
  `/annotate/formatted`, `GET /lookup`, `/formats`, `/semantic-types`, `/health`, with OpenAPI
  docs at `/docs`. Requests are serialised behind a lock — annotation is CPU-bound and the
  underlying objects are not thread-safe — so scale out with processes.
- **Opt-in REST authentication and CORS.** `MMLITE_API_TOKEN` makes every endpoint except
  `/health` require `Authorization: Bearer <token>` (constant-time comparison);
  `MMLITE_CORS_ORIGINS` adds `CORSMiddleware`. Both off unless set.
- **Request limits**: `MMLITE_MAX_TEXT_CHARS` (50,000, per document) and
  `MMLITE_MAX_FORMATTED_CHARS` (1,000,000, the raw `/annotate/formatted` body). Both are
  checked right after parsing, before anything is annotated.

### Configuration and extension

- **`strict_parity`** (default `true`). `true` reproduces MetaMapLite 3.6.2rc8 exactly, bugs
  included. `false` fixes the upstream bugs that give wrong answers, all in ConText:
  - Only the mentions inside the sentence being read are analysed, each within its own bullet
    item. Java applies every sentence to every mention in the document, so a later "Family
    history of hypertension in mother" turned the patient's own earlier "hypertension" into
    `Historical` / `Other`. This also makes ConText close to linear in document length rather
    than quadratic.
  - The longest trigger at a position wins, instead of whichever bucket is substituted first.
    "Pneumonia was ruled out." is negated by the *post* trigger "was ruled out"; Java tagged it
    with the shorter *pre* trigger "ruled out", which lands at the end of the sentence with
    nothing after it to negate. "may be ruled out for X" is `Possible`, not a negation.
  - Triggers stored with a trailing space (`denies `, `no `) fire anywhere, not only right before
    the concept, so a whole "Denies A, B, C" list is negated rather than its first item — and
    "no evidence of X" / "without evidence of X" negate at all.
  - Time patterns match ("3 months of", "since last march"); they were character classes where
    groups were meant. "previous" marks the next word historical. The first pseudo-trigger keeps
    its first letter. "Rule out X" / "r/o X" gives `Possible`.
  - Experiencer and temporality triggers inside a concept's own name count; negation triggers
    there deliberately do not ("No known allergies" is itself the finding).
  - `mmi`, `json` and `brat` list semantic types and sources alphabetically and documents in
    input order, instead of Java `HashSet` order.

  NegEx output is identical in both modes. `GET /health` reports `strict_parity`, and a server
  running strict-parity ConText logs a warning once.
- **Extra ConText triggers** (`context_triggers.EXTRA_TRIGGERS`, fixed mode only). `cannot` is a
  negation trigger on its own, so "cannot exclude pneumonia" came out *negated* — the opposite of
  what it says. Adds the exclusion family (`cannot exclude`, `cannot be excluded`,
  `cannot rule out`, `unable to exclude`, …), the hedges `concerning for`, `suspicious for`,
  `suspected`, `differential includes`, `low suspicion for`, `possible`, `probable`, and
  `absent` / `is absent` / `are absent`. `ConText(extra_triggers=...)` takes your own list.
- **`Entity.assertion`** carries ConText's verdict — `Affirmed`, `Negated` or `Possible`. Java
  keeps only a boolean and discards the verdict, so a hedged mention read exactly like an
  affirmed one. Shown as `[possible]` in `full` output and returned by the REST API; `negated` is
  unchanged, so `mmi`, `json` and `brat` are untouched. `None` under NegEx.
- **Pluggable assertion.** `AssertionClassifier` adapts any per-sentence assertion model
  (medspaCy's ConText, a transformer classifier, site rules) to the pipeline.
  `register_negation_detector(name, factory, aliases)` makes it selectable wherever `negex` /
  `context` are, and packages can declare one under the `mmlite.negation_detectors`
  entry-point group. `--negation-detector NAME` on `annotate` and `serve`.
- **Selectable POS tagger.** `Settings.postag_model` names the spaCy pipeline whose tagger gates
  candidate spans; the default is `en_core_web_sm`. scispaCy's `en_core_sci_sm` agrees with Java
  better (F1 0.977 vs 0.969) and is opt-in because its models need spaCy 3.7 / Python ≤ 3.12 —
  `requirements-scispacy-py312.txt` pins a tested environment for it.
- **POS verb rescue** (`Settings.postag_verb_rescue`, default `0.03`) re-tags a token the model
  calls a finite verb with its most likely noun or adjective reading. `en_core_web_sm` calls rare
  clinical words verbs far more readily than OpenNLP, and the POS gate then never looks them up.
  Raises parity F1 from 0.969 to 0.974 while *reducing* extras.

### Performance

- Annotation runs at ~500 sentences/s on one core; resident memory ~120 MB. An index build over
  UMLS 2026AA (all English sources) takes ~2.5 minutes and produces a 2.2 GB SQLite database:
  9.1 M terms, 7.3 M distinct keys, 3.5 M CUIs.
- Abbreviation propagation tokenizes each passage once rather than once per abbreviation-bearing
  entity (48 KB: 7.0 s → 2.1 s, byte-identical output), and NegEx receives only the entities
  starting inside each sentence.
- The four per-concept caches are LRU-capped at 50,000 entries each (~125 MB together) instead of
  growing for the life of the process.
- `--preload` / `MMLITE_PRELOAD_INDEX` is off by default and documented as rarely worth
  turning on: measured A/B with the index on local disk it is no faster, and costs ~4 s of
  start-up and ~1.65 GB RSS per process.

### Not implemented, deliberately

- **MetaMap-style scoring** (`EntityLookup5`, Java's `--enable_scoring`), off by default in Java.
  Measured rather than assumed: with NLM's own index the score takes one of two values, carrying
  ~0.04 bits per evidence. See the development plan, §10.1.
- **Word sense disambiguation, derivational variants, disjoint entities** — absent from
  MetaMapLite itself.

### Examples

Runnable scripts in [examples/](examples/) — a starting point to adapt, not a library. See
[examples/README.md](examples/README.md).

- **Annotation** — `analyze_file.py`, `analyze_folder.py` (one loaded index reused across a
  folder, `--workers` for a process pool) and `json_to_sqlite.py`, which loads `--outputformat
  json` files into a normalized `entities`/`concepts`/`evidence` database.
- **Corpus exports** — `parse_to_csv.py`, `parse_to_jsonl.py`, `parse_to_jsonl_batch.py`,
  `parse_to_parquet.py` and `parse_to_sqlite.py` write one row per (span, concept) in five
  formats from a single row shape (`_annotations.py`), so exports of one corpus agree and join
  on the same key. `compare_exports.py` checks that rather than asserting it, and exits non-zero
  on disagreement. `load_to_sqlite.py` completes the two-step route, sharing its schema with
  `parse_to_sqlite.py` so both end at the same database.
  - An attribute the run never assessed gets **no column**, and where one is kept it is
    empty/NULL rather than 0: NegEx assesses `negated` only, `--usecontext` adds `assertion`,
    `temporality` and `experiencer`, `--no-negation` assesses none. NULL means "not assessed"
    and 0 means "assessed, and absent".
  - Each run saves itself into a fresh `out/runs/<run-id>/` directory beside a `manifest.json`
    recording the settings that produced it (`_runs.py`), so no run overwrites another.
- **Phenotyping** — `notes_of_interest.py` finds the notes that *affirm* a concept, counting a
  document only when the mention is not negated, and matching by CUI so synonym and
  morphological variants collapse.
- **Terminology crosswalks** — `filter_mrconso.py` / `filter_mrrel.py` and the `add_*_codes.py`
  and `add_*_hierarchy.py` scripts map matched concepts to SNOMED CT, ICD-10-CM, RxNorm and
  LOINC, with `create_combined_codes_view.py` over the result.

### Documentation

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) — install to first results.
- [docs/UMLS identifiers.md](<docs/UMLS identifiers.md>) — CUI, LUI, SUI, AUI and TUI explained.

### Licensing

Released under the same licence as MetaMapLite — NLM's open-source BSD licence — with the
upstream Informational Notice retained in [NOTICE-MetaMapLite.md](NOTICE-MetaMapLite.md).
mmlite is an independent port and is **not** developed, funded or endorsed by NLM, NIH or
the U.S. Government. The UMLS data it indexes is licensed separately.

### Quality

706 tests, `ruff` clean, `mypy --strict` clean over `src/mmlite`, CI on Python 3.11–3.13.
Every `examples/` script has its own test file under `tests/examples/`, run against small
synthetic MRCONSO/MRSTY fixtures and an index built from them, so the suite needs no UMLS
download. The parity tests skip wherever licensed fixtures are absent.
