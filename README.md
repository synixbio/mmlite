# mmlite

A Python port of [MetaMapLite][metamaplite],
the U.S. National Library of Medicine's near-real-time UMLS concept recognizer — as a library,
a CLI, and a REST API, with no JVM.

*Courtesy of the U.S. National Library of Medicine.*

> **Not NLM's own Python port.** NLM's Lister Hill Center maintains one at
> [LHNCBC/pymetamaplite](https://github.com/LHNCBC/pymetamaplite), written by MetaMapLite's own
> maintainer — if you want the implementation NLM stands behind, use that one. This is an
> independent reimplementation, verified against the Java release rather than derived from their
> Python code, and it is not endorsed by NLM. This project was briefly published as
> `pymetamaplite`; it was renamed to `mmlite` in 0.1.1 to stop the two being confused, and the
> old name was released back to PyPI.

This README is a quick overview. **New here? Start with
[docs/GETTING_STARTED.md][getting-started]** — install, build the index and get your
first results in under an hour. New to UMLS identifiers (CUI, LUI, SUI, AUI, TUI)? See
[docs/UMLS identifiers.md][umls-ids]. Release notes are in
[CHANGELOG.md][changelog]. Every CLI flag has `--help`, and the REST API documents itself at
`/docs`.

## Verified against the original

Measured against **Java MetaMapLite 3.6.2rc8** on UMLS 2026AA (see
[tests/parity/README.md][parity] for the method and how to reproduce):

| | |
|---|---|
| Index tables (`cuisourceinfo`, `cuiconcept`, `meshtcrelaxed`) | **identical, row for row** |
| Tokenization | **exact**, token for token |
| Concepts found, `(CUI, positions)` | **P 0.975 / R 0.963 / F1 0.969** |
| MMI records byte-identical | 430/565 |
| `json` output, spans found by both | 285/296 agree on every field Java emits |

Index contents, normalization, candidate generation, restriction, subsumption, negation,
abbreviations, MMI scoring and field layout are reproduced deterministically. Most of the
residual gap is the two NLP models MetaMapLite uses — OpenNLP's sentence detector (F1 0.923 vs
ours) and its POS tagger (0.850 tag agreement, 0.960 on the decision that affects results) —
which have no Python equivalent. One known difference is deliberate: Java's index lookup is
**unreliable for dictionary keys containing non-ASCII characters** — its binary search compares
UTF-8 as signed bytes against a file sorted in `String` order, so it finds some and misses others
depending on where the key falls. Measured over a 600-key sample it resolves about 44% of them,
missing terms such as `Sjögren` and `Ménière disease` that this port finds. That accounts for five
of the 35 concept-level disagreements, each pinned by name in the test suite rather than averaged
into the figures above. See the development plan, §16 and §21.

## Install

```powershell
uv venv
uv pip install --python .venv -e ".[dev,nlp,server]"
uv pip install --python .venv en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Extras: `nlp` (spaCy — POS tagging and spaCy sentence detection), `server` (FastAPI + uvicorn),
`all` for both. The core package needs only `typer`; without `nlp`, pass `--no-postag`.

The POS tagger is a spaCy model chosen by name (`--postag-model`, `MMLITE_POSTAG_MODEL`).
On Python ≤ 3.12, scispaCy's `en_core_sci_sm` measures **closer to Java than the default**
(F1 0.977 vs 0.969) and finds 27/30 drugs in a medication list where `en_core_web_sm` finds 20 —
see `--postag-model` in `mmlite annotate --help`.

## Build an index

You supply the UMLS — a licensed download, never redistributed with this package. `MRCONSO.RRF`
and `MRSTY.RRF` are required, `MRSAT.RRF` adds MeSH tree codes (used by MMI scoring):

```powershell
mmlite -v build-index --mrconso ...\MRCONSO.RRF --mrsty ...\MRSTY.RRF --mrsat ...\MRSAT.RRF --out ivf
```

UMLS 2026AA (all English sources) takes ~2.5 min and produces a 2.2 GB SQLite database:
9.1 M terms / 7.3 M distinct keys / 3.5 M CUIs. `--sources MSH,SNOMEDCT_US` restricts it;
`--include-suppressed` keeps `SUPPRESS=O/E/Y` rows.

## Command line

```powershell
mmlite annotate "No history of myocardial infarction." --restrict-to-sts dsyn,sosy
mmlite annotate --input note.txt --outputformat json     # mmi (default) | json | brat | cuilist | full | bc
mmlite annotate --pipe --no-negation < note.txt
mmlite lookup "type 2 diabetes mellitus"                 # dictionary spot-check
mmlite normalize "Alzheimer's Disease"                   # show the lookup keys for a span
mmlite tokenize "Dr. Smith has type 2 diabetes."         # sentences + tokens + POS
mmlite index-stats
```

`annotate` mirrors `metamaplite.sh`: `--restrict-to-sts`, `--restrict-to-sources`,
`--segmentation SENTENCES|BLANKLINES|LINES`, `--inputformat` (`freetext`, `sli`, `sldi`,
`sldiwi`, `chemdner`, `chemdnersldi`, `ncbicorpus`, `pubtator`, `pubmed`, `medline`, `bioc`),
`--excluded-terms`, `--uda`,
`--cui-term-list`, `--usecontext`, `--keep-subsumed`, `--preload`. Restrictions take semantic type abbreviations or TUIs and UMLS
SABs, in any case (`dsyn`, `DSYN`, `T047` and `t047` all select the same concepts).

## Python API

```python
from mmlite import MetaMapLite

mml = MetaMapLite(index_directory="ivf")
entities = mml.process_text(
    "No history of type 2 diabetes mellitus (T2DM). T2DM is treated with metformin.",
    restrict_to_sts={"dsyn", "phsu"},
)
for e in entities:
    print(e.start, e.length, e.text, e.negated, [(ev.cui, ev.concept.preferred_name) for ev in e.evs])
print(mml.format(entities, "mmi"))
```

```
14 24 type 2 diabetes mellitus True  [('C0011860', 'Diabetes Mellitus, Non-Insulin-Dependent')]
40  4 T2DM                     False [('C0011860', 'Diabetes Mellitus, Non-Insulin-Dependent')]
47  4 T2DM                     False [('C0011860', 'Diabetes Mellitus, Non-Insulin-Dependent')]
68  9 metformin                False [('C0025598', 'metformin')]
```

`mml.process_file("note.txt")` annotates a file directly — offsets index the file verbatim
(line endings are never translated), so `brat` output lines up with a CRLF `.txt`.

`MetaMapLite(...)` takes a `Settings` object or `properties_file=` accepting MetaMapLite's own
`metamaplite.properties` keys (`metamaplite.index.directory`, `metamaplite.semanticgroup`,
`metamaplite.enable.postagging`, `metamaplite.detect.negations`, `metamaplite.postaglist`,
`metamaplite.uda.filename`, …); every key also reads from `MMLITE_<NAME>` env vars.

## REST API

```powershell
mmlite serve --indexdir ivf --port 8000        # OpenAPI docs at /docs
```

| Endpoint | Purpose |
|---|---|
| `POST /annotate` | one document → structured JSON entities |
| `POST /annotate/batch` | several documents in one request |
| `POST /annotate/text?format=mmi` | MetaMapLite's own output formats as `text/plain` |
| `POST /annotate/formatted` | a blob in any `--inputformat` (ChemDNER, PubMed XML, BioC, …) → one result per document |
| `GET /lookup?term=…` | dictionary lookup for a single term |
| `GET /formats` | every accepted input/output format, segmentation method and negation detector |
| `GET /semantic-types` | the UMLS semantic types, for building `restrict_to_sts` |
| `GET /health` | status, index metadata, uptime |

Any annotate request may pass `"detector": "context"` to use ConText for that request instead of
NegEx, which adds `temporality` and `experiencer` to each entity.

```bash
curl -X POST http://127.0.0.1:8000/annotate -H "Content-Type: application/json" \
  -d '{"text":"No chest pain.","restrict_to_sts":["sosy"]}'
```

The index and spaCy model load once at start-up (~10 s; `--preload` adds a 4 s term-map load and
~1.6 GB of RAM, and measured no faster, so leave it off). Requests take ~10–20 ms for a short clinical note. Annotation is
CPU-bound and the underlying objects are not thread-safe, so requests are serialised behind a
lock — **scale out with processes** (`uvicorn mmlite.server:app --workers N`, each holding
its own copy of the index), not threads.

## Performance

UMLS 2026AA, all English sources, on a laptop: index build 2.5 min; annotation ~500 sentences/s
(the absolute figure varies a lot with machine state);
resident memory ~120 MB, or ~1.8 GB with `preload_index=True` (a 419 MB pickled term map, 4 s to
load). Preloading measured no faster than the default SQLite lookups with the index on local disk,
so it is off by default and rarely worth turning on.

## What is and isn't implemented

Implemented: the whole `EntityLookup4` path — index building, normalization, segmentation,
tokenization, POS gating, longest-match candidate generation, semantic-type/source restriction,
subsumption removal, NegEx negation, Schwartz–Hearst abbreviations, user-defined acronyms,
custom concept lists, excluded terms, both negation detectors (NegEx by default, ConText via
`--usecontext`, which adds temporality and experiencer), every `--inputformat` MetaMapLite registers (`freetext`,
`sli`/`sldi`, `sldiwi`, `chemdner`, `chemdnersldi`, `ncbicorpus`, `pubtator`, `pubmed`, `medline`,
`bioc` — `pubtator` as a correction, since Java's own loader throws on every input), and the
MMI / JSON / BRAT / CuiList output formats.

Not implemented, deliberately:

- **MetaMap-style scoring** (`EntityLookup5`, Java's `--enable_scoring`) — off by default in Java,
  and measured rather than assumed: with NLM's own index the `vars` table is a single placeholder
  row, so the variation term is identically zero and the coverage and cohesiveness terms reduce to
  1.0 by construction. The score takes one of two values — 666.67, or 833.33 in the rare case that
  a defective head test fires (749 / 4 over the parity corpus). `EntityLookup5`'s substantive
  difference from `EntityLookup4` is its candidate generation, not its scores. See
  the development plan, §10.1, for the closed form and how to reproduce it.
- **Word sense disambiguation, derivational variants, disjoint entities** — absent from
  MetaMapLite itself.

## Development

```powershell
.venv\Scripts\python -m pytest -q        # 706 tests; the parity tests skip without Java fixtures
.venv\Scripts\ruff check src tests scripts examples
.venv\Scripts\ruff format --check src tests scripts examples
.venv\Scripts\mypy                       # strict, over src/mmlite (see pyproject.toml)
python scripts/compare_with_java.py --java-dir <public_mm_lite>      # parity vs the real thing
```

`ruff format` is the formatter; lint rules and the 100-character line length are configured in
`pyproject.toml`. [CI][ci-yml] runs the same lint, `mypy --strict` and the tests
on Python 3.11–3.13 on every push; the parity tests skip there, since their fixtures are
licensed. Anything that changes annotation output must keep the parity harness green — see
[tests/parity/README.md][parity].

## Releasing

[`release.yml`][release-yml] publishes via PyPI Trusted Publishing, so no API
token is stored anywhere — PyPI verifies that this workflow, in this repository, produced the
upload. Running the workflow by hand publishes to **TestPyPI**; pushing a `v*` tag publishes to
**PyPI**. Both build, run `twine check`, and install the wheel into a clean environment before
uploading, and a tagged run additionally refuses to publish if the tag disagrees with
`__version__`. A version number, once on PyPI, can never be reused — so bump
`src/mmlite/__init__.py` and tag it, rather than re-cutting one.

The project's development plan, an internal working document, records what each Java class does
and why each behaviour was reproduced — including deliberate quirks (a span must start and end with a
character Java's `CharUtils` calls alphanumeric, which is ASCII *and Greek*, so `β-blocker`
matches but `-blocker` cannot; sources and semantic types print in Java `HashSet` order).

## Licensing

This package is released under the **same licence as MetaMapLite** — NLM's open-source BSD
licence — reproduced in [LICENSE][license], with the upstream notice retained verbatim in
[NOTICE-MetaMapLite.md][notice]. That licence requires you to keep the
Informational Notice in redistributions and to acknowledge NLM as the source; hence the
"Courtesy of the U.S. National Library of Medicine" line at the top of this README.

mmlite is an independent port and is **not** developed, funded or endorsed by NLM, NIH or
the U.S. Government. The UMLS data it indexes is licensed separately: you need a UMLS
Metathesaurus License and a UTS account, and the index you build from it is yours alone to keep —
don't redistribute it.

[getting-started]: https://github.com/synixbio/mmlite/blob/main/docs/GETTING_STARTED.md
[umls-ids]: https://github.com/synixbio/mmlite/blob/main/docs/UMLS%20identifiers.md
[changelog]: https://github.com/synixbio/mmlite/blob/main/CHANGELOG.md
[parity]: https://github.com/synixbio/mmlite/blob/main/tests/parity/README.md
[ci-yml]: https://github.com/synixbio/mmlite/blob/main/.github/workflows/ci.yml
[release-yml]: https://github.com/synixbio/mmlite/blob/main/.github/workflows/release.yml
[license]: https://github.com/synixbio/mmlite/blob/main/LICENSE
[notice]: https://github.com/synixbio/mmlite/blob/main/NOTICE-MetaMapLite.md
[metamaplite]: https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/NLP/MetaMapLite.html
