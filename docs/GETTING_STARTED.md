# Getting started with mmlite

*A user manual for your first hour. For every flag, field and format, see the
`--help` on every command, and the REST API's own `/docs` page.*

mmlite reads clinical or biomedical text and tells you which UMLS concepts it mentions,
where, and whether they are negated. Give it *"No history of myocardial infarction"* and it
returns `C0027051 Myocardial Infarction`, at characters 14–35, **negated**. It is a Python port of
the U.S. National Library of Medicine's MetaMapLite and reproduces its results closely
(F1 0.97 against the Java tool on the same UMLS release), with no Java to install.

## What you need

| | |
|---|---|
| **A UMLS licence** | The dictionary is built from your own UMLS download — it is never shipped with the package. Get a free licence and the release files at [uts.nlm.nih.gov](https://uts.nlm.nih.gov/). You need `MRCONSO.RRF` and `MRSTY.RRF`; `MRSAT.RRF` is optional but improves ranking. |
| **Python 3.11 or newer** | 3.12 if you plan to use the scispaCy tagger (see the end of this guide). |
| **Disk** | ~2.3 GB for the built index; ~13 GB more while the raw UMLS files are on disk. |
| **RAM** | ~300 MB while annotating. |
| **`uv`** | The commands below use [uv](https://docs.astral.sh/uv/); plain `pip` works the same way. |

Everything runs on your machine. After installation no network access is needed.

## 1. Install (5 minutes)

From the source checkout:

```powershell
uv venv
uv pip install --python .venv -e ".[nlp,server]"
uv pip install --python .venv en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

Or from the wheel in `dist/`, into any environment:

```powershell
uv pip install "mmlite[nlp,server]@./dist/mmlite-0.1.0-py3-none-any.whl"
uv pip install en_core_web_sm@https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

The `nlp` extra is spaCy, which supplies part-of-speech tags; `server` is the REST API. Either
can be left out: without `nlp`, pass `--no-postag` to every `annotate` call.

On Windows, you can skip activating the environment and call the tools directly as
`.venv\Scripts\mmlite.exe`; the rest of this guide writes `mmlite` for short.

Check it is installed:

```powershell
mmlite --help
```

## 2. Build the index (3 minutes)

Point it at your UMLS files. This reads them once and writes a single SQLite file:

```powershell
mmlite -v build-index --mrconso D:\umls\2026AA\META\MRCONSO.RRF --mrsty D:\umls\2026AA\META\MRSTY.RRF --mrsat D:\umls\2026AA\META\MRSAT.RRF --out ivf
```

`-v` shows progress. UMLS 2026AA with every English source takes about 2.5 minutes and produces
`ivf/mmlite.sqlite`, 2.2 GB. Once it exists you can delete the raw `.RRF` files if you
want the disk back; the index is self-contained.

Confirm it:

```powershell
mmlite index-stats
```

You should see counts in the millions (`rows_cuisourceinfo: 9112359` for 2026AA) and a
`built_at` timestamp.

> Every command looks for the index in `ivf/` under the current directory. Working elsewhere?
> Pass `--indexdir D:\path\to\ivf`, or set `MMLITE_INDEX_DIRECTORY` once and forget it.

## 3. Your first annotation

```powershell
mmlite annotate "No history of myocardial infarction. Patient takes metformin for type 2 diabetes." --restrict-to-sts dsyn,phsu
```

The default output is MetaMapLite's `mmi` format, one line per concept:

```
00000000.tx|MMI|8.29|Myocardial Infarction|C0027051|[dsyn]|"Myocardial Infarction"-text-0-"myocardial infarction"-JJ-1|text|14/21|C23.550.717.489.750;C23.550.513.355.750;C14.280.647.500;C14.907.585.500|
00000000.tx|MMI|3.68|Diabetes Mellitus, Non-Insulin-Dependent|C0011860|[dsyn]|"Type 2 Diabetes"-text-0-"type 2 diabetes"-NN-0|text|65/15|C18.452.394.750.149;C19.246.300|
00000000.tx|MMI|2.30|metformin|C0025598|[orch, phsu]|"metformin"-text-0-"metformin"-NN-0|text|51/9|D02.078.370.141.450|
```

Reading a line, field by field: document id · `MMI` · score · **preferred name** · **CUI** ·
semantic types · the matched text and its part of speech (the trailing `-1` on the first line is
the **negation flag**; `-0` means affirmed) · field · **character offset/length** · MeSH tree
codes.

Try it once *without* `--restrict-to-sts` and it will look noisy — MetaMapLite maps *every*
word it can, so `for`, `history` and `Patient` each yield concepts too. That is expected, and
the next section is how everyone handles it.

## 4. The five things you'll do next

**Keep only the concept types you care about.** Restrict to UMLS semantic types — diseases
(`dsyn`), signs and symptoms (`sosy`), drugs (`phsu`), procedures (`topp`), and so on:

```powershell
mmlite annotate --input note.txt --restrict-to-sts dsyn,sosy,phsu
```

`mmlite lookup "<term>"` shows the semantic types of any concept, and the REST API's
`GET /semantic-types` lists all of them. A typo (`dysn`) is not an error — it quietly matches
nothing — so check the abbreviation if you get an empty result.

**Get structured output.** `--outputformat json` gives one object per matched span with its
offsets, `negated` flag and concept list; `brat` produces standoff annotations that line up with
the source file; `cuilist` is just the CUIs, one per line.

```powershell
mmlite annotate --input note.txt --outputformat json --restrict-to-sts dsyn,sosy > note.json
```

**Understand negation.** *"denies fever"*, *"no evidence of MI"*, *"no family history of
cancer"* all come back with `negated: true`. For richer context — was it the patient's mother?
is it hypothetical? — switch to the ConText detector, which adds `temporality` and
`experiencer` to every entity. The `full` output format shows them (`json` is Java's layout
and has no such fields; the REST API's own JSON carries them):

```powershell
mmlite annotate --usecontext --outputformat full "Mother had breast cancer. If she develops angina, order a stress test." --restrict-to-sts neop,sosy
```

```
11:13 'breast cancer' (NN) [Recent] [Other]
    C0006142 Malignant neoplasm of breast [neop] {...} <- 'Breast Cancer'
42:6 'angina' (JJ) [Hypothetical] [Patient]
    C0002962 Angina Pectoris [sosy] {...} <- 'Angina'
```

**Check a single term.** Before wondering why something wasn't found, see whether it's in the
dictionary at all and what it normalizes to:

```powershell
mmlite lookup "type 2 diabetes mellitus"
mmlite normalize "Alzheimer's Disease"
```

## 5. From Python

```python
from mmlite import MetaMapLite

mml = MetaMapLite(index_directory="ivf")        # loads once; reuse it for every document
entities = mml.process_text(
    "No history of type 2 diabetes mellitus (T2DM). T2DM is treated with metformin.",
    restrict_to_sts={"dsyn", "phsu"},
)
for e in entities:
    print(e.start, e.length, e.text, e.negated, [(ev.cui, ev.concept.preferred_name) for ev in e.evs])
```

```
14 24 type 2 diabetes mellitus True  [('C0011860', 'Diabetes Mellitus, Non-Insulin-Dependent')]
40  4 T2DM                     False [('C0011860', 'Diabetes Mellitus, Non-Insulin-Dependent')]
47  4 T2DM                     False [('C0011860', 'Diabetes Mellitus, Non-Insulin-Dependent')]
68  9 metformin                False [('C0025598', 'metformin')]
```

Notice `T2DM`: the abbreviation was defined in the first sentence and resolved to the same
concept afterwards. `mml.process_file("note.txt")` annotates a file, and
`mml.format(entities, "mmi")` renders any of the text formats.

## 6. Over HTTP

```powershell
mmlite serve --indexdir ivf --port 8000
```

Then, from anything that can speak JSON:

```bash
curl -X POST http://127.0.0.1:8000/annotate -H "Content-Type: application/json" \
  -d '{"text":"No chest pain.","restrict_to_sts":["sosy"]}'
```

Interactive documentation for every endpoint is at <http://127.0.0.1:8000/docs>. Requests take
10–50 ms for a short note. The server is single-tenant by design — no authentication, bound to
localhost — so review your deployment carefully before exposing it to anyone
else.

## 7. Things that surprise people

- **Offsets are zero-based character positions** into the text you gave, and they never move:
  a CRLF file keeps its `\r`s so that `brat` output lines up with the file byte for byte.
- **Line-initial drug names in medication lists can be missed** (`metformin 1000 mg PO BID`
  at the start of a line). A block of dose lines with no full stops reads as one long
  "sentence", and the tagger takes the drug names for verbs. If you process medication lists,
  see `--postag-model` in `mmlite annotate --help`:
  on Python 3.12 the scispaCy model finds 27/30 where the default finds 20, and there is a
  recipe for Python 3.13.
- **Accented and Greek terms work** — `Sjögren syndrome`, `Ménière disease`, `β-blocker` — and
  in fact work *better* than in the Java original, which mishandles non-ASCII dictionary keys.
- **The same text always gives the same answer.** There is no randomness anywhere in the
  pipeline, so results are reproducible run to run and machine to machine given the same index.
- **Semantic types and sources print in an odd order** (`[orch, phsu]`, not alphabetical). That
  is Java's `HashSet` order, reproduced on purpose so output can be diffed against MetaMapLite's.

## Where next

- `mmlite <command> --help` — every CLI flag, with its default and its properties key.
- `/docs` on a running server — the REST API's own OpenAPI reference.
- [UMLS identifiers](<UMLS identifiers.md>) — if CUI, LUI, SUI, AUI and TUI are new to you.
