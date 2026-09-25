# Parity with Java MetaMapLite

`corpus/` holds the input documents (written for this project, no licensed content). Two of them
exist for a specific regression: `crlf_note.txt` is the only one with CRLF line endings, and
`unicode_terms.txt` the only one with non-ASCII terms (Greek letters and accents).
`fixtures/` holds the *output of Java MetaMapLite 3.6.2rc8* on that corpus, plus its
`specialterms.txt`. Those files contain UMLS-derived concept names and are **not committed**
(see `.gitignore`); regenerate them from a local MetaMapLite install:

```powershell
python scripts/compare_with_java.py     --java-dir D:/Apps/jars/public_mm_lite_.../public_mm_lite --refresh
python scripts/compare_with_java.py     --java-dir D:/Apps/jars/public_mm_lite_.../public_mm_lite --refresh --format json
python scripts/compare_nlp_with_java.py --java-dir D:/Apps/jars/public_mm_lite_.../public_mm_lite --refresh
copy <java>\data\specialterms.txt tests\parity\fixtures\
```

`test_parity.py` then runs without Java and skips when the fixtures or the index are missing.

## Measured 2026-09-17, re-measured 2026-09-18 (UMLS 2026AA USAbase, spaCy `en_core_web_sm` 3.8)

"Byte-identical MMI records" counts Java records reproduced character for character, splitting
both outputs with `compare_with_java.records()`; ordered and unordered counting agree.

`json` is compared **structurally**, never byte for byte. Java builds it with `org.json`, whose
`JSONObject` is a `HashMap`, so key order carries no information and is not reproducible; it also
drops keys whose value is null, and prints `0` where Python prints `0.0`. Array order *is*
significant and is checked: `sources` and `evlist` are Java `HashSet` iteration order, which this
port reproduces. Our output is a superset of Java's — see the `fieldid` / `negated` note in
`test_parity.py`.

| Layer | Result |
|---|---|
| `cuisourceinfo` table | **identical**, all 9,112,359 rows |
| `cuiconcept` / `meshtcrelaxed` tables | **identical** (3,530,466 / 1,288,305 rows) |
| Tokenization | **exact**, token for token (every document but the CRLF one — see below) |
| MMI records | 430/565 byte-identical; 1/12 documents fully identical |
| Concepts `(CUI, positions)` | **P 0.975 / R 0.963 / F1 0.969** |
| `json` shared spans | 285/296 (0.963) agree on every field Java emits |
| Sentence segmentation | F1 0.923 (regex) / 0.924 (spaCy) vs OpenNLP |
| POS tags | 0.850 agreement; **0.960** on the span-start decision that matters |

The corpus grew from 10 documents to 12 on 2026-09-18 (`crlf_note.txt`, `unicode_terms.txt`;
DEVELOPMENT_PLAN §20), which is why these differ from the 390/515 and P 0.976 / R 0.965 measured
over the older corpus.

The residual gap is mostly model-dependent: MetaMapLite's OpenNLP sentence detector and POS tagger
have no Python equivalent. Five of the 35 concept-level disagreements are not that, but the
deliberate non-ASCII divergence of DEVELOPMENT_PLAN §16 and §21 — Java's index lookup is
unreliable for keys containing non-ASCII characters (it resolves roughly 44% of them) and this
port does not reproduce that, so it finds `γ-glutamyl transferase` and `Café au lait spots` in
`unicode_terms.txt` where Java falls back to an ASCII fragment. Those
five are pinned individually by `test_known_nonascii_divergences_are_exactly_these`, so that
implementing §16's option (b) or (c) shows up as an explicit change rather than a drifting average.

Offsets are additionally anchored to the files rather than to Java's output: every `start/length`
in every MMI record — ours *and* Java's — must cut its own recorded text out of the document read
verbatim (1,325 positions), and the same invariant runs at the API level for every document.
Comparing the two implementations to each other cannot catch a shared misreading of line endings;
this can, and was checked against the real bug to confirm it fails when it should.

`crlf_note.txt` is excluded from the tokenization comparison, not because tokenization differs but
because the fixture cannot represent it: Java's `--list_sentences_postags` dump writes tokens
inline and the harness normalizes line endings when caching, so a CR whitespace token cannot
survive the round trip. Its offsets are guarded instead by
`test_crlf_document_offsets_agree_with_java_exactly`, which requires *exact* concept agreement —
no false positives, no misses.
