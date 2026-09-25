# scripts

- `gen_greek_table.py` — regenerate `src/mmlite/greek.py` from MetaMapLite's
  `GreekCharacters.java` (`python scripts/gen_greek_table.py path/to/GreekCharacters.java`).
- `gen_negex_triggers.py` — regenerate `src/mmlite/pipeline/negex_triggers.py` from
  MetaMapLite's `NegExKeyMap.java` (`python scripts/gen_negex_triggers.py path/to/NegExKeyMap.java`).
- `compare_with_java.py` — parity harness: diffs MMI/JSON/BRAT/cuilist output against a local
  Java MetaMapLite install over `tests/parity/corpus/` (`--java-dir`, `--refresh`, `--format`).
  See [tests/parity/README.md](../tests/parity/README.md).
- `compare_nlp_with_java.py` — same idea for segmentation/tokenization/POS, via Java's
  `--list_sentences_postags`.

To build an index, use `mmlite build-index` (CLI) — there is no `build_index` script.
