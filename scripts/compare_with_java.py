"""Parity harness: run Java MetaMapLite and mmlite on the corpus and diff the output.

    python scripts/compare_with_java.py --java-dir D:/Apps/.../public_mm_lite
        [--refresh] [--format mmi]

* Java outputs are cached as fixtures in tests/parity/fixtures/<name>.<format>; ``--refresh``
  re-runs Java (slow: one JVM start per document, ~10 s each).
* Python output is produced in-process with the same options (specialterms.txt from the Java
  distribution, docid = file name).
* For every document the script prints IDENTICAL or a unified diff, and finishes with
  concept-level (CUI, positions) precision/recall over the whole corpus.

``--format json`` is compared **structurally**, not byte for byte.  Java builds its output with
``org.json``, whose ``JSONObject`` is backed by a ``HashMap``, so key order carries no meaning and
is not reproducible; ``org.json`` also drops keys whose value is null, and prints ``0`` where
Python prints ``0.0``.  Parsing both sides and comparing the data is the only comparison that
says anything true.  Array order *is* significant and is not normalized away: ``semantictypes``
and ``sources`` are emitted in Java ``HashSet`` iteration order, which this port reproduces
deliberately (:mod:`mmlite.javautil`), so a difference there is a real difference.
"""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests" / "parity" / "corpus"
FIXTURES = ROOT / "tests" / "parity" / "fixtures"


def run_java(java_dir: Path, doc: Path, fmt: str) -> str:
    """Run metamaplite.bat/sh on ``doc`` and return the output file's text."""
    launcher = java_dir / ("metamaplite.bat" if sys.platform == "win32" else "metamaplite.sh")
    cmd = [str(launcher), f"--outputformat={fmt}", doc.name]
    if sys.platform == "win32":
        cmd = ["cmd", "/c", str(launcher), f"--outputformat={fmt}", doc.name]
    subprocess.run(cmd, cwd=doc.parent, check=True, capture_output=True)
    ext = {"mmi": ".mmi", "json": ".json", "brat": ".ann", "cuilist": ".cuis"}.get(fmt, "." + fmt)
    out = doc.with_suffix(ext)
    text = out.read_text(encoding="utf-8").replace("\r\n", "\n")
    out.unlink()
    return text


def run_python(java_dir: Path, doc: Path, fmt: str, mml=None) -> str:
    """Annotate ``doc`` the way Java's ``--inputformat=freetext`` file path does.

    Going through the loader rather than ``process_text`` matters for more than tidiness: Java
    loads the file with ``FreeText``, which sets no ``section`` infon, so its entities carry no
    field id.  ``process_text`` hard-codes ``fieldid="text"``, which made the ``json`` comparison
    apples-to-oranges on that field (DEVELOPMENT_PLAN §12).  ``docid`` is pinned to the file name
    because Java's own basename logic splits on ``/`` only and would yield the whole path here.
    Reading goes through the loader too, so line endings survive as Java's do.
    """
    from mmlite import MetaMapLite

    if mml is None:
        mml = MetaMapLite(index_directory=ROOT / "ivf")
    (document,) = mml.load(doc, docid=doc.name)
    return mml.format(mml.process_document(document), fmt)


def records(text: str, docid: str) -> list[str]:
    """Split MMI output into records.  A record can contain embedded newlines when the matched
    text spans a line break, so a new record starts only at ``<docid>|MMI|``."""
    out: list[str] = []
    for line in text.splitlines():
        if line.startswith(docid + "|MMI|") or not out:
            out.append(line)
        else:
            out[-1] += "\n" + line
    return [r for r in out if r]


def mmi_keys(text: str, docid: str) -> set[tuple[str, str]]:
    """(CUI, positions) pairs — the concept-level content of MMI output, ignoring POS/score."""
    keys = set()
    for rec in records(text, docid):
        f = rec.split("|")
        if len(f) > 8 and f[1] == "MMI":
            keys.add((f[4], f[8]))
    return keys


def json_lines(text: str) -> list[str]:
    """One JSON document as stable, diffable lines: keys sorted, arrays left in place."""
    return json.dumps(json.loads(text), indent=2, sort_keys=True).splitlines()


def same(expected: str, actual: str, fmt: str) -> bool:
    """Whether two outputs agree — structurally for ``json``, byte for byte otherwise."""
    if fmt == "json":
        return json.loads(expected) == json.loads(actual)
    return expected == actual


def diff_lines(expected: str, actual: str, fmt: str) -> list[str]:
    a, b = (
        (json_lines(expected), json_lines(actual))
        if fmt == "json"
        else (expected.splitlines(), actual.splitlines())
    )
    return list(difflib.unified_diff(a, b, "java", "python", lineterm="", n=0))


def json_field_report(expected: str, actual: str) -> list[str]:
    """Which keys each side emits, and where the two disagree on a shared key.

    A whole-file diff of 500 concepts buries the finding; what one actually wants to know is
    "which fields differ, and how often".
    """
    e_ents, a_ents = json.loads(expected), json.loads(actual)
    out: list[str] = []
    e_keys = {k for ent in e_ents for k in ent}
    a_keys = {k for ent in a_ents for k in ent}
    if e_keys - a_keys:
        out.append(f"entity keys only in java:   {sorted(e_keys - a_keys)}")
    if a_keys - e_keys:
        out.append(f"entity keys only in python: {sorted(a_keys - e_keys)}")
    by_span = {(e["start"], e["length"]): e for e in e_ents}
    mismatched: dict[str, int] = {}
    for ent in a_ents:
        other = by_span.get((ent["start"], ent["length"]))
        if other is None:
            continue
        for key in e_keys & a_keys:
            if ent.get(key) != other.get(key):
                mismatched[key] = mismatched.get(key, 0) + 1
    for key, n in sorted(mismatched.items()):
        out.append(f"shared key differs: {key} ({n} entities)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--java-dir", type=Path, required=True)
    ap.add_argument("--format", default="mmi")
    ap.add_argument("--refresh", action="store_true", help="re-run Java even if fixtures exist")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    from mmlite import MetaMapLite
    from mmlite.config import Settings

    settings = Settings.load(
        index_directory=ROOT / "ivf",
        excluded_terms_file=args.java_dir / "data" / "specialterms.txt",
    )
    mml = MetaMapLite(settings)
    FIXTURES.mkdir(parents=True, exist_ok=True)

    identical = 0
    tp = fp = fn = 0
    docs = sorted(CORPUS.glob("*.txt"))
    for doc in docs:
        fixture = FIXTURES / f"{doc.stem}.{args.format}"
        if args.refresh or not fixture.exists():
            fixture.write_text(
                run_java(args.java_dir, doc, args.format), encoding="utf-8", newline="\n"
            )
        expected = fixture.read_text(encoding="utf-8")
        actual = run_python(args.java_dir, doc, args.format, mml)
        if args.format == "mmi":
            e, a = mmi_keys(expected, doc.name), mmi_keys(actual, doc.name)
            tp += len(e & a)
            fp += len(a - e)
            fn += len(e - a)
        if same(expected, actual, args.format):
            identical += 1
            print(f"IDENTICAL  {doc.name}")
            continue
        print(f"DIFFERS    {doc.name}")
        if args.format == "json":
            for line in json_field_report(expected, actual):
                print("    *", line)
        if not args.quiet:
            for line in diff_lines(expected, actual, args.format):
                print("   ", line)
    print(f"\n{identical}/{len(docs)} documents byte-identical")
    if tp + fp + fn:
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(
            f"concept level over all documents, (CUI, positions): "
            f"P={p:.3f} R={r:.3f} F1={f1:.3f} (tp={tp} fp={fp} fn={fn})"
        )
    return 0 if identical == len(docs) else 1


if __name__ == "__main__":
    sys.exit(main())
