"""Measure the two model-dependent gaps against Java MetaMapLite: sentence segmentation and
part-of-speech tagging.

    python scripts/compare_nlp_with_java.py --java-dir D:/Apps/.../public_mm_lite [--refresh]

Java's ``--list_sentences_postags`` dumps, per document, every sentence (offset|length|text)
followed by its tokens as ``text(TAG),``.  We compare:

* **segmentation** — Java's sentence spans vs each of our detectors (exact-span F1);
* **tokenization** — our tokenizer against Java's token texts (should be exact);
* **tagging** — spaCy's tags against OpenNLP's on Java's own tokens, plus agreement on the
  decision that actually matters, "is this token allowed to start a candidate span"
  (``ALLOWED_PART_OF_SPEECH``).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CORPUS = ROOT / "tests" / "parity" / "corpus"
FIXTURES = ROOT / "tests" / "parity" / "fixtures"

from mmlite.pipeline.postag import ALLOWED_PART_OF_SPEECH, add_part_of_speech
from mmlite.pipeline.segment import RegexSentenceDetector
from mmlite.pipeline.tokenize import tokenize


def java_dump(java_dir: Path, doc: Path) -> str:
    launcher = java_dir / ("metamaplite.bat" if sys.platform == "win32" else "metamaplite.sh")
    cmd = [str(launcher), "--list_sentences_postags", doc.name]
    if sys.platform == "win32":
        cmd = ["cmd", "/c", *cmd]
    subprocess.run(cmd, cwd=doc.parent, check=True, capture_output=True)
    out = doc.with_suffix(".sentences_postags")
    text = out.read_text(encoding="utf-8").replace("\r\n", "\n")
    out.unlink()
    return text


HEADER = re.compile(r"^(\d+)\|(\d+)\|")


def parse_dump(text: str) -> list[tuple[int, int, str, list[str]]]:
    """-> [(offset, length, sentence text, [tag per token])]; tags align with our tokenizer.

    Both the sentence text and its token dump can contain newlines (a sentence spanning a line
    break), so the sentence is taken as exactly ``length`` characters after the header and the
    token dump is everything up to the next header.
    """
    out = []
    lines = text.split("\n")
    heads = [i for i, line in enumerate(lines) if HEADER.match(line)]
    for n, i in enumerate(heads):
        offset, length = (int(g) for g in HEADER.match(lines[i]).groups())
        end = heads[n + 1] if n + 1 < len(heads) else len(lines)
        block = "\n".join(lines[i:end])
        body = block.split("|", 2)[2]
        sent, dump = body[:length], body[length:].lstrip("\n")
        tags = []
        pos = 0
        for tok in tokenize(sent):
            want = tok.text + "("
            k = dump.find(want, pos)
            if k < 0:
                tags.append("?")
                continue
            stop = dump.find("),", k + len(want))
            tags.append(dump[k + len(want) : stop] if stop > 0 else "?")
            pos = stop + 2 if stop > 0 else pos
        out.append((offset, length, sent, tags))
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--java-dir", type=Path, required=True)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    detectors = {"regex": RegexSentenceDetector()}
    taggers = {}
    try:
        from mmlite.pipeline.spacy_nlp import SpacyPosTagger, SpacySentenceDetector

        detectors["spacy"] = SpacySentenceDetector()
        detectors["spacy-rule"] = SpacySentenceDetector(rule_based=True)
        taggers["spacy"] = SpacyPosTagger()
    except (ImportError, OSError) as e:  # pragma: no cover
        print("spaCy unavailable:", e)

    seg = {name: [0, 0, 0] for name in detectors}  # tp, fp, fn
    tok_total = tok_bad = 0
    tag_total = tag_same = 0
    gate_total = gate_same = 0
    tag_confusion: dict[tuple[str, str], int] = {}

    for doc in sorted(CORPUS.glob("*.txt")):
        fixture = FIXTURES / f"{doc.stem}.sentences_postags"
        if args.refresh or not fixture.exists():
            fixture.write_text(java_dump(args.java_dir, doc), encoding="utf-8", newline="\n")
        text = doc.read_text(encoding="utf-8")
        sentences = parse_dump(fixture.read_text(encoding="utf-8"))
        gold_spans = {(o, o + len(s)) for o, _, s, _ in sentences}
        for name, det in detectors.items():
            ours = {(a, b) for a, b in det.spans(text)}
            seg[name][0] += len(gold_spans & ours)
            seg[name][1] += len(ours - gold_spans)
            seg[name][2] += len(gold_spans - ours)

        for _, _, sent, tags in sentences:
            toks = tokenize(sent)
            if len(toks) != len(tags):
                tok_bad += 1
            tok_total += 1
            if "spacy" not in taggers:
                continue
            add_part_of_speech(toks, taggers["spacy"])
            for tok, jtag in zip(toks, tags, strict=False):
                if tok.is_whitespace or jtag in ("?", "WS"):
                    continue
                tag_total += 1
                if tok.pos == jtag:
                    tag_same += 1
                else:
                    tag_confusion[(jtag, tok.pos)] = tag_confusion.get((jtag, tok.pos), 0) + 1
                gate_total += 1
                if (jtag in ALLOWED_PART_OF_SPEECH) == (tok.pos in ALLOWED_PART_OF_SPEECH):
                    gate_same += 1

    print("sentence segmentation vs OpenNLP (exact spans)")
    for name, (tp, fp, fn) in seg.items():
        p, r, f = prf(tp, fp, fn)
        print(f"  {name:11s} P={p:.3f} R={r:.3f} F1={f:.3f}  (tp={tp} fp={fp} fn={fn})")
    print(f"\ntokenization: {tok_total - tok_bad}/{tok_total} sentences token-for-token identical")
    if tag_total:
        print(
            f"POS tags (spaCy vs OpenNLP on Java's tokens): "
            f"{tag_same}/{tag_total} = {tag_same / tag_total:.3f}"
        )
        print(f"span-start gate agreement: {gate_same}/{gate_total} = {gate_same / gate_total:.3f}")
        print("top tag disagreements (java -> spacy):")
        for (j, s), n in sorted(tag_confusion.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {n:4d}  {j:6s} -> {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
