"""spaCy-backed sentence detection and POS tagging (replacement for the OpenNLP models).

spaCy is an optional dependency (``pip install mmlite[nlp]`` + ``python -m spacy download
en_core_web_sm``).  The tagger runs over MetaMapLite's own tokens (a pre-tokenized ``Doc``) so
tokenization stays identical to Java; only the tag assignment comes from spaCy.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # spaCy and numpy are optional; only the type checker needs them here
    import numpy as np
    from spacy.language import Language
    from spacy.pipeline import Tagger, TrainablePipe
    from spacy.tokens import Doc

DEFAULT_MODEL = "en_core_web_sm"


class TaggerImportError(ImportError):
    """spaCy itself is not installed."""


class TaggerModelNotFound(OSError):
    """spaCy is installed but the requested model is not."""


#: The two ways the tagger can be unavailable, and the only start-up failures ``--no-postag``
#: can do anything about. They keep their original base classes -- a missing package was an
#: ``ImportError`` and a missing model an ``OSError`` -- so existing handlers are unaffected;
#: a single class cannot subclass both, as their C layouts conflict. The distinction matters
#: because a *missing index* is an ``OSError`` too, and answering that with advice about
#: part-of-speech tagging sends the reader after the wrong problem.
TAGGER_UNAVAILABLE = (TaggerImportError, TaggerModelNotFound)


@lru_cache(maxsize=4)
def load_model(name: str = DEFAULT_MODEL) -> Language:
    try:
        import spacy
    except ImportError as e:  # pragma: no cover
        raise TaggerImportError("spaCy is not installed; run: uv pip install 'mmlite[nlp]'") from e
    try:
        return spacy.load(name, exclude=["ner", "lemmatizer"])
    except OSError as e:
        # spaCy's own models come from ``spacy download``; scispaCy's are plain packages
        # installed from a release URL (see USER_GUIDE "Choosing the POS tagger").
        how = (
            f"python -m spacy download {name}"
            if name.startswith("en_core_web_")
            else f"pip install the {name} package"
        )
        raise TaggerModelNotFound(f"spaCy model {name!r} not found; run: {how}") from e


# Penn Treebank tags the word "to" as TO in every use; spaCy's tagger emits IN for the
# prepositional reading, which OpenNLP (and therefore MetaMapLite) never does.  Since TO is not
# in ALLOWED_PART_OF_SPEECH and IN is, leaving it alone makes spans start at "to" that Java
# never considers.  Measured on the parity corpus this is the single largest source of
# gate disagreement (8 of 45).
PTB_FIXUPS = {"to": "TO"}

# A token spaCy tags as a finite verb is re-tagged with its most likely noun/adjective tag when
# that tag's probability is at least this.  Verbs cannot start a span (ALLOWED_PART_OF_SPEECH), and
# en_core_web_sm tags rare clinical words as verbs far more often than OpenNLP does - "diabetes"
# (VBZ) in "Quarterly diabetes follow-up", "Famotidine" (VB) opening a medication-list line - so
# those concepts vanished outright.  Measured against Java (USER_GUIDE "Choosing the POS tagger"):
# on the parity corpus F1 0.969 -> 0.974 (misses 21 -> 18, extras 14 -> 11); over two long
# clinical notes as well, misses 118 -> 94 and extras 128 -> 119.  Participles (VBN/VBG) are left
# alone: they are routinely adjectival ("was continued", "elevated"), OpenNLP keeps them verbs,
# and rescuing them added spans such as "continued" -> Continuous that Java never considers.
# 0.03-0.05 score alike; below 0.01 the extras start to outweigh the recovered concepts.
VERB_RESCUE_THRESHOLD = 0.03
RESCUE_FROM = frozenset({"VB", "VBD", "VBP", "VBZ"})
RESCUE_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"})


class SpacyPosTagger:
    """Penn Treebank tags (``token.tag_``) from spaCy's tagger, applied to pre-tokenized words.

    ``ptb_fixups=False`` disables the :data:`PTB_FIXUPS` corrections.  ``verb_rescue`` is the
    :data:`VERB_RESCUE_THRESHOLD` (``0`` tags exactly as spaCy does).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ptb_fixups: bool = True,
        verb_rescue: float = VERB_RESCUE_THRESHOLD,
    ):
        self.model = model
        self.fixups = PTB_FIXUPS if ptb_fixups else {}
        self.verb_rescue = verb_rescue
        self.nlp = load_model(model)
        from spacy.tokens import Doc

        self._Doc = Doc
        # tag_ needs only the tagger and whatever it listens to (a shared tok2vec in spaCy's own
        # pipelines); attribute_ruler maps tag -> pos, which nothing here reads.
        self._tagger = cast("Tagger", self.nlp.get_pipe("tagger"))
        self._pre = [
            cast("TrainablePipe", proc) for name, proc in self.nlp.pipeline if name == "tok2vec"
        ]
        self._labels = list(self._tagger.labels)
        self._rescue_idx = [i for i, lab in enumerate(self._labels) if lab in RESCUE_TAGS]

    def _scores(self, docs: list[Doc]) -> list[np.ndarray]:
        """Per-token tag scores, one (tokens x labels) array per doc.

        Calls ``predict`` directly rather than ``pipe``: the tagger's tok2vec listener only
        accepts the batch the tok2vec step just produced, so both see exactly ``docs``.
        """
        for proc in self._pre:
            proc.set_annotations(docs, proc.predict(docs))
        ops = self._tagger.model.ops
        return [ops.to_numpy(s) for s in self._tagger.model.predict(docs)]

    def _tag(self, doc: Doc, scores: np.ndarray) -> list[str]:
        import numpy as np

        tags = []
        for i, token in enumerate(doc):
            row = scores[i]
            tag = self._labels[int(row.argmax())]
            fixed = self.fixups.get(token.text.lower())
            if fixed is not None:
                tag = fixed
            elif self.verb_rescue > 0 and tag in RESCUE_FROM and self._rescue_idx:
                # spaCy 3.8's tagger emits unnormalized scores at prediction time.
                if row.min() < 0 or abs(float(row.sum()) - 1.0) > 1e-3:
                    row = np.exp(row - row.max())
                    row = row / row.sum()
                best = max(self._rescue_idx, key=lambda k: row[k])
                if row[best] >= self.verb_rescue:
                    tag = self._labels[best]
            tags.append(tag)
        return tags

    def tag(self, words: list[str]) -> list[str]:
        if not words:
            return []
        return self.tag_batch([words])[0]

    def tag_batch(self, sentences: list[list[str]]) -> list[list[str]]:
        """Tag several sentences in one pass through each pipe.

        Every component here is per-``Doc``, so the tags are exactly those of tagging one at a
        time; what changes is that thinc's matrix work happens once per batch instead of once per
        sentence, which is most of the cost on short clinical sentences.
        """
        docs = [self._Doc(self.nlp.vocab, words=words) for words in sentences if words]
        if not docs:
            return [[] for _ in sentences]
        tagged = iter(
            [self._tag(doc, scores) for doc, scores in zip(docs, self._scores(docs), strict=True)]
        )
        return [next(tagged) if words else [] for words in sentences]


class SpacySentenceDetector:
    """Sentence spans from spaCy.

    Parser-based by default; ``rule_based=True`` uses the sentencizer instead.
    """

    def __init__(self, model: str = DEFAULT_MODEL, rule_based: bool = False):
        self.nlp = load_model(model)
        if rule_based:
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer", first=True)
            self._disable = [n for n in self.nlp.pipe_names if n != "sentencizer"]
        else:
            self._disable = [
                n for n in self.nlp.pipe_names if n not in ("tok2vec", "parser", "senter")
            ]

    def spans(self, text: str) -> list[tuple[int, int]]:
        with self.nlp.select_pipes(disable=self._disable):
            doc = self.nlp(text)
        out = []
        for s in doc.sents:
            start, end = s.start_char, s.end_char
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            if start < end:
                out.append((start, end))
        return out
