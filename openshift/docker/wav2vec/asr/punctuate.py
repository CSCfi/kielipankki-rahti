"""Punctuation and truecasing of recognised words.

Languages in ``config.PUNCTUATE_MODEL_LANGUAGES`` go through the
xlm-roberta punctuation model via the ``punctuators`` package; the others get
rule-based sentence breaks at pauses and turn boundaries. Either way every
word gets ``word_with_punctuation`` (cased, with any attached marks) and
``punctuation`` (the marks that follow the word, or an empty string).
"""

import json
import logging
import os
import re
import unicodedata
from unittest import mock

from . import config

log = logging.getLogger(__name__)


def uses_model(lang, model_languages=None):
    languages = config.PUNCTUATE_MODEL_LANGUAGES if model_languages is None else model_languages
    return lang in languages or lang.split("-")[0] in languages


def capitalize(word):
    return word[:1].upper() + word[1:]


def trailing_punctuation(text):
    match = re.search(r"[^\w]+$", text)
    return match.group() if match else ""


def letters_only(text):
    return "".join(c for c in text.lower() if unicodedata.category(c)[0] in ("L", "N"))


def by_rules(turns, pause_s=None):
    """Sentence-initial capitals and full stops from the turn boundaries and
    from pauses of at least ``pause_s`` within a turn."""
    pause_s = config.PUNCTUATE_PAUSE_S if pause_s is None else pause_s
    out = []
    for words in turns:
        forms = []
        for i, word in enumerate(words):
            text = word["word"]
            if i == 0 or words[i - 1]["end"] + pause_s <= word["start"]:
                text = capitalize(text)
            if i == len(words) - 1 or word["end"] + pause_s <= words[i + 1]["start"]:
                text += "."
            forms.append({"word_with_punctuation": text, "punctuation": trailing_punctuation(text)})
        out.append(forms)
    return out


class Punctuator:
    """The ONNX model from a directory holding ``model.onnx``, ``sp.model``
    and ``config.yaml``, optionally with alternative graphs listed under
    ``files`` in ``kielipankki.json`` and picked by ``quantize``."""

    def __init__(
        self,
        model_dir=config.MODEL_DIR,
        quantize=config.QUANTIZE,
        threads=config.THREADS,
        model_languages=None,
    ):
        import onnxruntime as ort
        from punctuators.models import punc_cap_seg_model as pcs

        info = {}
        path = os.path.join(model_dir, "kielipankki.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                info = json.load(f)
        files = {
            q: f
            for q, f in info.get("files", {"none": "model.onnx"}).items()
            if os.path.exists(os.path.join(model_dir, f))
        }
        if quantize not in files:
            log.warning("no %s graph in the model directory, choosing from: %s", quantize, ", ".join(sorted(files)))
            quantize = "int8" if "int8" in files else sorted(files)[0]
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        original = ort.InferenceSession

        def session(graph, providers=None, **_):
            return original(graph, options, providers=providers)

        # The wrapper creates the session itself with no way to pass thread
        # settings; without them ONNX Runtime uses every core of the node.
        with mock.patch.object(pcs.ort, "InferenceSession", session):
            self.model = pcs.PunctCapSegModelONNX(
                pcs.PunctCapSegConfigONNX(directory=model_dir, model_filename=files[quantize]),
                ort_providers=["CPUExecutionProvider"],
            )
        self.model_languages = (
            config.PUNCTUATE_MODEL_LANGUAGES if model_languages is None else model_languages
        )
        self.description = {
            "name": info.get("name", os.path.basename(os.path.normpath(model_dir))),
            "backend": "onnx",
            "quantization": quantize,
            "model_languages": self.model_languages,
        }
        log.info("loaded %s", json.dumps(self.description))

    def by_model(self, turns):
        """Punctuate all the words as one text, so that the model has the
        whole context, then capitalise the first word of each turn. A word
        the model's output does not reproduce keeps its own letters."""
        words = [word for turn in turns for word in turn]
        forms = []
        if words:
            text = " ".join(w["word"] for w in words)
            forms = self.model.infer([text], apply_sbd=False)[0].split()
            if len(forms) != len(words):
                log.warning("punctuation changed the word count (%d to %d), keeping the raw words", len(words), len(forms))
                forms = [w["word"] for w in words]
        flat = []
        for word, form in zip(words, forms):
            if letters_only(form) != letters_only(word["word"]):
                form = word["word"] + trailing_punctuation(form)
            flat.append({"word_with_punctuation": form, "punctuation": trailing_punctuation(form)})
        out = []
        position = 0
        for turn in turns:
            chunk = flat[position : position + len(turn)]
            position += len(turn)
            if chunk:
                chunk[0]["word_with_punctuation"] = capitalize(chunk[0]["word_with_punctuation"])
            out.append(chunk)
        return out

    def punctuate(self, turns, lang):
        """``turns`` are lists of word dicts with ``word``, ``start`` and
        ``end``; returns the matching lists of punctuated forms."""
        if uses_model(lang, self.model_languages):
            return self.by_model(turns)
        return by_rules(turns)
