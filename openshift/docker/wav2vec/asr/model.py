"""Loading a CTC model and computing emissions for long audio.

Two backends share one interface: ``HuggingFaceRecognizer`` for wav2vec2
checkpoints in Hugging Face format, and ``OnnxRecognizer`` for models
exported with ``tools/convert_ebranchformer_to_onnx.py``. ``load_recognizer``
picks by the contents of the model directory.
"""

import json
import logging
import os

import numpy as np

from . import config
from .ctc import Normalizer

log = logging.getLogger(__name__)

# fairseq letter dictionaries list the tokens after these four specials.
FAIRSEQ_SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]


class Recognizer:
    """A loaded model with its vocabulary.

    ``emissions`` returns log-probabilities of shape [T, V]. Audio longer than
    ``chunk_s`` is processed in windows of that length that overlap by
    ``stride_s`` on each side; the overlapping frames are dropped so the
    result is one contiguous sequence.

    Subclasses set ``tokens``, ``blank``, ``delimiter``, ``ignore``,
    ``sample_rate``, ``frame_ratio`` and ``description`` and implement
    ``_forward``; then call ``_finish``.
    """

    def __init__(self, chunk_s=config.CHUNK_S, stride_s=config.STRIDE_S):
        self.chunk_s = chunk_s
        self.stride_s = stride_s

    def _finish(self):
        self.normalizer = Normalizer(
            self.tokens, self.blank, self.delimiter, self.ignore, lang=self.description["language"]
        )
        self.frame_s = self.frame_ratio / self.sample_rate
        self.chunk = int(self.chunk_s * self.sample_rate)
        self.stride = int(self.stride_s * self.sample_rate)
        log.info("loaded %s", json.dumps(self.description))

    def _forward(self, samples):
        raise NotImplementedError

    def emissions(self, samples, progress=None):
        """Log-probabilities [T, V] for float32 samples at the model's rate.
        ``progress`` is called after every window."""
        n = len(samples)
        if n <= self.chunk:
            out = self._forward(samples)
            if progress:
                progress()
            return out
        pieces = []
        step = self.chunk - 2 * self.stride
        starts = list(range(0, n - 2 * self.stride, step))
        for i, start in enumerate(starts):
            window = samples[start : start + self.chunk]
            lp = self._forward(window)
            left = self.stride // self.frame_ratio if i > 0 else 0
            right = self.stride // self.frame_ratio if i < len(starts) - 1 else 0
            pieces.append(lp[left : lp.shape[0] - right])
            if progress:
                progress()
        return np.concatenate(pieces)


class HuggingFaceRecognizer(Recognizer):
    def __init__(
        self,
        model_dir=config.MODEL_DIR,
        quantize=config.QUANTIZE,
        threads=config.THREADS,
        lang=config.LANG,
        **kwargs,
    ):
        super().__init__(**kwargs)
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        torch.set_num_threads(threads)
        self.torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(model_dir)
        model = Wav2Vec2ForCTC.from_pretrained(model_dir)
        model.eval()
        parameters = sum(p.numel() for p in model.parameters())
        if quantize == "int8":
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        elif quantize != "none":
            raise ValueError(f"unknown ASR_QUANTIZE value {quantize!r}")
        self.model = model

        tokenizer = self.processor.tokenizer
        vocab = tokenizer.get_vocab()
        self.tokens = [None] * (max(vocab.values()) + 1)
        for token, i in vocab.items():
            self.tokens[i] = token
        self.blank = tokenizer.pad_token_id
        self.delimiter = vocab[tokenizer.word_delimiter_token]
        self.ignore = {
            i
            for i in (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id)
            if i is not None
        }
        self.sample_rate = self.processor.feature_extractor.sampling_rate
        self.frame_ratio = model.config.inputs_to_logits_ratio
        self.description = {
            "name": model_name(model_dir, model.config),
            "language": lang,
            "backend": "wav2vec2",
            "quantization": quantize,
            "parameters": parameters,
        }
        self._finish()

    def _forward(self, samples):
        inputs = self.processor(samples, sampling_rate=self.sample_rate, return_tensors="pt")
        with self.torch.inference_mode():
            logits = self.model(inputs.input_values).logits[0]
            return self.torch.log_softmax(logits.float(), dim=-1).numpy()


class OnnxRecognizer(Recognizer):
    """A model directory written by the conversion tool: ``model.onnx``
    (int8 weights), optionally ``model.fp32.onnx``, ``dict.ltr.txt`` and a
    ``kielipankki.json``. The graphs take ``source`` [1, samples] and return
    log-probabilities [1, frames, V]. ``quantize`` picks the graph: "int8" or
    "none"."""

    def __init__(
        self,
        model_dir=config.MODEL_DIR,
        quantize=config.QUANTIZE,
        threads=config.THREADS,
        lang=config.LANG,
        **kwargs,
    ):
        super().__init__(**kwargs)
        import onnxruntime as ort

        with open(os.path.join(model_dir, "kielipankki.json"), encoding="utf-8") as f:
            info = json.load(f)
        files = {
            q: f
            for q, f in info.get("files", {"int8": "model.onnx"}).items()
            if os.path.exists(os.path.join(model_dir, f))
        }
        if quantize not in files:
            log.warning(
                "no %s graph in the model directory, choosing from: %s", quantize, ", ".join(sorted(files))
            )
            quantize = "int8" if "int8" in files else sorted(files)[0]
        graph = os.path.join(model_dir, files[quantize])
        self.normalize = bool(info.get("normalize", True))
        self.sample_rate = int(info.get("sample_rate", 16000))
        self.frame_ratio = int(info.get("frame_ratio", 320))
        max_s = float(info.get("max_seconds", 0) or 0)
        if max_s and self.chunk_s > max_s:
            log.warning("model accepts at most %.0f s; lowering the chunk length", max_s)
            self.chunk_s = max_s

        self.tokens = list(FAIRSEQ_SPECIALS)
        with open(os.path.join(model_dir, "dict.ltr.txt"), encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts:
                    self.tokens.append(parts[0])
        self.blank = 0
        self.delimiter = self.tokens.index("|")
        self.ignore = {1, 2, 3}

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(graph, options, providers=["CPUExecutionProvider"])
        self.description = {
            "name": info.get("name") or os.path.basename(os.path.normpath(model_dir)),
            "language": lang,
            "backend": "onnx",
            "quantization": quantize,
            "parameters": info.get("parameters"),
        }
        self._finish()

    def _forward(self, samples):
        x = np.asarray(samples, dtype=np.float32)
        if self.normalize:
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
        return self.session.run(None, {"source": x[None]})[0][0]


def load_recognizer(model_dir=config.MODEL_DIR, **kwargs):
    if os.path.exists(os.path.join(model_dir, "model.onnx")):
        return OnnxRecognizer(model_dir, **kwargs)
    return HuggingFaceRecognizer(model_dir, **kwargs)


def model_name(model_dir, model_config):
    """The packaging tool leaves a kielipankki.json with the source name;
    otherwise use what the checkpoint remembers or the directory name."""
    path = os.path.join(model_dir, "kielipankki.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            name = json.load(f).get("name")
            if name:
                return name
    name = getattr(model_config, "_name_or_path", "") or ""
    if name and not os.path.isabs(name):
        return name
    return os.path.basename(os.path.normpath(model_dir))
