"""The ONNX backend loads a model directory written by
tools/convert_ebranchformer_to_onnx.py. A tiny graph with the same interface
(source [1, samples] -> log-probabilities [1, frames, vocab], 320 samples per
frame) stands in for a converted model."""

import json
import os

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from asr import model  # noqa: E402

DICT = ["|", "a", "b", "c"]  # after the four fairseq specials: ids 4..7
VOCAB = 4 + len(DICT)
RATIO = 320


def tiny_graph(path):
    """Average each 320-sample frame, project it to the vocabulary and take
    the log-softmax, with the frame count derived from the input length."""
    rng = np.random.default_rng(1)
    W = numpy_helper.from_array(rng.standard_normal((RATIO, VOCAB)).astype(np.float32), "W")
    consts = [
        helper.make_tensor("ratio", TensorProto.INT64, [1], [RATIO]),
        helper.make_tensor("one", TensorProto.INT64, [1], [1]),
        helper.make_tensor("zero", TensorProto.INT64, [1], [0]),
        helper.make_tensor("axis1", TensorProto.INT64, [1], [1]),
        helper.make_tensor("minus1", TensorProto.INT64, [1], [-1]),
    ]
    nodes = [
        helper.make_node("Shape", ["source"], ["shape"]),
        helper.make_node("Gather", ["shape", "one"], ["samples"], axis=0),
        helper.make_node("Div", ["samples", "ratio"], ["frames"]),
        helper.make_node("Mul", ["frames", "ratio"], ["used"]),
        helper.make_node("Slice", ["source", "zero", "used", "axis1"], ["trimmed"]),
        helper.make_node("Concat", ["one", "frames", "ratio"], ["newshape"], axis=0),
        helper.make_node("Reshape", ["trimmed", "newshape"], ["framed"]),
        helper.make_node("MatMul", ["framed", "W"], ["logits"]),
        helper.make_node("LogSoftmax", ["logits"], ["log_probs"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, "samples"])],
        [helper.make_tensor_value_info("log_probs", TensorProto.FLOAT, [1, "frames", VOCAB])],
        initializer=[W] + consts,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, path)


@pytest.fixture
def model_dir(tmp_path):
    tiny_graph(tmp_path / "model.onnx")
    (tmp_path / "dict.ltr.txt").write_text("".join(f"{t} 1\n" for t in DICT), encoding="utf-8")
    (tmp_path / "kielipankki.json").write_text(json.dumps({
        "name": "GetmanY1/tiny-test", "backend": "onnx", "normalize": True,
        "sample_rate": 16000, "frame_ratio": RATIO, "max_seconds": 20, "parameters": 12,
        "files": {"int8": "model.onnx", "none": "model.fp32.onnx"},
    }), encoding="utf-8")
    return str(tmp_path)


def test_factory_picks_onnx_and_reads_dictionary(model_dir):
    r = model.load_recognizer(model_dir, threads=1, lang="sme", quantize="int8")
    assert isinstance(r, model.OnnxRecognizer)
    assert r.tokens == ["<s>", "<pad>", "</s>", "<unk>", "|", "a", "b", "c"]
    assert r.blank == 0 and r.delimiter == 4 and r.ignore == {1, 2, 3}
    assert r.frame_s == pytest.approx(0.02)
    assert r.description["name"] == "GetmanY1/tiny-test"
    assert r.description["backend"] == "onnx" and r.description["language"] == "sme"
    assert r.description["quantization"] == "int8"
    # The chunk length is capped by what the model was exported for.
    assert r.chunk_s == 20


def test_quantize_selects_graph(model_dir):
    # No fp32 graph in the directory: falls back to int8 with a warning.
    r = model.load_recognizer(model_dir, threads=1, quantize="none")
    assert r.description["quantization"] == "int8"
    import shutil
    shutil.copy(os.path.join(model_dir, "model.onnx"), os.path.join(model_dir, "model.fp32.onnx"))
    r = model.load_recognizer(model_dir, threads=1, quantize="none")
    assert r.description["quantization"] == "none"


def test_onnx_emissions_shape_and_chunking(model_dir):
    r = model.load_recognizer(model_dir, threads=1, chunk_s=20, stride_s=2)
    x = np.random.default_rng(0).standard_normal(16000 * 3).astype(np.float32)
    lp = r.emissions(x)
    assert lp.shape == (150, VOCAB)
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-4)
    # Longer than one window: contiguous frames from stitched windows.
    calls = []
    long = np.random.default_rng(0).standard_normal(16000 * 50).astype(np.float32)
    lp = r.emissions(long, progress=lambda: calls.append(1))
    assert lp.shape[0] == 2500 and len(calls) == 3
    # Normalisation happens in the backend, so scaling the input changes nothing.
    assert np.allclose(r.emissions(x), r.emissions(x * 10), atol=1e-4)


def test_factory_picks_hugging_face_without_onnx(tmp_path):
    # No model.onnx: the factory goes to the Hugging Face loader, which fails
    # on an empty directory. The point is only that dispatch is by content.
    with pytest.raises(Exception):
        model.load_recognizer(str(tmp_path), threads=1)
