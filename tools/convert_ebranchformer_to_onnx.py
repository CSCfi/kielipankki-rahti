#!/usr/bin/env python3
"""Convert an Aalto E-Branchformer fairseq checkpoint to an ONNX model
directory the wav2vec worker can serve.

    python tools/convert_ebranchformer_to_onnx.py <model repo dir> --out model-sme-ebranchformer.zip

The repo dir is a checkout of, for example,
GetmanY1/wav2vec2-large-ebranch-sami-18k-finetuned-experimental: the
checkpoint (*.pt), dict.ltr.txt and the fairseq_extra_encoders package. This
runs in a conversion environment, not in the service image: Python 3.10,
torch, hydra-core 1.0.7, omegaconf from Getmany1/omegaconf@2.0_branch and
fairseq from git, as in Aalto's demo Space (GetmanY1/sami_asr). onnx and
onnxruntime are needed as well; the test clip is read with the standard library.

Steps: write a slim copy of the checkpoint without the optimiser state and
with the relative positional encoding table (100 000 positions, 614 MB of
sinusoids) sized for the longest window the worker uses, load it with
fairseq, export the encoder plus CTC projection at opset 17 with a dynamic
time axis, quantise the weights to int8 with ONNX Runtime, check that the
int8 and fp32 models agree with torch on a test clip, and zip both graphs
with the dictionary and a kielipankki.json describing them. The worker's
ASR_QUANTIZE setting (int8 or none) chooses which graph to serve. Peak memory is about
3.5 GB; the checkpoint is never fully loaded at once.
"""

import argparse
import datetime
import gc
import json
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import torch

SAMPLE_RATE = 16000
FRAME_RATIO = 320


def slim_checkpoint(src, dst, max_frames):
    """Copy the checkpoint without optimiser state, with the encoder's
    max_positions set so that the sinusoidal relative position table (no
    learned parameters, not in the state dict) is built small."""
    ckpt = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    ckpt.pop("last_optimizer_state", None)
    w2v = ckpt["cfg"]["model"]["w2v_args"]["model"]
    print(f"max_positions {w2v['max_positions']} -> {max_frames}", flush=True)
    w2v["max_positions"] = max_frames
    torch.save(ckpt, dst)
    del ckpt
    gc.collect()


def load_model(repo_dir, checkpoint):
    sys.path.insert(0, repo_dir)
    import fairseq_extra_encoders  # noqa: F401  registers the model classes
    from fairseq import checkpoint_utils

    original_load = torch.load
    torch.load = lambda *a, **k: original_load(*a, **{**k, "weights_only": False})
    try:
        models, cfg, _task = checkpoint_utils.load_model_ensemble_and_task([checkpoint])
    finally:
        torch.load = original_load
    gc.collect()
    model = models[0].eval()
    return model, cfg


class CtcWrapper(torch.nn.Module):
    """source [1, samples] -> log-probabilities [1, frames, vocab]."""

    def __init__(self, model):
        super().__init__()
        self.encoder = model.w2v_encoder

    def forward(self, source):
        out = self.encoder(source=source, padding_mask=None)
        logits = out["encoder_out"].transpose(0, 1)  # T x B x V -> B x T x V
        return torch.log_softmax(logits.float(), dim=-1)


def test_signal(seconds):
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    x = 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
    x += 0.05 * rng.standard_normal(len(t))
    return x.astype(np.float32)


def normalise(x):
    return (x - x.mean()) / np.sqrt(x.var() + 1e-5)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_dir", help="directory with the checkpoint, dict.ltr.txt and fairseq_extra_encoders")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file, default the single *.pt in repo_dir")
    parser.add_argument("--name", default=None, help="model name to report, default the repo directory name")
    parser.add_argument("--out", required=True, help="zip file to write; its stem names the directory inside")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="longest input the model must accept, default 60")
    parser.add_argument("--test-clip", default=None, help="16 kHz WAV to compare torch and ONNX on, default a synthetic signal")
    args = parser.parse_args()

    repo_dir = os.path.abspath(args.repo_dir)
    checkpoint = args.checkpoint or next(
        os.path.join(repo_dir, f) for f in sorted(os.listdir(repo_dir)) if f.endswith(".pt")
    )
    name = args.name or os.path.basename(repo_dir)
    stem = os.path.splitext(os.path.basename(args.out))[0]

    work = tempfile.mkdtemp()
    max_frames = int(args.max_seconds * SAMPLE_RATE / FRAME_RATIO) + 16
    slim = os.path.join(work, "slim.pt")
    print("slimming", checkpoint, flush=True)
    slim_checkpoint(checkpoint, slim, max_frames)
    print("loading", flush=True)
    model, cfg = load_model(repo_dir, slim)
    os.remove(slim)
    normalize = bool(cfg.task.normalize)
    positions = model.w2v_encoder.w2v_model.encoder.embed_positions
    table = 0 if positions is None else positions.pe.numel() * 4 // 2**20
    wrapper = CtcWrapper(model).eval()
    parameters = sum(p.numel() for p in wrapper.parameters())
    print(f"{parameters/1e6:.0f}M parameters, normalize={normalize}, position table {table} MiB", flush=True)

    if args.test_clip:
        import wave
        with wave.open(args.test_clip) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (SAMPLE_RATE, 1, 2), \
                "test clip must be 16 kHz mono 16-bit PCM"
            clip = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    else:
        clip = test_signal(12.0)
    x = normalise(clip) if normalize else clip
    source = torch.from_numpy(x)[None]

    with torch.inference_mode():
        reference = wrapper(source).numpy()
    print("torch output", reference.shape, flush=True)

    fp32 = os.path.join(work, "model.fp32.onnx")
    int8 = os.path.join(work, "model.onnx")
    # Trace with a short input to keep the activations small; the time axis
    # is dynamic.
    dummy = torch.from_numpy(normalise(test_signal(8.0)))[None]
    print("exporting", flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy,),
            fp32,
            input_names=["source"],
            output_names=["log_probs"],
            dynamic_axes={"source": {1: "samples"}, "log_probs": {1: "frames"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    del wrapper, model
    gc.collect()

    print("quantising", flush=True)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    # Only the matrix multiplications, as with torch's dynamic quantisation of
    # Linear layers; ONNX Runtime has no CPU kernel for the quantised 1-D
    # convolutions of the feature extractor.
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"])

    import onnxruntime as ort
    for label, path in (("fp32", fp32), ("int8", int8)):
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = session.run(None, {"source": x[None]})[0]
        agree = float((out.argmax(-1) == reference.argmax(-1)).mean())
        diff = float(np.abs(out - reference).max())
        print(f"{label}: argmax agreement {agree:.4f}, max abs diff {diff:.3f}, size {os.path.getsize(path)/2**20:.0f} MiB", flush=True)
        if label == "fp32" and agree < 0.999:
            sys.exit("fp32 export does not reproduce torch; not writing the zip")
        # A different window length exercises the dynamic axes.
        y = x[: SAMPLE_RATE * 5]
        session.run(None, {"source": y[None]})

    description = {
        "name": name,
        "backend": "onnx",
        "source": checkpoint if not name.count("/") else f"https://huggingface.co/{name}",
        "normalize": normalize,
        "sample_rate": SAMPLE_RATE,
        "frame_ratio": FRAME_RATIO,
        "max_seconds": args.max_seconds,
        "parameters": parameters,
        "files": {"int8": "model.onnx", "none": "model.fp32.onnx"},
        "packaged": datetime.date.today().isoformat(),
    }
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        # Both precisions ship; the worker's ASR_QUANTIZE setting picks one.
        zf.write(int8, f"{stem}/model.onnx")
        zf.write(fp32, f"{stem}/model.fp32.onnx")
        zf.write(os.path.join(repo_dir, "dict.ltr.txt"), f"{stem}/dict.ltr.txt")
        zf.writestr(f"{stem}/kielipankki.json", json.dumps(description, indent=2))
    shutil.rmtree(work)
    print("wrote", args.out, f"{os.path.getsize(args.out)/2**20:.0f} MiB")


if __name__ == "__main__":
    main()
