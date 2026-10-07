#!/usr/bin/env python3
"""Build the model zips of the diarisation and punctuation workers for the
init container.

    python3 tools/package_stage_models.py diarize --out model-diarize.zip
    python3 tools/package_stage_models.py punctuate --out model-punctuate.zip

``diarize`` packages one of two backends, chosen with ``--backend``:

``sherpa`` (the default) fetches the pyannote segmentation 3.0 model (MIT)
and a speaker embedding model as sherpa-onnx redistributes them (no login),
and zips them as ``segmentation.onnx`` and ``embedding.onnx``. ``--embedding``
picks the embedding model. The default is NVIDIA's TitaNet small (CC BY
4.0): on the Finland Swedish test interview it found the interviewer's turns,
3D-Speaker's CAM++ came close, and WeSpeaker ResNet34 put both voices in one
cluster.

``pyannote`` fetches the pyannote/speaker-diarization-community-1 pipeline
(CC BY 4.0) for the pyannote.audio backend. The repository is gated: accept
its terms on huggingface.co once and set HF_TOKEN to a read token. The zip
holds the repository's config and models, which the worker loads offline.

``punctuate`` fetches the ONNX graph, SentencePiece model and config of
1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase (Apache 2.0), not
the 1.1 GB NeMo checkpoint beside them, and adds an int8 copy of the graph
(weights of MatMul and Gather nodes, made with ONNX Runtime, so ``pip
install onnx onnxruntime`` in a venv first or pass ``--no-quantize``; a
distribution's python3-onnx package next to pip's onnxruntime aborts on
import with "File already exists in database: onnx/onnx-ml.proto", because
each brings its own protobuf). The worker's
QUANTIZE setting picks the graph; ``none`` is the default for punctuation
because int8 changed about a tenth of the words on a Swedish test text.

Both zips carry a ``kielipankki.json`` naming the files and their sources.
Upload to the kielipankki_services_data bucket as described in
package_wav2vec_model.py; the init container finds the directory by that
file. Downloads go through curl, see package_wav2vec_model.py for why.
"""

import argparse
import datetime
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile

SHERPA_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION = {
    "name": "pyannote-segmentation-3.0",
    "url": f"{SHERPA_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    "member": "sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    "source": "https://huggingface.co/pyannote/segmentation-3.0",
}
EMBEDDINGS = {
    "titanet": {
        "name": "nemo-titanet-small",
        "file": "nemo_en_titanet_small.onnx",
        "source": "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small",
    },
    "campplus": {
        "name": "3dspeaker-campplus",
        "file": "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
        "source": "https://github.com/modelscope/3D-Speaker",
    },
    "eres2net": {
        "name": "3dspeaker-eres2net-base",
        "file": "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        "source": "https://github.com/modelscope/3D-Speaker",
    },
    "wespeaker": {
        "name": "wespeaker-resnet34-LM",
        "file": "wespeaker_en_voxceleb_resnet34_LM.onnx",
        "source": "https://github.com/wenet-e2e/wespeaker",
    },
}
PYANNOTE_REPO = "pyannote/speaker-diarization-community-1"
PYANNOTE_FILES = [
    "config.yaml",
    "segmentation/pytorch_model.bin",
    "embedding/pytorch_model.bin",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
]
PUNCTUATE_REPO = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
PUNCTUATE_FILES = ["config.yaml", "sp.model", "model.onnx"]


def fetch(url, target, token=None):
    print(f"fetching {url}", flush=True)
    command = ["curl", "-fsSL", "--retry", "3", "-o", target, url]
    if token:
        command[1:1] = ["-H", f"Authorization: Bearer {token}"]
    os.makedirs(os.path.dirname(target), exist_ok=True)
    subprocess.run(command, check=True)


def package_diarize(path, embedding):
    fetch(SEGMENTATION["url"], os.path.join(path, "seg.tar.bz2"))
    with tarfile.open(os.path.join(path, "seg.tar.bz2")) as tar:
        with tar.extractfile(SEGMENTATION["member"]) as src, open(
            os.path.join(path, "segmentation.onnx"), "wb"
        ) as dst:
            dst.write(src.read())
    os.remove(os.path.join(path, "seg.tar.bz2"))
    url = f"{SHERPA_RELEASES}/speaker-recongition-models/{embedding['file']}"
    fetch(url, os.path.join(path, "embedding.onnx"))
    return {
        "name": f"{SEGMENTATION['name']}+{embedding['name']}",
        "segmentation": "segmentation.onnx",
        "embedding": "embedding.onnx",
        "sources": {
            "segmentation": [SEGMENTATION["url"], SEGMENTATION["source"]],
            "embedding": [url, embedding["source"]],
        },
    }


def package_pyannote(path, revision):
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(f"{PYANNOTE_REPO} is gated: accept its terms on huggingface.co and set HF_TOKEN")
    for name in PYANNOTE_FILES:
        fetch(f"https://huggingface.co/{PYANNOTE_REPO}/resolve/{revision}/{name}", os.path.join(path, name), token)
    return {
        "name": PYANNOTE_REPO,
        "backend": "pyannote",
        "revision": revision,
        "source": f"https://huggingface.co/{PYANNOTE_REPO}",
    }


def package_punctuate(path, revision, quantize):
    if quantize:
        if importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxruntime") is None:
            sys.exit("the int8 copy needs onnx and onnxruntime: pip install them or pass --no-quantize")
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.shape_inference import quant_pre_process
    for name in PUNCTUATE_FILES:
        fetch(f"https://huggingface.co/{PUNCTUATE_REPO}/resolve/{revision}/{name}", os.path.join(path, name))
    info = {
        "name": PUNCTUATE_REPO,
        "revision": revision,
        "source": f"https://huggingface.co/{PUNCTUATE_REPO}",
        "files": {"none": "model.onnx"},
    }
    if quantize:
        os.rename(os.path.join(path, "model.onnx"), os.path.join(path, "model.fp32.onnx"))
        # ONNX Runtime's recommended optimisation pass before quantising;
        # its symbolic shape inference does not complete on this graph.
        print("optimising", flush=True)
        optimised = os.path.join(path, "model.optimised.onnx")
        quant_pre_process(os.path.join(path, "model.fp32.onnx"), optimised, skip_symbolic_shape=True)
        print("quantising", flush=True)
        quantize_dynamic(
            optimised,
            os.path.join(path, "model.onnx"),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gather"],
        )
        os.remove(optimised)
        info["files"] = {"int8": "model.onnx", "none": "model.fp32.onnx"}
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["diarize", "punctuate"])
    parser.add_argument("--out", default=None, help="zip file to write, default model-<stage>.zip")
    parser.add_argument("--revision", default="main", help="punctuate and pyannote: branch, tag or commit to fetch")
    parser.add_argument("--backend", choices=["sherpa", "pyannote"], default="sherpa", help="diarize: which worker backend to package for")
    parser.add_argument("--no-quantize", action="store_true", help="punctuate: ship only the fp32 graph")
    parser.add_argument("--embedding", choices=sorted(EMBEDDINGS), default="titanet", help="diarize: speaker embedding model")
    args = parser.parse_args()

    out = args.out or f"model-{args.stage}.zip"
    name = os.path.splitext(os.path.basename(out))[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name)
        os.makedirs(path)
        if args.stage == "diarize" and args.backend == "pyannote":
            info = package_pyannote(path, args.revision)
        elif args.stage == "diarize":
            info = package_diarize(path, EMBEDDINGS[args.embedding])
        else:
            info = package_punctuate(path, args.revision, not args.no_quantize)
        info["packaged"] = datetime.date.today().isoformat()
        with open(os.path.join(path, "kielipankki.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
            f.write("\n")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for directory, _, filenames in sorted(os.walk(path)):
                for filename in sorted(filenames):
                    full = os.path.join(directory, filename)
                    z.write(full, os.path.join(name, os.path.relpath(full, path)))
                    print(f"added {os.path.relpath(full, path)}", flush=True)
    print(f"wrote {out} ({os.path.getsize(out) / 2**20:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
