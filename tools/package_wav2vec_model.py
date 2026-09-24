#!/usr/bin/env python3
"""Download a Hugging Face wav2vec2 CTC checkpoint and zip it for the init
container.

    python3 tools/package_wav2vec_model.py GetmanY1/wav2vec2-large-fi-475k-finetuned-experimental \
        --out model-fi-wav2vec2.zip

Only the files the worker needs are fetched: the config, the safetensors
weights and the tokenizer and feature extractor configs. The zip holds one
directory named after the archive containing those files plus a
kielipankki.json that records where the model came from; the worker reports
that name in the ``model`` field of its responses.

Upload the result to the kielipankki_services_data bucket in Allas, for
example with ``swift upload kielipankki_services_data model-fi-wav2vec2.zip``
after ``source allas_conf`` on Puhti, and pass the public URL
``https://a3s.fi/kielipankki_services_data/<name>.zip`` as MODEL_URL to
openshift/templates/wav2vec_worker.yaml.

Files are fetched from huggingface.co with curl, the same way the init
container fetches the zip; nothing beyond the standard library is needed.
Set HF_TOKEN for a gated repository.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import zipfile

REQUIRED = ["config.json", "model.safetensors", "vocab.json", "tokenizer_config.json"]
OPTIONAL = ["special_tokens_map.json", "preprocessor_config.json", "added_tokens.json"]


def fetch(repo, revision, filename, target):
    """Download one file; False if the repository does not have it."""
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    command = ["curl", "-sS", "-L", "--retry", "3", "-o", target, "-w", "%{http_code}", url]
    if os.environ.get("HF_TOKEN"):
        command[1:1] = ["-H", f"Authorization: Bearer {os.environ['HF_TOKEN']}"]
    print(f"fetching {filename}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    status = result.stdout.strip()
    if status == "404":
        return False
    if result.returncode != 0 or status != "200":
        sys.exit(f"downloading {url} failed: HTTP {status} {result.stderr.strip()}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="Hugging Face repository, for example GetmanY1/wav2vec2-large-fi-475k-finetuned-experimental")
    parser.add_argument("--revision", default="main", help="branch, tag or commit to fetch, default main")
    parser.add_argument("--out", default=None, help="zip file to write, default <last part of repo>.zip")
    parser.add_argument("--keep-dir", default=None, help="also leave the unpacked model directory here")
    args = parser.parse_args()

    out = args.out or args.repo.rsplit("/", 1)[-1] + ".zip"
    name = os.path.splitext(os.path.basename(out))[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = args.keep_dir or os.path.join(tmp, name)
        os.makedirs(path, exist_ok=True)
        for filename in REQUIRED:
            if not fetch(args.repo, args.revision, filename, os.path.join(path, filename)):
                sys.exit(f"{filename} not found in {args.repo}; convert the checkpoint first")
        for filename in OPTIONAL:
            target = os.path.join(path, filename)
            if not fetch(args.repo, args.revision, filename, target):
                os.remove(target)
        with open(os.path.join(path, "kielipankki.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": args.repo,
                    "revision": args.revision,
                    "source": f"https://huggingface.co/{args.repo}",
                    "packaged": datetime.date.today().isoformat(),
                },
                f,
                indent=2,
            )
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename in sorted(os.listdir(path)):
                full = os.path.join(path, filename)
                if os.path.isfile(full):
                    zf.write(full, os.path.join(name, filename))
                    print("added", filename, f"{os.path.getsize(full) / 2**20:.1f} MiB")
    print("wrote", out, f"{os.path.getsize(out) / 2**20:.1f} MiB")
    print(f"upload it and use MODEL_URL=https://a3s.fi/kielipankki_services_data/{os.path.basename(out)}")


if __name__ == "__main__":
    main()
