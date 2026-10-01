"""Record reference responses from the live service into test/reference/ as JSON.

Run before changing the ASR or alignment services, then diff new responses
against these files (ignoring jobid and timing fields).
"""
import argparse
import json
import os
import time

import requests

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--domain", default="https://kielipankki.rahtiapp.fi")
parser.add_argument("--outdir", default="reference")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)


def poll(query_url, jobid, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        d = requests.post(query_url, data=jobid).json()
        pending = d.get("status") == "pending" or d.get("done") is False
        if not pending:
            return d
        time.sleep(2)
    raise TimeoutError(jobid)


def save(name, obj):
    path = os.path.join(args.outdir, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    print("wrote", path)


asr = args.domain + "/audio/asr/fi"
align = args.domain + "/audio/align/fi"

for clip in ("puhetta.mp3", "pohjantuuli_F1_1_22050.wav"):
    stem = clip.split(".")[0]
    for variant, query in (("split", ""), ("nosplit", ""), ("tekstiks", "/tekstiks")):
        url = asr + "/submit_file" + ("?nosplit=true" if variant == "nosplit" else "")
        with open(clip, "rb") as f:
            submitted = requests.post(url, files={"file": (clip, f)}).json()
        result = poll(asr + "/query_job" + query, submitted["jobid"])
        save(f"asr_{stem}_{variant}", {"submit": submitted, "result": result})

# The raw-body endpoint accepts only a canonical 44-byte WAV header; ffmpeg
# output carries a LIST chunk and is rejected, so both cases are recorded.
for clip in ("puhetta.wav", "puhetta_converted.wav"):
    with open(clip, "rb") as f:
        submitted = requests.post(asr + "/submit", data=f.read()).json()
    result = poll(asr + "/query_job", submitted["jobid"]) if "jobid" in submitted else None
    save(f"asr_{clip.split('.')[0]}_rawwav", {"submit": submitted, "result": result})

with open("puhetta.wav", "rb") as f:
    save("asr_puhetta_sync", requests.post(asr, data=f.read()).json())

save("asr_health", requests.get(asr + "/health").json())
save("align_health", requests.get(align + "/health").json())

with open("pohjantuuli_F1_1_22050.wav", "rb") as audio, open("pohjantuuli_F1_1_22050.txt", "rb") as text:
    submitted = requests.post(
        align + "/submit_file",
        files={"audio": ("pohjantuuli_F1_1_22050.wav", audio), "transcript": ("pohjantuuli_F1_1_22050.txt", text)},
    ).json()
result = poll(align + "/query_job", submitted["jobid"])
save("align_pohjantuuli", {"submit": submitted, "result": result})
for fmt, ext in (("ctm", "ctm"), ("eaf", "eaf"), ("TextGrid", "TextGrid")):
    path = os.path.join(args.outdir, "align_pohjantuuli." + ext)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.loads(result["results"])[fmt])
    print("wrote", path)
