import requests
import json
import time
import sys
import argparse
import os

parser = argparse.ArgumentParser(description="Test the forced align API")
parser.add_argument("--audio-file", default="pohjantuuli_F1_1_22050.wav")
parser.add_argument("--text-file", default="pohjantuuli_F1_1_22050.txt")
parser.add_argument("--lang", default="fi", help="language code in the path, e.g. fi, sme, sv-FI")
parser.add_argument("--domain", default="")
parser.add_argument("--local", action="store_true")
parser.add_argument(
    "--save-dir",
    metavar="DIR",
    help="write the ctm, eaf and TextGrid results there, named after the audio file",
)
args = parser.parse_args()

path = f"/audio/align/{args.lang}"
url = "https://kielipankki.rahtiapp.fi" + path
if args.domain:
    url = args.domain + path
if args.local:
    url = "http://localhost:1337" + path
submit_file_url = url + "/submit_file"
query_url = url + "/query_job"
response = requests.post(
    submit_file_url,
    files={
        "audio": (os.path.basename(args.audio_file), open(args.audio_file, "rb")),
        "transcript": (os.path.basename(args.text_file), open(args.text_file, "rb")),
    },
)
print(response.text)
response_d = json.loads(response.text)
time.sleep(1)
while True:
    query_response = requests.post(query_url, data=response_d["jobid"])
    query_response_d = json.loads(query_response.text)
    if ("status" in query_response_d and query_response_d["status"] == "pending") or (
        "done" in query_response_d and query_response_d["done"] == False
    ):
        time.sleep(1)
        continue
    else:
        print(json.dumps(query_response_d, indent=4))
        duration = (
            query_response_d["processing_finished"]
            - query_response_d["processing_started"]
        )
        print(f"Got result in {duration}")
        if args.save_dir and "results" in query_response_d:
            os.makedirs(args.save_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(args.audio_file))[0]
            results = json.loads(query_response_d["results"])
            for fmt in ("ctm", "eaf", "TextGrid"):
                out = os.path.join(args.save_dir, f"{stem}.{fmt}")
                with open(out, "w", encoding="utf-8") as f:
                    f.write(results[fmt])
                print(f"Wrote {out}")
        break
