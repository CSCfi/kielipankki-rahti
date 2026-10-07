import requests
import json
import time
import sys
import argparse

parser = argparse.ArgumentParser(description="Test the asr API")
parser.add_argument("--file", default="puhetta.mp3")
parser.add_argument("--lang", default="fi", help="language code in the path, e.g. fi, sme, sv-FI")
parser.add_argument("--local", action="store_true")
parser.add_argument("--nosplit", action="store_true")
parser.add_argument("--no-diarize", action="store_true", help="skip speaker diarisation")
parser.add_argument("--no-punctuate", action="store_true", help="skip punctuation and truecasing")
parser.add_argument("--num-speakers", type=int, help="fix the number of speakers for diarisation")
parser.add_argument("--domain", default="")
parser.add_argument("--query-path", default="")
parser.add_argument("--tekstiks", action="store_true", help="query the tekstiks view and print its sections and turns")
parser.add_argument("--json-out", action="store_true", help="print the full result as indented JSON")
parser.add_argument(
    "--save-transcript",
    metavar="PATH",
    help="write the recognised text, one line per segment, for use as an alignment transcript",
)
args = parser.parse_args()


path = f"/audio/asr/{args.lang}"
url = "https://kielipankki.rahtiapp.fi" + path
if args.domain:
    url = args.domain + path
if args.local:
    url = "http://localhost:1337" + path

filename = args.file
submit_file_url = url + "/submit_file"
options = []
if args.nosplit:
    options.append("nosplit=true")
if args.no_diarize:
    options.append("diarize=false")
if args.no_punctuate:
    options.append("punctuate=false")
if args.num_speakers:
    options.append(f"num_speakers={args.num_speakers}")
if options:
    submit_file_url += "?" + "&".join(options)
query_url = url + "/query_job" + ("/tekstiks" if args.tekstiks else args.query_path)
load_url = url + "/queue"

response = requests.post(
    submit_file_url, files={"file": (filename, open(filename, "rb"))}
)
print(response.text)
response_d = json.loads(response.text)
time.sleep(1)


def transcripts(result):
    """Recognised text per segment from a query_job or tekstiks result."""
    if "segments" in result:
        return [s["responses"][0]["transcript"] for s in result["segments"]]
    if "responses" in result:
        return [result["responses"][0]["transcript"]]
    if "result" in result and "sections" in result["result"]:
        return [s["transcript"] for s in result["result"]["sections"]]
    return []


while True:
    query_response = requests.post(query_url, data=response_d["jobid"])
    query_response_d = json.loads(query_response.text)
    if ("status" in query_response_d and query_response_d["status"] == "pending") or (
        "done" in query_response_d and query_response_d["done"] == False
    ):
        time.sleep(2)
        continue
    duration = (
        query_response_d["processing_finished"] - query_response_d["processing_started"]
    )
    if args.json_out:
        print(json.dumps(query_response_d, indent=2, ensure_ascii=False))
    print(f"Got result in {duration:.1f} s")
    if "checks" in query_response_d:
        print("Stages:", query_response_d["checks"], query_response_d.get("warnings", ""))
    for segment in query_response_d.get("segments", []):
        alternative = segment["responses"][0]
        speakers = sorted({w.get("speaker", "?") for w in alternative.get("words", [])})
        print(f"[{segment['start']}-{segment['stop']}] {','.join(speakers)}:",
              alternative.get("transcript_punctuated", alternative["transcript"]))
    tekstiks = query_response_d.get("result", {})
    if "sections" in tekstiks:
        print("Speakers:", ", ".join(tekstiks.get("speakers", {})))
        for section in tekstiks["sections"]:
            print(f"Section [{section['start']}-{section['end']}]")
            for turn in section.get("turns", []):
                print(f"  [{turn['start']}-{turn['end']}] {turn['speaker']}: {turn['transcript']}")
    if args.save_transcript:
        with open(args.save_transcript, "w", encoding="utf-8") as f:
            f.write("\n".join(transcripts(query_response_d)) + "\n")
        print(f"Wrote transcript to {args.save_transcript}")
    break
