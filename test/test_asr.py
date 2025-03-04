import requests
import json
import time
import sys
import argparse

parser = argparse.ArgumentParser(description="Test the asr API")
parser.add_argument("--file", default="puhetta.mp3")
parser.add_argument("--local", action="store_true")
parser.add_argument("--nosplit", action="store_true")
parser.add_argument("--domain", default="")
parser.add_argument("--query-path", default="")
args = parser.parse_args()


path = "/audio/asr/fi"
url = "https://kielipankki.rahtiapp.fi" + path
if args.domain:
    url = args.domain + path
if args.local:
    url = "http://localhost:1337" + path

filename = args.file
submit_file_url = url + "/submit_file"
if args.nosplit:
    submit_file_url += "?nosplit=true"
query_url = url + "/query_job" + args.query_path
load_url = url + "/queue"

response = requests.post(
    submit_file_url, files={"file": (filename, open(filename, "rb"))}
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
        time.sleep(2)
        continue
    duration = (
        query_response_d["processing_finished"] - query_response_d["processing_started"]
    )
    print(f"Got result in {duration}")
    break
