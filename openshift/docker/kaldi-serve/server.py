#!/usr/bin/python

from flask import Flask, request, Response, jsonify
import json
from io import BytesIO
from kaldiserve import ChainModel, Decoder, parse_model_specs, start_decoding
import sys
import logging
import toml
import redis
import uuid
import threading
import pydub
import pydub.silence
import requests
import time
import platform
import re
import subprocess
from tempfile import TemporaryFile, NamedTemporaryFile

MAX_CONTENT_LENGTH = 500 * 2**20

app = Flask("kaldi-serve")

base_url = "http://nginx:1337/audio/asr/fi"
submit_url = f"{base_url}/submit"

ASR_SEGMENTS = "asr_segments"
ASR = "asr"
expiry_time = 60 * 60 * 24 * 10

redis_conn = redis.Redis(host="redis", port=6379, decode_responses=True)

# REDIS DATA MODEL
# ----------------
# Jobids are UUID strings. These jobids are used as redis keys.
# The values for jobid keys are redis hashes. The following fields should ALWAYS
# be present:
#
# 1) type (ASR, ASR_SEGMENTS etc.)
# 2) status (done, pending etc.)
# 3) processing_started (unix timestamp)
#
# Additionally, if the following are known, they will be in the hash (rather than in
# a json blob somewhere)
#
# 4) processing_finished
# 5) segments (json list of {duration, jobid})
# 6) response (json object)


def update_response_from_redis_hash(response, redis_hash):
    response.update(
        {
            "status": redis_hash.get("status"),
            "processing_started": float(redis_hash.get("processing_started")),
        }
    )
    if "processing_finished" in redis_hash:
        response["processing_finished"] = float(redis_hash["processing_finished"])


# chain model contains all const components to be shared across multiple threads
model = ChainModel(parse_model_specs("model-spec.toml")[0])
model_params = toml.load("model-spec.toml")["model"][0]

# initialize a decoder that references the chain model
decoder = Decoder(model)
decoder_lock = threading.Lock()


def valid_wav_header(data):
    if len(data) < 44:
        return False
    if data[:4] != b"RIFF":
        return False
    if data[8:12] != b"WAVE":
        return False
    if data[12:15] != b"fmt":
        return False
    if data[36:40] != b"data":
        return False
    return True


def decode(data, lock):
    lock.acquire()
    with start_decoding(decoder):
        decoder.decode_wav_audio(data)
        res = decoder.get_decoded_results(1, word_level=True, bidi_streaming=False)
    lock.release()
    return res


def decode_and_commit(data, _id, lock):
    result = decode(data, lock)[0]  # only ever one result here
    response = {}
    response["responses"] = [
        {
            "transcript": result.transcript,
            "confidence": round(result.confidence, 5),
            "words": [
                {
                    "word": word.word,
                    "start": round(word.start_time, 3),
                    "end": round(word.end_time, 3),
                }
                for word in result.words
            ],
        }
    ]
    redis_conn.hset(
        _id,
        mapping={
            "status": "done",
            "processing_finished": round(time.time(), 3),
            "response": json.dumps(response),
        },
    )


def segmented(audio, _id):
    """Split audio, send parts to asr server, commit job ids to redis."""
    logging.error(f"Called segmented")
    min_segment = 5.0
    segments = pydub.silence.split_on_silence(
        audio, min_silence_len=360, silence_thresh=-36, keep_silence=True, seek_step=1
    )
    logging.error(f"Got {len(segments)} segments")
    while len(segments) > 1:
        smallest_duration = audio.duration_seconds
        smallest_duration_idx = 0
        for i, segment in enumerate(segments):
            if segment.duration_seconds < smallest_duration:
                smallest_duration = segment.duration_seconds
                smallest_duration_idx = i
        if smallest_duration >= min_segment:
            break
        if smallest_duration_idx == 0:
            segments[0] += segments[1]
            del segments[1]
        elif smallest_duration_idx == len(segments) - 1:
            segments[-2] += segments[-1]
            del segments[-1]
        else:
            if (
                segments[smallest_duration_idx - 1].duration_seconds
                < segments[smallest_duration_idx + 1].duration_seconds
            ):
                segments[smallest_duration_idx - 1] += segments[smallest_duration_idx]
                del segments[smallest_duration_idx]
            else:
                segments[smallest_duration_idx] += segments[smallest_duration_idx + 1]
                del segments[smallest_duration_idx + 1]
    jobs = []
    logging.error(f"Finally {len(segments)} segments")
    for i, segment in enumerate(segments):
        f = TemporaryFile()
        segment.export(f, format="wav")
        f.seek(0)
        audiobytes = f.read()
        logging.error(f"Posting segment {i}")
        response = requests.post(submit_url, data=audiobytes)
        logging.error(f"Response was {response.text}")
        jobs.append(
            {
                "duration": segment.duration_seconds,
                "jobid": json.loads(response.text)["jobid"],
            }
        )
    redis_conn.hset(_id, key="segments", value=json.dumps(jobs))


@app.route("/audio/asr/fi/submit", methods=["POST"])
def route_submit():
    audio_bytes = bytes(request.get_data(as_text=False))
    if not valid_wav_header(audio_bytes):
        return jsonify({"error": "invalid wav header"})
    _id = str(uuid.uuid4())
    redis_conn.hset(
        _id,
        mapping={
            "type": ASR,
            "status": "pending",
            "processing_started": round(time.time(), 3),
        },
    )
    redis_conn.expire(_id, expiry_time)
    job = threading.Thread(
        target=decode_and_commit, args=(audio_bytes, _id, decoder_lock)
    )
    job.start()
    return jsonify({"jobid": _id})


@app.route("/audio/asr/fi/submit_file", methods=["POST"])
def route_submit_file():
    logging.error(f"File submitted with length {request.content_length}")
    if request.content_length >= MAX_CONTENT_LENGTH:
        return jsonify(
            {"error": f"body size exceeded maximum of {MAX_CONTENT_LENGTH} bytes"}
        )
    args = request.args
    do_split = True
    if "nosplit" in args and args["nosplit"].lower() == "true":
        do_split = False
    if request.content_type.startswith("multipart/form-data"):
        file_name = request.files["file"].filename
        if "." not in file_name:
            return jsonify({"error": "could not determine file type"})
        extension = file_name[file_name.rindex(".") + 1 :]
        try:
            audio = pydub.AudioSegment.from_file(
                request.files["file"], format=extension
            )
        except Exception as ex:
            return jsonify({"error": "could not process file"})
    else:
        if request.content_type.startswith("audio/"):
            if request.content_type.startswith("audio/mpeg"):
                extension = "mp3"
            elif request.content_type.startswith(
                "audio/vorbis"
            ) or request.content_type.startswith("audio/ogg"):
                extension = "ogg"
            elif request.content_type.startswith(
                "audio/wav"
            ) or request.content_type.startswith("audio/x-wav"):
                extension = "wav"
        elif request.content_type.startswith("application/"):
            extension = "wav"
        else:
            return jsonify(
                {
                    "error": "expected either HTML form or mimetype audio/mpeg, audio/vorbis, audio/ogg, audio/wav or audio/x-wav"
                }
            )
        audio_file = BytesIO(request.get_data(as_text=False))

        file_name = ""
        if "Content-Disposition" in request.headers:
            match = re.search(
                r'filename="([^"]+)"', request.headers["Content-Disposition"]
            )
            if match:
                file_name = match.group(1)

        try:
            audio = pydub.AudioSegment.from_file(audio_file, format=extension)
        except Exception as ex:
            return jsonify({"error": "could not process file"})

    if extension == "wav":
        if audio.sample_width != 2 or audio.channels > 1 or audio.frame_rate != 16000:
            downsample_tmp_read_f = NamedTemporaryFile(suffix=".wav")
            audio.export(downsample_tmp_read_f.name, format="wav")
            downsample_tmp_write_f = NamedTemporaryFile(suffix=".wav")
            # For some reason passing arguments to ffmpeg through pydub doesn't seem to work, so we do it this way.
            # -y means overwrite the (temporary) output file, -ac 1 means make it mono if it isn't already, and -c:a pcm_s16le means to use the standard 16 bit encoder for the audio codec
            downsampler = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    f"{downsample_tmp_read_f.name}",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    f"{downsample_tmp_write_f.name}",
                ]
            )
            audio = pydub.AudioSegment.from_file(
                downsample_tmp_write_f.name, format=extension
            )
    _id = str(uuid.uuid4())

    if not do_split:
        f = TemporaryFile()
        audio.export(f, format="wav")
        f.seek(0)
        audio_bytes = f.read()
        redis_conn.hset(
            _id,
            mapping={
                "type": ASR,
                "status": "pending",
                "processing_started": round(time.time(), 3),
            },
        )
        redis_conn.expire(_id, expiry_time)
        job = threading.Thread(
            target=decode_and_commit, args=(audio_bytes, _id, decoder_lock)
        )
        job.start()
    else:
        redis_conn.hset(
            _id,
            mapping={
                "type": ASR_SEGMENTS,
                "status": "pending",
                "processing_started": round(time.time(), 3),
            },
        )
        redis_conn.expire(_id, expiry_time)
        logging.error(f"Setting up segmented thread")
        job = threading.Thread(target=segmented, args=(audio, _id))
        job.start()
    return jsonify({"jobid": _id, "file": file_name})


@app.route("/audio/asr/fi/query_job", methods=["POST"])
def route_query_job():
    _id = request.get_data(as_text=True)
    if _id not in redis_conn:
        return jsonify({"error": f"job id not available"})
    redis_hash = redis_conn.hgetall(_id)
    response = json.loads(redis_hash.get("response", "{}"))
    update_response_from_redis_hash(response, redis_hash)
    if redis_hash.get("type") == ASR:
        return jsonify(response)
    if redis_hash.get("type") != ASR_SEGMENTS:
        return jsonify({"error": "job id not available"})
    if "status" not in redis_hash:
        return jsonify({"error": "internal server error"})
    if "segments" not in redis_hash:
        if redis_hash.get("status") == "pending":
            return jsonify(response)
        else:
            logging.error(f"Got missing segments in redis hash {redis_hash}")
            return jsonify({"error": "internal server error"})
    response["segments"] = []
    segments = json.loads(redis_hash.get("segments"))
    processing_finished = 0.0
    running_time = 0.0
    for segment in segments:
        segment_id = segment["jobid"]
        if segment_id not in redis_conn:
            return jsonify({"error": f"job id not available"})
        segment_redis_hash = redis_conn.hgetall(segment_id)
        if segment_redis_hash.get("status") == "pending":
            return jsonify({"status": "pending"})
        duration = float(segment["duration"])
        segment_result = json.loads(segment_redis_hash.get("response", "{}"))
        update_response_from_redis_hash(segment_result, segment_redis_hash)
        segment_result["start"] = round(running_time, 3)
        running_time += duration
        segment_result["stop"] = round(running_time, 3)
        segment_result["duration"] = round(running_time, 3)
        if "processing_finished" in segment_result:
            segment_processing_finished = float(segment_result["processing_finished"])
            if segment_processing_finished > processing_finished:
                processing_finished = segment_processing_finished
        response["segments"].append(segment_result)
    response["processing_finished"] = processing_finished
    response["status"] = "done"
    response["model"] = model_params
    return jsonify(response)


@app.route("/audio/asr/fi/query_job/tekstiks", methods=["POST"])
def route_query_job_tekstiks():
    tekstiks_version = "KP 0.1"
    no_job_error = 40
    internal_error = 41
    transcribing_failed_error = 1
    _id = request.get_data(as_text=True)
    retval = {"id": _id, "metadata": {"version": tekstiks_version}}
    if _id not in redis_conn:
        retval["done"] = True
        retval["error"] = {"code": no_job_error, "message": "job id not found"}
        return jsonify(retval)
    redis_hash = redis_conn.hgetall(_id)
    update_response_from_redis_hash(retval, redis_hash)

    if redis_hash.get("type") not in (ASR, ASR_SEGMENTS):
        retval["done"] = True
        retval["error"] = {"code": no_job_error, "message": "job id not found"}
        return jsonify(retval)

    retval["result"] = {"speakers": {"S0": {}}, "sections": []}
    running_time = 0.0
    processing_finished = 0.0
    if "segments" not in redis_hash:
        if retval["status"] != "done":
            retval["done"] = False
            retval["message"] = "In progress"
            return jsonify(retval)
        retval["done"] = True
        retval["error"] = {
            "code": no_job_error,
            "message": "job has incompatible api request",
        }
        return jsonify(retval)
    segments = json.loads(redis_hash["segments"])
    for segment in segments:
        segment_id = segment["jobid"]
        if segment_id not in redis_conn:
            retval["error"] = {
                "code": no_job_error,
                "message": "one or more job segment id's not found",
            }
            retval["done"] = True
            return jsonify(retval)
        segment_redis_hash = redis_conn.hgetall(segment_id)
        segment_result = json.loads(segment_redis_hash.get("response", "{}"))
        update_response_from_redis_hash(segment_result, segment_redis_hash)
        if segment_result["status"] != "done":
            retval["done"] = False
            retval["message"] = "In progress"
            return jsonify(retval)
        if segment_result.get("processing_finished", 0.0) > processing_finished:
            processing_finished = segment_result["processing_finished"]
        duration = float(segment["duration"])
        retval["result"]["sections"].append(
            {
                "start": round(running_time, 3),
                "end": round(running_time + duration, 3),
                "transcript": segment_result["responses"][0]["transcript"],
                "words": [],
            }
        )
        running_time += duration
        if "words" in segment_result["responses"][0]:
            retval["result"]["sections"][-1]["words"] = segment_result["responses"][0][
                "words"
            ]
    retval["status"] = "done"
    retval["done"] = True
    retval["processing_finished"] = processing_finished
    return jsonify(retval)


@app.route("/audio/asr/fi/segmented", methods=["POST"])
def route_segmented():
    audio_bytes = bytes(request.get_data(as_text=False))
    if not valid_wav_header(audio_bytes):
        return jsonify({"error": "invalid wav header"})
    f = BytesIO(audio_bytes)
    audio = pydub.AudioSegment.from_file(f, format="wav")
    _id = str(uuid.uuid4())
    redis_conn.hset(
        _id,
        mapping={
            "type": ASR_SEGMENTS,
            "status": "pending",
            "processing_started": round(time.time(), 3),
        },
    )
    redis_conn.expire(_id, expiry_time)
    job = threading.Thread(target=segmented, args=(audio, _id))
    job.start()
    return jsonify({"jobid": _id})


@app.route("/audio/asr/fi", methods=["POST"])
def route_asr():
    audio_bytes = bytes(request.get_data(as_text=False))
    if not valid_wav_header(audio_bytes):
        return jsonify({"error": "invalid wav header"})
    alts = decode(audio_bytes, decoder_lock)
    retvals = []
    for alt in alts:
        retvals.append({"transcript": alt.transcript, "confidence": alt.confidence})
    response = dict(model_params)
    response["responses"] = sorted(retvals, key=lambda x: x["confidence"], reverse=True)

    return jsonify(response)


@app.route("/audio/asr/fi/queue", methods=["GET"])
def route_queue():
    return jsonify({})


@app.route("/audio/asr/fi/health", methods=["GET"])
def route_health():
    response = {"status": "UP", "checks": {"redis": "DOWN"}}
    try:
        if redis_conn.ping():
            response["checks"]["redis"] = "UP"
    except:
        pass
    return jsonify(response)


@app.route("/audio/asr/fi/self_test", methods=["GET"])
def route_self_test():
    try:
        response = json.loads(requests.get(f"{base_url}/health", timeout=3).text)
    except Exception as e:
        logging.error(str(e))
        return jsonify({"status": "DOWN"})
    response["checks"]["decoding"] = "DOWN"
    response["checks"]["submit_file"] = "DOWN"
    response["checks"]["query_response"] = "DOWN"
    test_mp3_filename = "test/puhetta.mp3"
    test_wav_filename = "test/puhetta.wav"
    immediate_asr_url = f"{base_url}"
    submit_file_url = f"{base_url}/submit_file"
    query_url = f"{base_url}/query_job"

    try:
        submit_file_response = requests.post(
            submit_file_url,
            files={
                "file": (
                    test_mp3_filename.split("/")[-1],
                    open(test_mp3_filename, "rb"),
                )
            },
            timeout=3,
        )
        submit_file_response = json.loads(submit_file_response.text)
        assert "jobid" in submit_file_response
        response["checks"]["submit_file"] = "UP"
    except Exception as e:
        logging.error(str(e))
        submit_file_response = None

    try:
        immediate_asr_response = requests.post(
            immediate_asr_url, data=open(test_wav_filename, "rb").read(), timeout=3
        )
        assert "responses" in json.loads(immediate_asr_response.text)
        response["checks"]["decoding"] = "UP"
    except Exception as e:
        logging.error(str(e))
        pass

    if not submit_file_response:
        return jsonify(response)
    try:
        time_increment = 0.5
        time_waited = 0
        while time_waited <= 3:
            time.sleep(time_increment)
            time_waited += time_increment
            query_file_response = requests.post(
                query_url, data=submit_file_response["jobid"]
            )
            query_file_response = json.loads(query_file_response.text)
            if query_file_response["status"] == "done":
                response["checks"]["query_response"] = "UP"
                return jsonify(response)
    except Exception as e:
        logging.error(str(e))
        pass
    return jsonify(response)
