"""HTTP API for ``/audio/asr/<lang>/...`` and ``/audio/align/<lang>/...``.

Stateless: uploads are converted, split and written to Redis, and workers
pick them up from the per-language queue. The routes and response shapes
are those of the Kaldi services this replaces.
"""

import json
import logging
import os
import re
import threading
import time

import requests
from flask import Flask, abort, jsonify, request

from . import audio, config
from .jobs import ALIGN, ASR, ASR_SEGMENTS, DONE, FAILED, PENDING, Store, align_task_name

log = logging.getLogger("asr.api")

app = Flask("asr")
store = Store()

TEKSTIKS_VERSION = "KP 0.1"
TEKSTIKS_NO_JOB = 40
TEKSTIKS_INTERNAL_ERROR = 41
TEKSTIKS_FAILED = 1


def check_lang(lang):
    if lang not in config.LANGUAGES:
        response = jsonify(
            {"error": f"unknown language {lang!r}; this API serves {', '.join(config.LANGUAGES)}"}
        )
        response.status_code = 404
        abort(response)


def error(message):
    return jsonify({"error": message})


def with_timing(response, job):
    response.update(
        {
            "status": job.get("status"),
            "processing_started": float(job.get("processing_started")),
        }
    )
    if "processing_finished" in job:
        response["processing_finished"] = float(job["processing_finished"])
    return response


# Submitting recognition jobs


def read_upload():
    """(bytes, extension hint, file name) from a multipart form with a
    ``file`` field or from a raw body with an audio content type."""
    content_type = request.content_type or ""
    if content_type.startswith("multipart/form-data"):
        upload = request.files.get("file")
        if upload is None:
            raise audio.AudioError("expected a form with a file field")
        file_name = upload.filename or ""
        if "." not in file_name:
            raise audio.AudioError("could not determine file type")
        return upload.read(), file_name.rsplit(".", 1)[1].lower(), file_name
    if content_type.startswith("audio/mpeg"):
        extension = "mp3"
    elif content_type.startswith(("audio/vorbis", "audio/ogg")):
        extension = "ogg"
    elif content_type.startswith(("audio/wav", "audio/x-wav", "application/")):
        extension = "wav"
    elif content_type.startswith("audio/"):
        extension = None
    else:
        raise audio.AudioError(
            "expected either HTML form or mimetype audio/mpeg, audio/vorbis, "
            "audio/ogg, audio/wav or audio/x-wav"
        )
    file_name = ""
    match = re.search(r'filename="([^"]+)"', request.headers.get("Content-Disposition", ""))
    if match:
        file_name = match.group(1)
    return request.get_data(), extension, file_name


def submit_single(lang, wav):
    jobid = store.create_job(ASR)
    store.store_audio(jobid, wav)
    store.enqueue(lang, jobid, "asr")
    return jobid


def split_and_enqueue(lang, parent, wav):
    """Split a converted WAV on silence and enqueue one child job per segment."""
    try:
        segments = audio.split_on_silence(audio.segment_from_wav(wav))
        children = []
        for segment in segments:
            jobid = submit_single(lang, audio.segment_to_wav(segment))
            children.append({"duration": segment.duration_seconds, "jobid": jobid})
        store.set_field(parent, "segments", json.dumps(children))
    except Exception:
        log.exception("splitting %s failed", parent)
        store.fail(parent, "could not split audio")


def submit_split(lang, wav):
    parent = store.create_job(ASR_SEGMENTS)
    threading.Thread(target=split_and_enqueue, args=(lang, parent, wav), daemon=True).start()
    return parent


def raw_wav_body():
    data = request.get_data()
    if not audio.is_riff_wave(data):
        return None, error("invalid wav header")
    try:
        return audio.to_canonical_wav(data, "wav"), None
    except audio.AudioError:
        return None, error("could not process file")


@app.route("/audio/asr/<lang>/submit", methods=["POST"])
def route_submit(lang):
    check_lang(lang)
    wav, err = raw_wav_body()
    if err:
        return err
    return jsonify({"jobid": submit_single(lang, wav)})


@app.route("/audio/asr/<lang>/segmented", methods=["POST"])
def route_segmented(lang):
    check_lang(lang)
    wav, err = raw_wav_body()
    if err:
        return err
    return jsonify({"jobid": submit_split(lang, wav)})


@app.route("/audio/asr/<lang>/submit_file", methods=["POST"])
def route_submit_file(lang):
    check_lang(lang)
    if (request.content_length or 0) >= config.MAX_CONTENT_LENGTH:
        return error(f"body size exceeded maximum of {config.MAX_CONTENT_LENGTH} bytes")
    do_split = request.args.get("nosplit", "").lower() != "true"
    try:
        data, extension, file_name = read_upload()
    except audio.AudioError as e:
        return error(str(e))
    try:
        wav = audio.to_canonical_wav(data, extension)
    except audio.AudioError as e:
        log.info("could not decode %r: %s", file_name, e)
        return error("could not process file")
    jobid = submit_split(lang, wav) if do_split else submit_single(lang, wav)
    return jsonify({"jobid": jobid, "file": file_name})


@app.route("/audio/asr/<lang>", methods=["POST"])
def route_asr_sync(lang):
    """Recognise a WAV body and wait for the result."""
    check_lang(lang)
    wav, err = raw_wav_body()
    if err:
        return err
    jobid = submit_single(lang, wav)
    deadline = time.time() + config.SYNC_TIMEOUT_S
    while time.time() < deadline:
        job = store.get_job(jobid)
        if job is None:
            return error("job id not available")
        if job.get("status") == DONE:
            response = dict(store.model_description(lang))
            response["responses"] = json.loads(job["response"])["responses"]
            return jsonify(response)
        if job.get("status") == FAILED:
            return error(job.get("error", "processing failed"))
        time.sleep(0.25)
    return jsonify({"error": "timed out waiting for the result", "jobid": jobid})


# Querying recognition jobs


@app.route("/audio/asr/<lang>/query_job", methods=["POST"])
def route_query_job(lang):
    check_lang(lang)
    jobid = request.get_data(as_text=True).strip()
    job = store.get_job(jobid) if jobid else None
    if job is None:
        return error("job id not available")
    response = with_timing(json.loads(job.get("response", "{}")), job)
    if job.get("type") == ASR:
        return jsonify(response)
    if job.get("type") != ASR_SEGMENTS:
        return error("job id not available")
    if job.get("status") == FAILED:
        return jsonify(response)
    if "segments" not in job:
        if job.get("status") == PENDING:
            return jsonify(response)
        log.error("segments missing from %s: %s", jobid, job)
        return error("internal server error")

    response["segments"] = []
    processing_finished = 0.0
    running_time = 0.0
    failed = None
    for segment in json.loads(job["segments"]):
        child = store.get_job(segment["jobid"])
        if child is None:
            return error("job id not available")
        if child.get("status") == PENDING:
            return jsonify({"status": PENDING})
        result = with_timing(json.loads(child.get("response", "{}")), child)
        if child.get("status") == FAILED and failed is None:
            failed = result.get("error", "processing failed")
        duration = float(segment["duration"])
        result["start"] = round(running_time, 3)
        running_time += duration
        result["stop"] = round(running_time, 3)
        result["duration"] = round(running_time, 3)
        processing_finished = max(processing_finished, result.get("processing_finished", 0.0))
        response["segments"].append(result)
    response["processing_finished"] = processing_finished
    response["status"] = DONE
    if failed:
        response["status"] = FAILED
        response["error"] = failed
    response["model"] = store.model_description(lang)
    return jsonify(response)


@app.route("/audio/asr/<lang>/query_job/tekstiks", methods=["POST"])
def route_query_job_tekstiks(lang):
    check_lang(lang)
    jobid = request.get_data(as_text=True).strip()
    retval = {"id": jobid, "metadata": {"version": TEKSTIKS_VERSION}}

    def fail(code, message):
        retval["done"] = True
        retval["error"] = {"code": code, "message": message}
        return jsonify(retval)

    def in_progress():
        retval["done"] = False
        retval["message"] = "In progress"
        return jsonify(retval)

    job = store.get_job(jobid) if jobid else None
    if job is None or job.get("type") not in (ASR, ASR_SEGMENTS):
        return fail(TEKSTIKS_NO_JOB, "job id not found")
    with_timing(retval, job)
    if job.get("status") == FAILED:
        return fail(TEKSTIKS_FAILED, job.get("error", "transcribing failed"))
    if "segments" not in job:
        if retval["status"] != DONE:
            return in_progress()
        return fail(TEKSTIKS_NO_JOB, "job has incompatible api request")

    retval["result"] = {"speakers": {"S0": {}}, "sections": []}
    running_time = 0.0
    processing_finished = 0.0
    for segment in json.loads(job["segments"]):
        child = store.get_job(segment["jobid"])
        if child is None:
            return fail(TEKSTIKS_NO_JOB, "one or more job segment id's not found")
        if child.get("status") == PENDING:
            return in_progress()
        if child.get("status") == FAILED:
            return fail(TEKSTIKS_FAILED, child.get("error", "transcribing failed"))
        result = with_timing(json.loads(child.get("response", "{}")), child)
        processing_finished = max(processing_finished, result.get("processing_finished", 0.0))
        duration = float(segment["duration"])
        alternative = result["responses"][0]
        retval["result"]["sections"].append(
            {
                "start": round(running_time, 3),
                "end": round(running_time + duration, 3),
                "transcript": alternative["transcript"],
                "words": alternative.get("words", []),
            }
        )
        running_time += duration
    retval["status"] = DONE
    retval["done"] = True
    retval["processing_finished"] = processing_finished
    return jsonify(retval)


# Forced alignment


@app.route("/audio/align/<lang>/submit_file", methods=["POST"])
def route_align_submit_file(lang):
    check_lang(lang)
    if (request.content_length or 0) >= config.MAX_CONTENT_LENGTH:
        return error(f"body size exceeded maximum of {config.MAX_CONTENT_LENGTH} bytes")
    if (
        not (request.content_type or "").startswith("multipart/form-data")
        or "audio" not in request.files
        or "transcript" not in request.files
    ):
        return error("expected multipart/form-data with audio and transcript file")
    upload = request.files["audio"]
    file_name = upload.filename or ""
    if "." not in file_name:
        return error("could not determine audio file type")
    try:
        wav = audio.to_canonical_wav(upload.read(), file_name.rsplit(".", 1)[1].lower())
    except audio.AudioError as e:
        log.info("could not decode %r: %s", file_name, e)
        return error("could not process audio file")
    duration = audio.duration_s(wav)
    if duration > config.MAX_ALIGN_S:
        return error(
            f"audio is {duration:.0f} s long; alignment accepts at most "
            f"{config.MAX_ALIGN_S:.0f} s"
        )
    try:
        transcript = request.files["transcript"].read().decode("utf-8")
    except UnicodeDecodeError:
        return error("transcript file appears invalid")
    if not transcript.strip():
        return error("transcript file appears invalid")
    jobid = store.create_job(ALIGN, task=align_task_name(lang))
    store.store_audio(jobid, wav)
    store.store_text(jobid, transcript)
    store.enqueue(lang, jobid, "align")
    return jsonify({"jobid": jobid, "file": file_name})


@app.route("/audio/align/<lang>/query_job", methods=["POST"])
def route_align_query_job(lang):
    check_lang(lang)
    jobid = request.get_data(as_text=True).strip()
    job = store.get_job(jobid) if jobid else None
    if job is None:
        return error("job id not available")
    for field in ("processing_started", "processing_finished"):
        if field in job:
            job[field] = float(job[field])
    return jsonify(job)


# Health


def health(lang):
    response = {"status": "UP", "checks": {"redis": "DOWN", "workers": "DOWN"}}
    try:
        if store.ping():
            response["checks"]["redis"] = "UP"
            response["queue_length"] = store.queue_length(lang)
            response["workers"] = store.worker_count(lang)
            if response["workers"]:
                response["checks"]["workers"] = "UP"
    except Exception:
        response["status"] = "DOWN"
    return jsonify(response)


@app.route("/audio/asr/<lang>/health", methods=["GET"])
def route_health(lang):
    check_lang(lang)
    return health(lang)


@app.route("/audio/align/<lang>/health", methods=["GET"])
def route_align_health(lang):
    check_lang(lang)
    return health(lang)


@app.route("/audio/asr/health", methods=["GET"])
def route_api_health():
    """Language independent, for the readiness probe."""
    try:
        store.ping()
        return jsonify({"status": "UP"})
    except Exception:
        return jsonify({"status": "DOWN"}), 503


@app.route("/audio/asr/<lang>/queue", methods=["GET"])
def route_queue(lang):
    check_lang(lang)
    return jsonify({"length": store.queue_length(lang)})


TEST_CLIPS = ("speech.mp3", "speech.wav", "align.wav", "align.txt")


def has_test_clips(directory):
    return all(os.path.exists(os.path.join(directory, name)) for name in TEST_CLIPS)


@app.route("/audio/asr/<lang>/self_test", methods=["GET"])
def route_self_test(lang):
    """Exercise the service through the public router with bundled clips."""
    check_lang(lang)
    base = f"{config.PUBLIC_BASE_URL}/audio/asr/{lang}"
    align_base = f"{config.PUBLIC_BASE_URL}/audio/align/{lang}"
    # Clips live in test/<lang>/ as speech.mp3, speech.wav, align.wav and
    # align.txt; a language without its own clips is tested with Finnish
    # audio, which still exercises the whole pipeline. The clips are not in
    # the repository; without them the checks are reported as SKIPPED.
    clips_lang = next(
        (l for l in (lang, "fi") if has_test_clips(f"{config.TEST_DATA_DIR}/{l}")), None
    )
    try:
        response = requests.get(f"{base}/health", timeout=3).json()
    except Exception as e:
        log.error("self test: health failed: %s", e)
        return jsonify({"status": "DOWN"})
    response["test_clips"] = clips_lang
    checks = response.setdefault("checks", {})
    for name in ("decoding", "submit_file", "query_response", "align"):
        checks[name] = "DOWN" if clips_lang else "SKIPPED"
    if clips_lang is None:
        return jsonify(response)
    data = f"{config.TEST_DATA_DIR}/{clips_lang}"

    def poll(url, jobid, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.5)
            result = requests.post(url, data=jobid, timeout=10).json()
            if result.get("status") in (DONE, FAILED):
                return result
        return None

    asr_job = align_job = None
    try:
        with open(f"{data}/speech.mp3", "rb") as f:
            submitted = requests.post(
                f"{base}/submit_file", files={"file": ("speech.mp3", f)}, timeout=10
            ).json()
        asr_job = submitted["jobid"]
        checks["submit_file"] = "UP"
    except Exception as e:
        log.error("self test: submit_file failed: %s", e)
    try:
        with open(f"{data}/align.wav", "rb") as a, open(f"{data}/align.txt", "rb") as t:
            submitted = requests.post(
                f"{align_base}/submit_file",
                files={"audio": ("align.wav", a), "transcript": ("align.txt", t)},
                timeout=10,
            ).json()
        align_job = submitted["jobid"]
    except Exception as e:
        log.error("self test: align submit failed: %s", e)
    try:
        with open(f"{data}/speech.wav", "rb") as f:
            result = requests.post(base, data=f.read(), timeout=config.SYNC_TIMEOUT_S + 5).json()
        assert result["responses"][0]["transcript"]
        checks["decoding"] = "UP"
    except Exception as e:
        log.error("self test: decoding failed: %s", e)
    if asr_job:
        try:
            result = poll(f"{base}/query_job", asr_job, 60)
            if result and result["status"] == DONE and result["segments"]:
                checks["query_response"] = "UP"
        except Exception as e:
            log.error("self test: query_job failed: %s", e)
    if align_job:
        try:
            result = poll(f"{align_base}/query_job", align_job, 120)
            if result and result["status"] == DONE and "ctm" in json.loads(result["results"]):
                checks["align"] = "UP"
        except Exception as e:
            log.error("self test: align failed: %s", e)
    return jsonify(response)
