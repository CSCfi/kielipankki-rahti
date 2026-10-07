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

from . import audio, config, jobs, results, stages
from .jobs import ALIGN, ASR, ASR_SEGMENTS, DONE, FAILED, PENDING, Store, align_task_name
from .results import with_timing

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


def stage_options():
    """The diarisation and punctuation options of a request, or an error
    response."""
    try:
        return stages.StageOptions.from_request(request.values), None
    except ValueError as e:
        return None, error(str(e))


def enqueue_recognition(lang, wav, parent=None):
    jobid = store.create_job(ASR, parent=parent) if parent else store.create_job(ASR)
    store.store_audio(jobid, wav)
    store.enqueue(lang, jobid, "asr")
    return jobid


def submit_single(lang, wav, options):
    jobid = store.create_job(ASR)
    stages.add(store, jobid, lang, wav, options)
    store.store_audio(jobid, wav)
    store.enqueue(lang, jobid, "asr")
    return jobid


def split_and_enqueue(lang, parent, wav):
    """Split a converted WAV on silence and enqueue one child job per segment."""
    try:
        segments = audio.split_on_silence(audio.segment_from_wav(wav))
        children = []
        for segment in segments:
            jobid = enqueue_recognition(lang, audio.segment_to_wav(segment), parent)
            children.append({"duration": segment.duration_seconds, "jobid": jobid})
        store.set_field(parent, "segments", json.dumps(children))
    except Exception:
        log.exception("splitting %s failed", parent)
        store.fail(parent, "could not split audio")
    # The children may all have finished before the segment list existed.
    stages.settle(store, parent)


def submit_split(lang, wav, options):
    parent = store.create_job(ASR_SEGMENTS)
    stages.add(store, parent, lang, wav, options)
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
    options, err = stage_options()
    if err:
        return err
    wav, err = raw_wav_body()
    if err:
        return err
    return jsonify({"jobid": submit_single(lang, wav, options)})


@app.route("/audio/asr/<lang>/segmented", methods=["POST"])
def route_segmented(lang):
    check_lang(lang)
    options, err = stage_options()
    if err:
        return err
    wav, err = raw_wav_body()
    if err:
        return err
    return jsonify({"jobid": submit_split(lang, wav, options)})


@app.route("/audio/asr/<lang>/submit_file", methods=["POST"])
def route_submit_file(lang):
    check_lang(lang)
    if (request.content_length or 0) >= config.MAX_CONTENT_LENGTH:
        return error(f"body size exceeded maximum of {config.MAX_CONTENT_LENGTH} bytes")
    do_split = request.args.get("nosplit", "").lower() != "true"
    options, err = stage_options()
    if err:
        return err
    try:
        data, extension, file_name = read_upload()
    except audio.AudioError as e:
        return error(str(e))
    try:
        wav = audio.to_canonical_wav(data, extension)
    except audio.AudioError as e:
        log.info("could not decode %r: %s", file_name, e)
        return error("could not process file")
    jobid = submit_split(lang, wav, options) if do_split else submit_single(lang, wav, options)
    return jsonify({"jobid": jobid, "file": file_name})


@app.route("/audio/asr/<lang>", methods=["POST"])
def route_asr_sync(lang):
    """Recognise a WAV body and wait for the result."""
    check_lang(lang)
    options, err = stage_options()
    if err:
        return err
    wav, err = raw_wav_body()
    if err:
        return err
    jobid = submit_single(lang, wav, options)
    deadline = time.time() + config.SYNC_TIMEOUT_S
    while time.time() < deadline:
        view = load_recognition(jobid)
        if view is None:
            return error("job id not available")
        status, failure = view.status()
        if status == DONE:
            response = dict(store.model_description(lang))
            response["responses"] = view.segment_results()[0]["responses"]
            add_stage_results(response, view)
            return jsonify(response)
        if status == FAILED:
            return error(failure or "processing failed")
        time.sleep(0.25)
    return jsonify({"error": "timed out waiting for the result", "jobid": jobid})


def load_recognition(jobid):
    """The query-time view of a recognition job. A pending job's punctuation
    trigger is re-checked first, so that a job whose last worker died
    between finishing and triggering still completes."""
    view = results.load(store, jobid)
    if view is not None and view.status()[0] == PENDING:
        stages.settle(store, jobid)
        view = results.load(store, jobid)
    return view


def add_stage_results(response, view):
    """Diarisation turns and the state of the requested stages."""
    checks, warnings = view.checks()
    if checks:
        response["checks"] = checks
    if warnings:
        response["warnings"] = warnings
    if view.turns() is not None:
        response["diarization"] = {"speakers": view.speakers(), "turns": view.turns()}
    return response


# Querying recognition jobs


@app.route("/audio/asr/<lang>/query_job", methods=["POST"])
def route_query_job(lang):
    check_lang(lang)
    jobid = request.get_data(as_text=True).strip()
    view = load_recognition(jobid)
    if view is None or view.missing_child:
        return error("job id not available")
    job = view.job
    status, failure = view.status()
    if not view.split:
        response = view.segment_results()[0]
        response["status"] = status
        if status == DONE:
            response["processing_finished"] = view.processing_finished()
        return jsonify(add_stage_results(response, view))
    if job.get("status") == FAILED or view.segments is None:
        return jsonify(with_timing(json.loads(job.get("response", "{}")), job))
    if status == PENDING:
        return jsonify({"status": PENDING})
    response = with_timing({}, job)
    response["segments"] = view.segment_results()
    response["processing_finished"] = view.processing_finished()
    response["status"] = status
    if failure:
        response["error"] = failure
    response["model"] = store.model_description(lang)
    return jsonify(add_stage_results(response, view))


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

    view = load_recognition(jobid)
    if view is None:
        return fail(TEKSTIKS_NO_JOB, "job id not found")
    job = view.job
    with_timing(retval, job)
    if job.get("status") == FAILED:
        return fail(TEKSTIKS_FAILED, job.get("error", "transcribing failed"))
    if not view.split:
        if job.get("status") != DONE:
            return in_progress()
        return fail(TEKSTIKS_NO_JOB, "job has incompatible api request")
    if view.segments is None:
        return in_progress()
    if view.missing_child:
        return fail(TEKSTIKS_NO_JOB, "one or more job segment id's not found")
    status, failure = view.status()
    if status == PENDING:
        return in_progress()
    if status == FAILED:
        return fail(TEKSTIKS_FAILED, failure or "transcribing failed")

    retval["result"] = {"speakers": {s: {} for s in view.speakers()}, "sections": []}
    segment_results = view.segment_results()
    placed = view.absolute_words(segment_results)
    for i, (segment, result) in enumerate(zip(view.segments, segment_results)):
        alternative = result["responses"][0]
        section = {"type": "speech", "start": segment["start"], "end": segment["stop"]}
        turns = results.turn_entries([w for s, _, w in placed if s == i])
        if turns:
            section["transcript"] = alternative["transcript"]
            section["turns"] = turns
        else:
            section["type"] = "non-speech"
        retval["result"]["sections"].append(section)
    checks, warnings = view.checks()
    if checks:
        retval["checks"] = checks
    if warnings:
        retval["warnings"] = warnings
    retval["status"] = DONE
    retval["done"] = True
    retval["processing_finished"] = view.processing_finished()
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
            response["stages"] = {
                queue: {"workers": store.worker_count(queue), "queue_length": store.queue_length(queue)}
                for queue in (jobs.DIARIZE_QUEUE, jobs.PUNCTUATE_QUEUE)
            }
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
    # Stages are reported as the query result reports them; one that has no
    # worker is skipped rather than failed.
    for name in ("diarization", "punctuation"):
        checks[name] = "SKIPPED"
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
                checks.update(result.get("checks", {}))
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
