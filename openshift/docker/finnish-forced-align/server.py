#!/usr/bin/python

from flask import Flask, request, Response, jsonify
import json
from io import BytesIO
import sys
import logging
import datetime
import redis
import uuid
import threading
import subprocess
import pydub
import requests
import time
import platform
import re
import os
import shutil
from tempfile import TemporaryFile

MAX_CONTENT_LENGTH = 500*2**20

app = Flask("finnish-forced-align")

STAGING_WRITE_BUSY = "STAGING_WRITE_BUSY"
DATA_DIR_EMPTY = "DATA_DIR_EMPTY"

DataInDir = '/opt/kaldi/egs/src_for_wav'
DataOutDir = '/opt/kaldi/egs/kohdistus'
DataInDirStaging = '/home/app/wav_staging'

redis_conn = redis.Redis(host='redis', port=6379, decode_responses = True)

expiry_time = 60*60*24*10

def validate_transcript(transcript):
    return True

def align():
    completed_process = subprocess.run(
        ["/opt/kaldi/egs/align/aligning_with_Docker/bin/align_in_singularity.sh",
         "phone-finnish-finnish.csv", "false", "false", DataInDir, "no"], # "textDirTrue" as 4th arg to have separate text and audio dirs
        cwd = '/opt/kaldi/egs/align', stderr = subprocess.PIPE, stdout = subprocess.PIPE) # to capture args, pass stdout = subprocess.PIPE, stderr = subprocess.PIPE
    submit_results()

def not_ready_for_processing():
    return os.path.isdir(DataInDir) or os.path.isdir(DataOutDir)

def submit_results():
    dirname = os.listdir(DataOutDir)[0]
    id2result = {}
    try:
        for filename in os.listdir(os.path.join(DataOutDir, dirname)):
            if '.' not in filename:
                continue
            prefix, suffix = filename.split('.')
            if prefix not in id2result:
                id2result[prefix] = {suffix: open(os.path.join(DataOutDir, dirname, filename), encoding="utf-8").read()}
            else:
                id2result[prefix][suffix] = open(os.path.join(DataOutDir, dirname, filename), encoding="utf-8").read()
    except Exception as ex:
        logging.error("tried to submit results, got exception " + str(ex))
    shutil.rmtree(DataInDir)
    shutil.rmtree(DataOutDir)
    for _id in id2result:
        response = {'status': 'done', 'processing_finished': round(time.time(), 3)}
        results = {}
        for suffix in id2result[_id]:
            results[suffix] = id2result[_id][suffix]
        response['results'] = json.dumps(results)
        redis_conn.hset(_id, mapping = response)

@app.route('/audio/align/fi/submit_file', methods=["POST"])
def route_submit_file():
    if request.content_length >= MAX_CONTENT_LENGTH:
        return jsonify({'error': 'body size exceeded maximum of {} bytes'}.format(MAX_CONTENT_LENGTH))
    if not request.content_type.startswith('multipart/form-data') or 'audio' not in request.files or 'transcript' not in request.files:
        return jsonify({'error': 'expected multipart/form-data with audio and transcript file'})
    audio_file_name = request.files['audio'].filename
    if '.' not in audio_file_name:
        return jsonify({'error': 'could not determine audio file type'})
    extension = audio_file_name[audio_file_name.rindex('.')+1:]
    try:
        audio = pydub.AudioSegment.from_file(request.files['audio'], format=extension)
    except Exception as ex:
        return jsonify({'error': 'could not process audio file'})
    transcript_bytes = request.files['transcript'].read()
    transcript = str(transcript_bytes, encoding='utf-8')
    if not validate_transcript(transcript):
        return jsonify({'error': 'transcript file appears invalid'})
    _id = str(uuid.uuid4())
    os.mkdir(DataInDirStaging)
    audio.export(os.path.join(DataInDirStaging, _id + '.wav'), format='wav')
    open(os.path.join(DataInDirStaging, _id + '.txt'), 'w', encoding="utf-8").write(transcript)
    redis_conn.hset(_id, mapping = {'status': 'pending', 'task': 'finnish-forced-align', 'processing_started': round(time.time(), 3)})
    redis_conn.expire(_id, expiry_time)
    wait_counter = 0
    while not_ready_for_processing():
        time.sleep(1)
        wait_counter += 1
        if wait_counter > 25:
            return jsonify({'error': "service unavailable due to load, try again later"})
    os.rename(DataInDirStaging, DataInDir)
    os.mkdir(DataOutDir)
    job = threading.Thread(target=align)
    job.start()
    return jsonify({'jobid': _id, 'file': audio_file_name})

@app.route('/audio/align/fi/query_job', methods=["POST"])
def route_query_job():
    _id = request.get_data(as_text = True)
    if _id not in redis_conn:
        return jsonify({'error': 'job id not available'})
    redis_hash = redis_conn.hgetall(_id)
    if 'processing_started' in redis_hash:
        redis_hash['processing_started'] = float(
            redis_hash['processing_started'])
    if 'processing_finished' in redis_hash:
        redis_hash['processing_finished'] = float(
            redis_hash['processing_finished'])
    return jsonify(redis_hash)

@app.route('/audio/align/fi/health', methods=["GET"])
def route_health():
    response = {"status": "UP",
                "checks": {"redis": "DOWN"}}
    try:
        if redis_conn.ping():
            response["checks"]["redis"] = "UP"
    except:
        pass
    return jsonify(response)
