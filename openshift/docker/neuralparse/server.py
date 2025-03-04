#!/usr/bin/python

from flask import Flask, request, Response, jsonify
import json
import logging
import datetime
import time

app = Flask("finnish-tnparser")

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
        response = json.loads(str(redis_conn.get(_id), encoding = 'utf-8'))
        response['status'] = 'done'
        response['results'] = {}
        response['processing_finished'] = round(time.time(), 3)
        for suffix in id2result[_id]:
            response['results'][suffix] = id2result[_id][suffix]
        redis_conn.set(_id, json.dumps(response))

@app.route('/text/fi/parse', methods=["POST"])
def route_submit():
    
    os.mkdir(DataInDirStaging)
    audio.export(os.path.join(DataInDirStaging, _id + '.wav'), format='wav')
    open(os.path.join(DataInDirStaging, _id + '.txt'), 'w', encoding="utf-8").write(transcript)
    redis_conn.set(_id, json.dumps({'status': 'pending', 'task': 'finnish-forced-align', 'processing_started': round(time.time(), 3)}))
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
    return jsonify(json.loads(str(redis_conn.get(_id), encoding='utf-8')))
