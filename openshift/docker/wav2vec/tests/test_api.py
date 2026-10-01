"""API and worker against fakeredis with a stand-in recogniser, so that the
request flow, the Redis data model and the response shapes are exercised
without a model. The clips are PCM WAV, so ffmpeg is not needed either."""

import io
import json
import threading
import wave

import fakeredis
import numpy as np
import pytest

from asr import api, audio, config, ctc, jobs, worker

TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "|", "a", "b", "c", "d"]


class FakeRecognizer:
    """Emits "ab" for the first second and "cd" after a pause, per window."""

    def __init__(self):
        self.tokens = TOKENS
        self.blank, self.delimiter, self.ignore = 0, 4, {1, 2, 3}
        self.frame_s = 0.02
        self.normalizer = ctc.Normalizer(TOKENS, 0, 4, {1, 2, 3})
        self.description = {"name": "fake", "language": "fi", "backend": "wav2vec2", "quantization": "none"}

    def emissions(self, samples, progress=None):
        frames = max(int(len(samples) / 16000 / self.frame_s), 10)
        path = np.zeros(frames, dtype=np.int64)
        path[1] = 5
        path[3] = 6
        path[4] = 4
        if frames > 40:
            path[30] = 7
            path[32] = 8
        lp = np.full((frames, len(TOKENS)), np.log(0.01), np.float32)
        lp[np.arange(frames), path] = np.log(0.9)
        return lp


@pytest.fixture
def store(monkeypatch):
    server = fakeredis.FakeServer()
    s = jobs.Store(
        fakeredis.FakeStrictRedis(server=server, decode_responses=True),
        fakeredis.FakeStrictRedis(server=server, decode_responses=False),
        fakeredis.FakeStrictRedis(server=server, decode_responses=True),
    )
    monkeypatch.setattr(api, "store", s)
    return s


@pytest.fixture
def client(store):
    api.app.config["TESTING"] = True
    return api.app.test_client()


def run_worker(store, jobs_expected):
    w = worker.Worker(store, FakeRecognizer(), lang="fi", name="test-worker")
    w.start()
    for _ in range(jobs_expected):
        raw, message = store.claim("fi", "test-worker")
        w.handle(message)
        store.release("fi", "test-worker", raw)


def wav_bytes(seconds, rate=16000, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate) * channels)
    return buf.getvalue()


def test_submit_file_nosplit_and_query(client, store):
    data = wav_bytes(3.0)
    r = client.post("/audio/asr/fi/submit_file?nosplit=true", data={"file": (io.BytesIO(data), "x.wav")},
                    content_type="multipart/form-data").get_json()
    assert set(r) == {"jobid", "file"} and r["file"] == "x.wav"
    assert client.post("/audio/asr/fi/query_job", data=r["jobid"]).get_json()["status"] == "pending"
    assert store.queue_length("fi") == 1
    assert client.get("/audio/asr/fi/queue").get_json() == {"length": 1}

    run_worker(store, 1)
    result = client.post("/audio/asr/fi/query_job", data=r["jobid"]).get_json()
    assert result["status"] == "done"
    assert set(result) == {"status", "processing_started", "processing_finished", "responses"}
    alt = result["responses"][0]
    assert alt["transcript"] == "ab cd"
    assert [w["word"] for w in alt["words"]] == ["ab", "cd"]
    assert set(alt["words"][0]) == {"word", "start", "end", "confidence"}
    assert store.load_audio(r["jobid"]) is None


def test_submit_file_split_segments_and_tekstiks(client, store):
    # 12 s with a two second silence in the middle: two segments.
    tone = (np.sin(np.arange(16000 * 5) * 0.1) * 8000).astype("<i2").tobytes()
    silence = b"\0\0" * 16000 * 2
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(tone + silence + tone)
    r = client.post("/audio/asr/fi/submit_file", data={"file": (io.BytesIO(buf.getvalue()), "x.wav")},
                    content_type="multipart/form-data").get_json()
    for t in threading.enumerate():
        if t is not threading.current_thread():
            t.join(10)
    parent = store.get_job(r["jobid"])
    segments = json.loads(parent["segments"])
    assert len(segments) == 2
    assert client.post("/audio/asr/fi/query_job", data=r["jobid"]).get_json() == {"status": "pending"}
    tk = client.post("/audio/asr/fi/query_job/tekstiks", data=r["jobid"]).get_json()
    assert tk["done"] is False and tk["message"] == "In progress"

    run_worker(store, 2)
    result = client.post("/audio/asr/fi/query_job", data=r["jobid"]).get_json()
    assert result["status"] == "done"
    assert result["model"]["backend"] == "wav2vec2"
    assert len(result["segments"]) == 2
    first, second = result["segments"]
    assert first["start"] == 0.0 and first["stop"] == pytest.approx(segments[0]["duration"], abs=0.01)
    assert second["start"] == first["stop"]
    assert second["duration"] == second["stop"]
    assert set(first) >= {"responses", "status", "processing_started", "processing_finished", "start", "stop", "duration"}
    tk = client.post("/audio/asr/fi/query_job/tekstiks", data=r["jobid"]).get_json()
    assert tk["done"] is True and tk["status"] == "done"
    assert tk["metadata"] == {"version": "KP 0.1"}
    assert tk["result"]["speakers"] == {"S0": {}}
    assert [s["transcript"] for s in tk["result"]["sections"]] == ["ab cd", "ab cd"]
    assert tk["result"]["sections"][1]["start"] == first["stop"]


def test_raw_submit_and_sync(client, store):
    assert client.post("/audio/asr/fi/submit", data=b"not a wav").get_json() == {"error": "invalid wav header"}
    r = client.post("/audio/asr/fi/submit", data=wav_bytes(0.5)).get_json()
    assert set(r) == {"jobid"}
    run_worker(store, 1)
    assert client.post("/audio/asr/fi/query_job", data=r["jobid"]).get_json()["responses"][0]["transcript"] == "ab"

    # The synchronous endpoint waits for a worker; run one in a thread.
    done = threading.Event()

    def later():
        while store.queue_length("fi") == 0:
            pass
        run_worker(store, 1)
        done.set()

    threading.Thread(target=later, daemon=True).start()
    result = client.post("/audio/asr/fi", data=wav_bytes(0.5)).get_json()
    done.wait(5)
    assert result["responses"][0]["transcript"] == "ab"
    assert result["backend"] == "wav2vec2" and result["name"] == "fake"


def test_query_errors(client, store):
    assert client.post("/audio/asr/fi/query_job", data="nope").get_json() == {"error": "job id not available"}
    tk = client.post("/audio/asr/fi/query_job/tekstiks", data="nope").get_json()
    assert tk["done"] is True and tk["error"]["code"] == 40
    r = client.get("/audio/asr/xx/health")
    assert r.status_code == 404 and r.get_json()["error"].startswith("unknown language 'xx'")
    assert client.post("/audio/asr/fi/submit_file", data=b"x", content_type="text/plain").get_json()["error"].startswith("expected either")


def test_expired_audio_fails_job(client, store):
    r = client.post("/audio/asr/fi/submit", data=wav_bytes(1.0)).get_json()
    store.drop_inputs(r["jobid"])
    run_worker(store, 1)
    result = client.post("/audio/asr/fi/query_job", data=r["jobid"]).get_json()
    assert result["status"] == "failed"
    assert result["error"] == "audio expired before processing"


def test_align_flow(client, store):
    data = {
        "audio": (io.BytesIO(wav_bytes(2.0)), "clip.wav"),
        "transcript": (io.BytesIO("ab cd".encode()), "clip.txt"),
    }
    r = client.post("/audio/align/fi/submit_file", data=data, content_type="multipart/form-data").get_json()
    assert r["file"] == "clip.wav"
    job = client.post("/audio/align/fi/query_job", data=r["jobid"]).get_json()
    assert job["status"] == "pending" and job["task"] == "finnish-forced-align" and job["type"] == "align"
    run_worker(store, 1)
    job = client.post("/audio/align/fi/query_job", data=r["jobid"]).get_json()
    assert job["status"] == "done"
    results = json.loads(job["results"])
    assert set(results) == {"ctm", "eaf", "TextGrid", "intervals"}
    assert [w["label"] for w in results["intervals"]["words"]] == ["ab", "cd"]
    assert [l["label"] for l in results["intervals"]["letters"]] == ["a", "b", "c", "d"]
    assert results["ctm"].splitlines()[0].endswith(" ab")
    assert 'name = "letters"' in results["TextGrid"]


def test_align_rejects_bad_input(client, store):
    r = client.post("/audio/align/fi/submit_file", data=b"x", content_type="text/plain").get_json()
    assert r["error"].startswith("expected multipart")
    data = {"audio": (io.BytesIO(wav_bytes(2.0)), "clip.wav"), "transcript": (io.BytesIO(b"ab 12"), "t.txt")}
    r = client.post("/audio/align/fi/submit_file", data=data, content_type="multipart/form-data").get_json()
    run_worker(store, 1)
    job = client.post("/audio/align/fi/query_job", data=r["jobid"]).get_json()
    assert job["status"] == "failed" and "digits" in job["error"]
    old = config.MAX_ALIGN_S
    config.MAX_ALIGN_S = 1.0
    try:
        data = {"audio": (io.BytesIO(wav_bytes(2.0)), "clip.wav"), "transcript": (io.BytesIO(b"ab"), "t.txt")}
        r = client.post("/audio/align/fi/submit_file", data=data, content_type="multipart/form-data").get_json()
        assert "at most 1 s" in r["error"]
    finally:
        config.MAX_ALIGN_S = old


def test_health_reports_workers_and_queue(client, store):
    h = client.get("/audio/asr/fi/health").get_json()
    assert h["status"] == "UP" and h["checks"] == {"redis": "UP", "workers": "DOWN"}
    assert h["queue_length"] == 0 and h["workers"] == 0
    store.heartbeat("fi", "w1")
    h = client.get("/audio/align/fi/health").get_json()
    assert h["checks"]["workers"] == "UP" and h["workers"] == 1
    assert client.get("/audio/asr/health").get_json() == {"status": "UP"}


def test_requeue_from_vanished_worker(store):
    store.enqueue("fi", "job1", "asr")
    store.claim("fi", "dead-worker")
    assert store.queue_length("fi") == 0
    w = worker.Worker(store, FakeRecognizer(), lang="fi", name="live-worker")
    w.start()
    assert store.queue_length("fi") == 1
    # A worker that is still alive keeps its claims.
    store.claim("fi", "busy-worker")
    store.heartbeat("fi", "busy-worker")
    w.sweep()
    assert store.queue_length("fi") == 0
    # Its own leftovers are always taken back.
    store.enqueue("fi", "job2", "asr")
    store.claim("fi", "live-worker")
    w.start()
    assert store.queue_length("fi") == 1


def test_split_on_silence_merges_short_segments():
    seg = audio.segment_from_wav(wav_bytes(1.0))
    assert len(audio.split_on_silence(seg)) == 1
