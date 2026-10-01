"""Redis data model: job hashes, per-language queues and audio blobs.

Job ids are UUID strings used as Redis hash keys. Every hash has ``type``,
``status`` and ``processing_started``; finished jobs also have
``processing_finished`` and either ``response`` (recognition, a JSON string)
or ``results`` (alignment, a JSON string). A split recognition job has
``type=asr_segments`` and a ``segments`` field listing its children.

Other keys:

``asr:<lang>``
    List of queue messages, JSON ``{"jobid": ..., "task": "asr" | "align"}``.
    The API pushes on the left, workers pop from the right.
``asr:<lang>:processing:<worker>``
    Messages a worker has popped but not finished. Put back on the queue if
    the worker disappears.
``worker:<lang>:<worker>``
    Heartbeat with an expiry, refreshed while the worker runs.
``model:<lang>``
    JSON describing the model the workers of a language serve.
``audio:<jobid>``, ``text:<jobid>``
    Input blobs for a job, with a short expiry.
"""

import json
import time
import uuid

import redis

from . import config

ASR = "asr"
ASR_SEGMENTS = "asr_segments"
ALIGN = "align"

PENDING = "pending"
DONE = "done"
FAILED = "failed"

# Alignment hashes carry a "task" field that existing clients know.
ALIGN_TASK_NAMES = {"fi": "finnish-forced-align"}


def align_task_name(lang):
    return ALIGN_TASK_NAMES.get(lang, f"{lang}-forced-align")


def queue_key(lang):
    return f"asr:{lang}"


def processing_key(lang, worker):
    return f"asr:{lang}:processing:{worker}"


def heartbeat_key(lang, worker):
    return f"worker:{lang}:{worker}"


def model_key(lang):
    return f"model:{lang}"


def now():
    return round(time.time(), 3)


# How long a worker blocks waiting for a queue message before it wakes up to
# refresh its heartbeat and sweep for vanished workers.
CLAIM_TIMEOUT_S = 30
# redis-py aborts any command that takes longer than the socket timeout,
# including blocking ones, so the blocking client gets a longer one. Audio
# blobs of an hour are around 100 MB, hence the generous ordinary timeout.
SOCKET_TIMEOUT_S = 60


class Store:
    """Thin wrapper over Redis clients: text, bytes, and one reserved for the
    blocking queue read."""

    def __init__(self, text_client=None, binary_client=None, blocking_client=None):
        def connect(**kwargs):
            return redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, **kwargs)

        self.r = text_client or connect(decode_responses=True, socket_timeout=SOCKET_TIMEOUT_S)
        self.rb = binary_client or connect(decode_responses=False, socket_timeout=SOCKET_TIMEOUT_S)
        self.blocking = blocking_client or connect(
            decode_responses=True, socket_timeout=CLAIM_TIMEOUT_S + 10
        )

    def ping(self):
        return self.r.ping()

    # Job hashes

    def create_job(self, job_type, **fields):
        jobid = str(uuid.uuid4())
        mapping = {"type": job_type, "status": PENDING, "processing_started": now()}
        mapping.update(fields)
        self.r.hset(jobid, mapping=mapping)
        self.r.expire(jobid, config.JOB_EXPIRY_S)
        return jobid

    def exists(self, jobid):
        return bool(self.r.exists(jobid))

    def get_job(self, jobid):
        job = self.r.hgetall(jobid)
        return job or None

    def set_field(self, jobid, field, value):
        self.r.hset(jobid, field, value)

    def finish(self, jobid, **fields):
        """Mark a job done. Nothing is written if the hash has expired, so
        that no hash without an expiry is left behind."""
        mapping = {"status": DONE, "processing_finished": now()}
        mapping.update(fields)
        if self.exists(jobid):
            self.r.hset(jobid, mapping=mapping)

    def fail(self, jobid, error):
        """Mark a job failed. Recognition jobs get the error in ``response``
        so that clients reading that field see it too."""
        job_type = self.r.hget(jobid, "type")
        if job_type is None:
            return
        mapping = {
            "status": FAILED,
            "processing_finished": now(),
            "error": error,
        }
        if job_type in (ASR, ASR_SEGMENTS):
            mapping["response"] = json.dumps({"error": error})
        self.r.hset(jobid, mapping=mapping)

    # Input blobs

    def store_audio(self, jobid, wav_bytes):
        self.rb.set(f"audio:{jobid}", wav_bytes, ex=config.AUDIO_EXPIRY_S)

    def load_audio(self, jobid):
        return self.rb.get(f"audio:{jobid}")

    def store_text(self, jobid, text):
        self.r.set(f"text:{jobid}", text, ex=config.AUDIO_EXPIRY_S)

    def load_text(self, jobid):
        return self.r.get(f"text:{jobid}")

    def drop_inputs(self, jobid):
        self.r.delete(f"audio:{jobid}", f"text:{jobid}")

    # Queues

    def enqueue(self, lang, jobid, task):
        self.r.lpush(queue_key(lang), json.dumps({"jobid": jobid, "task": task}))

    def queue_length(self, lang):
        return self.r.llen(queue_key(lang))

    def claim(self, lang, worker):
        """Move the oldest queue message to the worker's processing list and
        return it decoded, or None after CLAIM_TIMEOUT_S seconds."""
        raw = self.blocking.brpoplpush(
            queue_key(lang), processing_key(lang, worker), CLAIM_TIMEOUT_S
        )
        if raw is None:
            return None
        return raw, json.loads(raw)

    def release(self, lang, worker, raw):
        self.r.lrem(processing_key(lang, worker), 1, raw)

    def requeue_processing(self, lang, worker):
        """Put everything in a worker's processing list back on the queue."""
        key = processing_key(lang, worker)
        count = 0
        while True:
            raw = self.r.rpoplpush(key, queue_key(lang))
            if raw is None:
                return count
            count += 1

    def processing_workers(self, lang):
        prefix = processing_key(lang, "")
        return [key[len(prefix) :] for key in self.r.scan_iter(prefix + "*")]

    # Workers

    def heartbeat(self, lang, worker):
        self.r.set(heartbeat_key(lang, worker), now(), ex=config.WORKER_HEARTBEAT_S)

    def worker_alive(self, lang, worker):
        return bool(self.r.exists(heartbeat_key(lang, worker)))

    def worker_count(self, lang):
        return sum(1 for _ in self.r.scan_iter(heartbeat_key(lang, "*")))

    def publish_model(self, lang, description):
        self.r.set(model_key(lang), json.dumps(description))

    def model_description(self, lang):
        raw = self.r.get(model_key(lang))
        return json.loads(raw) if raw else {}
