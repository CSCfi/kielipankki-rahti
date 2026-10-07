"""Settings, all taken from the environment with defaults."""

import os


def _int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


def _float(name, default):
    value = os.environ.get(name)
    return float(value) if value else default


# "api" serves HTTP, "worker" consumes one language queue, "diarize" and
# "punctuate" consume the stage queues of those names.
ROLE = os.environ.get("ROLE", "api")

# Worker: language code used in public paths and as the queue name.
LANG = os.environ.get("ASR_LANG", "fi")
# API: languages it accepts requests for, comma separated.
LANGUAGES = [
    lang.strip()
    for lang in os.environ.get("ASR_LANGUAGES", "fi").split(",")
    if lang.strip()
]

# Worker: Hugging Face model directory and how to run it.
MODEL_DIR = os.environ.get("ASR_MODEL_DIR", "/data/model")
QUANTIZE = os.environ.get("ASR_QUANTIZE", "int8")  # "none" or "int8"
THREADS = _int("ASR_THREADS", 4)
CHUNK_S = _float("ASR_CHUNK_S", 30.0)
STRIDE_S = _float("ASR_STRIDE_S", 5.0)

# Diarisation worker, sherpa-onnx backend: clustering threshold for the
# speaker embeddings when the request does not fix the number of speakers
# (sherpa-onnx default 0.5; lower finds more speakers).
DIARIZE_THRESHOLD = _float("ASR_DIARIZE_THRESHOLD", 0.5)
# Diarisation worker, pyannote backend: windows per forward pass of the
# segmentation and embedding models. The pipeline's own default of 32 needs
# gigabytes of activations on CPU, where batching gains nothing.
DIARIZE_BATCH = _int("ASR_DIARIZE_BATCH", 4)

# Punctuation worker: languages the punctuation model is applied to, by
# language code or its first subtag; other languages get rule-based
# capitalisation. Finnish is in the model's training languages and Swedish
# is not, but zero-shot output on Finland Swedish was good enough to use.
PUNCTUATE_MODEL_LANGUAGES = [
    lang.strip()
    for lang in os.environ.get("ASR_PUNCTUATE_MODEL_LANGUAGES", "fi,sv").split(",")
    if lang.strip()
]
# Rule-based punctuation: a pause this long ends a sentence.
PUNCTUATE_PAUSE_S = _float("ASR_PUNCTUATE_PAUSE_S", 1.0)

# Prefixed because Kubernetes sets REDIS_PORT and friends from the redis
# Service (to strings like tcp://172.30.1.2:6379).
REDIS_HOST = os.environ.get("ASR_REDIS_HOST", "redis")
REDIS_PORT = _int("ASR_REDIS_PORT", 6379)

# Base URL the self test calls itself through, so that nginx is exercised too.
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://nginx:1337")

# Directory with the self-test clips, one subdirectory per language,
# relative to the working directory.
TEST_DATA_DIR = os.environ.get("ASR_TEST_DATA_DIR", "test")

SAMPLE_RATE = 16000
MAX_CONTENT_LENGTH = 500 * 2**20
JOB_EXPIRY_S = 60 * 60 * 24 * 10
AUDIO_EXPIRY_S = 60 * 60
# Forced alignment runs one trellis over the whole file.
MAX_ALIGN_S = _float("ASR_MAX_ALIGN_S", 30 * 60)
# How long the synchronous recognition endpoint waits for a result.
SYNC_TIMEOUT_S = 60.0
# A worker that has not refreshed its heartbeat for this long is considered
# gone and the jobs it had claimed are put back on the queue.
WORKER_HEARTBEAT_S = 300
# A job is attempted at most this many times; a job that keeps killing the
# worker that claims it (out of memory, say) then fails instead of looping.
MAX_ATTEMPTS = _int("ASR_MAX_ATTEMPTS", 2)
