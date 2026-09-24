"""Queue worker: loads one model and serves recognition and alignment jobs
for one language from the Redis list ``asr:<lang>``."""

import json
import logging
import os
import signal
import socket
import time

import redis

from . import audio, config, ctc, formats
from .jobs import Store

log = logging.getLogger("asr.worker")

REDIS_ERRORS = (redis.ConnectionError, redis.TimeoutError)


class Worker:
    def __init__(self, store, recognizer, lang=config.LANG, name=None):
        self.store = store
        self.recognizer = recognizer
        self.lang = lang
        self.name = name or os.environ.get("HOSTNAME") or socket.gethostname()
        self.stopping = False

    # Lifecycle

    def start(self):
        self.store.publish_model(self.lang, self.recognizer.description)
        self.store.heartbeat(self.lang, self.name)
        n = self.store.requeue_processing(self.lang, self.name)
        if n:
            log.info("requeued %d job(s) left over from a previous run", n)
        self.sweep()

    def sweep(self):
        """Requeue jobs claimed by workers whose heartbeat has expired."""
        for other in self.store.processing_workers(self.lang):
            if other != self.name and not self.store.worker_alive(self.lang, other):
                n = self.store.requeue_processing(self.lang, other)
                log.info("requeued %d job(s) from vanished worker %s", n, other)

    def run(self):
        while not self.stopping:
            try:
                self.start()
                self.loop()
            except REDIS_ERRORS as e:
                log.error("redis unavailable (%s), retrying", e)
                time.sleep(5)

    def loop(self):
        while not self.stopping:
            self.store.heartbeat(self.lang, self.name)
            claimed = self.store.claim(self.lang, self.name)
            if claimed is None:
                self.sweep()
                continue
            raw, message = claimed
            try:
                self.handle(message)
            except REDIS_ERRORS:
                raise
            except Exception:
                log.exception("job %s failed", message.get("jobid"))
                self.store.fail(message["jobid"], "internal error while processing")
            finally:
                self.store.release(self.lang, self.name, raw)

    def stop(self, *_):
        log.info("stopping after the current job")
        self.stopping = True

    # Jobs

    def handle(self, message):
        jobid = message["jobid"]
        task = message.get("task", "asr")
        if not self.store.exists(jobid):
            log.warning("job %s expired before processing", jobid)
            return
        started = time.time()
        wav = self.store.load_audio(jobid)
        if wav is None:
            self.store.fail(jobid, "audio expired before processing")
            return
        if task == "asr":
            self.recognize(jobid, wav)
        elif task == "align":
            text = self.store.load_text(jobid)
            if text is None:
                self.store.fail(jobid, "transcript expired before processing")
                return
            self.align(jobid, wav, text)
        else:
            self.store.fail(jobid, f"unknown task {task!r}")
            return
        self.store.drop_inputs(jobid)
        log.info(
            "%s %s: %.1f s of audio in %.1f s",
            task, jobid, audio.duration_s(wav), time.time() - started,
        )

    def _progress(self):
        self.store.heartbeat(self.lang, self.name)

    def recognize(self, jobid, wav):
        r = self.recognizer
        log_probs = r.emissions(audio.samples(wav), progress=self._progress)
        result = ctc.greedy_decode(
            log_probs, r.frame_s, r.tokens, r.blank, r.delimiter, r.ignore
        )
        self.store.finish(jobid, response=json.dumps({"responses": [result]}))

    def align(self, jobid, wav, text):
        r = self.recognizer
        try:
            words = r.normalizer.words(text)
            log_probs = r.emissions(audio.samples(wav), progress=self._progress)
            tiers = ctc.align_words(log_probs, r.frame_s, words, r.delimiter, r.blank)
        except ctc.AlignmentError as e:
            self.store.fail(jobid, str(e))
            return
        results = formats.all_formats(tiers, audio.duration_s(wav))
        self.store.finish(jobid, results=json.dumps(results, ensure_ascii=False))


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    from .model import load_recognizer

    recognizer = load_recognizer()
    worker = Worker(Store(), recognizer)
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    log.info("worker %s serving %s", worker.name, config.LANG)
    worker.run()


if __name__ == "__main__":
    main()
