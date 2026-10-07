"""Queue workers. One process serves one queue: a language queue with a
loaded CTC model (recognition and alignment), or one of the stage queues
``diarize`` and ``punctuate``. The claim loop, heartbeat and recovery of
jobs from vanished workers are shared; a handler does the work."""

import json
import logging
import os
import signal
import socket
import time

import redis

from . import audio, config, ctc, formats, jobs, results, stages
from .jobs import Store

log = logging.getLogger("asr.worker")

REDIS_ERRORS = (redis.ConnectionError, redis.TimeoutError)


class RecognitionHandler:
    """Recognition and forced alignment for one language."""

    def __init__(self, recognizer, lang=config.LANG):
        self.recognizer = recognizer
        self.queue = lang
        self.description = recognizer.description

    def handle(self, store, jobid, job, message, progress):
        task = message.get("task", "asr")
        wav = store.load_audio(jobid)
        if wav is None:
            store.fail(jobid, "audio expired before processing")
            return
        if task == "asr":
            self.recognize(store, jobid, wav, progress)
        elif task == "align":
            text = store.load_text(jobid)
            if text is None:
                store.fail(jobid, "transcript expired before processing")
                return
            self.align(store, jobid, wav, text, progress)
        else:
            store.fail(jobid, f"unknown task {task!r}")
            return
        store.drop_inputs(jobid)
        return audio.duration_s(wav)

    def recognize(self, store, jobid, wav, progress):
        r = self.recognizer
        log_probs = r.emissions(audio.samples(wav), progress=progress)
        result = ctc.greedy_decode(
            log_probs, r.frame_s, r.tokens, r.blank, r.delimiter, r.ignore
        )
        store.finish(jobid, response=json.dumps({"responses": [result]}))

    def align(self, store, jobid, wav, text, progress):
        r = self.recognizer
        try:
            words = r.normalizer.words(text)
            log_probs = r.emissions(audio.samples(wav), progress=progress)
            tiers = ctc.align_words(log_probs, r.frame_s, words, r.delimiter, r.blank)
        except ctc.AlignmentError as e:
            store.fail(jobid, str(e))
            return
        result = formats.all_formats(tiers, audio.duration_s(wav))
        store.finish(jobid, results=json.dumps(result, ensure_ascii=False))


class DiarizationHandler:
    queue = jobs.DIARIZE_QUEUE

    def __init__(self, diarizer):
        self.diarizer = diarizer
        self.description = diarizer.description

    def handle(self, store, jobid, job, message, progress):
        wav = store.load_audio(jobid)
        if wav is None:
            store.fail(jobid, "audio expired before processing")
            return
        num_speakers = int(job["num_speakers"]) if job.get("num_speakers") else None
        turns = self.diarizer.turns(audio.samples(wav), num_speakers, progress)
        speakers = []
        for turn in turns:
            if turn["speaker"] not in speakers:
                speakers.append(turn["speaker"])
        store.finish(jobid, response=json.dumps({"turns": turns, "speakers": speakers}))
        store.drop_inputs(jobid)
        return audio.duration_s(wav)


class PunctuationHandler:
    queue = jobs.PUNCTUATE_QUEUE

    def __init__(self, punctuator):
        self.punctuator = punctuator
        self.description = punctuator.description

    def handle(self, store, jobid, job, message, progress):
        view = results.load(store, job.get("parent"))
        if view is None or view.segments is None:
            store.fail(jobid, "recognition job expired before punctuation")
            return
        placed = view.absolute_words()
        groups = results.group_turns([word for _, _, word in placed])
        forms = self.punctuator.punctuate(groups, job.get("lang") or config.LANG)
        per_segment = [[] for _ in view.segments]
        for (i, _, _), form in zip(placed, [f for group in forms for f in group]):
            per_segment[i].append(form)
        store.finish(jobid, response=json.dumps({"words": per_segment}, ensure_ascii=False))
        return None


class Worker:
    def __init__(self, store, handler, name=None):
        self.store = store
        self.handler = handler
        self.queue = handler.queue
        self.name = name or os.environ.get("HOSTNAME") or socket.gethostname()
        self.stopping = False

    # Lifecycle

    def start(self):
        self.store.publish_model(self.queue, self.handler.description)
        self.store.heartbeat(self.queue, self.name)
        n = self.store.requeue_processing(self.queue, self.name)
        if n:
            log.info("requeued %d job(s) left over from a previous run", n)
        self.sweep()

    def sweep(self):
        """Requeue jobs claimed by workers whose heartbeat has expired."""
        for other in self.store.processing_workers(self.queue):
            if other != self.name and not self.store.worker_alive(self.queue, other):
                n = self.store.requeue_processing(self.queue, other)
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
            self.store.heartbeat(self.queue, self.name)
            if not self.process_next():
                self.sweep()

    def process_next(self):
        """Claim and process one message; False if none arrived in time."""
        claimed = self.store.claim(self.queue, self.name)
        if claimed is None:
            return False
        raw, message = claimed
        jobid = message.get("jobid")
        try:
            self.handle(message)
        except REDIS_ERRORS:
            raise
        except Exception:
            log.exception("job %s failed", jobid)
            self.store.fail(jobid, "internal error while processing")
        finally:
            self.store.release(self.queue, self.name, raw)
        # Whatever happened to this job, the punctuation of the recognition
        # job it belongs to may now be able to run.
        stages.settle(self.store, jobid)
        return True

    def stop(self, *_):
        log.info("stopping after the current job")
        self.stopping = True

    def _progress(self):
        self.store.heartbeat(self.queue, self.name)

    def handle(self, message):
        jobid = message["jobid"]
        job = self.store.get_job(jobid)
        if job is None:
            log.warning("job %s expired before processing", jobid)
            return
        attempt = self.store.count_attempt(jobid)
        if attempt > config.MAX_ATTEMPTS:
            log.error("job %s has already been attempted %d times, failing it", jobid, attempt - 1)
            self.store.fail(jobid, "processing failed repeatedly")
            return
        started = time.time()
        duration = self.handler.handle(self.store, jobid, job, message, self._progress)
        elapsed = time.time() - started
        if duration is None:
            log.info("%s %s: %.1f s", message.get("task"), jobid, elapsed)
        else:
            log.info("%s %s: %.1f s of audio in %.1f s", message.get("task"), jobid, duration, elapsed)


def load_handler(role=config.ROLE):
    if role == "worker":
        from .model import load_recognizer

        return RecognitionHandler(load_recognizer())
    if role == "diarize":
        from .diarize import load_diarizer

        return DiarizationHandler(load_diarizer())
    if role == "punctuate":
        from .punctuate import Punctuator

        return PunctuationHandler(Punctuator())
    raise ValueError(f"ROLE {role!r} is not a worker role")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    worker = Worker(Store(), load_handler())
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    log.info("worker %s serving queue %s", worker.name, worker.queue)
    worker.run()


if __name__ == "__main__":
    main()
