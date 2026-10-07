"""Optional stages of a recognition job: diarisation and punctuation.

Both are requested at submit time. Diarisation runs in parallel with
recognition on the whole audio, in a job of its own. Punctuation runs last,
over the words of all segments and the speaker turns, so it is enqueued by
whichever worker settles the last of those; ``settle`` is called after every
job and enqueues the punctuation job exactly once, when everything it reads
is final.
"""

import json
import logging

from . import jobs

log = logging.getLogger(__name__)


class StageOptions:
    """What a recognition request asked for."""

    def __init__(self, diarize=True, punctuate=True, num_speakers=None):
        self.diarize = diarize
        self.punctuate = punctuate
        self.num_speakers = num_speakers

    @classmethod
    def from_request(cls, values):
        """Parse ``diarize``, ``punctuate`` (anything but "false" means yes)
        and ``num_speakers`` (a positive integer) from a request's values."""

        def flag(name):
            return values.get(name, "").strip().lower() != "false"

        num_speakers = values.get("num_speakers", "").strip()
        if num_speakers:
            if not num_speakers.isdigit() or int(num_speakers) < 1:
                raise ValueError("num_speakers must be a positive integer")
            num_speakers = int(num_speakers)
        else:
            num_speakers = None
        return cls(flag("diarize"), flag("punctuate"), num_speakers)


def add(store, parent, lang, wav, options):
    """Create the stage jobs a recognition job asked for and record them in
    its hash. Must run before the recognition work is enqueued, so that the
    worker finishing it sees the punctuation job. A stage without a live
    worker is skipped at once rather than left waiting for one."""
    if options.diarize:
        jobid = store.create_job(jobs.DIARIZE, parent=parent, lang=lang)
        store.set_field(parent, "diarize_job", jobid)
        if store.worker_count(jobs.DIARIZE_QUEUE):
            if options.num_speakers:
                store.set_field(jobid, "num_speakers", options.num_speakers)
            store.store_audio(jobid, wav)
            store.enqueue(jobs.DIARIZE_QUEUE, jobid, "diarize")
        else:
            store.skip(jobid, "no diarisation worker available")
    if options.punctuate:
        jobid = store.create_job(jobs.PUNCTUATE, parent=parent, lang=lang)
        store.set_field(parent, "punctuate_job", jobid)
        if not store.worker_count(jobs.PUNCTUATE_QUEUE):
            store.skip(jobid, "no punctuation worker available")


def recognition_children(store, parent_id, parent):
    """Hashes of the recognition jobs a punctuation job reads, or None while
    a split job is still being split."""
    if parent.get("type") == jobs.ASR_SEGMENTS:
        if "segments" not in parent:
            return None
        return [store.get_job(s["jobid"]) for s in json.loads(parent["segments"])]
    return [parent]


def settle(store, jobid):
    """Called after a job of any kind finishes: enqueue the punctuation job of
    the recognition job it belongs to if everything that job reads is final."""
    job = store.get_job(jobid)
    if job is None:
        return
    parent_id = job.get("parent", jobid)
    parent = store.get_job(parent_id)
    if parent is None:
        return
    punctuate_id = parent.get("punctuate_job")
    if not punctuate_id:
        return
    punctuate_job = store.get_job(punctuate_id)
    if jobs.settled(punctuate_job) or punctuate_job.get("queued"):
        return
    if parent.get("status") == jobs.FAILED:
        store.skip(punctuate_id, "recognition failed")
        return
    if parent.get("status") == jobs.PENDING and parent.get("type") == jobs.ASR:
        return
    children = recognition_children(store, parent_id, parent)
    if children is None or not all(jobs.settled(child) for child in children):
        return
    diarize_id = parent.get("diarize_job")
    if diarize_id and not jobs.settled(store.get_job(diarize_id)):
        return
    if store.set_field_once(punctuate_id, "queued", "1"):
        store.enqueue(jobs.PUNCTUATE_QUEUE, punctuate_id, "punctuate")
        log.info("enqueued punctuation %s for %s", punctuate_id, parent_id)
