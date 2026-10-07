"""The query-time view of a recognition job: its segments, the speaker of
each word and the punctuated word forms, merged from the job hashes.

Word times inside a segment's ``words`` are relative to the segment, as they
always were. Speaker turns and the ``turns`` of the tekstiks view use
absolute times, as in the Estonian format that view follows.
"""

import json

from . import jobs

# A silence this long starts a new turn even when the speaker stays the same.
NEW_TURN_SIL_S = 2.0
DEFAULT_SPEAKER = "S0"


def load(store, jobid):
    """A Recognition for a job id, or None if it is unknown or not a
    recognition job."""
    job = store.get_job(jobid) if jobid else None
    if job is None or job.get("type") not in (jobs.ASR, jobs.ASR_SEGMENTS):
        return None
    return Recognition(store, jobid, job)


def with_timing(response, job):
    response["status"] = job.get("status")
    response["processing_started"] = float(job.get("processing_started"))
    if "processing_finished" in job:
        response["processing_finished"] = float(job["processing_finished"])
    return response


class Recognition:
    def __init__(self, store, jobid, job):
        self.jobid = jobid
        self.job = job
        self.split = job.get("type") == jobs.ASR_SEGMENTS
        self.segments = None
        self.missing_child = False
        if self.split:
            if "segments" in job:
                self.segments = []
                offset = 0.0
                for entry in json.loads(job["segments"]):
                    child = store.get_job(entry["jobid"])
                    if child is None:
                        self.missing_child = True
                    duration = float(entry["duration"])
                    self.segments.append(
                        {
                            "jobid": entry["jobid"],
                            "job": child,
                            "duration": duration,
                            "start": round(offset, 3),
                            "stop": round(offset + duration, 3),
                        }
                    )
                    offset += duration
        else:
            self.segments = [{"jobid": jobid, "job": job, "duration": None, "start": 0.0, "stop": None}]
        self.diarize = store.get_job(job["diarize_job"]) if job.get("diarize_job") else None
        self.punctuate = store.get_job(job["punctuate_job"]) if job.get("punctuate_job") else None
        self._turns = None
        if self.diarize and self.diarize.get("status") == jobs.DONE:
            self._turns = json.loads(self.diarize["response"])["turns"]
        self._punctuation = None
        if self.punctuate and self.punctuate.get("status") == jobs.DONE:
            self._punctuation = json.loads(self.punctuate["response"])["words"]

    # Status

    def status(self):
        """(status, error): failed with the first error, pending while any
        part is, done otherwise."""
        if self.job.get("status") == jobs.FAILED:
            return jobs.FAILED, self.job.get("error", "processing failed")
        if self.segments is None:
            return jobs.PENDING, None
        error = None
        for segment in self.segments:
            child = segment["job"]
            if child is None:
                continue
            if child.get("status") == jobs.PENDING:
                return jobs.PENDING, None
            if child.get("status") == jobs.FAILED and error is None:
                error = child.get("error", "processing failed")
        if error:
            return jobs.FAILED, error
        for stage in (self.diarize, self.punctuate):
            if stage is not None and not jobs.settled(stage):
                return jobs.PENDING, None
        return jobs.DONE, None

    def processing_finished(self):
        times = [float(s["job"]["processing_finished"]) for s in self.segments if s["job"] and "processing_finished" in s["job"]]
        for stage in (self.diarize, self.punctuate):
            if stage and "processing_finished" in stage:
                times.append(float(stage["processing_finished"]))
        return max(times, default=0.0)

    def checks(self):
        """``checks`` (UP, DOWN or SKIPPED per requested stage) and
        ``warnings`` (the reason for anything not UP)."""
        checks, warnings = {}, {}
        for name, stage in (("diarization", self.diarize), ("punctuation", self.punctuate)):
            if stage is None:
                continue
            status = stage.get("status")
            if status == jobs.DONE:
                checks[name] = "UP"
            elif status == jobs.PENDING:
                checks[name] = "PENDING"
            else:
                checks[name] = "SKIPPED" if status == jobs.SKIPPED else "DOWN"
                warnings[name] = stage.get("error", "failed")
        return checks, warnings

    # Speakers

    def turns(self):
        """Diarisation turns, ``[{"speaker", "start", "end"}]``, or None."""
        return self._turns

    def speakers(self):
        if self._turns is None:
            return [DEFAULT_SPEAKER]
        seen = []
        for turn in self._turns:
            if turn["speaker"] not in seen:
                seen.append(turn["speaker"])
        return seen or [DEFAULT_SPEAKER]

    def speaker_at(self, start, end):
        """Label of the turn overlapping a word the most, or the nearest
        turn when none overlaps. Turns can nest (a short interjection inside
        a long turn); on equal overlap the shorter turn wins."""
        if not self._turns:
            return DEFAULT_SPEAKER
        best, best_key, nearest, nearest_distance = None, (0.0, 0.0), None, None
        for turn in self._turns:
            overlap = min(end, turn["end"]) - max(start, turn["start"])
            key = (overlap, turn["start"] - turn["end"])
            if overlap > 0 and key > best_key:
                best, best_key = turn, key
            distance = max(turn["start"] - end, start - turn["end"], 0.0)
            if nearest_distance is None or distance < nearest_distance:
                nearest, nearest_distance = turn, distance
        return (best or nearest)["speaker"]

    # Words

    def segment_results(self):
        """One result per segment as ``query_job`` reports it: the child's
        response with timing, position in the file, and words carrying
        ``speaker`` and the punctuated forms when those stages are done."""
        results = []
        for i, segment in enumerate(self.segments):
            child = segment["job"] or {"status": None, "processing_started": 0}
            result = with_timing(json.loads(child.get("response", "{}")), child)
            if self.split:
                result["start"] = segment["start"]
                result["stop"] = segment["stop"]
                result["duration"] = segment["stop"]
            if child.get("status") == jobs.DONE:
                alternatives = result.get("responses") or [{}]
                result["type"] = "speech" if alternatives[0].get("words") else "non-speech"
            punctuation = self._punctuation[i] if self._punctuation and i < len(self._punctuation) else None
            for alternative in result.get("responses", []):
                words = alternative.get("words", [])
                for k, word in enumerate(words):
                    if self._turns is not None:
                        word["speaker"] = self.speaker_at(
                            segment["start"] + word["start"], segment["start"] + word["end"]
                        )
                    if punctuation and k < len(punctuation):
                        word.update(punctuation[k])
                if punctuation and len(punctuation) == len(words):
                    alternative["transcript_punctuated"] = " ".join(
                        w["word_with_punctuation"] for w in words
                    )
            results.append(result)
        return results

    def absolute_words(self, segment_results=None):
        """Every word of the job in order, with absolute times, as
        ``(segment index, word index, word)``; the words are copies."""
        out = []
        for i, result in enumerate(segment_results or self.segment_results()):
            offset = self.segments[i]["start"]
            alternatives = result.get("responses") or [{}]
            for k, word in enumerate(alternatives[0].get("words", [])):
                moved = dict(word)
                moved["start"] = round(offset + word["start"], 3)
                moved["end"] = round(offset + word["end"], 3)
                out.append((i, k, moved))
        return out


def group_turns(words):
    """Split absolute-time words into turns: a new turn starts at a speaker
    change or after NEW_TURN_SIL_S of silence. Returns lists of words."""
    turns = []
    for word in words:
        if (
            not turns
            or turns[-1][-1].get("speaker", DEFAULT_SPEAKER) != word.get("speaker", DEFAULT_SPEAKER)
            or turns[-1][-1]["end"] + NEW_TURN_SIL_S < word["start"]
        ):
            turns.append([])
        turns[-1].append(word)
    return turns


def turn_entries(words):
    """Turns in the tekstiks shape, each ending where the next begins."""
    entries = []
    for group in group_turns(words):
        entries.append(
            {
                "speaker": group[0].get("speaker", DEFAULT_SPEAKER),
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "transcript": " ".join(w.get("word_with_punctuation", w["word"]) for w in group),
                "words": group,
            }
        )
    for a, b in zip(entries, entries[1:]):
        a["end"] = b["start"]
    return entries
