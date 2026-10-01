"""CTC decoding, forced alignment and transcript normalisation.

Emissions are arrays of shape [T, V] of log-probabilities, one row per model
frame (20 ms for wav2vec2). torchaudio does the algorithmic work:
``forced_align`` finds the best path through a transcript and
``merge_tokens`` turns a frame path into token spans. What remains here is
grouping tokens into words, the timing corrections, and mapping transcript
text onto the model vocabulary.
"""

import re
import unicodedata

import numpy as np
import torch
import torchaudio.functional as F

# Timing corrections, chosen on the pohjantuuli clip against the Kaldi
# aligner's output (docs/wav2vec-migration.md). CTC emits a character about
# two frames after its onset, so every boundary is moved this much earlier.
LATENCY_S = 0.04
# When a unit's end is this close to the next unit's start, the two are made
# contiguous: the last character's frames end early and the rest of the word
# is the gap before the next word.
MAX_GAP_S = 0.3
# Before a longer pause (and at the end of the audio) a word is extended by
# this much instead, but never into the next word.
PAUSE_EXTENSION_S = 0.2

# Trellis cells (frames times transcript tokens) allowed for one forced
# alignment, bounding its memory.
MAX_TRELLIS_CELLS = 1_000_000_000


class AlignmentError(Exception):
    pass


def interval(start_frame, end_frame, frame_s):
    """Seconds for a frame span, corrected for emission latency; at least one
    frame long."""
    start = max(0.0, start_frame * frame_s - LATENCY_S)
    end = max(start + frame_s, end_frame * frame_s - LATENCY_S)
    return round(start, 3), round(end, 3)


def close_gaps(intervals, max_end):
    """Extend each interval's end to the next start when the gap is small,
    and by PAUSE_EXTENSION_S otherwise; ``max_end`` is the audio duration."""
    for a, b in zip(intervals, intervals[1:]):
        gap = b["start"] - a["end"]
        if 0 <= gap <= MAX_GAP_S:
            a["end"] = b["start"]
        elif gap > MAX_GAP_S:
            a["end"] = round(min(a["end"] + PAUSE_EXTENSION_S, b["start"]), 3)
    if intervals:
        last = intervals[-1]
        last["end"] = round(max(last["end"], min(last["end"] + PAUSE_EXTENSION_S, max_end)), 3)
    return intervals


def token_spans(path, scores, blank):
    """Spans of non-blank tokens on a frame path, repeats merged: a list of
    torchaudio TokenSpan(token, start, end, score) with ``end`` exclusive."""
    return F.merge_tokens(
        torch.as_tensor(np.asarray(path), dtype=torch.int64),
        torch.as_tensor(np.asarray(scores), dtype=torch.float32),
        blank=blank,
    )


def greedy_decode(log_probs, frame_s, tokens, blank, delimiter, ignore=()):
    """Best-path decoding with word times and confidences.

    Returns ``{"transcript", "confidence", "words": [{"word", "start",
    "end", "confidence"}]}``. A word's confidence is the mean over its frames
    of the highest softmax probability; the transcript's confidence is the
    mean of the word confidences.
    """
    best_path = log_probs.argmax(axis=1)
    frame_conf = np.exp(log_probs.max(axis=1))
    words = []
    chars, start, end = [], None, None

    def flush():
        if chars:
            s, e = interval(start, end, frame_s)
            words.append(
                {
                    "word": "".join(chars),
                    "start": s,
                    "end": e,
                    "confidence": round(float(frame_conf[start:end].mean()), 5),
                }
            )
        chars.clear()

    for span in token_spans(best_path, frame_conf, blank):
        if span.token == delimiter:
            flush()
        elif span.token in ignore:
            continue
        else:
            if not chars:
                start = span.start
            chars.append(tokens[span.token])
            end = span.end
    flush()
    close_gaps(words, log_probs.shape[0] * frame_s)
    confidence = float(np.mean([w["confidence"] for w in words])) if words else 0.0
    return {
        "transcript": " ".join(w["word"] for w in words),
        "confidence": round(confidence, 5),
        "words": words,
    }


def align_tokens(log_probs, targets, blank):
    """Frame spans of each target token on the best CTC path, as a list of
    (start, end) with ``end`` exclusive. Raises AlignmentError when no path
    exists or the trellis would be too large."""
    if not targets:
        return []
    if log_probs.shape[0] * len(targets) > MAX_TRELLIS_CELLS:
        raise AlignmentError("transcript and audio are too long to align in one pass")
    emissions = torch.as_tensor(log_probs, dtype=torch.float32)[None]
    target_tensor = torch.tensor(targets, dtype=torch.int32)[None]
    try:
        alignment, scores = F.forced_align(emissions, target_tensor, blank=blank)
    except RuntimeError as e:
        raise AlignmentError(f"audio is too short for the transcript ({e})") from e
    spans = F.merge_tokens(alignment[0], scores[0].exp(), blank=blank)
    if len(spans) != len(targets):
        raise AlignmentError("alignment did not cover the transcript")
    return [(span.start, span.end) for span in spans]


def align_words(log_probs, frame_s, words, delimiter, blank):
    """Align ``words``, a list of (label, token ids, source characters) as
    produced by ``Normalizer.words``, and return the ``words`` and
    ``letters`` tiers as lists of {"start", "end", "label"}."""
    targets = []
    word_slices = []
    for i, (_, ids, _) in enumerate(words):
        if i > 0:
            targets.append(delimiter)
        word_slices.append((len(targets), len(targets) + len(ids)))
        targets.extend(ids)
    spans = align_tokens(log_probs, targets, blank)

    word_tier = []
    letter_tier = []
    for (label, ids, chars), (a, b) in zip(words, word_slices):
        letters = [
            dict(zip(("start", "end"), interval(s, e, frame_s)), label=ch)
            for (s, e), ch in zip(spans[a:b], chars)
        ]
        # Letters within a word are contiguous: each ends where the next begins.
        for x, y in zip(letters, letters[1:]):
            x["end"] = y["start"]
        word_tier.append(
            {"start": letters[0]["start"], "end": letters[-1]["end"], "label": label}
        )
        letter_tier.extend(letters)
    close_gaps(word_tier, log_probs.shape[0] * frame_s)
    # The last letter of each word follows the word's possibly extended end.
    pos = 0
    for word, (_, ids, _) in zip(word_tier, words):
        pos += len(ids)
        letter_tier[pos - 1]["end"] = word["end"]
    return {"words": word_tier, "letters": letter_tier}


# Letters outside the model vocabulary that have an obvious replacement and
# do not decompose to it in Unicode.
FALLBACK_CHARS = {
    "ß": "ss", "æ": "ae", "ø": "o", "œ": "oe", "đ": "d", "ð": "d", "þ": "th",
    "ł": "l", "ŋ": "n", "ŧ": "t", "ı": "i",
}


def spell_numbers(text, lang):
    """Replace runs of digits with words in the given language, using
    num2words. Returns None if the language is not supported."""
    from num2words import num2words

    codes = [lang, lang.split("-")[0]] if lang else []
    for code in codes:
        try:
            return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang=code), text)
        except NotImplementedError:
            continue
    return None


class Normalizer:
    """Maps transcript words onto model tokens for alignment."""

    def __init__(self, tokens, blank, delimiter, ignore=(), lang=None):
        self.lang = lang
        self.char_to_id = {}
        for i, tok in enumerate(tokens):
            if i in (blank, delimiter) or i in ignore or tok is None:
                continue
            if len(tok) == 1:
                self.char_to_id[tok] = i

    def _map_char(self, c):
        """Token ids for one character, or None if it cannot be aligned."""
        if c in self.char_to_id:
            return [self.char_to_id[c]]
        if c.isdigit():
            return None
        for alt in (c.lower(), c.upper()):
            if alt in self.char_to_id:
                return [self.char_to_id[alt]]
        if c in FALLBACK_CHARS:
            ids = [self._map_char(x) for x in FALLBACK_CHARS[c]]
            if all(ids):
                return [i for sub in ids for i in sub]
        base = unicodedata.normalize("NFKD", c)
        base = "".join(x for x in base if not unicodedata.combining(x))
        if base != c and base and base in self.char_to_id:
            return [self.char_to_id[base]]
        if unicodedata.category(c)[0] in ("P", "S", "Z", "C"):
            return []
        return None

    def words(self, transcript):
        """(label, token ids, source character per token id) for each
        whitespace-delimited word. Digits are spelled out when num2words
        knows the language and the spelled words become the labels. Raises
        AlignmentError for digits in other languages or letters the model has
        no token for; words that consist only of punctuation are dropped."""
        if re.search(r"\d", transcript):
            spelled = spell_numbers(transcript, self.lang)
            if spelled is None:
                raise AlignmentError(
                    "transcript contains digits; write numbers out as words"
                )
            transcript = spelled
        result = []
        bad = set()
        for label in transcript.split():
            ids = []
            chars = []
            for c in label:
                mapped = self._map_char(c)
                if mapped is None:
                    bad.add(c)
                    continue
                ids.extend(mapped)
                chars.extend([c] * len(mapped))
            if ids:
                result.append((label, ids, chars))
        if bad:
            raise AlignmentError(
                "transcript contains characters the model cannot align: "
                + " ".join(sorted(bad))
            )
        if not result:
            raise AlignmentError("transcript is empty")
        return result
