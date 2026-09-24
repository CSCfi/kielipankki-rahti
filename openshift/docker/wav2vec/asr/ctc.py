"""Greedy CTC decoding and forced alignment over log-probabilities.

Emissions are arrays of shape [T, V] of log-probabilities, one row per model
frame (20 ms for wav2vec2). Token ids index the model vocabulary; ``blank``
is the CTC blank, ``delimiter`` the word boundary token.
"""

import unicodedata

import numpy as np

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

# Backpointer table cells (frames times trellis states) allowed for one
# forced alignment; one byte each.
MAX_TRELLIS_CELLS = 2_000_000_000


class AlignmentError(Exception):
    pass


def _collapse(best_path, blank):
    """Runs of identical non-blank tokens as (token, start, end) with ``end``
    exclusive, in frame indices."""
    units = []
    prev = blank
    for t, tok in enumerate(best_path):
        if tok == blank:
            prev = blank
            continue
        if tok == prev:
            units[-1][2] = t + 1
        else:
            units.append([tok, t, t + 1])
        prev = tok
    return units


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
            conf = float(frame_conf[start:end].mean())
            s, e = interval(start, end, frame_s)
            words.append(
                {
                    "word": "".join(chars),
                    "start": s,
                    "end": e,
                    "confidence": round(conf, 5),
                }
            )
        chars.clear()

    for tok, s, e in _collapse(best_path, blank):
        if tok == delimiter:
            flush()
            start = end = None
        elif tok in ignore:
            continue
        else:
            if not chars:
                start = s
            chars.append(tokens[tok])
            end = e
    flush()
    close_gaps(words, log_probs.shape[0] * frame_s)
    confidence = float(np.mean([w["confidence"] for w in words])) if words else 0.0
    return {
        "transcript": " ".join(w["word"] for w in words),
        "confidence": round(confidence, 5),
        "words": words,
    }


def viterbi_align(log_probs, targets, blank):
    """Frame spans of each target token on the best CTC path.

    Returns a list of (start, end) frame indices, end exclusive, one per
    target. Raises AlignmentError if no path exists or the trellis is too
    large.
    """
    T = log_probs.shape[0]
    N = len(targets)
    if N == 0:
        return []
    S = 2 * N + 1
    if T * S > MAX_TRELLIS_CELLS:
        raise AlignmentError("transcript and audio are too long to align in one pass")
    targets = np.asarray(targets, dtype=np.int64)
    state_tok = np.full(S, blank, dtype=np.int64)
    state_tok[1::2] = targets
    # Skipping the blank between two characters is allowed only when they
    # differ, otherwise the repeat would collapse.
    skip = np.zeros(S, dtype=bool)
    skip[3::2] = targets[1:] != targets[:-1]

    neg_inf = -np.inf
    alpha = np.full(S, neg_inf, dtype=np.float32)
    alpha[0] = log_probs[0, blank]
    alpha[1] = log_probs[0, targets[0]]
    backptr = np.zeros((T, S), dtype=np.uint8)
    idx = np.arange(S)
    for t in range(1, T):
        stay = alpha
        step = np.concatenate(([neg_inf], alpha[:-1]))
        jump = np.concatenate(([neg_inf, neg_inf], alpha[:-2]))
        jump = np.where(skip, jump, neg_inf)
        stacked = np.stack((stay, step, jump))
        choice = stacked.argmax(axis=0)
        alpha = stacked[choice, idx] + log_probs[t, state_tok]
        backptr[t] = choice

    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    if not np.isfinite(alpha[s]):
        raise AlignmentError("audio is too short for the transcript")
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, 0, -1):
        path[t] = s
        s -= int(backptr[t, s])
    path[0] = s

    spans = [None] * N
    for t, s in enumerate(path):
        if s % 2 == 1:
            i = (s - 1) // 2
            if spans[i] is None:
                spans[i] = [t, t + 1]
            else:
                spans[i][1] = t + 1
    if any(span is None for span in spans):
        raise AlignmentError("alignment did not cover the transcript")
    return [tuple(span) for span in spans]


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
    spans = viterbi_align(log_probs, targets, blank)

    word_tier = []
    letter_tier = []
    for (label, ids, chars), (a, b) in zip(words, word_slices):
        word_spans = spans[a:b]
        letters = [
            dict(zip(("start", "end"), interval(s, e, frame_s)), label=ch)
            for (s, e), ch in zip(word_spans, chars)
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


class Normalizer:
    """Maps transcript words onto model tokens for alignment."""

    def __init__(self, tokens, blank, delimiter, ignore=()):
        self.char_to_id = {}
        for i, tok in enumerate(tokens):
            if i in (blank, delimiter) or i in ignore or tok is None:
                continue
            if len(tok) == 1:
                self.char_to_id[tok] = i
        self.lower = all(not c.isupper() for c in self.char_to_id)

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
        whitespace-delimited word. Raises AlignmentError for digits or
        letters the model has no token for; words that consist only of
        punctuation are dropped."""
        result = []
        bad = set()
        digits = False
        for label in transcript.split():
            ids = []
            chars = []
            for c in label:
                mapped = self._map_char(c)
                if mapped is None:
                    if c.isdigit():
                        digits = True
                    else:
                        bad.add(c)
                    continue
                ids.extend(mapped)
                chars.extend([c] * len(mapped))
            if ids:
                result.append((label, ids, chars))
        if digits:
            raise AlignmentError(
                "transcript contains digits; write numbers out as words"
            )
        if bad:
            raise AlignmentError(
                "transcript contains characters the model cannot align: "
                + " ".join(sorted(bad))
            )
        if not result:
            raise AlignmentError("transcript is empty")
        return result
