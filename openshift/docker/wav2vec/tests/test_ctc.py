"""Unit tests for word grouping, forced alignment and normalisation on
hand-made emissions. Run with ``pytest`` from openshift/docker/wav2vec."""

import numpy as np
import pytest

from asr import ctc

# Vocabulary: 0 blank, 1 "|", 2 a, 3 b, 4 c, 5 <unk>
TOKENS = ["<pad>", "|", "a", "b", "c", "<unk>"]
BLANK, DELIM = 0, 1
FRAME_S = 0.02


def emissions(path, peak=0.9):
    """Log-probabilities with ``peak`` on the given token per frame and the
    rest of the mass spread over the other tokens."""
    lp = np.full((len(path), len(TOKENS)), np.log((1 - peak) / (len(TOKENS) - 1)), np.float32)
    for t, tok in enumerate(path):
        lp[t, tok] = np.log(peak)
    return lp


def test_greedy_decode_collapses_repeats_and_blanks():
    # "ab" then "ca": _ _ _ _ a a _ b | c _ a a _ _ _
    path = [0, 0, 0, 0, 2, 2, 0, 3, 1, 4, 0, 2, 2, 0, 0, 0]
    result = ctc.greedy_decode(emissions(path), FRAME_S, TOKENS, BLANK, DELIM, {5})
    assert result["transcript"] == "ab ca"
    assert [w["word"] for w in result["words"]] == ["ab", "ca"]
    # Frame 4 is 0.08 s; boundaries are moved LATENCY_S earlier.
    assert result["words"][0]["start"] == pytest.approx(0.08 - ctc.LATENCY_S)
    # Word ends are extended to the next word's start when the gap is small.
    assert result["words"][0]["end"] == result["words"][1]["start"] == pytest.approx(0.18 - ctc.LATENCY_S)
    # The last word ends at frame 13 and is extended, but not past the audio.
    assert result["words"][1]["end"] == pytest.approx(min(0.26 - ctc.LATENCY_S + ctc.PAUSE_EXTENSION_S, 16 * FRAME_S))
    assert 0 < result["confidence"] <= 1


def test_interval_clamps_at_start_and_keeps_one_frame():
    assert ctc.interval(0, 1, FRAME_S) == (0.0, FRAME_S)
    assert ctc.interval(10, 12, FRAME_S) == (pytest.approx(0.2 - ctc.LATENCY_S), pytest.approx(0.24 - ctc.LATENCY_S))


def test_greedy_decode_repeat_needs_blank():
    # a _ a is "aa"; a a is "a".
    r = ctc.greedy_decode(emissions([2, 0, 2]), FRAME_S, TOKENS, BLANK, DELIM)
    assert r["transcript"] == "aa"
    r = ctc.greedy_decode(emissions([2, 2]), FRAME_S, TOKENS, BLANK, DELIM)
    assert r["transcript"] == "a"


def test_greedy_decode_ignores_special_tokens_and_empty_audio():
    r = ctc.greedy_decode(emissions([5, 0, 2, 5]), FRAME_S, TOKENS, BLANK, DELIM, {5})
    assert r["transcript"] == "a"
    r = ctc.greedy_decode(emissions([0, 0, 0]), FRAME_S, TOKENS, BLANK, DELIM)
    assert r == {"transcript": "", "confidence": 0.0, "words": []}


def test_close_gaps_pauses_get_a_bounded_extension():
    words = [
        {"start": 0.0, "end": 0.5},
        {"start": 0.6, "end": 1.0},
        {"start": 1.5, "end": 1.9},
        {"start": 2.0, "end": 2.5},
    ]
    ctc.close_gaps(words, 2.6)
    assert words[0]["end"] == 0.6
    assert words[1]["end"] == pytest.approx(1.0 + ctc.PAUSE_EXTENSION_S)
    # Never into the next word.
    assert words[2]["end"] == 2.0
    # The last word stops at the end of the audio.
    assert words[3]["end"] == 2.6


def test_align_tokens_recovers_planted_path():
    # a a _ b b b _ _ c c, with a distractor token in the silence.
    path = [2, 2, 0, 3, 3, 3, 0, 0, 4, 4]
    spans = ctc.align_tokens(emissions(path), [2, 3, 4], BLANK)
    assert spans == [(0, 2), (3, 6), (8, 10)]


def test_align_tokens_repeated_character():
    # "aa" must pass through a blank between the two a's.
    path = [2, 0, 2]
    spans = ctc.align_tokens(emissions(path), [2, 2], BLANK)
    assert spans == [(0, 1), (2, 3)]
    with pytest.raises(ctc.AlignmentError):
        ctc.align_tokens(emissions([2, 2]), [2, 2], BLANK)


def test_align_tokens_forces_transcript_through_unlikely_frames():
    # The audio "says" b but the transcript is "c"; c still gets a span.
    spans = ctc.align_tokens(emissions([3, 3, 3]), [4], BLANK)
    assert len(spans) == 1
    start, end = spans[0]
    assert 0 <= start < end <= 3


def test_align_words_tiers():
    # "ab" | "c": _ _ _ a _ b | _ c c _ _ _ _ _ _ _ _ _ _ _
    path = [0, 0, 0, 2, 0, 3, 1, 0, 4, 4] + [0] * 10
    words = [("ab", [2, 3], ["a", "b"]), ("c!", [4], ["c"])]
    tiers = ctc.align_words(emissions(path), FRAME_S, words, DELIM, BLANK)
    assert [w["label"] for w in tiers["words"]] == ["ab", "c!"]
    assert tiers["words"][0]["start"] == pytest.approx(0.06 - ctc.LATENCY_S)
    assert tiers["words"][1]["start"] == pytest.approx(0.16 - ctc.LATENCY_S)
    assert tiers["words"][1]["end"] == pytest.approx(0.2 - ctc.LATENCY_S + ctc.PAUSE_EXTENSION_S)
    # The first word is extended up to the second.
    assert tiers["words"][0]["end"] == tiers["words"][1]["start"]
    letters = tiers["letters"]
    assert [l["label"] for l in letters] == ["a", "b", "c"]
    # Letters are contiguous within the word and the last one ends with it.
    assert letters[0]["end"] == letters[1]["start"]
    assert letters[1]["end"] == tiers["words"][0]["end"]
    assert letters[2]["end"] == tiers["words"][1]["end"]


def test_normalizer_maps_and_rejects():
    tokens = ["<pad>", "|", "a", "b", "c", "s", "z", "e", "<unk>", "ä"]
    n = ctc.Normalizer(tokens, 0, 1, {8})
    words = n.words("Abc, šzé ää -")
    assert [label for label, _, _ in words] == ["Abc,", "šzé", "ää"]
    assert words[0][1] == [2, 3, 4]
    assert words[0][2] == ["A", "b", "c"]
    assert words[1][1] == [5, 6, 7]
    assert words[2][1] == [9, 9]
    # A letter that maps to two tokens is the source of both.
    tokens2 = ["<pad>", "|", "s", "a"]
    assert ctc.Normalizer(tokens2, 0, 1).words("aß")[0][2] == ["a", "ß", "ß"]
    # Without a language, digits cannot be spelled out.
    with pytest.raises(ctc.AlignmentError, match="digits"):
        n.words("abc 12")
    with pytest.raises(ctc.AlignmentError, match="cannot align: x"):
        n.words("abc x")
    with pytest.raises(ctc.AlignmentError, match="empty"):
        n.words(" ... ")


def test_normalizer_spells_digits_in_known_languages():
    tokens = ["<pad>", "|"] + list("abcdefghijklmnopqrstuvwxyzäöå")
    fi = ctc.Normalizer(tokens, 0, 1, lang="fi")
    labels = [label for label, _, _ in fi.words("sivu 12 ja 3")]
    assert labels == ["sivu", "kaksitoista", "ja", "kolme"]
    sv = ctc.Normalizer(tokens, 0, 1, lang="sv-FI")
    assert [label for label, _, _ in sv.words("sida 12")] == ["sida", "tolv"]
    sme = ctc.Normalizer(tokens, 0, 1, lang="sme")
    with pytest.raises(ctc.AlignmentError, match="digits"):
        sme.words("siidu 12")
