"""Speaker assignment, turn grouping and rule-based punctuation."""

from asr import punctuate, results


class Turns(results.Recognition):
    """A Recognition with only diarisation turns set, for speaker_at."""

    def __init__(self, turns):
        self._turns = turns


def test_speaker_at_prefers_overlap_then_nearest():
    r = Turns([
        {"speaker": "S0", "start": 0.0, "end": 10.0},
        {"speaker": "S1", "start": 4.0, "end": 5.0},
    ])
    assert r.speaker_at(1.0, 2.0) == "S0"
    assert r.speaker_at(4.2, 4.8) == "S1"
    assert r.speaker_at(3.9, 4.3) == "S0"
    assert r.speaker_at(12.0, 13.0) == "S0"
    assert Turns([]).speaker_at(0.0, 1.0) == "S0"


def words(spec):
    return [{"word": w, "start": s, "end": e, **({"speaker": spk} if spk else {})} for w, s, e, spk in spec]


def test_group_turns_on_speaker_change_and_silence():
    ws = words([("a", 0.0, 0.5, "S0"), ("b", 0.6, 1.0, "S0"), ("c", 1.1, 1.5, "S1"), ("d", 4.0, 4.5, "S1")])
    groups = results.group_turns(ws)
    assert [[w["word"] for w in g] for g in groups] == [["a", "b"], ["c"], ["d"]]
    entries = results.turn_entries(ws)
    assert [(t["speaker"], t["start"], t["end"]) for t in entries] == [("S0", 0.0, 1.1), ("S1", 1.1, 4.0), ("S1", 4.0, 4.5)]
    assert entries[0]["transcript"] == "a b"
    assert results.turn_entries([]) == []


def test_rules_capitalise_after_pauses_and_at_turn_starts():
    turns = [words([("a", 0.0, 0.5, None), ("b", 0.6, 1.0, None), ("c", 2.5, 3.0, None)]), words([("d", 3.1, 3.5, None)])]
    out = punctuate.by_rules(turns, pause_s=1.0)
    assert [w["word_with_punctuation"] for w in out[0]] == ["A", "b.", "C."]
    assert [w["punctuation"] for w in out[0]] == ["", ".", "."]
    assert out[1] == [{"word_with_punctuation": "D.", "punctuation": "."}]


def test_uses_model_matches_language_subtags():
    assert punctuate.uses_model("sv-FI", ["fi", "sv"])
    assert punctuate.uses_model("fi", ["fi", "sv"])
    assert not punctuate.uses_model("sme", ["fi", "sv"])

