"""The diarisation backends' shared output shape, with a stand-in for the
pyannote pipeline since its models are gated."""

import json

from asr import diarize


class Segment:
    def __init__(self, start, end):
        self.start, self.end = start, end


class Annotation:
    def __init__(self, tracks):
        self.tracks = tracks

    def itertracks(self, yield_label=False):
        return [(Segment(s, e), "_", label) for s, e, label in self.tracks]


class FakeOutput:
    def __init__(self, exclusive):
        self.exclusive_speaker_diarization = Annotation(exclusive)
        self.speaker_diarization = Annotation([(0.0, 9.0, "SPEAKER_01")])


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.segmentation_batch_size = 32
        self.embedding_batch_size = 32

    def __call__(self, file, num_speakers=None, hook=None):
        self.calls.append((file["waveform"].shape, file["sample_rate"], num_speakers))
        hook("segmentation", None, file=file)
        return FakeOutput([(3.0, 4.0, "SPEAKER_00"), (0.0, 3.0, "SPEAKER_01"), (4.0, 9.0, "SPEAKER_01")])


def test_pyannote_backend_uses_exclusive_turns_and_relabels(tmp_path):
    (tmp_path / "kielipankki.json").write_text(json.dumps({"name": "community-1", "backend": "pyannote"}))
    pipeline = FakePipeline()
    d = diarize.PyannoteDiarizer(str(tmp_path), pipeline=pipeline, batch_size=4)
    assert pipeline.segmentation_batch_size == 4 and pipeline.embedding_batch_size == 4
    beats = []
    turns = d.turns([0.0] * 16000 * 9, num_speakers=2, progress=lambda: beats.append(1))
    assert turns == [
        {"speaker": "S0", "start": 0.0, "end": 3.0},
        {"speaker": "S1", "start": 3.0, "end": 4.0},
        {"speaker": "S0", "start": 4.0, "end": 9.0},
    ]
    assert pipeline.calls == [((1, 16000 * 9), 16000, 2)] and beats
    assert d.description["backend"] == "pyannote.audio" and d.description["name"] == "community-1"


def test_label_turns_orders_by_start():
    assert diarize.label_turns([("b", 2.0, 3.0), ("a", 0.0, 1.0), ("b", 1.0, 2.0)]) == [
        {"speaker": "S0", "start": 0.0, "end": 1.0},
        {"speaker": "S1", "start": 1.0, "end": 2.0},
        {"speaker": "S1", "start": 2.0, "end": 3.0},
    ]
