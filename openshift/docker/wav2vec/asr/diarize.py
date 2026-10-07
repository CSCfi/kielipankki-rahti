"""Speaker diarisation: a list of turns ``[{"speaker": "S0", "start",
"end"}]`` sorted by start, labels numbered in order of first appearance.

Two backends share that interface and ``load_diarizer`` picks by the model
directory: ``SherpaDiarizer`` runs pyannote's segmentation 3.0 model and a
speaker embedding model through sherpa-onnx, ``PyannoteDiarizer`` runs a
pyannote.audio pipeline such as speaker-diarization-community-1.
"""

import json
import logging
import os

import numpy as np

from . import config

log = logging.getLogger(__name__)

SPEAKER_PREFIX = "S"


def read_info(model_dir):
    path = os.path.join(model_dir, "kielipankki.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def label_turns(turns):
    """Sort raw ``(speaker, start, end)`` turns by start and relabel the
    speakers S0, S1, ... in order of first appearance."""
    labels = {}
    out = []
    for speaker, start, end in sorted(turns, key=lambda t: (t[1], t[2])):
        if speaker not in labels:
            labels[speaker] = f"{SPEAKER_PREFIX}{len(labels)}"
        out.append({"speaker": labels[speaker], "start": round(start, 3), "end": round(end, 3)})
    return out


class SherpaDiarizer:
    """Loads ``segmentation.onnx`` and ``embedding.onnx`` from the model
    directory, or the file names given in ``kielipankki.json``."""

    def __init__(
        self,
        model_dir=config.MODEL_DIR,
        threads=config.THREADS,
        threshold=config.DIARIZE_THRESHOLD,
    ):
        import sherpa_onnx

        self.sherpa_onnx = sherpa_onnx
        info = read_info(model_dir)
        segmentation = os.path.join(model_dir, info.get("segmentation", "segmentation.onnx"))
        embedding = os.path.join(model_dir, info.get("embedding", "embedding.onnx"))
        self.threshold = threshold
        self.segmentation_config = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=segmentation),
            num_threads=threads,
        )
        self.embedding_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding, num_threads=threads
        )
        self.diarizer = sherpa_onnx.OfflineSpeakerDiarization(self._config(None))
        self.sample_rate = self.diarizer.sample_rate
        self.description = {
            "name": info.get("name", os.path.basename(os.path.normpath(model_dir))),
            "backend": "sherpa-onnx",
            "segmentation": os.path.basename(segmentation),
            "embedding": os.path.basename(embedding),
            "threshold": threshold,
        }
        log.info("loaded %s", json.dumps(self.description))

    def _config(self, num_speakers):
        clustering = self.sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers or -1, threshold=self.threshold
        )
        return self.sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=self.segmentation_config,
            embedding=self.embedding_config,
            clustering=clustering,
        )

    def turns(self, samples, num_speakers=None, progress=None):
        """``samples`` are float32 at ``sample_rate``. ``num_speakers`` fixes
        the number of clusters; otherwise the threshold decides."""
        # set_config only changes the clustering; the models stay loaded.
        self.diarizer.set_config(self._config(num_speakers))

        # Called after every segmentation window with the progress counts,
        # whose number differs between sherpa-onnx versions.
        def callback(*_):
            if progress:
                progress()
            return 0

        result = self.diarizer.process(np.asarray(samples, dtype=np.float32), callback=callback)
        return label_turns(
            (segment.speaker, segment.start, segment.end) for segment in result.sort_by_start_time()
        )


class PyannoteDiarizer:
    """A pyannote.audio pipeline checkpoint (``config.yaml`` with its models
    beside it) loaded offline. Turns come from the pipeline's exclusive
    speaker diarisation, which has no overlapping turns, so that every word
    has one speaker."""

    def __init__(
        self,
        model_dir=config.MODEL_DIR,
        threads=config.THREADS,
        batch_size=config.DIARIZE_BATCH,
        pipeline=None,
    ):
        info = read_info(model_dir)
        self.sample_rate = config.SAMPLE_RATE
        if pipeline is None:
            import torch
            from pyannote.audio import Pipeline

            torch.set_num_threads(threads)
            pipeline = Pipeline.from_pretrained(model_dir)
            if pipeline is None:
                raise RuntimeError(f"no pyannote pipeline in {model_dir}")
        for name in ("segmentation_batch_size", "embedding_batch_size"):
            if hasattr(pipeline, name):
                setattr(pipeline, name, batch_size)
        self.pipeline = pipeline
        self.description = {
            "name": info.get("name", os.path.basename(os.path.normpath(model_dir))),
            "backend": "pyannote.audio",
            "revision": info.get("revision"),
            "batch_size": batch_size,
        }
        log.info("loaded %s", json.dumps(self.description))

    def turns(self, samples, num_speakers=None, progress=None):
        import torch

        # The pipeline calls the hook after every step and every batch
        # within a step, with the step name and its artifact.
        def hook(*_, **__):
            if progress:
                progress()

        waveform = torch.as_tensor(np.asarray(samples, dtype=np.float32))[None]
        output = self.pipeline(
            {"waveform": waveform, "sample_rate": self.sample_rate},
            num_speakers=num_speakers,
            hook=hook,
        )
        annotation = getattr(output, "exclusive_speaker_diarization", output)
        return label_turns(
            (speaker, segment.start, segment.end)
            for segment, _, speaker in annotation.itertracks(yield_label=True)
        )


def load_diarizer(model_dir=config.MODEL_DIR, **kwargs):
    info = read_info(model_dir)
    if info.get("backend") == "pyannote" or os.path.exists(os.path.join(model_dir, "config.yaml")):
        return PyannoteDiarizer(model_dir, **kwargs)
    return SherpaDiarizer(model_dir, **kwargs)
