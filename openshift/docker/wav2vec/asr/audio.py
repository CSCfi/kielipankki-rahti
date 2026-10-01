"""Decoding uploads to 16 kHz mono 16-bit WAV and splitting on silence.

ffmpeg does all decoding, as a subprocess with the same arguments the
previous service used. pydub is used only for reading and writing WAV and for
silence detection, neither of which needs ffmpeg.
"""

import io
import os
import subprocess
import tempfile
import wave

import numpy as np
import pydub
import pydub.silence

from . import config

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

# Silence splitting parameters, unchanged from the Kaldi service.
MIN_SILENCE_MS = 360
SILENCE_THRESH_DB = -36
MIN_SEGMENT_S = 5.0


class AudioError(Exception):
    """The upload could not be decoded."""


def is_riff_wave(data):
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def wav_params(data):
    """(sample rate, channels, sample width in bytes, frames) of a WAV, or
    None if the ``wave`` module cannot read it."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
    except (wave.Error, EOFError):
        return None


def is_canonical(data):
    params = wav_params(data)
    return params is not None and params[:3] == (config.SAMPLE_RATE, 1, 2)


def to_canonical_wav(data, hint=None):
    """Convert anything ffmpeg understands to 16 kHz mono 16-bit PCM WAV.

    ``hint`` is a file extension; it is passed to ffmpeg only if the content
    cannot be identified without it. Raises AudioError on failure.
    """
    if is_canonical(data):
        return data
    with tempfile.TemporaryDirectory() as tmp:
        suffix = f".{hint}" if hint and hint.isalnum() else ""
        src = os.path.join(tmp, "in" + suffix)
        dst = os.path.join(tmp, "out.wav")
        with open(src, "wb") as f:
            f.write(data)
        # -y overwrites the output, -ac 1 downmixes to mono, -c:a pcm_s16le
        # is 16-bit PCM and -ar 16000 resamples. -vn drops any cover art.
        result = subprocess.run(
            [
                FFMPEG, "-nostdin", "-y", "-loglevel", "error",
                "-i", src, "-vn", "-ac", "1", "-c:a", "pcm_s16le", "-ar", "16000",
                dst,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0 or not os.path.exists(dst):
            raise AudioError(result.stderr.decode("utf-8", "replace").strip())
        with open(dst, "rb") as f:
            out = f.read()
    if not is_canonical(out):
        raise AudioError("conversion did not produce 16 kHz mono PCM")
    return out


def segment_from_wav(data):
    return pydub.AudioSegment.from_wav(io.BytesIO(data))


def segment_to_wav(segment):
    buf = io.BytesIO()
    segment.export(buf, format="wav")
    return buf.getvalue()


def duration_s(data):
    params = wav_params(data)
    if params is None:
        raise AudioError("not a PCM WAV file")
    rate, _, _, frames = params
    return frames / rate


def samples(data):
    """Float32 samples in [-1, 1] at 16 kHz mono from a WAV, converting first
    if the WAV is not already canonical."""
    data = to_canonical_wav(data)
    with wave.open(io.BytesIO(data)) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def split_on_silence(audio):
    """Split on silence, then merge the shortest segments into their shorter
    neighbour until every segment is at least MIN_SEGMENT_S long."""
    segments = pydub.silence.split_on_silence(
        audio,
        min_silence_len=MIN_SILENCE_MS,
        silence_thresh=SILENCE_THRESH_DB,
        keep_silence=True,
        seek_step=1,
    )
    if not segments:
        return [audio]
    while len(segments) > 1:
        idx = min(range(len(segments)), key=lambda i: segments[i].duration_seconds)
        if segments[idx].duration_seconds >= MIN_SEGMENT_S:
            break
        if idx == 0:
            segments[0] += segments[1]
            del segments[1]
        elif idx == len(segments) - 1:
            segments[-2] += segments[-1]
            del segments[-1]
        elif segments[idx - 1].duration_seconds < segments[idx + 1].duration_seconds:
            segments[idx - 1] += segments[idx]
            del segments[idx]
        else:
            segments[idx] += segments[idx + 1]
            del segments[idx + 1]
    return segments
