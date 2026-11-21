"""Decode audio with PyAV and transcribe it locally with faster-whisper."""
import io
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import BinaryIO

import av
import numpy as np

from config import WHISPER_DEVICE, WHISPER_LANGUAGE, WHISPER_MODEL

SAMPLE_RATE = 16_000                      # Whisper expects 16 kHz mono audio.
BYTES_PER_SAMPLE = 4                      # float32
S16_TO_FLOAT32 = np.float32(1 / 32768)

AudioSource = str | Path | BinaryIO
StopCheck = Callable[[], None]            # Raises to abort; the web server uses it for "Stop generation".


class InvalidAudioError(ValueError):
    """The file has no audio track or it cannot be decoded."""


class AudioTooLongError(ValueError):
    """The decoded audio is longer than the allowed duration."""


def _register_pip_cuda_libraries() -> list:
    """On Windows, let CTranslate2 find CUDA DLLs installed with pip (nvidia-* packages).

    Returns the DLL directory handles, which must stay alive for the directories to remain registered.
    """
    handles = []
    if os.name != "nt":
        return handles
    for package in ("cublas", "cudnn", "cuda_nvrtc"):
        directory = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / package / "bin"
        if directory.is_dir():
            handles.append(os.add_dll_directory(str(directory)))
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ["PATH"]
    return handles


_cuda_dll_handles = _register_pip_cuda_libraries()


@contextmanager
def open_recording(source: AudioSource) -> Iterator[av.container.InputContainer]:
    """Open a recording from a path or an open binary file."""
    recording = Path(source).open("rb") if isinstance(source, (str, Path)) else nullcontext(source)
    # An empty protocol whitelist stops FFmpeg from following references inside the file
    # (for example playlists pointing at other files or URLs). Only the supplied bytes are read.
    with recording as file, av.open(file, mode="r", options={"protocol_whitelist": ""}) as container:
        yield container


def _audio_frames(container) -> Iterator[av.AudioFrame]:
    if not container.streams.audio:
        raise InvalidAudioError("The file contains no audio track.")
    for packet in container.demux(container.streams.audio[0]):
        try:
            frames = packet.decode()
        except av.error.InvalidDataError:
            continue  # Skip a damaged packet instead of discarding the rest of the lecture.
        for frame in frames:
            frame.pts = None  # Let the resampler ignore gaps from skipped packets.
            yield frame


def decode_audio(source: AudioSource, *, max_seconds: int | None = None,
                 check_stop: StopCheck | None = None) -> np.ndarray:
    """Decode the first audio track to the mono float32 16 kHz waveform Whisper expects."""
    byte_limit = None if max_seconds is None else max_seconds * SAMPLE_RATE * BYTES_PER_SAMPLE
    pcm = io.BytesIO()
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    try:
        with open_recording(source) as container:
            for frame in chain(_audio_frames(container), (None,)):  # None flushes the resampler.
                if check_stop:
                    check_stop()
                for output in resampler.resample(frame):
                    if byte_limit is not None and pcm.tell() + output.samples * BYTES_PER_SAMPLE > byte_limit:
                        raise AudioTooLongError("The recording exceeds the audio duration limit.")
                    pcm.write(output.to_ndarray() * S16_TO_FLOAT32)
    except av.FFmpegError as exc:
        raise InvalidAudioError("The file could not be decoded as audio.") from exc
    if not pcm.tell():
        raise InvalidAudioError("The audio track contains no decodable samples.")
    return np.frombuffer(pcm.getbuffer(), dtype=np.float32)


class Transcriber:
    """Speech-to-text with one faster-whisper model, loaded on first use."""

    def __init__(self, model_size: str = WHISPER_MODEL, device: str = WHISPER_DEVICE,
                 language: str = WHISPER_LANGUAGE):
        self.model_size = model_size
        self.device = device
        self.language = language

    @cached_property
    def model(self):
        # Imported here so that importing this module stays fast and does not touch the GPU.
        from ctranslate2 import get_cuda_device_count
        from faster_whisper import WhisperModel

        device = self.device
        if device == "auto":
            device = "cuda" if get_cuda_device_count() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        return WhisperModel(self.model_size, device=device, compute_type=compute_type)

    def transcribe(self, source: AudioSource, *, max_seconds: int | None = None,
                   on_segment: Callable[[str], None] | None = None,
                   check_stop: StopCheck | None = None) -> str:
        """Return the transcript as one string. Calls `on_segment` with each piece as it is recognized."""
        check_stop = check_stop or (lambda: None)
        waveform = decode_audio(source, max_seconds=max_seconds, check_stop=check_stop)
        check_stop()
        model = self.model  # Loading can take a while the first time.
        check_stop()
        # faster-whisper runs inference lazily, one chunk per iteration of `segments`.
        segments, _ = model.transcribe(
            waveform,
            language=self.language,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        texts = []
        for segment in segments:
            check_stop()  # Discard a chunk that finished after a stop request.
            text = segment.text.strip()
            if text:
                texts.append(text)
                if on_segment is not None:
                    on_segment(text)
            check_stop()  # Don't start inference on the next chunk after a stop request.
        return " ".join(texts)
