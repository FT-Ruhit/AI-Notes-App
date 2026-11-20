import io
import os
import sys
from contextlib import contextmanager, nullcontext
from collections.abc import Callable
from functools import cached_property
from itertools import chain
from pathlib import Path
from typing import BinaryIO

import av
import numpy as np

from config import WHISPER_DEVICE, WHISPER_LANGUAGE, WHISPER_MODEL

SAMPLE_RATE = 16_000
S16_TO_FLOAT32 = np.float32(1 / 32768)
AudioSource = str | Path | BinaryIO

# Keep the handles alive so CTranslate2 can load the pip-installed CUDA DLLs.
_cuda_dll_handles = []
if os.name == "nt":
    for package in ("cublas", "cudnn", "cuda_nvrtc"):
        directory = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / package / "bin"
        if directory.is_dir():
            _cuda_dll_handles.append(os.add_dll_directory(str(directory)))
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ["PATH"]


class InvalidAudioError(ValueError):
    pass


class AudioTooLongError(ValueError):
    pass


@contextmanager
def open_recording(source: AudioSource):
    recording = Path(source).open("rb") if isinstance(source, (str, Path)) else nullcontext(source)
    # Read only the supplied recording, including when a demuxer encounters an external reference.
    with recording as file, av.open(file, mode="r", options={"protocol_whitelist": ""}) as container:
        yield container


def _audio_frames(container):
    if not container.streams.audio:
        raise InvalidAudioError("The file contains no audio track.")
    for packet in container.demux(container.streams.audio[0]):
        try:
            frames = packet.decode()
        except av.error.InvalidDataError:
            # A damaged AAC packet need not discard the rest of the lecture.
            continue
        for frame in frames:
            frame.pts = None
            yield frame


def decode_audio(source: AudioSource, *, max_seconds: int | None = None,
                 check_stop: Callable[[], None] | None = None) -> np.ndarray:
    """Decode the first audio track to the mono float32 waveform Whisper expects."""
    byte_limit = None if max_seconds is None else max_seconds * SAMPLE_RATE * 4
    pcm = io.BytesIO()
    # Keep the original 16-bit downmix and normalize one frame at a time.
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    try:
        with open_recording(source) as container:
            for frame in chain(_audio_frames(container), (None,)):  # None flushes the resampler.
                if check_stop:
                    check_stop()
                for output in resampler.resample(frame):
                    if byte_limit is not None and pcm.tell() + output.samples * 4 > byte_limit:
                        raise AudioTooLongError("The recording exceeds the audio duration limit.")
                    pcm.write(output.to_ndarray() * S16_TO_FLOAT32)
    except av.FFmpegError as exc:
        raise InvalidAudioError("The file could not be decoded as audio.") from exc
    if not pcm.tell():
        raise InvalidAudioError("The audio track contains no decodable samples.")
    return np.frombuffer(pcm.getbuffer(), dtype=np.float32)


def decode_aac(audio_path: str) -> np.ndarray:
    return decode_audio(audio_path)


class Transcriber:
    def __init__(self, model_size="small", device=WHISPER_DEVICE, language=WHISPER_LANGUAGE):
        self.model_size = model_size
        self.device = device
        self.language = language

    @cached_property
    def model(self):
        from ctranslate2 import get_cuda_device_count
        from faster_whisper import WhisperModel

        device = self.device
        if device == "auto":
            device = "cuda" if get_cuda_device_count() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        return WhisperModel(self.model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: AudioSource, *, max_seconds: int | None = None,
                   on_segment: Callable[[str], None] | None = None,
                   check_stop: Callable[[], None] | None = None) -> str:
        waveform = decode_audio(audio_path, max_seconds=max_seconds, check_stop=check_stop)
        if check_stop:
            check_stop()
        model = self.model
        if check_stop:
            check_stop()
        segments, _ = model.transcribe(
            waveform,
            language=self.language,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        texts = []
        for segment in segments:
            if check_stop:
                check_stop()
            text = segment.text.strip()
            if text:
                texts.append(text)
                if on_segment is not None:
                    on_segment(text)
            if check_stop:
                check_stop()
        return " ".join(texts)


if __name__ == "__main__":
    transcript = Transcriber(WHISPER_MODEL).transcribe("./Wednesday_HUM.aac")
    Path("temp.txt").write_text(transcript, encoding="utf-8")
