import os
import sys
from pathlib import Path

import av
import numpy as np
from faster_whisper import WhisperModel


_cuda_dll_handles = []
if os.name == "nt":
    _site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    for _pkg in ("cublas", "cudnn", "cuda_nvrtc"):
        _bin = _site_packages / "nvidia" / _pkg / "bin"
        if _bin.is_dir():
            _cuda_dll_handles.append(os.add_dll_directory(str(_bin)))
            os.environ["PATH"] = str(_bin) + os.pathsep + os.environ["PATH"]


_av_open = av.open


def _open_media(file, mode = "r", **kwargs):
    kwargs.pop("metadata_errors", None)
    return _av_open(file, mode=mode, **kwargs)


av.open = _open_media


def decode_aac(audio_path: str) -> np.ndarray:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    chunks = []
    with av.open(audio_path) as container:
        for packet in container.demux(audio=0):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                continue
            for frame in frames:
                frame.pts = None
                for output in resampler.resample(frame):
                    chunks.append(output.to_ndarray().reshape(-1))
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError(f"No decodable audio found: {audio_path}")
    return np.concatenate(chunks).astype(np.float32) / 32768.0


class Transcriber:
    def __init__(self, model_size="small", device="cuda", language="en"):
        compute_type = "float16" if device == "cuda" else "int8"
        self.language = language
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> str:
        if not Path(audio_path).is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        audio = decode_aac(audio_path) if Path(audio_path).suffix.lower() == ".aac" else audio_path
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments)


if __name__ == "__main__":

    transcriber = Transcriber(model_size="medium")
    text = transcriber.transcribe("./Wednesday_HUM.aac")
    with open("./temp.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
