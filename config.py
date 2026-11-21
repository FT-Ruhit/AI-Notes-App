"""All settings for AI Notes, read once from the environment or a `.env` file.

Copy `.env.example` to `.env` to change them. Restart the server after editing.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))


def _positive_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        number = int(value)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {value!r}") from None
    if number <= 0:
        raise ValueError(f"{name} must be greater than zero, got {number}")
    return number


# Note generation (Anthropic). ANTHROPIC_API_KEY is read directly by the Anthropic SDK.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

# Transcription (faster-whisper, runs locally).
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # "auto", "cuda" or "cpu"
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

# Web server limits.
MAX_UPLOAD_BYTES = _positive_int("MAX_UPLOAD_MIB", 256) * 1024**2
MAX_AUDIO_MINUTES = _positive_int("MAX_AUDIO_MINUTES", 240)
MAX_CONTEXT_CHARS = 4000            # Lecture context typed by the user.
MAX_ACTIVE_RECORDINGS = 4           # Uploads being processed at once, across /notes and /jobs.
MAX_RETAINED_JOBS = 64              # Background jobs kept in memory, finished or not.
JOB_RETENTION_SECONDS = 60 * 60     # How long a finished job stays available.
