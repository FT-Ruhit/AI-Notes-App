"""Command-line version: pick a recording from input/, write the transcript and notes to output/."""
import logging
from pathlib import Path

import av

from notes import make_notes
from transcription import Transcriber, open_recording

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")


def find_recordings(directory: Path) -> list[Path]:
    """Return the files in `directory` that contain an audio track, sorted by name."""
    recordings = []
    for path in directory.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            with open_recording(path) as container:
                if container.streams.audio:
                    recordings.append(path)
        except (OSError, av.FFmpegError):
            continue
    return sorted(recordings, key=lambda path: path.name.casefold())


def choose(recordings: list[Path]) -> Path:
    print("Select a recording:")
    for index, recording in enumerate(recordings):
        print(f"{index} - {recording.name}")
    while True:
        try:
            index = int(input("Recording number: "))
        except ValueError:
            index = -1
        if 0 <= index < len(recordings):
            return recordings[index]
        print(f"Enter a number from 0 to {len(recordings) - 1}.")


def main():
    INPUT_DIR.mkdir(exist_ok=True)
    recordings = find_recordings(INPUT_DIR)
    if not recordings:
        print(f"Place a recording in {INPUT_DIR}/ and run again.")
        return

    recording = choose(recordings)
    context = input("Lecture context (optional): ")

    transcript = Transcriber().transcribe(recording)
    # Save the transcript first so it survives a failed note request.
    OUTPUT_DIR.mkdir(exist_ok=True)
    transcript_path = OUTPUT_DIR / f"{recording.stem}.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    print(f"Saved {transcript_path}")
    if not transcript:
        print("No speech was detected.")
        return

    notes_path = OUTPUT_DIR / f"{recording.stem}.md"
    notes_path.write_text(make_notes(transcript, context), encoding="utf-8")
    print(f"Saved {notes_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
