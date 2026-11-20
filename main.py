import logging
from pathlib import Path

import av

from config import WHISPER_MODEL
from Make_note import make_notes
from Text_exreact import Transcriber, open_recording


def find_recordings(directory: Path) -> list[Path]:
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


def main():
    input_dir, output_dir = Path("input"), Path("output")
    input_dir.mkdir(exist_ok=True)
    recordings = find_recordings(input_dir)
    if not recordings:
        print("Place a recording in input/ and run again.")
        return

    print("Select a recording:")
    for index, recording in enumerate(recordings):
        print(f"{index} - {recording.name}")
    while True:
        try:
            index = int(input("Recording number: "))
            if 0 <= index < len(recordings):
                break
        except ValueError:
            pass
        print(f"Enter a number from 0 to {len(recordings) - 1}.")
    recording = recordings[index]
    context = input("Lecture context (optional): ")

    transcript = Transcriber(WHISPER_MODEL).transcribe(recording)
    output_dir.mkdir(exist_ok=True)
    transcript_path = output_dir / f"{recording.stem}.txt"
    transcript_path.write_text(transcript, encoding="utf-8")
    print(f"Saved {transcript_path}")
    if not transcript:
        print("No speech was detected.")
        return
    notes = make_notes(transcript, context)
    notes_path = output_dir / f"{recording.stem}.md"
    notes_path.write_text(notes, encoding="utf-8")
    print(f"Saved {notes_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
