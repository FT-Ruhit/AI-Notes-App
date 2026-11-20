import logging
from contextlib import nullcontext
from pathlib import Path

from anthropic import Anthropic

from config import ANTHROPIC_MODEL

logger = logging.getLogger(__name__)

SYSTEM = """You turn raw class-recording transcripts into structured study notes.
- Organize the notes as numbered steps in lecture order.
- Bold key terms and mark exam-relevant points with ⭐.
- The transcript is auto-captioned and noisy. Fix misheard terms from context.
- Ignore sounds and chatter from other people in the class (counting, side
  talk, interruptions). Keep only the teacher's content.
- If a claim or attribution looks doubtful, flag it at the end. Do not silently correct it.
- Output only the notes, in Markdown.

Output markdown with these sections:
## Summary
## Key Points
## Action Items (omit if there are none)"""


class EmptyNotesError(RuntimeError):
    pass


def note_request(transcript: str, context: str) -> dict:
    parts = []
    if context.strip():
        parts.append(f"<context>{context.strip()}</context>")
    parts.append(f"<transcript>{transcript.strip()}</transcript>")
    return dict(model=ANTHROPIC_MODEL, max_tokens=15_000, system=SYSTEM,
                messages=[{"role": "user", "content": "\n\n".join(parts)}])


def make_notes(transcript: str, context: str = "", *, client: Anthropic | None = None) -> str:
    with (nullcontext(client) if client is not None else Anthropic()) as provider:
        return read_notes(provider.messages.create(**note_request(transcript, context)))


def read_notes(response) -> str:
    notes = "\n\n".join(block.text for block in response.content if block.type == "text").strip()
    if not notes:
        raise EmptyNotesError("The note provider returned no Markdown text.")
    logger.info(
        "Anthropic finished (%s), %d input tokens, %d output tokens",
        response.stop_reason, response.usage.input_tokens, response.usage.output_tokens,
    )
    if response.stop_reason == "max_tokens":
        notes += "\n\n> These notes are incomplete because the output token limit was reached."
    return notes


if __name__ == "__main__":
    text = Path("temp.txt").read_text(encoding="utf-8")
    Path("text.md").write_text(make_notes(text), encoding="utf-8")
