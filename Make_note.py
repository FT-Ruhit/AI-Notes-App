from dotenv import load_dotenv
import anthropic

def make_notes(transcript: str, context: str = "") -> str:
    load_dotenv()

    client = anthropic.Anthropic()

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

    parts = list()
    if context.strip():
        parts.append(f"<context>{context.strip()}</context>")
    parts.append(f"<transcript>{transcript.strip()}</transcript>")
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=15000,
        system=SYSTEM,
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
    )
    
    notes = "\n\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    )
    print(response.stop_reason)
    print(response.usage.input_tokens)
    print(response.usage.output_tokens)
    if response.stop_reason == "max_tokens":
        notes += "\n\n[Notes were cut off. This is for test purpose. try a shorter recording.]"
    return notes
        
if __name__ == "__main__":
    with open("./temp.txt", 'r', encoding='utf-8') as f:
        text = f.read()
        
    notes = make_notes(text)
    # print(notes)
    
    with open("./text.md", "w", encoding='utf-8') as f:
        f.write(notes)

