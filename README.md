# AI Notes

Turn class recordings into transcripts and structured study notes. Audio is transcribed locally with Faster Whisper, then the transcript is sent to the Anthropic API to generate Markdown notes.

## Features

- Select a recording from the `input/` folder.
- Provide optional context, such as the subject, lecture topic, or unfamiliar terms.
- Save a plain-text transcript and Markdown notes in `output/`.
- Generate a summary, key points, and action items when applicable, with instructions to highlight exam-relevant material and flag doubtful claims.

## Setup

Install Python (3.10 or newer recommended), then clone this repository and open a terminal in its root folder.

### 1. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install faster-whisper av numpy anthropic python-dotenv
```

The repository does not currently pin dependency versions.

### 3. Create the `.env` file

Create a file named exactly `.env` in the project root, alongside `main.py`. Add your Anthropic API key:

```dotenv
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Replace the placeholder with a key from your Anthropic account. Do not name the file `.env.txt`.

`Make_note.py` loads this file using `python-dotenv`, and the Anthropic client reads `ANTHROPIC_API_KEY` automatically. This is the only environment variable required by the application. Keep the key private; `.env` is already excluded by `.gitignore`.

Note generation requires internet access and Anthropic API access, and may incur usage charges. The model ID is currently hardcoded as `claude-sonnet-5-5` in `Make_note.py`. Check that it is available to your account; if the API rejects it, change the `model` argument to a model ID supported by your account. Setting a model name in `.env` will not change it.

### 4. Choose GPU or CPU transcription

The default in `main.py` is:

```python
extractor = Transcriber("medium")
```

This uses CUDA with `float16` computation and requires a compatible NVIDIA GPU and CUDA/cuDNN runtime libraries for the installed Faster Whisper/CTranslate2 versions. On Windows, the code also looks for NVIDIA DLLs inside the active environment's `site-packages/nvidia/` folders.

To use the CPU instead, replace that line with:

```python
extractor = Transcriber("medium", device="cpu")
```

CPU mode uses `int8` computation. You can select a smaller model, such as `"small"`, to reduce resource requirements. Transcription defaults to English; pass `language="en"` or another supported language code to `Transcriber` when needed.

## Usage

1. Create `input/` and `output/` folders in the project root if they do not exist.
2. Place a recording in `input/`, for example `lecture.mp3` or `lecture.aac`.
3. Run the application from the project root:

   ```bash
   python main.py
   ```

4. Enter the recording's displayed index (starting at `0`).
5. Enter optional lecture context, or press Enter to leave it blank.

The first use of a Whisper model may download its files. After processing `lecture.mp3`, the application writes:

```text
output/lecture.txt    # Transcript
output/lecture.md     # Study notes
```

Files with the same base name overwrite the corresponding outputs. The transcript is saved before note generation, so it remains available if the API request fails. The application also prints the API stop reason and input/output token counts.

## Project structure

```text
main.py           # Interactive file selection and processing pipeline
Text_exreact.py   # Faster Whisper transcription and AAC decoding
Make_note.py      # Anthropic prompt and Markdown note generation
input/            # Your recordings (create locally)
output/           # Generated transcripts and notes (create locally)
.env              # Your API key (create locally; ignored by Git)
.gitignore        # Local files and generated-content exclusions
README.md         # Setup and usage instructions
```

`test.py` is a small input experiment, not an automated test suite. Running `Make_note.py` directly reads `temp.txt` and writes `text.md`. Running `Text_exreact.py` directly expects `Wednesday_HUM.aac` in the project root and writes `temp.txt`; use `main.py` for the normal workflow.

## Troubleshooting

- **Missing input/output folder:** Create both folders and run the command from the project root.
- **No audio file available:** Add a recording to `input/`. File discovery currently compares filenames against PyAV's format names and is case-sensitive, so some extensions may not appear. Try a lowercase `.mp3` or `.aac` extension for a compatible recording.
- **CUDA or DLL errors:** Verify the runtime libraries required by your installed transcription dependencies, or switch to CPU mode as shown above.
- **API authentication or model errors:** Check `ANTHROPIC_API_KEY` in `.env`, your account's API access, and the model ID in `Make_note.py`.
- **Incomplete notes:** The API request has a 15,000-token output limit. If reached, the app appends a cutoff notice; try a shorter recording.

Recordings are processed locally for transcription, but transcripts and any supplied context are sent to Anthropic for note generation. Review generated notes against the lecture material for accuracy.
