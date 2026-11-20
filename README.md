# AI Notes

Turn a lecture recording into downloadable Markdown study notes. Faster Whisper transcribes the audio locally. Anthropic receives the transcript and optional lecture context and produces a summary, key points in lecture order, and action items when applicable.

## Setup

Use Python 3.12 or newer. From the repository root, create a virtual environment and install the dependencies. In Windows PowerShell, run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, use `python3 -m venv .venv` and `source .venv/bin/activate` before the install command.

Create a `.env` file alongside `api.py` with your Anthropic API key. Restart the server after changing settings.

```dotenv
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

The default transcription model is `medium`. Its first use may download model files. On a Windows NVIDIA setup, install the CUDA runtime libraries below if they are missing. To use CPU transcription, set `WHISPER_DEVICE=cpu` in `.env`.

```powershell
python -m pip install nvidia-cublas-cu12==12.9.2.10 nvidia-cudnn-cu12==9.28.1.4 nvidia-cuda-nvrtc-cu12==12.9.86
```

## Run the API

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open [the upload page](http://127.0.0.1:8000/) for long recordings. After the upload finishes, the page shows the transcription as Whisper recognizes speech, then offers a Markdown download when notes are ready. Pause or resume updates, refresh the page, or reopen its job URL while the server is running. Finished jobs remain available for one hour. Restarting the server clears jobs. Run one server worker for this workflow.

The studio lets students select a transcription model, review formatted study notes, and copy or download the original Markdown. It gets the available faster-whisper models and the configured default from `GET /models`. The server keeps the most recently used transcription model in memory.

For API clients, multipart `POST /jobs` accepts `file`, optional `context`, and an optional `model` from `GET /models`. Omitting the model uses `WHISPER_MODEL`. A successful upload returns HTTP 202 with an `id`. Poll `GET /jobs/{id}?after=0` for the recording name, model, stage, and transcript segments. Pass the returned `cursor` as the next `after` value to receive new segments. When the stage is `ready`, download `GET /jobs/{id}/notes`. A `failed` stage includes an error message.

The [interactive API docs](http://127.0.0.1:8000/docs) also expose the original synchronous `POST /notes` endpoint. Its required `file` field accepts a recording. The optional `context` field accepts text such as a course code, lecture topic, or names the speaker uses.

```powershell
curl.exe --fail -F "file=@input/lecture.m4a" -F "context=Physics, electric dipoles" -o lecture.md http://127.0.0.1:8000/notes
```

Use `curl` in place of `curl.exe` on macOS or Linux. A successful request returns the UTF-8 Markdown bytes with a `text/markdown` content type and the recording's base name in the attachment header. The command above saves them to `lecture.md`. The request stays open through transcription and note generation.

Audio is identified from its contents. Supported examples include WAV, MP3, AAC, M4A, Ogg/Opus, WebM, and video files with an audio track. The API releases the upload after transcription and returns the notes directly. Each server process loads Whisper on its first valid recording and reuses that model and one Anthropic client for subsequent requests. Use one Uvicorn worker to share a single GPU model. The next transcription can start while the previous lecture's notes are being generated.

| HTTP status | Meaning |
| --- | --- |
| 413 | The multipart request or decoded recording exceeds the configured limit. |
| 415 | The recording has no decodable audio track. |
| 422 | The file field is missing, the recording is empty, or Whisper detected no speech. |
| 502 | Anthropic failed to generate notes. Check the server key and model access. |
| 503 | The server needs an API key or its transcription backend is unavailable. |
| 504 | The note provider timed out. |

## Optional configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | Choose a notes model available to your Anthropic account. |
| `WHISPER_MODEL` | `medium` | Choose a Whisper model size or local model directory. |
| `WHISPER_DEVICE` | `auto` | Select CUDA when detected or CPU otherwise. Set `cpu` or `cuda` to choose explicitly. |
| `WHISPER_LANGUAGE` | `en` | Set the spoken language code, such as `bn` for Bengali. |
| `MAX_UPLOAD_MIB` | `256` | Maximum size of the entire HTTP request, in MiB. |
| `MAX_AUDIO_MINUTES` | `240` | Maximum decoded recording duration for an upload, in minutes. |

## Run the CLI

Place a recording in `input/` and run `python main.py`. Select the recording and enter any lecture context. For `lecture.m4a`, the CLI writes `output/lecture.txt` before requesting notes, so the transcript is available even if note generation fails. A successful run also writes `output/lecture.md`. Files with the same base name overwrite the corresponding outputs.

Both entry points mark the Markdown as incomplete if the provider reaches the 15,000-token output limit.

## Tests

Run `python -m unittest discover -s tests -v`. The tests use generated audio and stand-ins for the models and run offline.
