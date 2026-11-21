# AI Notes

Turn a lecture recording into Markdown study notes. [faster-whisper](https://github.com/SYSTRAN/faster-whisper) transcribes the audio on your machine. Only the transcript and your optional lecture context are sent to Anthropic, which writes a summary, key points in lecture order, and action items.

## Project layout

| Path | What it does |
| --- | --- |
| [`api.py`](api.py) | Web server (FastAPI): routes, security middleware, background jobs. Start reading here. |
| [`transcription.py`](transcription.py) | Decodes audio with PyAV and transcribes it with faster-whisper. |
| [`notes.py`](notes.py) | Builds the Anthropic prompt and reads the Markdown notes from the response. |
| [`config.py`](config.py) | Every setting and limit, read from the environment or `.env`. |
| [`cli.py`](cli.py) | Command-line version: reads from `input/`, writes to `output/`. |
| [`frontend/`](frontend) | The browser studio: `index.html`, `studio.js` (upload and progress), `markdown.js` (safe Markdown preview), `studio.css`. |
| [`tests/`](tests) | Offline tests with generated audio and mocked models. |

How a recording flows through the app:

```
browser ──POST /jobs──▶ api.py ──▶ transcription.py (Whisper, local) ──▶ notes.py (Anthropic) ──▶ Markdown
   ▲                       │
   └──GET /jobs/{id}───────┘  progress, live transcript, then GET /jobs/{id}/notes
```

## Setup

Use Python 3.12 or newer. In Windows PowerShell, run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS or Linux, use `python3 -m venv .venv`, `source .venv/bin/activate` and `cp .env.example .env`.

Put your Anthropic API key in `.env`. The file is ignored by git. Restart the server after changing any setting.

The first transcription downloads the Whisper model. Transcription uses an NVIDIA GPU when one is detected. On Windows, install the CUDA runtime libraries if they are missing:

```powershell
python -m pip install nvidia-cublas-cu12==12.9.2.10 nvidia-cudnn-cu12==9.28.1.4 nvidia-cuda-nvrtc-cu12==12.9.86
```

To force CPU transcription, set `WHISPER_DEVICE=cpu`.

## Run the web studio

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/, choose a recording and a transcription model, and select **Generate notes**. The page shows the transcript as Whisper recognizes speech, then shows the notes with copy and download buttons. You can pause updates, stop the job, or refresh the page. The job id in the URL reconnects to the same job.

Finished jobs stay available for one hour. Restarting the server clears all jobs. Run **one** Uvicorn worker. Jobs and the loaded Whisper model live in that process's memory.

## Security

The server has **no login**. It is built for one person on their own computer.

- Keep `--host 127.0.0.1`. The server only answers requests addressed to `127.0.0.1`, `localhost` or `[::1]`, which blocks DNS-rebinding attacks.
- Browsers cannot submit uploads or stop jobs from another site (cross-site request forgery). The server rejects state-changing requests whose `Origin` is not the server itself.
- Pages get a strict Content Security Policy and cannot be framed. The notes preview builds DOM nodes from text and never inserts generated HTML.
- A job id is a random UUID and works like a password for that job's transcript, notes and stop button until the job expires.
- Uploads are limited in size, duration, number of fields, and concurrency. FFmpeg may only read the uploaded bytes and cannot follow references to other files or URLs.
- Lecture context and transcripts are XML-escaped in the prompt and the model is told to ignore instructions inside them.

Don't expose the server to a network as it is. To share it, put it behind a reverse proxy that adds authentication and HTTPS, and add your hostname to `LOCAL_HOSTS` in `api.py`.

## HTTP API

| Route | Purpose |
| --- | --- |
| `GET /models` | Transcription models the server accepts, and the default. |
| `POST /jobs` | Multipart upload with `file`, optional `context` (up to 4,000 characters), and optional `model`. Returns HTTP 202 with `{"id": ...}`. |
| `GET /jobs/{id}?after=0` | Stage (`queued`, `transcribing`, `generating`, `ready`, `failed`, `stopping`, `stopped`), new transcript segments, and a `cursor`. Pass the cursor as the next `after` to receive only new segments. |
| `POST /jobs/{id}/stop` | Stop a queued or running job. |
| `GET /jobs/{id}/notes` | The finished Markdown, once the stage is `ready`. |
| `POST /notes` | One-shot: upload `file` and optional `context`, and the response is the Markdown. The request stays open until the notes are done. |

Interactive docs are at http://127.0.0.1:8000/docs. A one-shot example:

```powershell
curl.exe --fail -F "file=@input/lecture.m4a" -F "context=Physics, electric dipoles" -o lecture.md http://127.0.0.1:8000/notes
```

Use `curl` instead of `curl.exe` on macOS or Linux.

Audio is identified from its contents. WAV, MP3, AAC, M4A, Ogg/Opus, WebM, and video files with an audio track all work. Up to four recordings are processed at once across both upload routes, and up to 64 jobs are kept. Only one transcription runs at a time, but the next recording can be transcribed while notes for the previous one are being written.

| HTTP status | Meaning |
| --- | --- |
| 403 | The request came from another website. |
| 413 | The upload or the decoded recording exceeds the configured limit. |
| 415 | The file is not multipart form data, or has no decodable audio track. |
| 422 | The file is missing or empty, the model is not offered, or Whisper detected no speech. |
| 429 | The recording queue or the job store is full. Try again later. |
| 502 | Anthropic failed to generate notes. Check the key and model access. |
| 503 | The API key is missing or the transcription backend is unavailable. |
| 504 | Anthropic timed out. |

Notes end with a warning if Anthropic reaches the 15,000-token output limit.

## Configuration

Set these in `.env`. See [`.env.example`](.env.example).

| Variable | Default | Purpose |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | (none) | Required for note generation. |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | A notes model available to your Anthropic account. |
| `WHISPER_MODEL` | `medium` | Default Whisper model size or a local model directory. |
| `WHISPER_DEVICE` | `auto` | `auto` uses CUDA when detected. Set `cpu` or `cuda` to choose. |
| `WHISPER_LANGUAGE` | `en` | Spoken language code, such as `bn` for Bengali. |
| `MAX_UPLOAD_MIB` | `256` | Maximum size of an upload request, in MiB. |
| `MAX_AUDIO_MINUTES` | `240` | Maximum recording length, in minutes. |

Fixed limits, such as concurrency and job retention, are at the bottom of [`config.py`](config.py).

## Command line

Put a recording in `input/` and run `python cli.py`. Pick the recording and type any lecture context. For `lecture.m4a`, the CLI writes `output/lecture.txt` first, so the transcript is kept even if note generation fails, and then `output/lecture.md`. Existing files with the same name are overwritten.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The tests run offline with generated audio and stand-ins for Whisper and Anthropic.
