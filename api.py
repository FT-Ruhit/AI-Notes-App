import asyncio
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache, partial
from shutil import copyfileobj
from tempfile import TemporaryFile
from threading import Event
from time import monotonic
from uuid import uuid4
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path, PureWindowsPath
from typing import Annotated
from urllib.parse import quote

from anthropic import AsyncAnthropic, APIError, APITimeoutError
from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import available_models
from starlette.middleware.body_limit import RequestBodyLimitMiddleware

from config import WHISPER_MODEL
from Make_note import EmptyNotesError, note_request, read_notes
from Text_exreact import AudioTooLongError, InvalidAudioError, Transcriber

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MIB", "256")) * 1024**2
MAX_AUDIO_MINUTES = int(os.getenv("MAX_AUDIO_MINUTES", "240"))
MODEL_OPTIONS = list(dict.fromkeys([*available_models(), WHISPER_MODEL]))


@asynccontextmanager
async def lifespan(app: FastAPI):
    key = os.getenv("ANTHROPIC_API_KEY")
    async with (AsyncAnthropic(api_key=key) if key else nullcontext()) as client:
        async with asyncio.TaskGroup() as workers:
            yield {
                "notes_client": client,
                "transcriber_for": lru_cache(maxsize=1)(Transcriber),
                "transcription_slot": asyncio.Lock(),
                "jobs": {},
                "workers": workers,
            }


app = FastAPI(title="AI Notes", lifespan=lifespan)
app.mount("/frontend", StaticFiles(directory=Path(__file__).with_name("frontend")), name="frontend")
app.add_middleware(RequestBodyLimitMiddleware, max_body_size=MAX_UPLOAD_BYTES)


class MarkdownResponse(PlainTextResponse):
    media_type = "text/markdown"


def download_header(filename: str | None) -> str:
    stem = PureWindowsPath(filename or "notes").stem
    stem = "".join(c for c in stem if c.isprintable() and c not in '<>:"/\\|?*').strip(" .")
    name = f"{stem or 'notes'}.md"
    if name.isascii():
        return f'attachment; filename="{name}"'
    return f"attachment; filename=\"notes.md\"; filename*=UTF-8''{quote(name, safe='')}"


@app.post(
    "/notes",
    response_class=MarkdownResponse,
    summary="Turn a recording into Markdown study notes",
    responses={
        413: {"description": "Upload or decoded recording exceeds its limit"},
        415: {"description": "File contains no decodable audio"},
        422: {"description": "Missing file, empty recording, or no speech detected"},
        502: {"description": "Note generation failed"},
        503: {"description": "Transcription or note generation is unavailable"},
        504: {"description": "Note generation timed out"},
    },
)
async def create_notes(request: Request, file: UploadFile, context: Annotated[str, Form()] = "") -> MarkdownResponse:
    validate_upload(request, file)
    try:
        notes = await process_recording(request.state, file.file, context)
    finally:
        await file.close()
    return MarkdownResponse(notes, headers={"Content-Disposition": download_header(file.filename)})


def validate_upload(request, file):
    if file.size == 0:
        raise HTTPException(422, "The uploaded recording is empty.")
    if request.state.notes_client is None:
        raise HTTPException(503, "Set ANTHROPIC_API_KEY on the server to enable note generation.")


async def process_recording(state, recording, context, job=None):
    loop = asyncio.get_running_loop()
    try:
        # Queue only decoding and inference. Other requests can generate notes concurrently.
        async with state.transcription_slot:
            options = {"max_seconds": MAX_AUDIO_MINUTES * 60}
            if job is not None:
                job.check_stop()
                job.stage = "transcribing"
                options["check_stop"] = job.check_stop
                options["on_segment"] = partial(loop.call_soon_threadsafe, job.add_segment)
            transcript = await run_in_threadpool(
                state.transcriber_for(job.model if job else WHISPER_MODEL).transcribe,
                recording, **options,
            )
    except AudioTooLongError as exc:
        raise HTTPException(413, f"Recordings can be up to {MAX_AUDIO_MINUTES} minutes long.") from exc
    except InvalidAudioError as exc:
        raise HTTPException(415, str(exc)) from exc
    except (OSError, RuntimeError, ValueError) as exc:
        logger.exception("Transcription failed")
        raise HTTPException(503, "Transcription failed. Check the server's Whisper configuration.") from exc
    finally:
        recording.close()

    if not transcript:
        raise HTTPException(422, "No speech was detected in the recording.")
    try:
        if job is not None:
            job.check_stop()
            job.stage = "generating"
        response = await state.notes_client.messages.create(**note_request(transcript, context))
        return read_notes(response)
    except APITimeoutError as exc:
        raise HTTPException(504, "The note provider timed out.") from exc
    except (APIError, EmptyNotesError) as exc:
        logger.warning("Note generation failed (%s)", type(exc).__name__)
        raise HTTPException(502, "The note provider could not generate notes for this recording.") from exc


class JobStopped(Exception):
    pass


@dataclass
class Job:
    filename: str | None
    model: str
    started: float = field(default_factory=monotonic)
    finished: float | None = None
    stage: str = "queued"
    segments: list[str] = field(default_factory=list)
    notes: str = ""
    error: str | None = None
    stop: Event = field(default_factory=Event)
    task: asyncio.Task | None = None

    def check_stop(self):
        if self.stop.is_set():
            raise JobStopped()

    def add_segment(self, segment):
        if not self.stop.is_set():
            self.segments.append(segment)


@app.get("/", include_in_schema=False)
async def upload_page():
    return FileResponse(Path(__file__).with_name("upload.html"))


@app.get("/models")
async def models():
    return {"options": MODEL_OPTIONS, "default": WHISPER_MODEL}


async def finish_job(state, job_id, recording, context):
    job = state.jobs[job_id]
    try:
        with recording:
            job.task = asyncio.current_task()
            job.check_stop()
            job.notes = await process_recording(state, recording, context, job)
        job.stage = "ready"
    except (JobStopped, asyncio.CancelledError):
        if not job.stop.is_set():
            raise
    except HTTPException as exc:
        job.error = exc.detail
        job.stage = "failed"
    except Exception:
        logger.exception("Background recording job failed")
        job.error = "The server could not finish this recording."
        job.stage = "failed"
    finally:
        if job.stop.is_set():
            job.stage = "stopped"
            job.error = None
            job.notes = ""
        job.task = None
        job.finished = monotonic()
        asyncio.get_running_loop().call_later(3600, state.jobs.pop, job_id, None)


@app.post("/jobs", status_code=202)
async def queue_job(request: Request, file: UploadFile, context: Annotated[str, Form()] = "",
                    model: Annotated[str, Form()] = WHISPER_MODEL):
    validate_upload(request, file)
    if model not in MODEL_OPTIONS:
        raise HTTPException(422, "Choose a supported transcription model.")
    state = request.state
    job_id = uuid4().hex
    recording = TemporaryFile()
    try:
        await run_in_threadpool(copyfileobj, file.file, recording)
        recording.seek(0)
        state.jobs[job_id] = Job(file.filename, model)
        state.workers.create_task(finish_job(state, job_id, recording, context))
    except BaseException:
        recording.close()
        state.jobs.pop(job_id, None)
        raise
    return {"id": job_id}


def get_job(request, job_id):
    job = request.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "This job expired or the server restarted. Upload the recording again.")
    return job


@app.post("/jobs/{job_id}/stop")
async def stop_job(request: Request, job_id: str):
    job = get_job(request, job_id)
    if job.stage in ("queued", "transcribing", "generating"):
        previous_stage = job.stage
        job.stop.set()
        job.stage = "stopping"
        # A running Whisper thread owns the recording and model lock until its next checkpoint.
        if job.task is not None and previous_stage != "transcribing":
            job.task.cancel()
    return {"stage": job.stage}


@app.get("/jobs/{job_id}")
async def job_status(request: Request, job_id: str, after: Annotated[int, Query(ge=0)] = 0):
    job = get_job(request, job_id)
    return JSONResponse({"stage": job.stage, "segments": job.segments[after:],
                         "filename": job.filename, "model": job.model,
                         "elapsed_seconds": int((job.finished or monotonic()) - job.started),
                         "cursor": len(job.segments), "error": job.error},
                        headers={"Cache-Control": "no-store"})


@app.get("/jobs/{job_id}/notes", response_class=MarkdownResponse)
async def job_download(request: Request, job_id: str):
    job = get_job(request, job_id)
    if job.stage in ("stopping", "stopped"):
        raise HTTPException(409, "Generation was stopped.")
    if job.stage != "ready":
        raise HTTPException(409, job.error or "Notes are still processing.")
    return MarkdownResponse(job.notes, headers={"Content-Disposition": download_header(job.filename),
                                                "Cache-Control": "no-store"})
