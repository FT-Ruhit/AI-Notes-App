"""Web server for the AI Notes lecture studio.

    GET  /                   Studio page (frontend/index.html)
    GET  /models             Transcription models the studio can offer
    POST /notes              Upload a recording and wait for the Markdown (simple API clients)
    POST /jobs               Upload a recording and get a job id right away (used by the studio)
    GET  /jobs/{id}          Job stage and new transcript segments
    POST /jobs/{id}/stop     Stop a queued or running job
    GET  /jobs/{id}/notes    Download finished notes

Security model: there are no user accounts. The server is meant for one person on
loopback (127.0.0.1); the middleware below rejects other hosts and cross-site browser
requests. Job ids are random UUIDs (122 random bits), so a job URL acts as the password
for that job until it expires.

Run a single worker so every request shares one Whisper model and one job store:
    python -m uvicorn api:app --host 127.0.0.1 --port 8000
"""
import asyncio
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from functools import lru_cache, partial
from pathlib import Path, PureWindowsPath
from shutil import copyfileobj
from tempfile import TemporaryFile
from threading import Event
from time import monotonic
from typing import Annotated, BinaryIO, Literal
from urllib.parse import quote
from uuid import uuid4

from anthropic import APIError, APITimeoutError, AsyncAnthropic
from anyio import CapacityLimiter, WouldBlock
from fastapi import APIRouter, FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from faster_whisper import available_models
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from config import (
    JOB_RETENTION_SECONDS,
    MAX_ACTIVE_RECORDINGS,
    MAX_AUDIO_MINUTES,
    MAX_CONTEXT_CHARS,
    MAX_RETAINED_JOBS,
    MAX_UPLOAD_BYTES,
    WHISPER_MODEL,
)
from notes import EmptyNotesError, build_note_request, read_notes
from transcription import AudioTooLongError, InvalidAudioError, Transcriber

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).with_name("frontend")
LOCAL_HOSTS = ["127.0.0.1", "localhost", "[::1]"]
MODEL_OPTIONS = list(dict.fromkeys([*available_models(), WHISPER_MODEL]))

LectureContext = Annotated[str, Form(max_length=MAX_CONTEXT_CHARS)]


# --------------------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------------------

@dataclass
class Studio:
    """Everything the routes share. Created at startup and available as `request.state.studio`."""

    notes_client: AsyncAnthropic | None    # None when ANTHROPIC_API_KEY is not set.
    workers: asyncio.TaskGroup             # Runs background jobs.
    jobs: dict[str, "Job"] = field(default_factory=dict)
    # One transcription at a time, so a single GPU is never shared by two Whisper runs.
    transcription_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Places for recordings being processed, shared by POST /notes and POST /jobs.
    recording_slots: CapacityLimiter = field(default_factory=lambda: CapacityLimiter(MAX_ACTIVE_RECORDINGS))
    # Keeps the most recently used Whisper model loaded and swaps it when another is requested.
    transcriber_for: Callable[[str], Transcriber] = field(default_factory=lambda: lru_cache(maxsize=1)(Transcriber))


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    async with (AsyncAnthropic(api_key=api_key) if api_key else nullcontext()) as notes_client:
        async with asyncio.TaskGroup() as workers:
            studio = Studio(notes_client, workers)
            try:
                yield {"studio": studio}
            finally:
                # Stop running jobs so shutdown does not wait for long recordings to finish.
                for job in studio.jobs.values():
                    job.request_stop()


# --------------------------------------------------------------------------------------
# Security
# --------------------------------------------------------------------------------------

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}
STRICT_CSP = "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
# FastAPI's interactive docs load scripts from a CDN, so they only get the anti-framing rule.
DOCS_CSP = "frame-ancestors 'none'"
DOCS_PATHS = {"/docs", "/docs/oauth2-redirect", "/redoc"}


def is_cross_site(request: Request) -> bool:
    own_origin = f"{request.url.scheme}://{request.headers['host']}"
    return (any(origin != own_origin for origin in request.headers.getlist("origin"))
            or request.headers.get("sec-fetch-site") == "cross-site")


async def local_only(request: Request, call_next):
    """Reject cross-site browser requests that change state (CSRF) and add security headers.

    TrustedHostMiddleware runs first and rejects unknown Host headers (DNS rebinding),
    so the Host header can be trusted when building our own origin here.
    """
    if request.method not in SAFE_METHODS and is_cross_site(request):
        return JSONResponse({"detail": "Use the studio on this server to submit requests."}, status_code=403)
    response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    response.headers["Content-Security-Policy"] = DOCS_CSP if request.url.path in DOCS_PATHS else STRICT_CSP
    return response


app = FastAPI(
    title="AI Notes",
    lifespan=lifespan,
    middleware=[  # Listed outermost first.
        Middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_HOSTS, www_redirect=False),
        Middleware(BaseHTTPMiddleware, dispatch=local_only),
        Middleware(RequestBodyLimitMiddleware, max_body_size=MAX_UPLOAD_BYTES),
    ],
)
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


# --------------------------------------------------------------------------------------
# Upload handling shared by POST /notes and POST /jobs
# --------------------------------------------------------------------------------------

class RecordingSlot:
    """A reserved place in the recording queue.

    The upload request releases it when it ends, unless a background job takes it over
    by setting `owned_by_job`. The job then releases it when it finishes.
    """

    def __init__(self, limiter: CapacityLimiter):
        try:
            limiter.acquire_on_behalf_of_nowait(self)
        except WouldBlock:
            raise HTTPException(429, "Recording queue is full. Try again later.") from None
        self._limiter = limiter
        self.owned_by_job = False

    def release(self):
        self._limiter.release_on_behalf_of(self)


class RecordingRoute(APIRoute):
    """Route class for the upload endpoints.

    It checks the content type and reserves a RecordingSlot *before* reading the body,
    so a busy server turns uploads away without receiving them. It then parses the form
    with tight limits: one file and two small text fields (`context` and `model`).
    FastAPI reuses the parsed form to fill in the endpoint's parameters.
    """

    def get_route_handler(self):
        handle = super().get_route_handler()

        async def handle_recording(request: Request):
            media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
            if media_type and media_type != "multipart/form-data":
                raise HTTPException(415, "Upload the recording as multipart/form-data.")
            slot = request.state.recording_slot = RecordingSlot(request.state.studio.recording_slots)
            try:
                text_part_bytes = MAX_CONTEXT_CHARS * 4  # Worst case for UTF-8.
                async with request.form(max_files=1, max_fields=2, max_part_size=text_part_bytes):
                    return await handle(request)
            finally:
                if not slot.owned_by_job:
                    slot.release()

        return handle_recording


uploads = APIRouter(route_class=RecordingRoute)


class MarkdownResponse(PlainTextResponse):
    media_type = "text/markdown"


def attachment_header(filename: str | None) -> str:
    """Content-Disposition for `<recording name>.md`, safe against header injection and paths."""
    stem = PureWindowsPath(filename or "notes").stem  # Also strips "C:\\..." and "../" prefixes.
    stem = "".join(c for c in stem if c.isprintable() and c not in '<>:"/\\|?*').strip(" .")
    name = f"{stem or 'notes'}.md"
    if name.isascii():
        return f'attachment; filename="{name}"'
    return f"attachment; filename=\"notes.md\"; filename*=UTF-8''{quote(name, safe='')}"


def check_upload(studio: Studio, file: UploadFile):
    if file.size == 0:
        raise HTTPException(422, "The uploaded recording is empty.")
    if studio.notes_client is None:
        raise HTTPException(503, "Set ANTHROPIC_API_KEY on the server to enable note generation.")


# --------------------------------------------------------------------------------------
# Recording -> transcript -> notes
# --------------------------------------------------------------------------------------

async def process_recording(studio: Studio, recording: BinaryIO, context: str, job: "Job | None" = None) -> str:
    """Transcribe a recording, then turn the transcript into Markdown notes.

    `recording` is closed as soon as transcription ends, freeing its temporary file while
    notes are generated. Failures are raised as HTTPException with a user-facing message.
    When `job` is given, its stage and live transcript are updated and it can be stopped.
    """
    model = job.model if job else WHISPER_MODEL
    options = {"max_seconds": MAX_AUDIO_MINUTES * 60}
    if job is not None:
        loop = asyncio.get_running_loop()
        options["check_stop"] = job.check_stop
        # Whisper runs in a worker thread; hand each segment back to the event loop thread.
        options["on_segment"] = partial(loop.call_soon_threadsafe, job.add_segment)
    try:
        # Recordings wait here for their turn at transcription. Notes for an earlier
        # recording can be generated while the next one is transcribed.
        async with studio.transcription_lock:
            if job is not None:
                job.check_stop()
                job.stage = "transcribing"
            transcribe = studio.transcriber_for(model).transcribe
            transcript = await run_in_threadpool(transcribe, recording, **options)
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
    if job is not None:
        job.check_stop()
        job.stage = "generating"
    try:
        response = await studio.notes_client.messages.create(**build_note_request(transcript, context))
        return read_notes(response)
    except APITimeoutError as exc:
        raise HTTPException(504, "The note provider timed out.") from exc
    except (APIError, EmptyNotesError) as exc:
        logger.warning("Note generation failed (%s)", type(exc).__name__)
        raise HTTPException(502, "The note provider could not generate notes for this recording.") from exc


# --------------------------------------------------------------------------------------
# Background jobs
# --------------------------------------------------------------------------------------

JobStage = Literal["queued", "transcribing", "generating", "ready", "failed", "stopping", "stopped"]
ACTIVE_STAGES = {"queued", "transcribing", "generating"}


class JobStopped(Exception):
    """Raised inside a job after the user asked it to stop."""


@dataclass
class Job:
    """A recording processed in the background while the studio polls its progress."""

    filename: str | None
    model: str
    stage: JobStage = "queued"
    segments: list[str] = field(default_factory=list)  # Transcript pieces, in order, as Whisper finds them.
    notes: str = ""
    error: str | None = None
    started: float = field(default_factory=monotonic)
    finished: float | None = None
    # A threading.Event because the Whisper worker thread checks it too.
    stop_requested: Event = field(default_factory=Event)
    task: asyncio.Task | None = None

    def check_stop(self):
        if self.stop_requested.is_set():
            raise JobStopped()

    def add_segment(self, text: str):
        if not self.stop_requested.is_set():
            self.segments.append(text)

    def request_stop(self):
        if self.stage not in ACTIVE_STAGES:
            return
        transcribing = self.stage == "transcribing"
        self.stop_requested.set()
        self.stage = "stopping"
        # A Whisper thread cannot be cancelled. It holds the recording and the transcription
        # lock until its next check_stop(), so let it exit on its own. Otherwise cancel now.
        if self.task is not None and not transcribing:
            self.task.cancel()

    def status(self, after: int) -> dict:
        return {
            "stage": self.stage,
            "segments": self.segments[after:],
            "cursor": len(self.segments),  # Pass back as `after` to receive only new segments.
            "filename": self.filename,
            "model": self.model,
            "elapsed_seconds": int((self.finished or monotonic()) - self.started),
            "error": self.error,
        }


async def run_job(studio: Studio, job_id: str, recording: BinaryIO, context: str, slot: RecordingSlot):
    job = studio.jobs[job_id]
    job.task = asyncio.current_task()
    try:
        with recording:  # Closes the file even if the job is stopped before transcription.
            job.check_stop()
            job.notes = await process_recording(studio, recording, context, job)
        job.stage = "ready"
    except (JobStopped, asyncio.CancelledError):
        if not job.stop_requested.is_set():
            raise  # Not a user stop, so let the cancellation propagate.
    except HTTPException as exc:
        job.stage, job.error = "failed", exc.detail
    except Exception:
        logger.exception("Background recording job failed")
        job.stage, job.error = "failed", "The server could not finish this recording."
    finally:
        if job.stop_requested.is_set():
            job.stage, job.error, job.notes = "stopped", None, ""
        job.task = None
        job.finished = monotonic()
        slot.release()
        asyncio.get_running_loop().call_later(JOB_RETENTION_SECONDS, studio.jobs.pop, job_id, None)


def get_job(request: Request, job_id: str) -> Job:
    job = request.state.studio.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "This job expired or the server restarted. Upload the recording again.")
    return job


# --------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def studio_page():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/models")
async def models():
    return {"options": MODEL_OPTIONS, "default": WHISPER_MODEL}


@uploads.post(
    "/notes",
    response_class=MarkdownResponse,
    summary="Turn a recording into Markdown study notes",
    responses={
        413: {"description": "Upload or decoded recording exceeds its limit"},
        415: {"description": "File contains no decodable audio"},
        422: {"description": "Missing file, empty recording, or no speech detected"},
        429: {"description": "Recording queue is full"},
        502: {"description": "Note generation failed"},
        503: {"description": "Transcription or note generation is unavailable"},
        504: {"description": "Note generation timed out"},
    },
)
async def create_notes(request: Request, file: UploadFile, context: LectureContext = "") -> MarkdownResponse:
    studio = request.state.studio
    check_upload(studio, file)
    notes = await process_recording(studio, file.file, context)
    return MarkdownResponse(notes, headers={"Content-Disposition": attachment_header(file.filename)})


@uploads.post("/jobs", status_code=202, summary="Queue a recording for background processing")
async def queue_job(request: Request, file: UploadFile, context: LectureContext = "",
                    model: Annotated[str, Form()] = WHISPER_MODEL):
    studio = request.state.studio
    check_upload(studio, file)
    if model not in MODEL_OPTIONS:
        raise HTTPException(422, "Choose a supported transcription model.")
    if len(studio.jobs) >= MAX_RETAINED_JOBS:
        raise HTTPException(429, "Job storage is full. Try again after older jobs expire.")

    job_id = uuid4().hex
    studio.jobs[job_id] = Job(file.filename, model)  # Claim storage before the copy below yields.
    # Starlette deletes the upload when this request ends, so the job needs its own copy.
    recording = TemporaryFile()
    try:
        await run_in_threadpool(copyfileobj, file.file, recording)
        recording.seek(0)
        slot = request.state.recording_slot
        studio.workers.create_task(run_job(studio, job_id, recording, context, slot))
        slot.owned_by_job = True
    except BaseException:
        recording.close()
        studio.jobs.pop(job_id, None)
        raise
    return {"id": job_id}


app.include_router(uploads)


@app.get("/jobs/{job_id}", summary="Job progress and new transcript segments")
async def job_status(request: Request, job_id: str, after: Annotated[int, Query(ge=0)] = 0):
    return JSONResponse(get_job(request, job_id).status(after), headers={"Cache-Control": "no-store"})


@app.post("/jobs/{job_id}/stop", summary="Stop a queued or running job")
async def stop_job(request: Request, job_id: str):
    job = get_job(request, job_id)
    job.request_stop()
    return {"stage": job.stage}


@app.get("/jobs/{job_id}/notes", response_class=MarkdownResponse, summary="Download finished notes")
async def job_notes(request: Request, job_id: str):
    job = get_job(request, job_id)
    if job.stage in ("stopping", "stopped"):
        raise HTTPException(409, "Generation was stopped.")
    if job.stage != "ready":
        raise HTTPException(409, job.error or "Notes are still processing.")
    return MarkdownResponse(job.notes, headers={"Content-Disposition": attachment_header(job.filename),
                                                "Cache-Control": "no-store"})
