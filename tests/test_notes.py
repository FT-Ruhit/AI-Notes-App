import asyncio
import io
import os
import struct
import unittest
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import chdir, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import quote

from fastapi.testclient import TestClient

import api
import main


def wav_bytes(seconds=1):
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x10" * (8000 * seconds))
    return audio.getvalue()


class NotesAPITests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"ANTHROPIC_API_KEY": "offline-test"}))
        self.whisper = self.enterContext(patch("faster_whisper.WhisperModel"))
        self.whisper.return_value.transcribe.return_value = ([SimpleNamespace(text="  Charge  ")], None)
        self.anthropic = self.enterContext(patch("api.AsyncAnthropic"))
        self.create_message = AsyncMock()
        self.anthropic.return_value.__aenter__.return_value.messages.create = self.create_message
        self.markdown = "## Summary\n\nবৈদ্যুতিক আধান ⭐"
        self.message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self.markdown)],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=8),
        )
        self.create_message.return_value = self.message
        self.client = self.enterContext(TestClient(api.app))

    def upload(self, filename="lecture.wav", content=None, context=""):
        return self.client.post(
            "/notes",
            files={"file": (filename, wav_bytes() if content is None else content, "application/octet-stream")},
            data={"context": context},
        )

    def test_download_contains_raw_utf8_notes_and_context_reaches_provider(self):
        response = self.upload("Lecture 1.WAV", context="  Physics  ")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, self.markdown.encode("utf-8"))
        self.assertEqual(response.headers["content-type"], "text/markdown; charset=utf-8")
        self.assertEqual(response.headers["content-disposition"], 'attachment; filename="Lecture 1.md"')
        waveform = self.whisper.return_value.transcribe.call_args.args[0]
        self.assertEqual(self.whisper.call_args.args[0], api.WHISPER_MODEL)
        self.assertEqual(waveform.shape, (16_000,))
        self.assertEqual(waveform.dtype.name, "float32")
        sent = self.create_message.call_args.kwargs["messages"][0]["content"]
        self.assertEqual(sent, "<context>Physics</context>\n\n<transcript>Charge</transcript>")

    def test_job_shows_segments_before_completion_and_resumes_from_cursor(self):
        release_segment, release_notes = Event(), Event()

        def segments():
            yield SimpleNamespace(text="First.")
            if not release_segment.wait(5):
                raise AssertionError("Transcription was blocked")
            yield SimpleNamespace(text="Second.")

        async def generate(**kwargs):
            if not await asyncio.to_thread(release_notes.wait, 5):
                raise AssertionError("Note generation was blocked")
            return self.message

        self.whisper.return_value.transcribe.return_value = (segments(), None)
        self.create_message.side_effect = generate
        self.assertEqual(self.client.get("/").status_code, 200)
        models = self.client.get("/models").json()
        self.assertEqual(models["default"], api.WHISPER_MODEL)
        selected = next(name for name in models["options"] if name != models["default"])
        response = self.client.post("/jobs", files={"file": ("lecture.wav", wav_bytes())},
                                    data={"model": selected, "context": ""})
        self.assertEqual(response.status_code, 202)
        url = "/jobs/" + response.json()["id"]

        def until(stage, cursor, after=0):
            deadline = monotonic() + 4
            while monotonic() < deadline:
                status = self.client.get(url, params={"after": after}).json()
                if status["stage"] == stage and status["cursor"] == cursor:
                    return status["segments"]
                sleep(.01)
            self.fail(f"Job did not reach {stage}: {status}")

        try:
            self.assertEqual(until("transcribing", 1), ["First."])
            self.assertEqual(self.client.get(url + "/notes").status_code, 409)
            release_segment.set()
            self.assertEqual(until("generating", 2, after=1), ["Second."])
        finally:
            release_segment.set()
            release_notes.set()
        self.assertEqual(until("ready", 2), ["First.", "Second."])
        self.assertEqual(self.whisper.call_args.args[0], selected)
        self.assertEqual(self.client.get(url).json()["filename"], "lecture.wav")
        prompt = self.create_message.call_args.kwargs["messages"][0]["content"]
        self.assertEqual(prompt, "<transcript>First. Second.</transcript>")
        download = self.client.get(url + "/notes")
        self.assertEqual(download.text, self.markdown)
        self.assertIn("lecture.md", download.headers["content-disposition"])
        self.create_message.assert_called_once()

    def start_job(self):
        response = self.client.post("/jobs", files={"file": ("lecture.wav", wav_bytes())})
        self.assertEqual(response.status_code, 202)
        return "/jobs/" + response.json()["id"]

    def wait_for_stage(self, url, stage):
        deadline = monotonic() + 4
        while monotonic() < deadline:
            state = self.client.get(url).json()
            if state["stage"] == stage:
                return state
            sleep(.01)
        self.fail(f"Expected {stage}, got {state}")

    def test_stop_transcription_and_queued_job(self):
        processing, release = Event(), Event()

        def segments():
            yield SimpleNamespace(text="First.")
            processing.set()
            release.wait(5)
            yield SimpleNamespace(text="Discarded.")
            self.fail("Inference continued after cancellation")

        self.whisper.return_value.transcribe.return_value = (segments(), None)
        first = self.start_job()
        try:
            self.assertTrue(processing.wait(3))
            second = self.start_job()
            self.assertEqual(self.client.get(second).json()["stage"], "queued")
            self.client.post(second + "/stop")
            self.wait_for_stage(second, "stopped")
            self.assertEqual(self.client.post(first + "/stop").json()["stage"], "stopping")
        finally:
            release.set()
        self.assertEqual(self.wait_for_stage(first, "stopped")["segments"], ["First."])
        self.assertEqual(self.client.post(first + "/stop").json()["stage"], "stopped")
        self.assertEqual(self.client.get(first + "/notes").status_code, 409)
        self.create_message.assert_not_awaited()
        self.whisper.return_value.transcribe.assert_called_once()

    def test_stop_cancels_inflight_note_request(self):
        started, disconnected = Event(), Event()

        async def generate(**kwargs):
            started.set()
            try:
                await asyncio.sleep(5)
            finally:
                disconnected.set()
            return self.message

        self.create_message.side_effect = generate
        job = self.start_job()
        self.assertTrue(started.wait(3))
        self.client.post(job + "/stop")
        self.wait_for_stage(job, "stopped")
        self.assertTrue(disconnected.is_set())
        self.assertEqual(self.client.get(job + "/notes").status_code, 409)

    def test_job_uses_default_and_rejects_unknown_model(self):
        files = {"file": ("lecture.wav", wav_bytes())}
        rejected = self.client.post("/jobs", files=files, data={"model": "unlisted"})
        self.assertEqual(rejected.status_code, 422)
        self.whisper.assert_not_called()
        queued = self.client.post("/jobs", files=files)
        self.assertEqual(queued.status_code, 202)
        self.assertEqual(self.client.get("/jobs/" + queued.json()["id"]).json()["model"], api.WHISPER_MODEL)

    def test_studio_assets_are_served(self):
        page = self.client.get("/")
        for name in ("studio.css", "studio.js", "markdown.js"):
            path = "/frontend/" + name
            self.assertIn(path, page.text)
            self.assertEqual(self.client.get(path).status_code, 200)

    def test_unicode_filename_uses_only_the_basename(self):
        name = "পাঠ"
        header = api.download_header(f"C:\\recordings\\{name}.aac")
        self.assertIn("filename*=UTF-8''" + quote(name + ".md"), header)
        self.assertNotIn("recordings", header)

    def test_stereo_downmix_preserves_the_original_signal_level(self):
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setparams((2, 2, 16_000, 0, "NONE", "not compressed"))
            wav.writeframes(struct.pack("<hh", 8192, 16384) * 320)
        self.assertEqual(self.upload(content=audio.getvalue()).status_code, 200)
        waveform = self.whisper.return_value.transcribe.call_args.args[0]
        self.assertEqual(waveform.shape, (320,))
        self.assertTrue((waveform == 0.375).all())

    def test_missing_empty_and_unreadable_files_skip_inference(self):
        self.assertEqual(self.client.post("/notes").status_code, 422)
        self.assertEqual(self.upload(content=b"").status_code, 422)
        self.assertEqual(self.upload(content=b"not an audio recording").status_code, 415)
        self.whisper.assert_not_called()
        self.create_message.assert_not_called()

    def test_upload_and_duration_limits_skip_inference(self):
        response = self.client.post(
            "/notes", files={"file": ("lecture.wav", wav_bytes())},
            headers={"content-length": str(api.MAX_UPLOAD_BYTES + 1)},
        )
        self.assertEqual(response.status_code, 413)
        with patch("api.MAX_AUDIO_MINUTES", 1):
            self.assertEqual(self.upload(content=wav_bytes(61)).status_code, 413)
        self.whisper.assert_not_called()

    def test_silence_skips_note_generation(self):
        self.whisper.return_value.transcribe.return_value = ([], None)
        response = self.upload()
        self.assertEqual(response.status_code, 422)
        self.assertIn("speech", response.json()["detail"])
        self.create_message.assert_not_called()

    def test_provider_returning_no_notes_is_an_error(self):
        self.message.content.clear()
        response = self.upload()
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("content-disposition", response.headers)

    def test_missing_api_key_skips_transcription(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}), TestClient(api.app) as client:
            response = client.post("/notes", files={"file": ("lecture.wav", wav_bytes())})
        self.assertEqual(response.status_code, 503)
        self.whisper.assert_not_called()

    def test_second_upload_uses_same_model_while_first_waits_for_notes(self):
        first_in_notes, release_first = Event(), Event()

        async def make_message(**kwargs):
            if "<context>first</context>" in kwargs["messages"][0]["content"]:
                first_in_notes.set()
                if not await asyncio.to_thread(release_first.wait, 10):
                    raise AssertionError("First note request timed out")
            return self.message

        self.create_message.side_effect = make_message
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.upload, context="first")
            try:
                self.assertTrue(first_in_notes.wait(10))
                second = pool.submit(self.upload, context="second")
                self.assertEqual(second.result(timeout=5).status_code, 200)
            finally:
                release_first.set()
            self.assertEqual(first.result(timeout=5).status_code, 200)
        self.whisper.assert_called_once()
        self.anthropic.assert_called_once()


class CLITests(unittest.TestCase):
    def test_cli_keeps_the_transcript_if_note_generation_fails(self):
        with TemporaryDirectory() as tmp, chdir(tmp):
            folder = Path("input")
            folder.mkdir()
            (folder / "lecture.WAV").write_bytes(wav_bytes())
            (folder / "unreadable.mp3").write_text("not audio", encoding="utf-8")
            self.assertEqual(main.find_recordings(folder), [folder / "lecture.WAV"])
            with (
                patch("main.Transcriber") as transcriber,
                patch("main.make_notes", side_effect=RuntimeError("offline")),
                patch("builtins.input", side_effect=["0", "Physics"]),
                redirect_stdout(io.StringIO()),
            ):
                transcriber.return_value.transcribe.return_value = "Charge"
                with self.assertRaisesRegex(RuntimeError, "offline"):
                    main.main()
            self.assertEqual(Path("output/lecture.txt").read_text(encoding="utf-8"), "Charge")
            self.assertFalse(Path("output/lecture.md").exists())
