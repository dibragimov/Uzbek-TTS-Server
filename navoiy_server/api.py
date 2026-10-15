"""Speaches/OpenAI-compatible speech endpoint for Dograh.

Request lifecycle (the main change from RC4):
  handler: validate -> clean+normalize -> take a replica lease -> start a
           Session task -> wait for the FIRST audio -> return 200 + stream.
  Session task (owned by the server, not by the HTTP connection): opens the
           Triton stream, reads it to the end, closes it, releases the lease.
Because the session task always finishes on its own, a client that disconnects
at any moment (barge-in) can no longer leak a slot or a stream, and HTTP 200 is
only sent once audio exists, so failures before audio are real error codes.

Dograh contract (verified against its Speaches provider): it sends input,
model, voice, response_format="pcm" and speed; reads 24 kHz PCM16 mono.
"""
from __future__ import annotations

import asyncio
import hmac
import io
import json
import logging
import statistics
import time
import wave
from collections import deque
from dataclasses import dataclass, field

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

from . import config
from .replica_pool import NoCapacity, ReplicaPool
from .text_cleanup import clean, looks_russian
from .triton_stream import InferenceError

log = logging.getLogger("navoiy.api")
MODEL_IDS = ("navoiy-tts-v1", "tts-1")


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    input: str
    model: str = "navoiy-tts-v1"
    voice: str = "calm"
    response_format: str = "pcm"
    speed: float | None = None
    stream_format: str | None = None
    instructions: str | None = None


def openai_error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind}})


def pcm_to_wav(data: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(config.SAMPLE_RATE)
        f.writeframes(data)
    return buf.getvalue()


def _summary(values) -> dict | None:
    vals = sorted(values)
    if not vals:
        return None
    p95 = vals[min(len(vals) - 1, max(0, round(0.95 * len(vals)) - 1))]
    return {"n": len(vals), "p50": round(statistics.median(vals), 1), "p95": round(p95, 1),
            "max": round(vals[-1], 1)}


@dataclass
class Stats:
    counters: dict = field(default_factory=lambda: {
        "requests": 0, "completed": 0, "client_disconnected": 0, "failed": 0,
        "rejected_capacity": 0, "empty_after_cleanup": 0, "russian_text": 0})
    first_audio_ms: deque = field(default_factory=lambda: deque(maxlen=500))
    first_half_second_ms: deque = field(default_factory=lambda: deque(maxlen=500))
    queue_ms: deque = field(default_factory=lambda: deque(maxlen=500))
    normalize_ms: deque = field(default_factory=lambda: deque(maxlen=500))
    rtf: deque = field(default_factory=lambda: deque(maxlen=500))

    def snapshot(self) -> dict:
        return {"counters": dict(self.counters),
                "first_audio_ms": _summary(self.first_audio_ms),
                "first_half_second_ms": _summary(self.first_half_second_ms),
                "queue_ms": _summary(self.queue_ms),
                "normalize_ms": _summary(self.normalize_ms),
                "rtf": _summary(self.rtf)}


_END = object()


class Session:
    """Owns one Triton request from open to close, independent of the client."""

    def __init__(self, *, text, lease, stream_factory, settings, stats, arrival, meta):
        self.text = text
        self.lease = lease
        self.stream_factory = stream_factory
        self.settings = settings
        self.stats = stats
        self.arrival = arrival
        self.meta = meta
        loop = asyncio.get_running_loop()
        self.loop = loop
        self.first_audio: asyncio.Future = loop.create_future()
        # Mark exceptions as retrieved: nobody may be waiting if the client left.
        self.first_audio.add_done_callback(lambda f: f.cancelled() or f.exception())
        self.queue: asyncio.Queue = asyncio.Queue()
        self.gone = asyncio.Event()
        self.request_id = None

    def consumer_gone(self) -> None:
        self.gone.set()

    async def run(self) -> None:
        settings = self.settings
        stream = self.stream_factory(self.lease.slot.endpoint, self.loop)
        self.request_id = stream.request_id
        deadline = time.monotonic() + settings.request_timeout_s
        audio_bytes = 0
        first_at = half_at = None
        outcome = "failed"
        error_text = None
        cancel_close = True
        try:
            await asyncio.to_thread(stream.open, self.text)
            async for chunk in stream.chunks(deadline):
                audio_bytes += len(chunk.pcm)
                if first_at is None:
                    first_at = chunk.arrived_at
                if half_at is None and audio_bytes >= config.PIPECAT_FIRST_CHUNK_BYTES:
                    half_at = chunk.arrived_at
                if not self.first_audio.done():
                    self.first_audio.set_result(chunk.pcm)
                elif not self.gone.is_set():
                    self.queue.put_nowait(chunk.pcm)
                if self.gone.is_set() and settings.cancel_on_disconnect:
                    break  # experimental: cancel instead of draining
            cancel_close = not stream.finished
            outcome = "client_disconnected" if self.gone.is_set() else "completed"
        except BaseException as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            if not self.first_audio.done():
                self.first_audio.set_exception(
                    exc if isinstance(exc, InferenceError) else InferenceError(error_text))
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            self.queue.put_nowait(_END if outcome != "failed" else InferenceError(error_text))
            # Normal end -> plain close (the path RC4 validated, which lets the
            # vocoder clean its per-request cache); otherwise cancel the request.
            await asyncio.to_thread(stream.close, cancel_close)
            await self.lease.release()
            self._record(outcome, audio_bytes, first_at, half_at, error_text)

    def _record(self, outcome, audio_bytes, first_at, half_at, error_text) -> None:
        s = self.stats
        s.counters[outcome] = s.counters.get(outcome, 0) + 1
        wall_ms = (time.perf_counter() - self.arrival) * 1000
        audio_s = audio_bytes / config.BYTES_PER_SECOND
        row = {"event": "tts_request", "request_id": self.request_id, "replica": self.lease.slot.index,
               "outcome": outcome, "audio_s": round(audio_s, 2), "wall_ms": round(wall_ms, 1), **self.meta}
        if first_at is not None:
            row["first_audio_ms"] = round((first_at - self.arrival) * 1000, 1)
            s.first_audio_ms.append(row["first_audio_ms"])
        if half_at is not None:
            row["first_half_second_ms"] = round((half_at - self.arrival) * 1000, 1)
            s.first_half_second_ms.append(row["first_half_second_ms"])
        if outcome == "completed" and audio_s > 0:
            s.rtf.append(round((wall_ms / 1000) / audio_s, 3))
        if error_text:
            row["error"] = error_text[:300]
        (log.warning if outcome == "failed" else log.info)(json.dumps(row, ensure_ascii=False))

    async def body(self):
        """HTTP body for PCM: first chunk, then the rest as it arrives."""
        try:
            yield self.first_audio.result()
            while True:
                item = await self.queue.get()
                if item is _END:
                    return
                if isinstance(item, BaseException):
                    # Status is already 200: the stream ends early; logged by run().
                    return
                yield item
        finally:
            self.consumer_gone()

    async def collect(self) -> bytes:
        parts = [self.first_audio.result()]
        try:
            while True:
                item = await self.queue.get()
                if item is _END:
                    return b"".join(parts)
                if isinstance(item, BaseException):
                    raise HTTPException(502, f"TTS synthesis failed: {item}")
                parts.append(item)
        finally:
            self.consumer_gone()


def build_app(*, settings: config.Settings, pool: ReplicaPool, normalizer, stream_factory,
              ready_check=None, info: dict | None = None) -> FastAPI:
    app = FastAPI(title="Navoiy Uzbek TTS", version=config.VERSION,
                  docs_url=None, redoc_url=None, openapi_url=None)
    stats = Stats()
    tasks: set[asyncio.Task] = set()
    warned: set[str] = set()
    app.state.pool, app.state.stats, app.state.tasks = pool, stats, tasks

    def warn_once(key: str, message: str) -> None:
        if key not in warned:
            warned.add(key)
            log.warning(message)

    def require_bearer(authorization: str | None) -> None:
        scheme, _, supplied = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(settings.api_key.encode(),
                                                                 supplied.strip().encode()):
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        # 400, not FastAPI's 422: Pipecat's OpenAI client only turns
        # BadRequestError (400) into a clean ErrorFrame.
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        return openai_error(400, f"Invalid request: {where} {first.get('msg', '')}".strip())

    @app.exception_handler(HTTPException)
    async def _http(_: Request, exc: HTTPException):
        resp = openai_error(exc.status_code, str(exc.detail),
                            "invalid_request_error" if exc.status_code < 500 else "server_error")
        for k, v in (exc.headers or {}).items():
            resp.headers[k] = v
        return resp

    @app.get("/")
    async def root():
        return {"service": "navoiy-tts", "version": config.VERSION}

    @app.get("/health")
    async def health():
        healthy = sum(1 for s in pool.slots if s.healthy)
        body = {"status": "ok" if healthy else "down", "healthy_replicas": healthy,
                "replicas": len(pool.slots)}
        return JSONResponse(body, status_code=200 if healthy else 503)

    @app.get("/ready")
    async def ready(authorization: str | None = Header(default=None)):
        require_bearer(authorization)
        ok = await asyncio.to_thread(ready_check) if ready_check else True
        if not ok:
            raise HTTPException(503, "Triton models not ready")
        return {"status": "ready", "sample_rate_hz": config.SAMPLE_RATE, **(info or {})}

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(default=None)):
        require_bearer(authorization)
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "navoiy"}
                                           for m in MODEL_IDS]}

    @app.get("/metrics")
    async def metrics(authorization: str | None = Header(default=None)):
        require_bearer(authorization)
        return {"version": config.VERSION, "pool": pool.snapshot(), "requests": stats.snapshot(),
                "in_flight_sessions": len(tasks), **(info or {})}

    @app.post("/v1/audio/speech")
    async def speech(payload: SpeechRequest, authorization: str | None = Header(default=None)):
        arrival = time.perf_counter()
        require_bearer(authorization)
        stats.counters["requests"] += 1
        if payload.voice not in settings.accepted_voices:
            raise HTTPException(400, f"Unknown voice {payload.voice!r}; accepted: "
                                     f"{', '.join(sorted(settings.accepted_voices))}")
        if payload.response_format not in ("pcm", "wav"):
            raise HTTPException(400, "response_format must be 'pcm' (streamed) or 'wav' (buffered)")
        if payload.stream_format not in (None, "audio"):
            raise HTTPException(400, "Only stream_format='audio' is supported")
        if not payload.input.strip():
            raise HTTPException(400, "input is empty")
        if len(payload.input) > settings.max_input_chars:
            raise HTTPException(400, f"input longer than {settings.max_input_chars} characters")
        if payload.speed is not None and abs(payload.speed - 1.0) > 1e-6:
            warn_once(f"speed:{payload.speed}", f"speed={payload.speed} requested; Navoiy has no speed "
                      "control, synthesizing at 1.0 (set Speed to 1.0 in Dograh)")
        if payload.instructions:
            warn_once("instructions", "instructions ignored: the voice is fixed")
        if payload.model not in MODEL_IDS:
            warn_once(f"model:{payload.model}", f"model={payload.model!r} accepted as navoiy-tts-v1")

        t0 = time.perf_counter()
        cleaned = clean(payload.input)
        russian = looks_russian(cleaned)
        if russian:
            stats.counters["russian_text"] += 1
            warn_once("russian", "Russian (Cyrillic) text received: Navoiy is Uzbek-only and will read "
                      "it transliterated. Make the agent answer in Latin Uzbek.")
        try:
            normalized = (await asyncio.to_thread(normalizer.normalize, cleaned)) if cleaned else ""
        except Exception as exc:
            raise HTTPException(400, f"Text normalization failed: {type(exc).__name__}") from exc
        normalize_ms = (time.perf_counter() - t0) * 1000
        stats.normalize_ms.append(round(normalize_ms, 2))
        if not normalized.strip():
            # e.g. a sentence that was only an emoji: speak nothing, don't error.
            stats.counters["empty_after_cleanup"] += 1
            return Response(b"" if payload.response_format == "pcm" else pcm_to_wav(b""),
                            media_type="application/octet-stream" if payload.response_format == "pcm"
                            else "audio/wav")

        t1 = time.perf_counter()
        try:
            lease = await pool.acquire(settings.queue_timeout_s)
        except NoCapacity as exc:
            stats.counters["rejected_capacity"] += 1
            # Deliberately no Retry-After: the OpenAI SDK would honour it and
            # stall the call for that long before retrying.
            raise HTTPException(exc.status, exc.detail) from None
        queue_ms = (time.perf_counter() - t1) * 1000
        stats.queue_ms.append(round(queue_ms, 2))

        session = Session(text=normalized, lease=lease, stream_factory=stream_factory,
                          settings=settings, stats=stats, arrival=arrival,
                          meta={"chars": len(payload.input), "russian": russian,
                                "normalize_ms": round(normalize_ms, 1), "queue_ms": round(queue_ms, 1)})
        task = asyncio.create_task(session.run())
        tasks.add(task)
        task.add_done_callback(tasks.discard)

        try:
            await asyncio.shield(session.first_audio)
        except asyncio.CancelledError:
            session.consumer_gone()  # client left before audio; session drains and releases
            raise
        except Exception as exc:
            raise HTTPException(502, f"TTS synthesis failed: {exc}") from None
        first_ms = (time.perf_counter() - arrival) * 1000
        headers = {
            "X-Navoiy-Request-Id": session.request_id or "",
            "X-Navoiy-Replica": str(lease.slot.index),
            "X-Navoiy-Audio-Rate": str(config.SAMPLE_RATE),
            "Server-Timing": (f"normalize;dur={normalize_ms:.1f}, queue;dur={queue_ms:.1f}, "
                              f"first_audio;dur={first_ms:.1f}"),
            "Cache-Control": "no-store",
        }
        if payload.response_format == "pcm":
            return StreamingResponse(session.body(), media_type="application/octet-stream",
                                     headers=headers)
        try:
            pcm = await session.collect()
        except asyncio.CancelledError:
            session.consumer_gone()
            raise
        return Response(pcm_to_wav(pcm), media_type="audio/wav", headers=headers)

    return app
