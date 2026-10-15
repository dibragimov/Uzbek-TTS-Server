"""One isolated Triton bidirectional gRPC stream per speech request.

Changes from RC4's triton_stream.py:
- gRPC callbacks hand results to the event loop with call_soon_threadsafe, so a
  waiting request no longer occupies a worker thread polling a blocking queue.
- The hand-off queue is unbounded. RC4's 128-slot queue with put_nowait could
  silently drop audio, or the end marker, if it ever filled.
Unchanged: unique request id per stream, request-id check, float->PCM16
conversion in the callback thread, final-response detection.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np


class InferenceError(RuntimeError):
    pass


@dataclass
class TritonChunk:
    pcm: bytes
    arrived_at: float


def pcm16(samples) -> bytes:
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not x.size or not np.isfinite(x).all():
        raise InferenceError("Empty or invalid waveform from Triton")
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class SpeechStream:
    def __init__(self, endpoint: str, loop: asyncio.AbstractEventLoop, grpc_module=None):
        self.endpoint = endpoint
        self.request_id = str(uuid.uuid4())
        self._loop = loop
        self._queue: asyncio.Queue = asyncio.Queue()
        self._grpc = grpc_module
        self.client = None
        self.finished = False
        self._closed = False

    # --- called on the gRPC callback thread ---------------------------------
    def _push(self, kind: str, value) -> None:
        try:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, (kind, value))
        except RuntimeError:
            pass  # event loop already closed (shutdown); nothing to deliver to

    def _on_response(self, result=None, error=None) -> None:
        try:
            if error is not None:
                self._push("error", str(error))
                return
            if result is None:
                self._push("error", "Empty gRPC result")
                return
            response = result.get_response()
            if response.id != self.request_id:
                self._push("error", "Triton request_id mismatch")
                return
            final = response.parameters.get("triton_final_response")
            if final is not None and bool(final.bool_param):
                self._push("end", None)
                return
            audio = result.as_numpy("waveform")
            if audio is not None and np.asarray(audio).size:
                self._push("audio", TritonChunk(pcm16(audio), time.perf_counter()))
        except BaseException as exc:  # never let an exception escape into gRPC
            self._push("error", f"Callback failure: {type(exc).__name__}: {exc}")

    # --- blocking; call via asyncio.to_thread -------------------------------
    def open(self, text: str) -> None:
        grpc = self._grpc
        if grpc is None:
            import tritonclient.grpc as grpc
        if self.client is not None:
            raise RuntimeError("SpeechStream already open")
        self.client = grpc.InferenceServerClient(url=self.endpoint, verbose=False)
        self.client.start_stream(callback=self._on_response)
        inp = grpc.InferInput("target_text", [1, 1], "BYTES")
        inp.set_data_from_numpy(np.array([[text]], dtype=object))
        self.client.async_stream_infer(
            model_name="cosyvoice2", inputs=[inp],
            outputs=[grpc.InferRequestedOutput("waveform")],
            request_id=self.request_id, enable_empty_final_response=True,
        )

    def close(self, cancel: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if self.client is not None:
            try:
                self.client.stop_stream(cancel_requests=cancel)
            except BaseException:
                pass
            try:
                self.client.close()
            except BaseException:
                pass

    # --- event loop ----------------------------------------------------------
    async def chunks(self, deadline: float) -> AsyncIterator[TritonChunk]:
        """Yield audio until Triton's final response; `deadline` is time.monotonic()."""
        seen = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise InferenceError("Triton request timed out")
            try:
                kind, value = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                raise InferenceError("Triton request timed out") from None
            if kind == "error":
                raise InferenceError(str(value))
            if kind == "end":
                if seen == 0:
                    raise InferenceError("Triton stream finalized without audio")
                self.finished = True
                return
            seen += 1
            yield value
