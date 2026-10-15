"""Test doubles: a fake Triton stream with the real SpeechStream interface."""
from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

from navoiy_server import config
from navoiy_server.triton_stream import InferenceError, TritonChunk


class Behaviour:
    def __init__(self, first_delay=0.03, chunks=4, chunk_bytes=12000, interval=0.03,
                 fail_open=False, fail_before_first=False, fail_after=None, hang=False):
        self.__dict__.update(locals())
        del self.__dict__["self"]


class FakeStream:
    behaviour = Behaviour()
    opened: list["FakeStream"] = []
    live = 0
    peak = 0

    @classmethod
    def reset(cls, **kw):
        cls.behaviour = Behaviour(**kw)
        cls.opened = []
        cls.live = 0
        cls.peak = 0

    def __init__(self, endpoint, loop):
        self.endpoint = endpoint
        self.request_id = str(uuid.uuid4())
        self.finished = False
        self.closed = None          # None = never closed; else the `cancel` flag used
        self.text = None
        self.b = FakeStream.behaviour

    def open(self, text):
        time.sleep(0.002)
        if self.b.fail_open:
            raise ConnectionError("cannot connect to Triton")
        self.text = text
        FakeStream.opened.append(self)
        FakeStream.live += 1
        FakeStream.peak = max(FakeStream.peak, FakeStream.live)

    async def chunks(self, deadline):
        b = self.b
        await asyncio.sleep(b.first_delay)
        if b.hang:
            await asyncio.sleep(max(0, deadline - time.monotonic()))
            raise InferenceError("Triton request timed out")
        if b.fail_before_first:
            raise InferenceError("TRT returned no speech tokens")
        for i in range(b.chunks):
            if b.fail_after is not None and i >= b.fail_after:
                raise InferenceError("Token2Wav failure mid-stream")
            yield TritonChunk(bytes([i % 251]) * b.chunk_bytes, time.perf_counter())
            await asyncio.sleep(b.interval)
        self.finished = True

    def close(self, cancel=False):
        if self.closed is None and self.text is not None:
            FakeStream.live -= 1
        self.closed = cancel


class UpperNormalizer:
    """Stands in for RC4's normalizer; marks its output so tests can see it ran."""
    def normalize(self, text):
        return text.strip()


def settings(**kw) -> config.Settings:
    base = dict(data_dir=Path("/tmp/navoiy-test"), hf_token=None, llm_repo="r", llm_revision="main",
                api_key="test-key-0123456789", host="127.0.0.1", port=8080, replicas=1, gpus=("0",),
                cosy_instances=2, max_active_per_replica=2, max_queue=8, queue_timeout_s=2.0,
                request_timeout_s=10.0, kv_cache_tokens=2560, max_input_chars=1200,
                accepted_voices=frozenset({"calm", "alloy", "uz", "auto", "default"}))
    base.update(kw)
    return config.Settings(**base)


AUTH = {"Authorization": "Bearer test-key-0123456789"}
DOGRAH_BODY = {  # exactly what Dograh's Speaches provider sends (Pipecat OpenAITTSService)
    "input": "Sizning balansingiz 25 000 so'm.", "model": "kokoro", "voice": "calm",
    "response_format": "pcm", "speed": 1.0}


async def settle(app, timeout=3.0):
    """Wait until every background session has finished and released its lease."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not app.state.tasks and all(s.active == 0 for s in app.state.pool.slots):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"sessions did not settle: tasks={len(app.state.tasks)} "
                         f"pool={app.state.pool.snapshot()}")


FIRST_CHUNK = config.PIPECAT_FIRST_CHUNK_BYTES
