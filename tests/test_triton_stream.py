"""SpeechStream against a fake tritonclient.grpc whose callbacks run on another thread."""
import asyncio
import threading
import time
import unittest

import numpy as np

from navoiy_server.triton_stream import InferenceError, SpeechStream


class _Param:
    def __init__(self, v):
        self.bool_param = v


class _Resp:
    def __init__(self, rid, final):
        self.id = rid
        self.parameters = {"triton_final_response": _Param(final)} if final else {}


class _Result:
    def __init__(self, rid, wave=None, final=False):
        self._r, self._w = _Resp(rid, final), wave

    def get_response(self):
        return self._r

    def as_numpy(self, name):
        return self._w


class FakeGrpc:
    """Minimal tritonclient.grpc stand-in. `script(rid, cb)` runs on a thread."""
    script = None
    stopped_with = []

    class InferInput:
        def __init__(self, *a):
            pass

        def set_data_from_numpy(self, arr):
            self.text = arr[0][0]

    class InferRequestedOutput:
        def __init__(self, *a):
            pass

    class InferenceServerClient:
        def __init__(self, url, verbose=False):
            self.url = url

        def start_stream(self, callback):
            self.cb = callback

        def async_stream_infer(self, model_name, inputs, outputs, request_id, enable_empty_final_response):
            assert model_name == "cosyvoice2" and enable_empty_final_response
            threading.Thread(target=FakeGrpc.script, args=(request_id, self.cb), daemon=True).start()

        def stop_stream(self, cancel_requests):
            FakeGrpc.stopped_with.append(cancel_requests)

        def close(self):
            pass


def wave(n=2400, value=0.5):
    return np.full((1, n), value, dtype=np.float32)


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def run_stream(self, script, timeout=2.0):
        FakeGrpc.script = staticmethod(script)
        FakeGrpc.stopped_with = []
        s = SpeechStream("127.0.0.1:9001", asyncio.get_running_loop(), grpc_module=FakeGrpc)
        await asyncio.to_thread(s.open, "salom")
        out = []
        try:
            async for c in s.chunks(time.monotonic() + timeout):
                out.append(c)
        finally:
            await asyncio.to_thread(s.close, not s.finished)
        return s, out

    async def test_chunks_in_order_then_end(self):
        def script(rid, cb):
            for i in range(50):
                cb(_Result(rid, wave(value=i / 100)))
            cb(_Result(rid, final=True))
        s, out = await self.run_stream(script)
        self.assertEqual(len(out), 50)  # nothing dropped (RC4's bounded queue could drop)
        firsts = [np.frombuffer(c.pcm, "<i2")[0] for c in out]
        self.assertEqual(firsts, sorted(firsts))
        self.assertTrue(s.finished)
        self.assertEqual(FakeGrpc.stopped_with, [False])

    async def test_many_chunks_burst_not_dropped(self):
        def script(rid, cb):
            for _ in range(1000):
                cb(_Result(rid, wave(240)))
            cb(_Result(rid, final=True))
        _, out = await self.run_stream(script)
        self.assertEqual(len(out), 1000)

    async def test_pcm16_conversion_and_clipping(self):
        def script(rid, cb):
            cb(_Result(rid, np.array([[0.0, 1.5, -1.5, 0.5]], dtype=np.float32)))
            cb(_Result(rid, final=True))
        _, out = await self.run_stream(script)
        self.assertEqual(list(np.frombuffer(out[0].pcm, "<i2")), [0, 32767, -32767, 16383])

    async def test_error_mismatch_and_empty_end(self):
        cases = [
            (lambda rid, cb: cb(None, error="model crashed"), "model crashed"),
            (lambda rid, cb: cb(_Result("other-id", wave())), "request_id mismatch"),
            (lambda rid, cb: cb(_Result(rid, final=True)), "without audio"),
            (lambda rid, cb: cb(_Result(rid, np.array([[np.nan]], dtype=np.float32))), "invalid waveform"),
        ]
        for script, needle in cases:
            with self.assertRaises(InferenceError) as ctx:
                await self.run_stream(script)
            self.assertIn(needle, str(ctx.exception))
            self.assertEqual(FakeGrpc.stopped_with, [True], "unfinished stream must be cancelled")

    async def test_deadline(self):
        def script(rid, cb):
            cb(_Result(rid, wave()))  # then silence forever
        with self.assertRaises(InferenceError) as ctx:
            await self.run_stream(script, timeout=0.3)
        self.assertIn("timed out", str(ctx.exception))

    async def test_callback_after_loop_closed_does_not_raise(self):
        loop = asyncio.new_event_loop()
        s = SpeechStream("x", loop, grpc_module=FakeGrpc)
        loop.close()
        s._on_response(_Result(s.request_id, wave()))  # must not raise into gRPC's thread


if __name__ == "__main__":
    unittest.main(verbosity=2)
