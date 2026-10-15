"""Real HTTP server on a socket, real clients, real disconnects."""
import asyncio
import random
import socket
import threading
import time
import unittest

import httpx
import openai
import uvicorn

from navoiy_server.api import build_app
from navoiy_server.replica_pool import ReplicaPool
from tests.fakes import AUTH, DOGRAH_BODY, FIRST_CHUNK, FakeStream, UpperNormalizer, settings


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, replicas=1, **kw):
        self.s = settings(**kw)
        self.port = free_port()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(replicas,), daemon=True)
        self.thread.start()
        assert self.ready.wait(10)

    def _run(self, replicas):
        async def main():
            self.loop = asyncio.get_running_loop()
            self.pool = ReplicaPool([f"r{i}" for i in range(replicas)],
                                    self.s.max_active_per_replica, self.s.max_queue)
            self.app = build_app(settings=self.s, pool=self.pool, normalizer=UpperNormalizer(),
                                 stream_factory=FakeStream)
            self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port,
                                                        log_level="error", timeout_keep_alive=75))
            task = asyncio.create_task(self.server.serve())
            while not self.server.started:
                await asyncio.sleep(0.01)
            self.ready.set()
            await task
        asyncio.run(main())

    def settled(self, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.app.state.tasks and all(s.active == 0 for s in self.pool.slots):
                return True
            time.sleep(0.02)
        return False

    def stop(self):
        self.server.should_exit = True
        self.thread.join(10)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"


class Integration(unittest.TestCase):
    def setUp(self):
        FakeStream.reset(first_delay=0.03, chunks=6, chunk_bytes=12000, interval=0.03)

    def tearDown(self):
        if hasattr(self, "srv"):
            self.assertTrue(self.srv.settled(), f"leak: {self.srv.pool.snapshot()} tasks={len(self.srv.app.state.tasks)}")
            self.assertTrue(all(s.closed is not None for s in FakeStream.opened), "unclosed stream")
            self.assertEqual(FakeStream.live, 0)
            self.srv.stop()

    def test_openai_sdk_exactly_like_pipecat(self):
        """Pipecat OpenAITTSService: with_streaming_response + iter_bytes(0.5 s of audio)."""
        self.srv = Server()

        async def pipecat_like():
            client = openai.AsyncOpenAI(api_key="test-key-0123456789", base_url=self.srv.url + "/v1")
            sizes = []
            for sentence in ("Assalomu alaykum.", "Balansingiz 25 000 so'm.", "Rahmat!"):
                async with client.audio.speech.with_streaming_response.create(
                        input=sentence, model="kokoro", voice="calm", response_format="pcm",
                        speed=1.0) as r:
                    self.assertEqual(r.status_code, 200)
                    async for chunk in r.iter_bytes(FIRST_CHUNK):
                        sizes.append(len(chunk))
            await client.close()
            return sizes

        sizes = asyncio.run(pipecat_like())
        self.assertEqual(sum(sizes), 3 * 6 * 12000)
        self.assertTrue(all(n % 2 == 0 for n in sizes), "odd-length chunk would split a PCM sample")

    def test_openai_sdk_error_surfaces_as_bad_request(self):
        self.srv = Server()

        async def go():
            client = openai.AsyncOpenAI(api_key="test-key-0123456789", base_url=self.srv.url + "/v1",
                                        max_retries=0)
            try:
                async with client.audio.speech.with_streaming_response.create(
                        input="x", model="kokoro", voice="nigora", response_format="pcm") as r:
                    await r.read()
            finally:
                await client.close()

        with self.assertRaises(openai.BadRequestError):  # Pipecat catches exactly this
            asyncio.run(go())

    def test_barge_in_mid_stream_releases_and_drains(self):
        self.srv = Server(max_active_per_replica=1)
        FakeStream.reset(first_delay=0.03, chunks=20, chunk_bytes=12000, interval=0.03)
        for _ in range(5):
            with httpx.Client(timeout=10) as c:
                with c.stream("POST", self.srv.url + "/v1/audio/speech", headers=AUTH, json=DOGRAH_BODY) as r:
                    self.assertEqual(r.status_code, 200)
                    next(r.iter_bytes(FIRST_CHUNK))
                # leaving the block closes the connection mid-stream (barge-in)
            self.assertTrue(self.srv.settled(), self.srv.pool.snapshot())
        counters = self.srv.app.state.stats.counters
        self.assertEqual(counters["client_disconnected"], 5)
        # Drained to Triton's normal end, so closed WITHOUT cancel (RC4-validated cleanup path)
        self.assertTrue(all(s.closed is False and s.finished for s in FakeStream.opened))

    def test_disconnect_before_first_audio_releases(self):
        self.srv = Server(max_active_per_replica=1)
        FakeStream.reset(first_delay=0.4, chunks=3, interval=0.02)

        async def go():
            async with httpx.AsyncClient(timeout=10) as c:
                task = asyncio.create_task(c.post(self.srv.url + "/v1/audio/speech", headers=AUTH,
                                                  json=DOGRAH_BODY))
                await asyncio.sleep(0.1)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        asyncio.run(go())
        self.assertTrue(self.srv.settled())
        # The slot is usable again: a normal request still succeeds.
        r = httpx.post(self.srv.url + "/v1/audio/speech", headers=AUTH, json=DOGRAH_BODY, timeout=10)
        self.assertEqual(r.status_code, 200)

    def test_stress_mixed_disconnects_no_leaks(self):
        self.srv = Server(replicas=2, max_active_per_replica=2, max_queue=64, queue_timeout_s=20)
        FakeStream.reset(first_delay=0.02, chunks=8, chunk_bytes=6000, interval=0.01)
        rng = random.Random(7)
        outcomes = {"full": 0, "barge": 0, "early": 0}

        async def one(i):
            async with httpx.AsyncClient(timeout=30) as c:
                mode = rng.choice(list(outcomes))
                outcomes[mode] += 1
                if mode == "early":
                    t = asyncio.create_task(c.post(self.srv.url + "/v1/audio/speech", headers=AUTH,
                                                   json=DOGRAH_BODY))
                    await asyncio.sleep(rng.uniform(0, 0.05))
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, httpx.HTTPError):
                        pass
                    return
                async with c.stream("POST", self.srv.url + "/v1/audio/speech", headers=AUTH,
                                    json=DOGRAH_BODY) as r:
                    assert r.status_code == 200, r.status_code
                    async for _ in r.aiter_bytes(FIRST_CHUNK):
                        if mode == "barge":
                            break

        async def go():
            await asyncio.gather(*[one(i) for i in range(60)])
        asyncio.run(go())
        self.assertTrue(self.srv.settled(10), self.srv.pool.snapshot())
        self.assertLessEqual(FakeStream.peak, 4, "more concurrent Triton streams than 2 replicas x 2")
        print(f"\n  stress: {outcomes}, peak concurrent streams {FakeStream.peak}, "
              f"counters {self.srv.app.state.stats.counters}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
