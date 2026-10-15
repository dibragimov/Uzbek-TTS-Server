import asyncio
import io
import unittest
import wave

from httpx import ASGITransport, AsyncClient

from navoiy_server.api import build_app
from navoiy_server.replica_pool import ReplicaPool
from tests.fakes import AUTH, DOGRAH_BODY, FakeStream, UpperNormalizer, settings, settle


class ApiTest(unittest.IsolatedAsyncioTestCase):
    replicas = 1

    async def asyncSetUp(self):
        FakeStream.reset()
        await self.make()

    async def make(self, **kw):
        self.s = settings(**kw)
        self.pool = ReplicaPool([f"127.0.0.1:{9001 + 10 * i}" for i in range(self.replicas)],
                                self.s.max_active_per_replica, self.s.max_queue)
        self.app = build_app(settings=self.s, pool=self.pool, normalizer=UpperNormalizer(),
                             stream_factory=FakeStream, ready_check=lambda: True)
        self.client = AsyncClient(transport=ASGITransport(app=self.app), base_url="http://t")

    async def asyncTearDown(self):
        await settle(self.app)
        self.assertTrue(all(s.closed is not None for s in FakeStream.opened), "a stream was never closed")
        self.assertEqual(FakeStream.live, 0)
        await self.client.aclose()

    async def speech(self, **overrides):
        return await self.client.post("/v1/audio/speech", headers=AUTH, json={**DOGRAH_BODY, **overrides})


class Contract(ApiTest):
    async def test_dograh_request_streams_24k_pcm(self):
        r = await self.speech()
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.content), 4 * 12000)
        self.assertEqual(r.headers["x-navoiy-audio-rate"], "24000")
        self.assertIn("first_audio;dur=", r.headers["server-timing"])
        self.assertNotIn("retry-after", r.headers)
        await settle(self.app)
        self.assertEqual(FakeStream.opened[0].closed, False, "completed stream must close without cancel")
        self.assertEqual(self.app.state.stats.counters["completed"], 1)

    async def test_any_model_label_and_voice_aliases(self):
        for model, voice in (("kokoro", "calm"), ("navoiy-tts-v1", "uz"), ("tts-1", "alloy"), ("x", "auto")):
            r = await self.speech(model=model, voice=voice)
            self.assertEqual(r.status_code, 200, (model, voice, r.text))

    async def test_speed_from_dograh_slider_is_ignored_not_rejected(self):
        self.assertEqual((await self.speech(speed=1.5)).status_code, 200)
        self.assertEqual((await self.speech(speed=None)).status_code, 200)

    async def test_wav_is_valid_and_complete(self):
        r = await self.speech(response_format="wav")
        self.assertEqual(r.status_code, 200)
        with wave.open(io.BytesIO(r.content)) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()), (1, 2, 24000))
            self.assertEqual(w.getnframes(), 4 * 12000 // 2)

    async def test_markdown_and_emoji_are_cleaned_before_normalizer(self):
        r = await self.speech(input="**Rahmat!** 😊\n- Balans: 25 000 so'm")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(FakeStream.opened[0].text, "Rahmat! Balans: 25 000 so'm")

    async def test_emoji_only_sentence_returns_empty_audio(self):
        r = await self.speech(input="😊👍")
        self.assertEqual((r.status_code, r.content), (200, b""))
        self.assertEqual(FakeStream.opened, [])

    async def test_russian_text_is_flagged_but_synthesized(self):
        r = await self.speech(input="Ваш баланс 25 000 сумов.")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.app.state.stats.counters["russian_text"], 1)


class Validation(ApiTest):
    async def test_errors_are_400_openai_shaped(self):
        cases = [dict(voice="nigora"), dict(response_format="mp3"), dict(stream_format="sse"),
                 dict(input="   "), dict(input="a" * 1201)]
        for case in cases:
            r = await self.speech(**case)
            self.assertEqual(r.status_code, 400, (case, r.text))
            self.assertIn("message", r.json()["error"])

    async def test_missing_input_is_400_not_422(self):
        r = await self.client.post("/v1/audio/speech", headers=AUTH, json={"model": "x"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("input", r.json()["error"]["message"])

    async def test_auth(self):
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic test-key-0123456789"}):
            r = await self.client.post("/v1/audio/speech", headers=headers, json=DOGRAH_BODY)
            self.assertEqual(r.status_code, 401)
        self.assertEqual((await self.client.get("/v1/models")).status_code, 401)
        self.assertEqual((await self.client.get("/v1/models", headers=AUTH)).status_code, 200)
        self.assertEqual((await self.client.get("/metrics", headers=AUTH)).status_code, 200)
        self.assertEqual((await self.client.get("/health")).status_code, 200)  # no auth by design


class Failures(ApiTest):
    async def test_open_failure_is_502_and_releases(self):
        FakeStream.reset(fail_open=True)
        r = await self.speech()
        self.assertEqual(r.status_code, 502)
        await settle(self.app)

    async def test_failure_before_audio_is_502_not_truncated_200(self):
        FakeStream.reset(fail_before_first=True)
        r = await self.speech()
        self.assertEqual(r.status_code, 502)
        self.assertIn("no speech tokens", r.json()["error"]["message"])
        await settle(self.app)
        self.assertEqual(FakeStream.opened[0].closed, True, "failed request must be cancelled on close")

    async def test_failure_mid_stream_truncates_and_releases(self):
        FakeStream.reset(fail_after=2)
        r = await self.speech()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.content), 2 * 12000)
        await settle(self.app)
        self.assertEqual(self.app.state.stats.counters["failed"], 1)

    async def test_wav_failure_mid_stream_is_502(self):
        FakeStream.reset(fail_after=2)
        r = await self.speech(response_format="wav")
        self.assertEqual(r.status_code, 502)

    async def test_hung_triton_times_out_and_releases(self):
        await self.make(request_timeout_s=5.0)
        FakeStream.reset(hang=True)
        r = await asyncio.wait_for(self.speech(), timeout=8)
        self.assertEqual(r.status_code, 502)
        self.assertIn("timed out", r.json()["error"]["message"])


class Capacity(ApiTest):
    async def test_limit_queue_and_reject_without_retry_after(self):
        await self.make(max_active_per_replica=2, max_queue=1, queue_timeout_s=0.3)
        FakeStream.reset(first_delay=0.05, chunks=10, interval=0.05)  # ~0.55 s each
        rs = await asyncio.gather(*[self.speech() for _ in range(4)])
        codes = sorted(r.status_code for r in rs)
        # 2 run, 1 queues then times out (0.3 s < 0.55 s), 1 is rejected at once (queue full)
        self.assertEqual(codes, [200, 200, 503, 503], [r.text for r in rs])
        self.assertTrue(all("retry-after" not in r.headers for r in rs))
        self.assertEqual(FakeStream.peak, 2)

    async def test_queued_request_runs_when_slot_frees(self):
        await self.make(max_active_per_replica=1, max_queue=4, queue_timeout_s=5)
        FakeStream.reset(first_delay=0.02, chunks=3, interval=0.02)
        rs = await asyncio.gather(*[self.speech() for _ in range(4)])
        self.assertEqual([r.status_code for r in rs], [200] * 4)
        self.assertEqual(FakeStream.peak, 1)


class TwoReplicas(ApiTest):
    replicas = 2

    async def test_load_spreads_across_replicas(self):
        FakeStream.reset(first_delay=0.05, chunks=4, interval=0.03)
        rs = await asyncio.gather(*[self.speech() for _ in range(4)])
        self.assertTrue(all(r.status_code == 200 for r in rs))
        self.assertEqual(sorted(r.headers["x-navoiy-replica"] for r in rs), ["0", "0", "1", "1"])
        self.assertEqual(FakeStream.peak, 4)

    async def test_unhealthy_replica_is_skipped_then_all_down_is_503(self):
        await self.pool.mark_unhealthy(0)
        rs = await asyncio.gather(*[self.speech() for _ in range(3)])
        self.assertTrue(all(r.headers.get("x-navoiy-replica") == "1" for r in rs if r.status_code == 200))
        await self.pool.mark_unhealthy(1)
        self.assertEqual((await self.speech()).status_code, 503)
        self.assertEqual((await self.client.get("/health")).status_code, 503)


if __name__ == "__main__":
    unittest.main(verbosity=2)
