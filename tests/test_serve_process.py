import os, signal, socket, subprocess, sys, tempfile, time, unittest
from pathlib import Path
import httpx
from tests.fakes import AUTH, DOGRAH_BODY

ROOT = Path(__file__).resolve().parents[1]

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

class ServeProcess(unittest.TestCase):
    def start(self, replicas=2):
        self.flags = Path(tempfile.mkdtemp())
        self.port = free_port()
        env = {**os.environ, "HARNESS_FLAGS": str(self.flags), "NAVOIY_API_KEY": "test-key-0123456789",
               "NAVOIY_PORT": str(self.port), "NAVOIY_REPLICAS": str(replicas), "NAVOIY_HOST": "127.0.0.1",
               "NAVOIY_DATA_DIR": str(self.flags / "data"), "PYTHONPATH": str(ROOT)}
        self.proc = subprocess.Popen([sys.executable, str(ROOT / "tests/serve_harness.py")], env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.url = f"http://127.0.0.1:{self.port}"
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                if httpx.get(self.url + "/health", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.1)
        self.proc.kill(); self.fail("server did not start:\n" + self.proc.stdout.read())

    def output(self, timeout=15):
        out, _ = self.proc.communicate(timeout=timeout)
        return out

    def test_requests_then_sigterm_stops_replicas(self):
        self.start(replicas=2)
        r = httpx.post(self.url + "/v1/audio/speech", headers=AUTH, json=DOGRAH_BODY, timeout=10)
        self.assertEqual((r.status_code, len(r.content)), (200, 72000))
        self.assertEqual(httpx.get(self.url + "/ready", headers=AUTH).status_code, 200)
        m = httpx.get(self.url + "/metrics", headers=AUTH).json()
        self.assertEqual((m["gpu"], m["replicas"], len(m["pool"]["replicas"])), ("FAKE GPU", 2, 2))
        self.proc.send_signal(signal.SIGTERM)
        out = self.output()
        self.assertEqual(self.proc.returncode, 0, out)
        for needle in ("HARNESS start_replica 0", "HARNESS start_replica 1", "listening on",
                       "HARNESS stop_replica 0", "HARNESS stop_replica 1", "navoiy.serve: stopped"):
            self.assertIn(needle, out)

    def test_dead_replica_is_routed_around_then_all_dead_exits(self):
        self.start(replicas=2)
        (self.flags / "dead0").touch()
        time.sleep(0.6)
        h = httpx.get(self.url + "/health").json()
        self.assertEqual(h["healthy_replicas"], 1)
        r = httpx.post(self.url + "/v1/audio/speech", headers=AUTH, json=DOGRAH_BODY, timeout=10)
        self.assertEqual((r.status_code, r.headers["x-navoiy-replica"]), (200, "1"))
        (self.flags / "dead1").touch()
        out = self.output(timeout=10)
        self.assertEqual(self.proc.returncode, 70, out)        # container restarts on RunPod
        self.assertIn("replica 0 log line", out)               # log tail of the dead replica shown
        self.assertIn("all Triton replicas are down", out)

if __name__ == "__main__":
    unittest.main(verbosity=2)
