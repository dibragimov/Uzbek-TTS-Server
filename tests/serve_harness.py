"""Runs the REAL server.serve() with GPU-bound pieces replaced. Used by test_serve_process."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from navoiy_server import config, server, triton_runtime
from navoiy_server.artifacts import Artifacts
from tests.fakes import FakeStream

FLAGS = Path(os.environ["HARNESS_FLAGS"])

class FakeReplica:
    def __init__(self, i):
        self.index, self.grpc_endpoint = i, f"127.0.0.1:{9001 + 10 * i}"
        self.log_path = FLAGS / f"replica{i}.log"; self.log_path.write_text(f"replica {i} log line\n")
        self.process = type("P", (), {"returncode": -9})()
    def alive(self):
        return not (FLAGS / f"dead{self.index}").exists()

def start_replica(i, *a, **k):
    print(f"HARNESS start_replica {i}", flush=True); return FakeReplica(i)

def stop_replica(r):
    print(f"HARNESS stop_replica {r.index}", flush=True)

async def smoke(*a, **k):
    return [{"first_audio_ms": 5.0, "audio_s": 1.0, "wall_ms": 9.0}]

arts = Artifacts(FLAGS, FLAGS, FLAGS, "sha", FLAGS)
server.ensure_all = lambda s: arts
server.ready_profile = lambda s, a: (FLAGS / "profile", {"gpu": {"name": "FAKE GPU"}})
server.make_normalizer = lambda a: type("N", (), {"normalize": lambda self, t: t})()
server.smoke_test = smoke
server.SpeechStream = FakeStream
server.triton_ready_check = lambda reps: (lambda: all(r.alive() for r in reps))
triton_runtime.start_replica = start_replica
triton_runtime.stop_replica = stop_replica
triton_runtime.wait_ready = lambda r, timeout_s: 0.5
import asyncio, navoiy_server.server as srv
srv.watch_replicas.__defaults__ = (0.2,)          # poll faster in the test
FakeStream.reset(first_delay=0.01, chunks=3, chunk_bytes=24000, interval=0.01)
import logging; logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(name)s: %(message)s")
server.serve(config.load())
