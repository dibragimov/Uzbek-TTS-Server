"""Build pipeline with the GPU tools simulated (convert, trtllm-build, Triton)."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from navoiy_server import hardware, profile_build
from navoiy_server.artifacts import Artifacts
from tests.fakes import FakeStream, settings

A100 = hardware.GpuInfo("NVIDIA A100-SXM4-80GB", (8, 0), 81920)
H100 = hardware.GpuInfo("NVIDIA H100 80GB HBM3", (9, 0), 81559)
VERSIONS = lambda: {"tensorrt": "10.11.0", "tensorrt_llm": "0.20.0"}  # noqa: E731


class FakeTools:
    """Simulates the external commands by producing their output files."""
    def __init__(self, fail_trtllm_build=False):
        self.calls = []
        self.fail_trtllm_build = fail_trtllm_build

    def __call__(self, cmd, check=True, **kw):
        cmd = [str(c) for c in cmd]
        name = Path(cmd[1]).name if cmd[0].endswith("python3") or "python" in cmd[0] else cmd[0]
        self.calls.append(name)
        arg = lambda flag: Path(cmd[cmd.index(flag) + 1])  # noqa: E731
        if name == "convert_checkpoint.py":
            arg("--output_dir").mkdir(parents=True)
            (arg("--output_dir") / "config.json").write_text("{}")
        elif name == "trtllm-build":
            if self.fail_trtllm_build:
                raise subprocess.CalledProcessError(1, cmd)
            out = arg("--output_dir")
            out.mkdir(parents=True)
            (out / "config.json").write_text("{}")
            (out / "rank0.engine").write_bytes(b"engine")
        elif name == "build_spk_cache.py":
            arg("--output").write_bytes(b"spk")
        return subprocess.CompletedProcess(cmd, 0)


class FakeReplica:
    def __init__(self, profile_dir, make_plan=True):
        self.index, self.grpc_endpoint, self.log_path = 0, "127.0.0.1:9001", Path("/dev/null")
        self.alive_flag = True
        if make_plan:
            (profile_dir / "runtime_model" / profile_build.FLOW_PLAN).write_bytes(b"plan")

    def alive(self):
        return self.alive_flag


class PrepareTest(unittest.TestCase):
    def setUp(self):
        FakeStream.reset(first_delay=0.01, chunks=3, chunk_bytes=24000, interval=0.0)
        self.root = Path(tempfile.mkdtemp())
        self.s = settings(data_dir=self.root / "data")
        base = self.root / "base"
        base.mkdir()
        for f in ("cosyvoice2.yaml", "flow.pt", "hift.pt", "flow.decoder.estimator.fp32.onnx"):
            (base / f).write_text("x")
        (base / "CosyVoice-BlankEN").mkdir()
        self.arts = Artifacts(base, self.root / "src", self.root / "llm", "sha-llm-1", self.root / "cosy")
        self.started = []
        import navoiy_server.triton_runtime as tr
        self._orig = (tr.wait_ready, tr.stop_replica, profile_build.make_normalizer)
        tr.wait_ready = lambda r, timeout_s: 12.0
        tr.stop_replica = lambda r: None
        profile_build.make_normalizer = lambda arts: type("N", (), {"normalize": lambda self, t: t})()

    def tearDown(self):
        import navoiy_server.triton_runtime as tr
        tr.wait_ready, tr.stop_replica, profile_build.make_normalizer = self._orig

    def prepare(self, tools, gpu=A100, make_plan=True):
        def start_replica(i, arts, profile_dir, settings, run_dir, runner=None):
            self.started.append(profile_dir.name)
            return FakeReplica(profile_dir, make_plan)
        return profile_build.prepare(self.s, runner=tools, stream_factory=FakeStream,
                                     start_replica=start_replica, gpu_query=lambda d: gpu,
                                     versions=VERSIONS, artifacts=self.arts)

    def test_full_build_then_idempotent(self):
        tools = FakeTools()
        profile = self.prepare(tools)
        self.assertEqual(tools.calls, ["convert_checkpoint.py", "trtllm-build", "build_spk_cache.py"])
        ready = json.loads((profile / "READY.json").read_text())
        self.assertEqual(ready["gpu"]["name"], A100.name)
        self.assertEqual(len(ready["smoke"]), 3)
        self.assertFalse(list(profile.glob("engine-build-*")), "temp build dir left behind")
        self.assertTrue((profile / "engine" / "BUILD_OK.json").is_file())
        self.assertTrue((profile / "runtime_model" / "flow.pt").is_symlink())
        # Second start: nothing rebuilt, Triton not even started.
        tools2, n = FakeTools(), len(self.started)
        self.assertEqual(self.prepare(tools2), profile)
        self.assertEqual((tools2.calls, len(self.started)), ([], n))

    def test_failed_engine_build_leaves_no_partial_state_and_resumes(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare(FakeTools(fail_trtllm_build=True))
        profile = next((self.s.profiles_dir).iterdir())
        self.assertFalse((profile / "engine").exists())
        self.assertFalse(list(profile.glob("engine-build-*")))
        self.assertFalse((profile / "READY.json").exists())
        tools = FakeTools()
        self.prepare(tools)
        self.assertIn("trtllm-build", tools.calls)

    def test_missing_flow_plan_is_an_error_not_ready(self):
        with self.assertRaises(RuntimeError):
            self.prepare(FakeTools(), make_plan=False)
        profile = next(self.s.profiles_dir.iterdir())
        self.assertFalse((profile / "READY.json").exists())
        # Engine and speaker cache are kept; only the Triton stage is retried.
        tools = FakeTools()
        self.prepare(tools)
        self.assertEqual(tools.calls, [])

    def test_each_gpu_type_gets_its_own_profile(self):
        a = self.prepare(FakeTools(), gpu=A100)
        h = self.prepare(FakeTools(), gpu=H100)
        self.assertNotEqual(a, h)
        self.assertTrue(a.name.startswith("nvidia-a100-sxm4-80gb-sm80-"))
        self.assertTrue(h.name.startswith("nvidia-h100-80gb-hbm3-sm90-"))
        self.assertEqual(self.prepare(FakeTools(), gpu=A100), a)

    def test_serve_refuses_unbuilt_profile(self):
        with self.assertRaises(RuntimeError) as ctx:
            profile_build.ready_profile(self.s, self.arts, gpu_query=lambda d: A100, versions=VERSIONS)
        self.assertIn("prepare", str(ctx.exception))

    def test_smoke_rejects_too_little_audio(self):
        FakeStream.reset(first_delay=0.0, chunks=1, chunk_bytes=2000, interval=0.0)
        with self.assertRaises(RuntimeError) as ctx:
            self.prepare(FakeTools())
        self.assertIn("smoke test", str(ctx.exception))


class Hardware(unittest.TestCase):
    def test_nvidia_smi_parsing(self):
        out = subprocess.CompletedProcess([], 0, stdout="NVIDIA H100 80GB HBM3, 9.0, 81559\n")
        gpu = hardware.query_gpu("0", runner=lambda *a, **k: out)
        self.assertEqual(gpu, H100)

    def test_key_changes_with_anything_that_changes_the_build(self):
        s = settings()
        k = lambda gpu=A100, v=VERSIONS(), sha="a": hardware.profile_key(gpu, v, sha, s)[0]  # noqa: E731
        base = k()
        self.assertEqual(base, k())
        self.assertNotEqual(base, k(gpu=H100))
        self.assertNotEqual(base, k(sha="b"))
        self.assertNotEqual(base, k(v={"tensorrt": "10.11.0", "tensorrt_llm": "0.21.0"}))

    def test_pre_ampere_rejected(self):
        with self.assertRaises(RuntimeError):
            hardware.check_supported(hardware.GpuInfo("Tesla V100-SXM2-32GB", (7, 0), 32768))
        hardware.check_supported(A100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
