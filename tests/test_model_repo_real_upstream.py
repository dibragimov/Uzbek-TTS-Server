"""Model-repo generation against the REAL pinned upstream CosyVoice files.

Needs the upstream runtime at COSYVOICE_DIR (checked out at the pinned commit);
skipped otherwise. Runs the unchanged RC4 patch and upstream fill_template.py.
"""
import os, py_compile, re, tempfile, unittest
from pathlib import Path
from navoiy_server import config, triton_runtime
from navoiy_server.artifacts import Artifacts

COSY = Path(os.environ.get("COSYVOICE_DIR", "/home/claude/cv"))

def settings(**kw):
    base = dict(data_dir=Path("/tmp/x"), hf_token=None, llm_repo="r", llm_revision="main", api_key="k"*16,
                host="0.0.0.0", port=8080, replicas=2, gpus=("0",), cosy_instances=2,
                max_active_per_replica=2, max_queue=4, queue_timeout_s=5, request_timeout_s=60,
                kv_cache_tokens=2560, max_input_chars=1200, accepted_voices=frozenset({"calm"}))
    base.update(kw)
    return config.Settings(**base)

@unittest.skipUnless((COSY / "runtime/triton_trtllm/model_repo").is_dir(), "upstream checkout missing")
class RealUpstream(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.arts = Artifacts(self.tmp / "base", self.tmp / "src", self.tmp / "llm", "abc", COSY)
        self.profile = self.tmp / "profiles" / "p1"

    def build(self, s):
        return triton_runtime.build_model_repo(self.arts, self.profile, s, self.tmp / "repo")

    def test_rc4_patch_applies_and_configs_match_rc4(self):
        repo = self.build(settings())
        model_py = repo / "cosyvoice2/1/model.py"
        py_compile.compile(str(model_py), doraise=True)  # patched orchestrator is valid Python
        src = model_py.read_text()
        for marker in ("NAVOIY_INSTRUCTION", "NAVOIY_LLM_GRPC_ENDPOINT", "NAVOIY_ENDING_MODE",
                       "Navoiy instruct2: do NOT append prompt speech tokens"):
            self.assertIn(marker, src)
        trt = (repo / "tensorrt_llm/config.pbtxt").read_text()
        for expected in (f'"{self.profile}/engine"', '"2560"', '"0.5"', '"inflight_fused_batching"',
                         'backend: "tensorrtllm"', "decoupled: True"):
            self.assertIn(expected, trt)
        cosy = (repo / "cosyvoice2/config.pbtxt").read_text()
        self.assertIn(f'"{self.profile}/runtime_model"', cosy)
        self.assertIn(f'"{self.tmp}/llm"', cosy)
        self.assertEqual(triton_runtime._instance_count(cosy), 2)
        wav = (repo / "token2wav/config.pbtxt").read_text()
        self.assertEqual(triton_runtime._instance_count(wav), 1)
        # Every placeholder RC4 filled is filled here too.
        for name in triton_runtime.MODELS:
            text = (repo / name / "config.pbtxt").read_text()
            for p in ("model_dir", "engine_dir", "llm_tokenizer_dir", "bls_instance_num",
                      "triton_max_batch_size", "decoupled_mode", "max_tokens_in_paged_kv_cache"):
                self.assertNotIn("${" + p + "}", text, f"{name}: {p} unfilled")

    def test_kv_cache_and_instances_are_settings(self):
        repo = self.build(settings(kv_cache_tokens=8192, cosy_instances=4))
        self.assertIn('"8192"', (repo / "tensorrt_llm/config.pbtxt").read_text())
        self.assertEqual(triton_runtime._instance_count((repo / "cosyvoice2/config.pbtxt").read_text()), 4)

    def test_same_unfilled_placeholders_as_rc4(self):
        """Placeholders left unfilled must be exactly those RC4 also left (TRT-LLM treats them as unset)."""
        repo = self.build(settings())
        left = set(re.findall(r"\$\{(\w+)\}", (repo / "tensorrt_llm/config.pbtxt").read_text()))
        self.assertNotIn("engine_dir", left)
        self.assertNotIn("max_tokens_in_paged_kv_cache", left)
        print("\n  placeholders left for TRT-LLM defaults (same as RC4):", sorted(left))

    def test_ports_and_env_per_replica(self):
        s = settings(replicas=3, gpus=("0", "1"))
        self.assertEqual(s.replica_ports(0), (9000, 9001, 9002))
        self.assertEqual(s.replica_ports(2), (9020, 9021, 9022))
        os.environ["NAVOIY_API_KEY"] = "secret-should-not-leak"
        env = triton_runtime.replica_env(self.arts, s, "1", 9011)
        self.assertEqual(env["NAVOIY_LLM_GRPC_ENDPOINT"], "127.0.0.1:9011")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertNotIn("NAVOIY_API_KEY", env)
        self.assertIn(str(COSY), env["PYTHONPATH"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
