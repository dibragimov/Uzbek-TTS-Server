"""Triton replicas: model repository generation and process lifecycle.

Each replica is one tritonserver process with the three upstream models
(cosyvoice2 orchestrator, tensorrt_llm, token2wav). Config values are exactly
RC4's, except paths/ports and the KV-cache size, which are settings.
Token2Wav must stay single-instance per replica: it owns per-request vocoder
caches (RC4 invariant), which is why parallelism is done with more replicas.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .artifacts import Artifacts

log = logging.getLogger("navoiy.triton")
RC4_DIR = Path(__file__).resolve().parent / "rc4"
MODELS = ("cosyvoice2", "tensorrt_llm", "token2wav")


def upstream_runtime(arts: Artifacts) -> Path:
    return arts.cosyvoice_dir / "runtime/triton_trtllm"


def build_model_repo(arts: Artifacts, profile_dir: Path, settings: config.Settings,
                     dest: Path, runner=subprocess.run) -> Path:
    upstream = upstream_runtime(arts)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for name in MODELS:
        shutil.copytree(upstream / "model_repo" / name, dest / name)
    # Unchanged RC4 patch; it refuses to run if upstream text differs.
    runner([sys.executable, str(RC4_DIR / "patch_orchestrator.py"),
            "--model-py", str(dest / "cosyvoice2/1/model.py")], check=True)
    runtime_model = profile_dir / "runtime_model"
    engine = profile_dir / "engine"
    fill = upstream / "scripts/fill_template.py"
    params = {
        "token2wav": f"model_dir:{runtime_model},triton_max_batch_size:16,max_queue_delay_microseconds:0",
        "cosyvoice2": (f"model_dir:{runtime_model},bls_instance_num:{settings.cosy_instances},"
                       f"llm_tokenizer_dir:{arts.llm_dir},triton_max_batch_size:16,"
                       "decoupled_mode:True,max_queue_delay_microseconds:0"),
        "tensorrt_llm": ("triton_backend:tensorrtllm,triton_max_batch_size:16,decoupled_mode:True,"
                         f"max_beam_width:1,engine_dir:{engine},"
                         f"max_tokens_in_paged_kv_cache:{settings.kv_cache_tokens},"
                         "max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,"
                         "exclude_input_in_output:True,enable_kv_cache_reuse:False,"
                         "batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0,"
                         "encoder_input_features_data_type:TYPE_FP16,logits_datatype:TYPE_FP32"),
    }
    for name, value in params.items():
        runner([sys.executable, str(fill), "-i", str(dest / name / "config.pbtxt"), value], check=True)
    verify_model_repo(dest, settings.cosy_instances)
    return dest


def _instance_count(pbtxt: str) -> int | None:
    m = re.search(r"instance_group\s*\[\s*\{\s*count\s*:\s*(\d+)", pbtxt)
    return int(m.group(1)) if m else None


def verify_model_repo(repo: Path, cosy_instances: int) -> None:
    cosy = (repo / "cosyvoice2/config.pbtxt").read_text()
    wav = (repo / "token2wav/config.pbtxt").read_text()
    if _instance_count(cosy) != cosy_instances:
        raise RuntimeError("cosyvoice2 instance count not configured as requested")
    if _instance_count(wav) != 1:
        raise RuntimeError("token2wav must keep ONE instance (per-request vocoder cache affinity)")
    patched = (repo / "cosyvoice2/1/model.py").read_text()
    for marker in ("NAVOIY_INSTRUCTION", "NAVOIY_LLM_GRPC_ENDPOINT", "NAVOIY_ENDING_MODE"):
        if marker not in patched:
            raise RuntimeError(f"orchestrator patch marker missing: {marker}")
    for name in MODELS:
        leftover = re.findall(r"\$\{(model_dir|engine_dir|llm_tokenizer_dir|bls_instance_num)\}",
                              (repo / name / "config.pbtxt").read_text())
        if leftover:
            raise RuntimeError(f"{name}/config.pbtxt has unfilled parameters: {leftover}")


@dataclass
class Replica:
    index: int
    gpu: str
    http_port: int
    grpc_port: int
    metrics_port: int
    repo: Path
    log_path: Path
    process: subprocess.Popen | None = None
    _log_handle: object = field(default=None, repr=False)

    @property
    def grpc_endpoint(self) -> str:
        return f"127.0.0.1:{self.grpc_port}"

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


def replica_env(arts: Artifacts, settings: config.Settings, gpu: str, grpc_port: int) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = (f"{arts.cosyvoice_dir}:{arts.cosyvoice_dir}/third_party/Matcha-TTS:"
                         + env.get("PYTHONPATH", ""))
    env.update({  # RC4 runtime environment
        "NAVOIY_INSTRUCTION": config.INSTRUCTION,
        "CUDA_VISIBLE_DEVICES": gpu,
        "NAVOIY_LLM_GRPC_ENDPOINT": f"127.0.0.1:{grpc_port}",
        "NAVOIY_LLM_TIMEOUT_S": "35",
        "NAVOIY_ENDING_MODE": "soften",
        "NAVOIY_FINAL_FADE_MS": "15",
        "NAVOIY_FINAL_SILENCE_MS": "200",
    })
    env.pop("NAVOIY_TOKEN_CAPTURE_DIR", None)
    for secret in ("NAVOIY_API_KEY", "HF_TOKEN", "GITHUB_TOKEN"):
        env.pop(secret, None)  # Triton does not need credentials
    return env


def start_replica(index: int, arts: Artifacts, profile_dir: Path, settings: config.Settings,
                  run_dir: Path, runner=subprocess.run) -> Replica:
    http, grpc, metrics = settings.replica_ports(index)
    gpu = settings.gpus[index % len(settings.gpus)]
    repo = build_model_repo(arts, profile_dir, settings, run_dir / f"replica{index}" / "model_repo",
                            runner=runner)
    log_path = run_dir / f"triton-replica{index}.log"
    replica = Replica(index, gpu, http, grpc, metrics, repo, log_path)
    cmd = ["tritonserver", "--model-repository", str(repo),
           f"--http-port={http}", f"--grpc-port={grpc}", f"--metrics-port={metrics}",
           f"--backend-config=python,shm-region-prefix-name=navoiy_r{index}_"]
    log.info("replica %d: GPU %s, gRPC %d, log %s", index, gpu, grpc, log_path)
    replica._log_handle = open(log_path, "w", encoding="utf-8")
    replica.process = subprocess.Popen(cmd, stdout=replica._log_handle, stderr=subprocess.STDOUT,
                                       env=replica_env(arts, settings, gpu, grpc),
                                       start_new_session=True)
    return replica


def tail(path: Path, n: int = 80) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def wait_ready(replica: Replica, timeout_s: float) -> float:
    began = time.perf_counter()
    url = f"http://127.0.0.1:{replica.http_port}/v2/health/ready"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not replica.alive():
            raise RuntimeError(f"Triton replica {replica.index} exited with "
                               f"{replica.process.returncode}\n--- log tail ---\n{tail(replica.log_path)}")
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return time.perf_counter() - began
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Triton replica {replica.index} not ready after {timeout_s:.0f}s\n"
                       f"--- log tail ---\n{tail(replica.log_path)}")


def stop_replica(replica: Replica, timeout_s: float = 20) -> None:
    proc = replica.process
    try:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
    finally:
        if replica._log_handle is not None:
            replica._log_handle.close()
            replica._log_handle = None
