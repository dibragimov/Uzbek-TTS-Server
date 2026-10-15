"""Build (once per GPU type) everything that is compiled for the hardware.

Stages, each resumable and only marked done after it fully succeeds:
  1. TensorRT-LLM engine: upstream run.sh stage 1 (convert_checkpoint.py +
     trtllm-build) with RC4's settings, built into a temp dir then renamed.
  2. Runtime model dir: links to the base CosyVoice2 files + spk2info.pt built
     from the fixed reference voice (unchanged RC4 build_spk_cache.py).
  3. Flow TensorRT plan: compiled by upstream Token2Wav on its first load, so we
     start one Triton replica, then synthesize smoke sentences end to end.
  4. READY.json: written last; serving refuses a profile without it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from . import config, hardware, triton_runtime
from .artifacts import Artifacts, ensure_all

log = logging.getLogger("navoiy.prepare")
READY = "READY.json"
FLOW_PLAN = "flow.decoder.estimator.fp16.mygpu.plan"
SMOKE_SENTENCES = [  # from the RC4 telecom regression set
    "Assalomu alaykum, Beeline call-markaziga xush kelibsiz.",
    "Sizning aloqa raqamingiz 90 123 45 67.",
    "Balansingizda 25 000 so'm mablag' bor.",
]


def run(cmd, *, env=None, cwd=None, runner=subprocess.run, log_path: Path | None = None):
    cmd = [str(c) for c in cmd]
    log.info("$ %s", " ".join(cmd))
    began = time.perf_counter()
    if log_path is None:
        runner(cmd, check=True, env=env, cwd=cwd)
    else:
        with open(log_path, "a", encoding="utf-8") as fh:
            try:
                runner(cmd, check=True, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                fh.flush()
                log.error("command failed; last lines of %s:\n%s", log_path, triton_runtime.tail(log_path, 60))
                raise
    log.info("  done in %.1fs", time.perf_counter() - began)


def detect_profile(settings: config.Settings, arts: Artifacts, gpu_query=hardware.query_gpu,
                   versions=hardware.library_versions):
    gpu = gpu_query(settings.gpus[0])
    hardware.check_supported(gpu)
    for other in settings.gpus[1:]:
        if gpu_query(other).name != gpu.name:
            raise RuntimeError("All NAVOIY_GPUS must be the same GPU model (one engine profile)")
    key, identity = hardware.profile_key(gpu, versions(), arts.llm_sha, settings)
    return gpu, key, identity


def build_engine(arts: Artifacts, profile_dir: Path, log_path: Path, runner=subprocess.run) -> Path:
    engine = profile_dir / "engine"
    if (engine / "BUILD_OK.json").is_file():
        log.info("engine: cached")
        return engine
    scripts = triton_runtime.upstream_runtime(arts) / "scripts"
    tmp = profile_dir / f"engine-build-{uuid.uuid4().hex[:8]}"
    try:
        run([sys.executable, scripts / "convert_checkpoint.py", "--model_dir", arts.llm_dir,
             "--output_dir", tmp / "checkpoint", "--dtype", config.ENGINE_DTYPE],
            runner=runner, log_path=log_path)
        run(["trtllm-build", "--checkpoint_dir", tmp / "checkpoint", "--output_dir", tmp / "engine",
             "--max_batch_size", config.ENGINE_MAX_BATCH_SIZE,
             "--max_num_tokens", config.ENGINE_MAX_NUM_TOKENS,
             "--gemm_plugin", config.ENGINE_DTYPE], runner=runner, log_path=log_path)
        built = tmp / "engine"
        if not (built / "config.json").is_file() or not list(built.glob("*.engine")):
            raise RuntimeError(f"trtllm-build produced no engine in {built}")
        (built / "BUILD_OK.json").write_text(json.dumps({
            "dtype": config.ENGINE_DTYPE, "max_batch_size": config.ENGINE_MAX_BATCH_SIZE,
            "max_num_tokens": config.ENGINE_MAX_NUM_TOKENS, "llm_sha": arts.llm_sha,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
        if engine.exists():
            shutil.rmtree(engine)
        os.replace(built, engine)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # checkpoint is ~1.3 GB, not needed after build
    return engine


def build_runtime_model(arts: Artifacts, profile_dir: Path, log_path: Path,
                        runner=subprocess.run) -> Path:
    runtime = profile_dir / "runtime_model"
    runtime.mkdir(parents=True, exist_ok=True)
    for src in arts.base_model_dir.iterdir():
        if src.name.startswith("."):
            continue
        dst = runtime / src.name
        if dst.is_symlink() and os.readlink(dst) == str(src):
            continue
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    spk = runtime / "spk2info.pt"
    if spk.is_file() and spk.stat().st_size > 0:
        log.info("speaker cache: cached")
        return runtime
    tmp = runtime / f"spk2info.{uuid.uuid4().hex[:8]}.tmp"
    env = dict(os.environ)
    env["PYTHONPATH"] = (f"{arts.cosyvoice_dir}:{arts.cosyvoice_dir}/third_party/Matcha-TTS:"
                         + env.get("PYTHONPATH", ""))
    try:
        run([sys.executable, triton_runtime.RC4_DIR / "build_spk_cache.py",
             "--cosyvoice-root", arts.cosyvoice_dir, "--model-dir", runtime,
             "--reference", arts.reference_wav, "--instruction", config.INSTRUCTION,
             "--output", tmp], env=env, runner=runner, log_path=log_path)
        os.replace(tmp, spk)
    finally:
        if tmp.exists():
            tmp.unlink()
    return runtime


async def smoke_test(endpoint: str, normalizer, stream_factory, timeout_s: float = 90,
                     sentences=tuple(SMOKE_SENTENCES)) -> list[dict]:
    results = []
    loop = asyncio.get_running_loop()
    for text in sentences:
        stream = stream_factory(endpoint, loop)
        began = time.perf_counter()
        first = None
        audio = 0
        try:
            await asyncio.to_thread(stream.open, normalizer.normalize(text))
            async for chunk in stream.chunks(time.monotonic() + timeout_s):
                first = first or chunk.arrived_at
                audio += len(chunk.pcm)
        finally:
            await asyncio.to_thread(stream.close, not stream.finished)
        seconds = audio / config.BYTES_PER_SECOND
        if seconds < 0.5:
            raise RuntimeError(f"smoke test produced only {seconds:.2f}s of audio for {text!r}")
        results.append({"first_audio_ms": round((first - began) * 1000, 1), "audio_s": round(seconds, 2),
                        "wall_ms": round((time.perf_counter() - began) * 1000, 1)})
        log.info("smoke: %s", results[-1])
    return results


def make_normalizer(arts: Artifacts):
    """RC4 normalizer: telecom layer over Navoiy's uztts (unchanged)."""
    for path in (str(arts.navoiy_src_dir), str(triton_runtime.RC4_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from uztts.normalize import normalize as uztts_normalize
    from telecom_normalizer import TelecomUzbekNormalizer
    return TelecomUzbekNormalizer(lambda text: uztts_normalize(text, mode="infer"))


def prepare(settings: config.Settings, *, runner=subprocess.run, stream_factory=None,
            start_replica=triton_runtime.start_replica, gpu_query=hardware.query_gpu,
            versions=hardware.library_versions, artifacts=None) -> Path:
    began = time.perf_counter()
    arts = artifacts or ensure_all(settings)
    gpu, key, identity = detect_profile(settings, arts, gpu_query, versions)
    profile_dir = settings.profiles_dir / key
    ready = profile_dir / READY
    if ready.is_file():
        log.info("profile %s is ready (built %s)", key, json.loads(ready.read_text()).get("built_at"))
        return profile_dir
    log.info("building profile %s for %s (first start on this GPU type; this takes a while)", key, gpu.name)
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "identity.json").write_text(json.dumps(identity, indent=2))
    build_log = profile_dir / "build.log"

    build_engine(arts, profile_dir, build_log, runner=runner)
    runtime = build_runtime_model(arts, profile_dir, build_log, runner=runner)
    # A plan from an interrupted earlier attempt may be partial: rebuild it.
    plan = runtime / FLOW_PLAN
    if plan.exists():
        plan.unlink()

    if stream_factory is None:
        from .triton_stream import SpeechStream as stream_factory
    run_dir = settings.data_dir / "run" / f"prepare-{uuid.uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    replica = start_replica(0, arts, profile_dir, settings, run_dir, runner=runner)
    try:
        # First load compiles the flow plan, so allow much longer than a normal start.
        startup_s = triton_runtime.wait_ready(replica, timeout_s=1800)
        log.info("Triton ready in %.0fs (includes flow-plan compilation)", startup_s)
        if not plan.is_file() or plan.stat().st_size == 0:
            raise RuntimeError(f"Triton started but the flow plan was not created at {plan}")
        smoke = asyncio.run(smoke_test(replica.grpc_endpoint, make_normalizer(arts), stream_factory))
    finally:
        triton_runtime.stop_replica(replica)
    ready.write_text(json.dumps({
        "profile": key, "gpu": hardware.describe(gpu), "identity": identity,
        "triton_first_start_s": round(startup_s, 1), "smoke": smoke,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.perf_counter() - began, 1)}, indent=2))
    log.info("profile %s READY in %.0fs", key, time.perf_counter() - began)
    return profile_dir


def ready_profile(settings: config.Settings, arts: Artifacts, gpu_query=hardware.query_gpu,
                  versions=hardware.library_versions) -> tuple[Path, dict]:
    _, key, _ = detect_profile(settings, arts, gpu_query, versions)
    ready = settings.profiles_dir / key / READY
    if not ready.is_file():
        raise RuntimeError(f"No ready engine profile {key} for this GPU. Run `python -m navoiy_server "
                           "prepare` first (runpod/start.sh does this automatically).")
    return settings.profiles_dir / key, json.loads(ready.read_text())
