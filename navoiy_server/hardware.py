"""Hardware profile: which compiled engines are valid on this machine.

TensorRT engines are specific to the GPU architecture and TensorRT version, so
cached engines are keyed by GPU model, compute capability, library versions,
model revisions and build settings. A new GPU type gets its own profile,
built once; the code itself never changes per GPU.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from importlib import metadata

from . import config

MIN_COMPUTE_CAPABILITY = (8, 0)  # BF16 engine needs Ampere or newer


@dataclass(frozen=True)
class GpuInfo:
    name: str
    compute_capability: tuple[int, int]
    memory_mib: int


def query_gpu(device: str = "0", runner=subprocess.run) -> GpuInfo:
    """Read the GPU via nvidia-smi (fast, no torch import)."""
    out = runner(
        ["nvidia-smi", f"--id={device}", "--query-gpu=name,compute_cap,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    if not out:
        raise RuntimeError(f"nvidia-smi returned nothing for GPU {device}")
    name, cap, mem = [p.strip() for p in out[0].split(",")]
    major, minor = cap.split(".")
    return GpuInfo(name=name, compute_capability=(int(major), int(minor)), memory_mib=int(float(mem)))


_DISTRIBUTIONS = {
    # Images ship TensorRT under different distribution names; first match wins.
    "tensorrt": ("tensorrt", "tensorrt-cu12", "tensorrt_cu12", "tensorrt-cu13"),
    "tensorrt_llm": ("tensorrt_llm", "tensorrt-llm"),
}


def library_versions(version_of=metadata.version) -> dict[str, str]:
    versions = {}
    for key, names in _DISTRIBUTIONS.items():
        for name in names:
            try:
                versions[key] = version_of(name)
                break
            except metadata.PackageNotFoundError:
                continue
        else:
            raise RuntimeError(f"{key} is not installed; this must run inside the Triton TRT-LLM image")
    return versions


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def profile_key(gpu: GpuInfo, versions: dict[str, str], llm_sha: str,
                settings: config.Settings) -> tuple[str, dict]:
    """Readable prefix plus a hash of everything that changes the build."""
    identity = {
        "gpu_name": gpu.name,
        "compute_capability": list(gpu.compute_capability),
        "versions": versions,
        "cosyvoice_rev": config.COSYVOICE_REV,
        "base_model_rev": config.BASE_MODEL_REV,
        "navoiy_src_rev": config.NAVOIY_SRC_REV,
        "llm_repo": settings.llm_repo,
        "llm_sha": llm_sha,
        "engine": {"dtype": config.ENGINE_DTYPE, "max_batch_size": config.ENGINE_MAX_BATCH_SIZE,
                   "max_num_tokens": config.ENGINE_MAX_NUM_TOKENS},
        "instruction": config.INSTRUCTION,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    major, minor = gpu.compute_capability
    key = f"{slug(gpu.name)}-sm{major}{minor}-{digest}"
    return key, identity


def check_supported(gpu: GpuInfo) -> None:
    if gpu.compute_capability < MIN_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"{gpu.name} (compute capability {gpu.compute_capability}) is not supported: "
            "the engine runs in BF16, which needs Ampere (8.0) or newer, e.g. A100/H100.")


def describe(gpu: GpuInfo) -> dict:
    return asdict(gpu)
