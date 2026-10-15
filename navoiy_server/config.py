"""All settings in one place.

Pinned artifacts are constants on purpose: changing them changes what was
validated, so it should be a code change with a commit, not an env tweak.
Everything that legitimately differs per deployment comes from the environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

VERSION = "1.1.0"

# --- Pinned upstream artifacts (from the RC4 handover) -----------------------
COSYVOICE_REPO_URL = "https://github.com/QwenAudio/CosyVoice.git"
COSYVOICE_REV = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
BASE_MODEL_REPO = "FunAudioLLM/CosyVoice2-0.5B"
BASE_MODEL_REV = "eec1ae6c79877dbd9379285cf8789c9e0879293d"
NAVOIY_SRC_REPO = "aisha-org/navoiy-tts"
NAVOIY_SRC_REV = "912e6bb6663f11e5dec5de830ca06ea8dd4c985f"
# Only what serving needs from the Navoiy source repo: the normalizer package
# and the fixed reference voice. The 600h .pt checkpoint is not needed because
# the converted LLM comes from the private HF repo.
NAVOIY_SRC_PATTERNS = ["uztts/**", "demo/calm_intro.wav"]
REFERENCE_WAV_REL = "demo/calm_intro.wav"
INSTRUCTION = ("Speak in a calm, steady, neutral voice with even pacing and "
               "relaxed intonation.<|endofprompt|>")

# --- TensorRT-LLM engine build (upstream run.sh stage 1, as used for L4) ----
ENGINE_DTYPE = "bfloat16"
ENGINE_MAX_BATCH_SIZE = 16
ENGINE_MAX_NUM_TOKENS = 32768

SAMPLE_RATE = 24000
BYTES_PER_SECOND = SAMPLE_RATE * 2
# Pipecat's OpenAI/Speaches TTS client reads 0.5 s chunks before pushing the
# first frame, so "time to 0.5 s of audio" is what a Dograh caller waits for.
PIPECAT_FIRST_CHUNK_BYTES = BYTES_PER_SECOND // 2


def _int(name: str, default: int, lo: int, hi: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
    return value


def _float(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value}")
    return value


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    hf_token: str | None
    llm_repo: str
    llm_revision: str
    api_key: str
    host: str
    port: int
    replicas: int
    gpus: tuple[str, ...]
    cosy_instances: int
    max_active_per_replica: int
    max_queue: int
    queue_timeout_s: float
    request_timeout_s: float
    kv_cache_tokens: int
    max_input_chars: int
    accepted_voices: frozenset[str] = field(default_factory=frozenset)
    cancel_on_disconnect: bool = False
    triton_base_port: int = 9000

    # Derived layout on the persistent volume.
    @property
    def hf_dir(self) -> Path:
        return self.data_dir / "hf"

    @property
    def cosyvoice_dir(self) -> Path:
        return self.data_dir / "src" / "CosyVoice"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    def replica_ports(self, index: int) -> tuple[int, int, int]:
        """(http, grpc, metrics) for Triton replica `index`; app port is separate."""
        base = self.triton_base_port + 10 * index
        return base, base + 1, base + 2


def load(require_api_key: bool = True) -> Settings:
    api_key = os.environ.get("NAVOIY_API_KEY", "").strip()
    if require_api_key and len(api_key) < 16:
        raise ValueError("NAVOIY_API_KEY must be set (at least 16 characters)")
    voices = os.environ.get("NAVOIY_ACCEPTED_VOICES", "calm,alloy,uz,auto,default")
    accepted = frozenset(v.strip() for v in voices.split(",") if v.strip())
    if not accepted:
        raise ValueError("NAVOIY_ACCEPTED_VOICES must list at least one voice")
    gpus = tuple(g.strip() for g in os.environ.get("NAVOIY_GPUS", "0").split(",") if g.strip())
    if not gpus:
        raise ValueError("NAVOIY_GPUS must list at least one device id")
    port = _int("NAVOIY_PORT", 8080, 1, 65535)
    settings = Settings(
        data_dir=Path(os.environ.get("NAVOIY_DATA_DIR", "/workspace/navoiy-data")),
        hf_token=(os.environ.get("HF_TOKEN") or None),
        llm_repo=os.environ.get("NAVOIY_LLM_REPO", "dlshod-ibragimov/navoiy-tts-telecom-uz-v1"),
        llm_revision=os.environ.get("NAVOIY_LLM_REVISION", "main").strip() or "main",
        api_key=api_key,
        host=os.environ.get("NAVOIY_HOST", "0.0.0.0"),
        port=port,
        replicas=_int("NAVOIY_REPLICAS", 1, 1, 16),
        gpus=gpus,
        # RC4 validated 2 orchestrator instances; upstream allows up to 4.
        cosy_instances=_int("NAVOIY_COSY_INSTANCES", 2, 1, 4),
        max_active_per_replica=_int("NAVOIY_MAX_ACTIVE_PER_REPLICA", 2, 1, 16),
        max_queue=_int("NAVOIY_MAX_QUEUE", 16, 0, 1000),
        queue_timeout_s=_float("NAVOIY_QUEUE_TIMEOUT_S", 5.0, 0.1, 120.0),
        request_timeout_s=_float("NAVOIY_REQUEST_TIMEOUT_S", 60.0, 5.0, 600.0),
        # RC4 value. It caps the KV cache shared by ALL in-flight requests of a
        # replica, so raise it before raising sessions per replica on big GPUs.
        kv_cache_tokens=_int("NAVOIY_KV_CACHE_TOKENS", 2560, 1024, 262144),
        max_input_chars=_int("NAVOIY_MAX_INPUT_CHARS", 1200, 50, 10000),
        accepted_voices=accepted,
        cancel_on_disconnect=_bool("NAVOIY_CANCEL_ON_DISCONNECT", False),
        triton_base_port=_int("NAVOIY_TRITON_BASE_PORT", 9000, 1024, 60000),
    )
    last_port = settings.replica_ports(settings.replicas - 1)[2]
    if settings.triton_base_port <= port <= last_port:
        raise ValueError(f"NAVOIY_PORT {port} collides with Triton ports "
                         f"{settings.triton_base_port}-{last_port}")
    return settings
