"""Fetch pinned inputs onto the persistent volume, once.

Each Hugging Face snapshot lives in a directory named by its commit hash and is
only marked complete after the download finishes, so an interrupted download is
resumed rather than used, and a changed revision never overwrites an old one.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import config

log = logging.getLogger("navoiy.artifacts")
MARKER = ".navoiy_complete.json"


@dataclass(frozen=True)
class Artifacts:
    base_model_dir: Path
    navoiy_src_dir: Path
    llm_dir: Path
    llm_sha: str
    cosyvoice_dir: Path

    @property
    def reference_wav(self) -> Path:
        return self.navoiy_src_dir / config.REFERENCE_WAV_REL


def _complete(path: Path, sha: str) -> bool:
    marker = path / MARKER
    if not marker.is_file():
        return False
    try:
        return json.loads(marker.read_text()).get("sha") == sha
    except (OSError, ValueError):
        return False


def resolve_llm_sha(settings: config.Settings, api=None) -> str:
    """Resolve the requested LLM revision (branch or commit) to a commit hash.

    If the Hub is unreachable but a completed snapshot exists, fall back to the
    newest cached one so a restart without network still works.
    """
    rev = settings.llm_revision
    try:
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        sha = api.model_info(settings.llm_repo, revision=rev, token=settings.hf_token).sha
        if not sha:
            raise RuntimeError("Hub returned no commit hash")
        if rev != sha:
            log.warning("NAVOIY_LLM_REVISION=%s resolved to %s; pin the hash for reproducible builds",
                        rev, sha)
        return sha
    except Exception as exc:  # network/auth problems
        cached = sorted(settings.hf_dir.glob("navoiy-llm@*"), key=lambda p: p.stat().st_mtime,
                        reverse=True) if settings.hf_dir.exists() else []
        for path in cached:
            try:
                info = json.loads((path / MARKER).read_text())
            except (OSError, ValueError):
                continue
            if info.get("repo") == settings.llm_repo and (rev in ("main", info.get("sha"))):
                log.warning("Could not reach the Hub (%s); using cached LLM %s", exc, info["sha"])
                return info["sha"]
        raise RuntimeError(
            f"Cannot resolve {settings.llm_repo}@{rev}: {exc}. Check HF_TOKEN has read access "
            "to the private repo.") from exc


def snapshot(repo: str, sha: str, dest: Path, token: str | None, patterns=None,
             downloader=None) -> Path:
    if _complete(dest, sha):
        log.info("cached: %s@%s", repo, sha[:12])
        return dest
    if downloader is None:
        from huggingface_hub import snapshot_download as downloader
    log.info("downloading %s@%s -> %s", repo, sha[:12], dest)
    dest.mkdir(parents=True, exist_ok=True)
    kwargs = {"repo_id": repo, "revision": sha, "local_dir": str(dest), "token": token}
    if patterns:
        kwargs["allow_patterns"] = patterns
    downloader(**kwargs)
    (dest / MARKER).write_text(json.dumps({"repo": repo, "sha": sha}))
    return dest


def _git(args, cwd=None, runner=subprocess.run):
    return runner(["git", *map(str, args)], cwd=cwd, check=True, capture_output=True, text=True)


def cosyvoice_source(dest: Path, runner=subprocess.run) -> Path:
    """Pinned upstream CosyVoice (Triton runtime + python package) with submodules."""
    if (dest / ".git").exists():
        head = _git(["-C", dest, "rev-parse", "HEAD"], runner=runner).stdout.strip()
        if head == config.COSYVOICE_REV and (dest / MARKER).is_file():
            log.info("cached: CosyVoice@%s", head[:12])
            return dest
        log.warning("CosyVoice checkout at %s is incomplete or at %s; re-cloning", dest, head[:12])
    tmp = dest.with_name(f"{dest.name}.tmp-{uuid.uuid4().hex[:8]}")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        _git(["clone", "--no-checkout", "--depth", "1", config.COSYVOICE_REPO_URL, tmp], runner=runner)
        _git(["-C", tmp, "fetch", "--depth", "1", "origin", config.COSYVOICE_REV], runner=runner)
        _git(["-C", tmp, "checkout", "--detach", config.COSYVOICE_REV], runner=runner)
        _git(["-C", tmp, "submodule", "update", "--init", "--recursive", "--depth", "1"], runner=runner)
        head = _git(["-C", tmp, "rev-parse", "HEAD"], runner=runner).stdout.strip()
        if head != config.COSYVOICE_REV:
            raise RuntimeError(f"CosyVoice checkout is {head}, expected {config.COSYVOICE_REV}")
        (tmp / MARKER).write_text(json.dumps({"sha": head}))
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return dest


def ensure_all(settings: config.Settings, *, api=None, downloader=None,
               git_runner=subprocess.run) -> Artifacts:
    settings.hf_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(settings.data_dir).free / 1e9
    if free_gb < 15:
        log.warning("only %.1f GB free on %s; first build needs about 15 GB", free_gb, settings.data_dir)
    llm_sha = resolve_llm_sha(settings, api=api)
    base = snapshot(config.BASE_MODEL_REPO, config.BASE_MODEL_REV,
                    settings.hf_dir / f"CosyVoice2-0.5B@{config.BASE_MODEL_REV[:12]}",
                    settings.hf_token, downloader=downloader)
    src = snapshot(config.NAVOIY_SRC_REPO, config.NAVOIY_SRC_REV,
                   settings.hf_dir / f"navoiy-tts@{config.NAVOIY_SRC_REV[:12]}",
                   settings.hf_token, patterns=config.NAVOIY_SRC_PATTERNS, downloader=downloader)
    llm = snapshot(settings.llm_repo, llm_sha, settings.hf_dir / f"navoiy-llm@{llm_sha[:12]}",
                   settings.hf_token, downloader=downloader)
    cosy = cosyvoice_source(settings.cosyvoice_dir, runner=git_runner)
    arts = Artifacts(base, src, llm, llm_sha, cosy)
    for required in (arts.reference_wav, base / "cosyvoice2.yaml",
                     base / "flow.decoder.estimator.fp32.onnx", base / "CosyVoice-BlankEN",
                     llm / "config.json", src / "uztts",
                     cosy / "runtime/triton_trtllm/scripts/convert_checkpoint.py"):
        if not required.exists():
            raise RuntimeError(f"Expected artifact missing after download: {required}")
    return arts
