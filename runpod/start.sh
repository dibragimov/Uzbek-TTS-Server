#!/usr/bin/env bash
# Container entry point on RunPod (or any host running the Triton image).
# Idempotent: first start on a new GPU type builds engines (long); later starts reuse them.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${NAVOIY_DATA_DIR:-/workspace/navoiy-data}"
mkdir -p "$DATA_DIR/logs"
LOG="$DATA_DIR/logs/start-$(date +%Y%m%d-%H%M%S).log"
# Everything goes to the RunPod log view AND to a file on the volume.
exec > >(tee -a "$LOG") 2>&1

echo "=== navoiy-tts-server $(cat VERSION) | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || {
  echo "FATAL: no GPU visible (nvidia-smi failed)"; exit 1; }

for v in NAVOIY_API_KEY HF_TOKEN; do
  if [ -z "${!v:-}" ]; then echo "FATAL: environment variable $v is not set"; exit 1; fi
done

echo "--- python dependencies"
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m pip install --no-cache-dir -q -r requirements-serving.txt
python3 -c "import huggingface_hub" 2>/dev/null || python3 -m pip install --no-cache-dir -q "huggingface_hub>=0.23,<1.0"
python3 -c "import fastapi, uvicorn, pydantic, inflect, httpx, huggingface_hub, tritonclient.grpc, numpy; print('imports OK')"

if [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "=== public URL (Dograh base URL is this + /v1): https://${RUNPOD_POD_ID}-${NAVOIY_PORT:-8080}.proxy.runpod.net"
fi
echo "=== volume free space: $(df -h "$DATA_DIR" | awk 'NR==2 {print $4}')"

echo "--- prepare (downloads + per-GPU engine build on first start) then serve"
exec python3 -m navoiy_server run
