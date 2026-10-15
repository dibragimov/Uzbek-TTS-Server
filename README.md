# navoiy-tts-server 1.1.0

Uzbek text-to-speech for Dograh, serving the Navoiy model (fine-tuned CosyVoice2-0.5B, fixed
"calm" voice) through TensorRT-LLM + Triton. It exposes a Speaches/OpenAI-compatible
`POST /v1/audio/speech` that streams 24 kHz, 16-bit mono PCM.

- **Deploying on RunPod:** see [DEPLOY_RUNPOD.md](DEPLOY_RUNPOD.md).
- **What changed since RC4, and why:** see [CHANGES.md](CHANGES.md).

## How it works

```text
Dograh (Speaches provider) --HTTPS--> FastAPI :8080  /v1/audio/speech
                                        |  bearer auth, validation, markdown/emoji cleanup,
                                        |  RC4 telecom normalizer + uztts
                                        |  least-loaded router (sessions per replica, bounded queue)
                     +------------------+------------------+
                     v                                     v
           Triton replica 0 (gRPC 9001)   ...    Triton replica N-1 (gRPC 9000+10(N-1)+1)
           cosyvoice2 orchestrator (RC4 patch)    one tritonserver process each, same GPU
           tensorrt_llm  (BF16 engine)            or one per GPU / MIG slice (NAVOIY_GPUS)
           token2wav     (flow TRT plan + HiFT, exactly ONE instance per replica)
```

The validated RC4 pieces are used unchanged; their SHA-256 checksums match RC4:

- `patch_orchestrator.py`
- `build_spk_cache.py`
- `telecom_normalizer.py`

The generated Triton configs and the patched orchestrator are byte-identical to RC4's, apart from
paths, ports and the KV-cache size. A test generates them both ways and compares them.

### Hardware profiles

TensorRT engines only run on the GPU type they were built for, so the code builds them itself on
first start. The cache key covers:

- the GPU model and compute capability;
- the TensorRT and TensorRT-LLM versions;
- all model revisions;
- the build settings.

A new GPU type (A100 → H100) gets its own profile automatically; no code changes. A profile counts
as ready only after the smoke test passes, which writes `READY.json`. Serving refuses any profile
without it.

### Request lifecycle

Each request is owned by a background session task, not by the HTTP connection. The task opens the
Triton stream, reads it to the end, closes it and releases its slot, whatever the client does. HTTP
200 is sent only once the first audio exists. A client that disconnects (barge-in) stops receiving
audio, but the Triton request finishes normally in the background. That is the cleanup path RC4
validated, because the Triton models do not honour cancellation.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `NAVOIY_API_KEY` | **required** (16+ chars) | Bearer key Dograh must send |
| `HF_TOKEN` | **required** | Read access to the private Navoiy LLM repo |
| `NAVOIY_LLM_REPO` | `dlshod-ibragimov/navoiy-tts-telecom-uz-v1` | Converted Navoiy LLM on Hugging Face |
| `NAVOIY_LLM_REVISION` | `main` | **Pin a commit hash**; a branch is resolved and logged with a warning |
| `NAVOIY_DATA_DIR` | `/workspace/navoiy-data` | Persistent cache: downloads, engines, logs |
| `NAVOIY_HOST` / `NAVOIY_PORT` | `0.0.0.0` / `8080` | API listen address |
| `NAVOIY_REPLICAS` | `1` | Triton servers to run (1-16) |
| `NAVOIY_GPUS` | `0` | Device ids or MIG UUIDs; replica *i* uses entry *i mod n* |
| `NAVOIY_MAX_ACTIVE_PER_REPLICA` | `2` | Concurrent syntheses per replica (RC4-validated value) |
| `NAVOIY_COSY_INSTANCES` | `2` | Orchestrator instances per replica (1-4; RC4 used 2) |
| `NAVOIY_KV_CACHE_TOKENS` | `2560` | TRT-LLM KV cache **shared by all requests of a replica**; raise before raising sessions |
| `NAVOIY_MAX_QUEUE` | `16` | Requests allowed to wait for a slot |
| `NAVOIY_QUEUE_TIMEOUT_S` | `5` | Max wait for a slot before 503 |
| `NAVOIY_REQUEST_TIMEOUT_S` | `60` | Max time for one synthesis |
| `NAVOIY_MAX_INPUT_CHARS` | `1200` | Longer input is rejected with 400 |
| `NAVOIY_ACCEPTED_VOICES` | `calm,alloy,uz,auto,default` | All map to the one fixed voice |
| `NAVOIY_CANCEL_ON_DISCONNECT` | `false` | Experimental: cancel instead of draining on barge-in |
| `NAVOIY_TRITON_BASE_PORT` | `9000` | Replica *i* uses ports base+10*i* (+0 HTTP, +1 gRPC, +2 metrics) |

RunPod bootstrap only (read by the start command, not by the service):

| Variable | Meaning |
|---|---|
| `NAVOIY_GIT_URL` | Git URL of this repository |
| `NAVOIY_GIT_REF` | Branch, tag or commit to run (default `main`) |
| `GITHUB_TOKEN` | Read-only token for a private repo. It is sent as a header and never written to disk. |

Pinned in code (`navoiy_server/config.py`), on purpose, because changing them changes what was
validated:

| Artifact | Pinned revision |
|---|---|
| CosyVoice | `074ca6d…` |
| CosyVoice2-0.5B | `eec1ae6…` |
| aisha-org/navoiy-tts | `912e6bb…` |
| Engine build settings | BF16, max batch 16, max tokens 32768 |
| Instruction | the fixed "calm voice" instruction |

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /v1/audio/speech` | Bearer | Synthesize; body below |
| `GET /health` | none | 200 if at least one replica is up, otherwise 503 |
| `GET /ready` | Bearer | Checks every live Triton replica and its models |
| `GET /metrics` | Bearer | Pool state, counters, p50/p95 timings, profile, GPU, LLM hash |
| `GET /v1/models` | Bearer | `navoiy-tts-v1`, `tts-1` |

`POST /v1/audio/speech` accepts this body:

```json
{"input": "Sizning balansingiz 25 000 so'm.", "model": "navoiy-tts-v1", "voice": "calm",
 "response_format": "pcm", "speed": 1.0}
```

Fields:

- `response_format`: `pcm` streams the audio (this is what Dograh uses); `wav` is buffered.
- `model`: any label is accepted.
- `voice`: must be one of `NAVOIY_ACCEPTED_VOICES`.
- `speed` and `instructions`: ignored with a logged warning.

Errors use OpenAI's shape, `{"error": {"message", "type"}}`:

- Invalid requests return **400**, never 422, because Pipecat only handles 400 cleanly.
- A synthesis failure before any audio returns **502**.
- Capacity limits return **503**, with no `Retry-After` header.

Response headers: `X-Navoiy-Request-Id`, `X-Navoiy-Replica`, and `Server-Timing` with `normalize`,
`queue` and `first_audio` (server-side milliseconds).

Each request logs one JSON line, without the text: outcome, replica, character count,
normalize/queue/first-audio/first-0.5-s timings, audio seconds, and errors. "First 0.5 s" is what a
Dograh caller actually waits for, because Pipecat reads 0.5 s of audio before playing anything.

## Layout on the volume

```text
/workspace/navoiy-tts-server/            code checkout (RunPod start command)
/workspace/navoiy-data/hf/               pinned downloads, one dir per commit:
    CosyVoice2-0.5B@eec1ae6c7987/  navoiy-tts@912e6bb6663f/  navoiy-llm@<sha12>/
/workspace/navoiy-data/src/CosyVoice/    pinned upstream runtime + python package
/workspace/navoiy-data/profiles/<gpu>-sm<cc>-<hash>/
    engine/  runtime_model/  identity.json  build.log  READY.json
/workspace/navoiy-data/run/serve-*/      per-start model repos + triton-replicaN.log
/workspace/navoiy-data/logs/start-*.log  full console log of every container start
```

## Commands

```bash
python3 -m navoiy_server doctor    # config, GPU, cached profiles; starts nothing
python3 -m navoiy_server prepare   # downloads + per-GPU build (idempotent)
python3 -m navoiy_server serve     # start replicas + API (needs a READY profile)
python3 -m navoiy_server run       # prepare, then serve (what runpod/start.sh runs)
```

Test client (standard library only; works on WSL as-is):

```bash
python3 tools/client.py say "Assalomu alaykum!" --out hello.wav
python3 tools/client.py suite                 # RC4's 20 telecom sentences -> WAVs + timings
python3 tools/client.py load --callers 8 --requests 5
```

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

There are 61 tests, none of which need a GPU:

- configs generated from the real pinned upstream and compared byte-for-byte with RC4;
- the `openai` SDK called exactly as Pipecat calls it;
- barge-in and stress tests on a real socket;
- the real `serve()` handling SIGTERM and replica crashes;
- the RunPod start command run against real git;
- the build pipeline with simulated GPU tools.

`tests/test_model_repo_real_upstream.py` needs the pinned CosyVoice checkout at `$COSYVOICE_DIR` and
is skipped without it.

**Not covered by tests:**

- the TensorRT engine build, flow-plan compilation and real Triton inference;
- the `uztts` import and the Hugging Face downloads;
- several replicas sharing one GPU.

The first RunPod start exercises all of these.

## Known limits

- **Uzbek only.** Russian (Cyrillic) input is counted and logged, then read transliterated. Prompt
  the agent to answer in Latin Uzbek.
- **Barge-in costs GPU time.** The rest of an interrupted sentence is still generated in the
  background, at most about one sentence of GPU time.
- **Scaling per GPU.** A single replica is limited by its single Token2Wav instance and the shared
  KV-cache cap. Scale with `NAVOIY_REPLICAS`, then measure.
- **Voice rights.** The fixed voice is Navoiy's demo recording. Decide on voice rights before any
  production use.
