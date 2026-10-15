# Changes from RC4 (navoiy-tts-v1-multisession-modal-server-rc4-headroom)

## Unchanged, deliberately

**Same files.** These match RC4's SHA-256 checksums:

- `patch_orchestrator.py`
- `build_spk_cache.py`
- `telecom_normalizer.py`
- `test_v1_hotfix.py`
- `telecom20_suite.json`
- `telecom20_expected_normalized.json`

**Same pins:**

- Container image `soar97/triton-cosyvoice:25.06`.
- CosyVoice commit `074ca6d`.
- The fixed instruction and reference voice.
- Engine build settings: upstream `run.sh` stage 1 with BF16, max batch 16, max tokens 32768.

**Same Triton configuration.** For the same paths, the generated `config.pbtxt` files and the
patched orchestrator are byte-identical to what RC4's `prepare_runtime.py` produced. A test proves
it.

**Same runtime environment:**

- gRPC route to TRT-LLM.
- Ending mode `soften` with a 15 ms fade and 200 ms of silence.
- LLM timeout of 35 s.
- Token captures disabled.

## Problems fixed

### 1. Slot leaks on client disconnect

RC4 could lose capacity on client disconnect, in two ways:

- **Disconnect while the gRPC stream was opening.** The cancellation is a `BaseException`, which
  `except Exception` doesn't catch, so the slot was never released.
- **Disconnect after the response was created, before streaming began.** The generator's `finally`
  never ran.

Barge-in makes these common in Dograh, and each one permanently lost a session slot. Now a
server-owned task holds each request from open to close and always releases it. This is tested with
60 mixed disconnects on a real socket.

### 2. Stream client

RC4 had two problems here:

- **A thread per waiting request.** Each waiting request occupied a worker thread that polled a
  blocking queue.
- **Audio could be dropped.** The 128-slot queue was fed with `put_nowait`, so if it filled, audio
  or the end marker could be lost silently. A lost end marker meant a hang until timeout.

Callbacks now hand results to the event loop directly, with no bounded queue.

### 3. HTTP 200 sent before any audio existed

A failure before the first audio arrived as a truncated "successful" response. Now 200 is only sent
after the first audio; earlier failures return 502.

### 4. `Retry-After: 2` on capacity errors

The OpenAI SDK used by Pipecat honours this header, so it would have paused a call for 2 s before
retrying. The header is removed.

### 5. Validation errors returned 422

Pipecat's client only turns `BadRequestError` (400) into a clean error frame. All request errors
now return 400 in OpenAI's error shape.

### 6. Speed other than 1.0 rejected

Dograh always sends the speed from its UI slider (0.25-4.0), so a moved slider would break every
TTS request. It is now ignored with a one-time warning. The same applies to `instructions`.

### 7. Cancel-on-disconnect while Triton ignores cancellation

Neither upstream Triton model checks for cancellation. RC4 cancelled the stream and freed the slot
while the GPU most likely kept generating, which overloads the GPU after barge-ins. The vocoder's
per-request cache cleanup also happens on the normal end path.

Now an interrupted request finishes normally in the background before its slot is freed. The old
behaviour remains available as `NAVOIY_CANCEL_ON_DISCONNECT=true` (experimental).

### 8. `/health` returned 200 with `"down"` when Triton had died

It now returns 503 when no replica is healthy. Dead replicas are routed around, and if all of them
die the process exits with code 70 so the container restarts.

### 9. Shutdown was skipped on SIGTERM

Found while testing. Uvicorn re-raises SIGTERM after its own shutdown, which killed the process
before in-flight sessions drained and Triton replicas were stopped.

## New

- **Hardware profiles.** The same code runs on any Ampere-or-newer GPU. Engines are built on first
  start for whatever GPU is present, cached under a key of GPU, compute capability, library
  versions, model revisions and build settings, and only marked `READY` after a smoke test. Every
  build stage is resumable and atomic.
- **Pinned downloads.** These replace RC4's assumption that a Modal volume was already populated.
  Each snapshot is kept in its own commit-named directory and only counted complete after it
  finishes. Your Hugging Face model is resolved to a commit hash and recorded.
- **Replicas.** `NAVOIY_REPLICAS=N` runs N Triton servers, each with its own ports and
  shared-memory prefix. A least-loaded router with a bounded queue sits in front. `NAVOIY_GPUS`
  maps replicas to GPUs or MIG slices.
- **Text cleanup.** Paired markdown markers, headers, list bullets, links and emoji are stripped
  before the RC4 normalizer. USSD and GSM codes and every Uzbek apostrophe form, including the
  backtick, pass through untouched. All 20 RC4 telecom sentences are verified unchanged.
- **Russian text detection.** Cyrillic Russian is counted and logged with a warning.
- **Pipecat-relevant metric.** "First 0.5 s of audio" is measured server-side and by the client
  tool.
- **Bootstrap, tooling and settings:**
  - RunPod start command and `runpod/start.sh`, including a public-URL hint.
  - `doctor` command.
  - Stdlib-only test client.
  - Uvicorn keep-alive raised to 75 s, because Pipecat sends one request per sentence on one
    connection.
  - `NAVOIY_KV_CACHE_TOKENS` is now a setting. RC4's 2560 caps the KV cache shared by all requests
    of a replica.

## Not yet validated: the first GPU run is the test

1. **TRT-LLM engine build** with `convert_checkpoint.py` and `trtllm-build` on A100/H100. The recipe
   is upstream's, the one RC4 used on L4, but it has not run on these GPUs.
2. **Flow-plan compilation and real Triton inference** on A100/H100.
3. **Downloads:** the Hugging Face snapshot downloads, the `uztts` import from the pinned Navoiy
   revision, and TensorRT version detection inside the image.
4. **RunPod's HTTPS proxy:** whether it streams chunked responses without buffering.
5. **Several Triton replicas on one GPU:** memory use, `/dev/shm` size, and the latency effect of
   contention.
6. **Audio quality on the new GPUs.** Listen to the telecom20 suite output, as was done on L4.
