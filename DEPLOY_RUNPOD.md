# Deploying navoiy-tts-server on RunPod

This guide gets an A100 (or H100) pod serving Navoiy TTS to Dograh through the Speaches provider.

## How it works

The pod runs the public Triton image `soar97/triton-cosyvoice:25.06`, the same image RC4 ran on
Modal. On every start, RunPod's start command fetches this repository from GitHub at the ref you
choose, then runs `runpod/start.sh`.

- **First start on a GPU type:** downloads the pinned models and builds the engines for that GPU.
  This takes long, plausibly tens of minutes; it hasn't been measured yet.
- **Later starts:** reuse everything from the network volume and only start Triton.

Commands are given one per block so they paste cleanly.

## 0. What you need

- A RunPod account with credit.
- A GitHub account. The code goes in a **private** repo.
- A Hugging Face **read** token that can read `dlshod-ibragimov/navoiy-tts-telecom-uz-v1`.
- On WSL: `git`, `python3`, `openssl`.

## 1. Put the code in a private GitHub repository

1. On github.com, create an empty **private** repository named `navoiy-tts-server`. Don't add a
   README or license, so the first push is clean.

2. On WSL, unzip the package and commit:

```bash
unzip navoiy-tts-server-v1.1.0.zip
```
```bash
cd navoiy-tts-server
```
```bash
git init -b main
```
```bash
git add .
```
```bash
git commit -m "navoiy-tts-server v1.1.0"
```
```bash
git remote add origin https://github.com/<your-user>/navoiy-tts-server.git
```
```bash
git push -u origin main
```

3. Record the commit hash. Pinning `NAVOIY_GIT_REF` to it later makes deployments reproducible:

```bash
git rev-parse HEAD
```

4. Create a token RunPod can use to read the repo. On GitHub, go to **Settings → Developer
   settings → Personal access tokens → Fine-grained tokens → Generate new token** and set:
   - **Repository access:** only `navoiy-tts-server`.
   - **Permissions:** Contents → Read-only. Nothing else.
   - **Expiry:** a short one; this is a test deployment.

## 2. Pin the Navoiy model revision

With `HF_TOKEN` set in your WSL shell, print the current commit of the model repo:

```bash
python3 -c "from huggingface_hub import HfApi; print(HfApi().model_info('dlshod-ibragimov/navoiy-tts-telecom-uz-v1').sha)"
```

Keep the 40-character hash for `NAVOIY_LLM_REVISION`. Without it the server resolves `main` on every
start and logs a warning. If the repo then changes, a new engine is built.

## 3. Generate the API key

```bash
openssl rand -hex 24
```

This is the key Dograh will send. Keep it for steps 4 and 9.

## 4. Create RunPod secrets

In the RunPod console, go to **Secrets → Create Secret** and create three secrets:

| Secret name | Value |
|---|---|
| `navoiy_api_key` | the key from step 3 |
| `hf_token` | your Hugging Face read token |
| `github_token` | the fine-grained GitHub token from step 1 |

Secret names may not start with `RUNPOD`.

## 5. Create a network volume

Go to **Storage → New Network Volume**:

- **Size:** 50 GB. The downloads take a few GB, and each GPU profile adds about 2-3 GB.
- **Datacenter:** pick one that currently lists **A100 80GB**, ideally one that also has **H100**.
  A network volume can only attach to pods in its own datacenter, and H100 can then reuse the same
  downloads.

## 6. Create the pod template

Go to **Templates → New Template**:

| Field | Value |
|---|---|
| Template type | Pod |
| Container image | `soar97/triton-cosyvoice:25.06` |
| Container disk | 30 GB |
| Volume mount path | `/workspace` |
| Expose HTTP ports | `8080` |
| Expose TCP ports | *(leave empty)* |

**Container start command** — paste exactly this one line (it is the tested
`runpod/start-command.txt`):

```bash
bash -c 'set -euo pipefail; R=/workspace/navoiy-tts-server; H=(); if [ -n "${GITHUB_TOKEN:-}" ]; then H=(-c "http.extraHeader=Authorization: Basic $(printf "x-access-token:%s" "$GITHUB_TOKEN" | base64 -w0)"); fi; [ -d "$R/.git" ] || git init -q "$R"; git -C "$R" remote remove origin 2>/dev/null || true; git -C "$R" remote add origin "$NAVOIY_GIT_URL"; git -C "$R" "${H[@]}" fetch -q --depth 1 origin "${NAVOIY_GIT_REF:-main}"; git -C "$R" checkout -q -f FETCH_HEAD; cd "$R"; exec bash runpod/start.sh'
```

**Environment variables:**

| Key | Value |
|---|---|
| `NAVOIY_API_KEY` | `{{ RUNPOD_SECRET_navoiy_api_key }}` |
| `HF_TOKEN` | `{{ RUNPOD_SECRET_hf_token }}` |
| `GITHUB_TOKEN` | `{{ RUNPOD_SECRET_github_token }}` |
| `NAVOIY_GIT_URL` | `https://github.com/<your-user>/navoiy-tts-server.git` |
| `NAVOIY_GIT_REF` | `main`, or better, the commit hash from step 1 |
| `NAVOIY_LLM_REVISION` | the 40-character hash from step 2 |

You can type the secret references or insert them with the key icon in the template editor. Leave
every other `NAVOIY_*` setting at its default for the first deployment; see the README table.

## 7. Deploy the pod

1. Go to **Pods → Deploy**, choose your network volume, and pick **A100 80GB** (PCIe or SXM; each
   gets its own engine profile).
2. Select the template from step 6. One GPU is enough.
3. Click **Deploy**.

## 8. Watch the first start

Open the pod's **Logs** tab. The image probably has no SSH or web terminal, so the logs are your
window into the pod. Every start is also saved to `/workspace/navoiy-data/logs/start-*.log`.

In order, you should see:

1. The image being pulled. It is large, so the first pull takes several minutes.
2. `=== navoiy-tts-server 1.1.0 | commit …`, then the GPU line.
3. `=== public URL (Dograh base URL is this + /v1): https://<POD_ID>-8080.proxy.runpod.net`.
   Copy it.
4. `imports OK`, then the downloads:
   - `downloading FunAudioLLM/CosyVoice2-0.5B@eec1ae6c7987`
   - `downloading aisha-org/navoiy-tts@912e6bb6663f`
   - `downloading dlshod-ibragimov/navoiy-tts-telecom-uz-v1@…`
5. `building profile nvidia-a100-…-sm80-… (first start on this GPU type; this takes a while)`.
6. `$ … convert_checkpoint.py …`, then `$ trtllm-build …`, then `build_spk_cache.py`. Each line is
   followed by `done in Ns`, and command output goes to the profile's `build.log`.
7. `replica 0: GPU 0, gRPC 9001, …`, then
   `Triton ready in Ns (includes flow-plan compilation)`.
8. Three `smoke: {...}` lines, then **`profile … READY`**.
9. The serve phase:
   - `replica 0 ready in Ns, warm-up first audio N ms`
   - **`Navoiy TTS 1.1.0 listening on 0.0.0.0:8080 (1 replica(s) x 2 sessions, queue 16)`**

If the start fails, the log says why. See the troubleshooting table at the end.

## 9. Verify from WSL

Set the URL and key once per shell:

```bash
export NAVOIY_URL=https://<POD_ID>-8080.proxy.runpod.net
```
```bash
export NAVOIY_API_KEY=<key from step 3>
```

Health (no auth):

```bash
curl -s $NAVOIY_URL/health
```

Readiness of every Triton model:

```bash
curl -s -H "Authorization: Bearer $NAVOIY_API_KEY" $NAVOIY_URL/ready
```

One sentence to a WAV you can play:

```bash
curl -s -H "Authorization: Bearer $NAVOIY_API_KEY" -H "Content-Type: application/json" -d '{"input":"Sizning aloqa raqamingiz 90 123 45 67.","model":"navoiy-tts-v1","voice":"calm","response_format":"wav"}' -o test.wav $NAVOIY_URL/v1/audio/speech
```

The 20 RC4 telecom sentences, saved as WAVs, with Pipecat-style timings. Listen to them; this is the
same set you confirmed on L4:

```bash
python3 tools/client.py suite
```

A small load test with 4 parallel "calls" of 5 sentences each:

```bash
python3 tools/client.py load --callers 4 --requests 5
```

Server-side view of the same traffic:

```bash
curl -s -H "Authorization: Bearer $NAVOIY_API_KEY" $NAVOIY_URL/metrics
```

**How to read the timings:**

- The client measures **first audio** and **first 0.5 s**. The second is what a Dograh caller waits
  for.
- Both include the network round trip from Tashkent to the RunPod datacenter. For server-only time,
  compare with `first_audio;dur=` in `Server-Timing`, or the `first_audio_ms` in `/metrics`.
- If the client numbers are much larger than the server's, most of the gap is network or proxy, not
  the model.

**Corporate network:** if `curl` or `client.py` fails with an SSL certificate error, the Beeline
network is re-signing HTTPS. Pass your corporate CA bundle: `curl --cacert <bundle.pem>` and
`client.py --cafile <bundle.pem>`.

## 10. Point Dograh at it

In Dograh, edit the agent's TTS settings:

| Field | Value |
|---|---|
| Provider | **Local Models (Speaches)** |
| Base URL | `https://<POD_ID>-8080.proxy.runpod.net/v1` (note the `/v1`) |
| API key | the key from step 3 |
| Model | `navoiy-tts-v1` |
| Voice | `calm` |
| Speed | `1.0` |

Leaving Model as Dograh's default `kokoro` still works, but the server logs a warning once. Make a
Web Call and check the pod logs: each spoken sentence produces one
`{"event": "tts_request", … "outcome": "completed"}` line.

The agent prompt must make the LLM answer in **Latin-script Uzbek**. Navoiy cannot speak Russian;
Cyrillic Russian is counted as `russian_text` in `/metrics` and read transliterated.

## 11. Day-to-day operations

**Deploy a code change.** Push to GitHub, update `NAVOIY_GIT_REF` if you pin commits, then
**restart** the pod. The start command fetches the new code. Engines are reused unless something in
the hardware key changed.

**Change a setting** (for example `NAVOIY_REPLICAS=2`). Edit the pod's environment variables. RunPod
restarts the pod; `/workspace` is kept.

**Try parallelism on one GPU:**

1. Set `NAVOIY_REPLICAS=2`.
2. Run `client.py load --callers 8 --requests 5`.
3. Compare the p95 times and the per-replica counts with the 1-replica run.

Raise `NAVOIY_MAX_ACTIVE_PER_REPLICA` only together with `NAVOIY_KV_CACHE_TOKENS`.

**Move to H100.** Deploy an H100 pod in the same datacenter with the same volume and template. It
builds its own profile on first start; the A100 profile stays cached for later.

**Save money.** Stop or terminate the pod when you aren't testing. The network volume keeps all
downloads and engines and is billed separately.

**Clean up old builds.** Old profiles under `/workspace/navoiy-data/profiles/` and `run/` folders can
be deleted safely; a missing profile is simply rebuilt.

## Troubleshooting

| Symptom in the log | Cause / fix |
|---|---|
| Pod exits right away; git `fetch` error | `NAVOIY_GIT_URL`/`NAVOIY_GIT_REF` wrong, or `GITHUB_TOKEN` lacks Contents:read on the repo |
| `FATAL: environment variable … is not set` | Missing env var or secret reference in the template |
| `FATAL: no GPU visible` | Pod deployed without a GPU |
| `Cannot resolve …@…: … Check HF_TOKEN` | Token can't read the private model repo, or the revision hash is wrong |
| `tensorrt is not installed` | Wrong container image; must be `soar97/triton-cosyvoice:25.06` |
| `… is not supported: the engine runs in BF16` | GPU older than Ampere (for example V100); use A100/H100 |
| `command failed; last lines of …/build.log` | Engine or speaker-cache build failed; the tail is printed, and the full log is in the profile directory |
| `Triton replica 0 exited` / `not ready after …` | Triton log tail is printed; full log in `/workspace/navoiy-data/run/*/triton-replica0.log` |
| `smoke test produced only …s of audio` | Model produced near-silence; check the LLM revision and the smoke lines |
| `all Triton replicas are down; exiting` | Crash after start; the log tail of each replica is printed before exit |
| Dograh: 401 | API key in Dograh differs from `NAVOIY_API_KEY` |
| Dograh: 400 `Unknown voice` | Voice field not one of `calm, alloy, uz, auto, default` |
| Calls stall, 503 `queue is full` / `Timed out waiting` | More parallel sentences than capacity; raise `NAVOIY_REPLICAS` (then measure) |
| First words of a reply arrive late | Compare the client's "first 0.5 s" with `first_half_second_ms` in `/metrics`. If the server is fast, the time is network or proxy. |

If the RunPod HTTPS proxy turns out to buffer streamed responses, expose port 8080 as a **TCP** port
instead and use `http://<public-ip>:<port>/v1` for a latency comparison. Do this for testing only:
that path is not encrypted.
