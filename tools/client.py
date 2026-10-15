#!/usr/bin/env python3
"""Test client for the Navoiy TTS server. Standard library only (runs on WSL as-is).

  say   - synthesize one sentence to a WAV and print timings
  suite - synthesize RC4's 20 telecom sentences to WAVs (listening check + timings)
  load  - N parallel callers, each sending sentences back to back (like Dograh calls)

Timings are measured the way Pipecat reads the stream:
  first_audio = HTTP headers received (the server only answers once audio exists)
  first_0.5s  = 24,000 bytes received (Pipecat's first read before playback starts)
Environment: NAVOIY_URL (e.g. https://<pod>-8080.proxy.runpod.net), NAVOIY_API_KEY.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import http.client
import json
import os
import ssl
import statistics
import sys
import time
import urllib.parse
import wave
from pathlib import Path

SAMPLE_RATE = 24000
HALF_SECOND = SAMPLE_RATE  # bytes in 0.5 s of 16-bit mono
SUITE = Path(__file__).resolve().parents[1] / "navoiy_server/rc4/telecom20_suite.json"


def connect(url: str, cafile: str | None, timeout: float):
    u = urllib.parse.urlsplit(url)
    if u.scheme == "https":
        ctx = ssl.create_default_context(cafile=cafile)
        return http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout, context=ctx), u.path
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout), u.path


def synth(conn, base_path: str, key: str, text: str, voice: str = "calm") -> dict:
    body = json.dumps({"input": text, "model": "navoiy-tts-v1", "voice": voice,
                       "response_format": "pcm", "speed": 1.0}).encode()
    t0 = time.perf_counter()
    conn.request("POST", base_path.rstrip("/") + "/v1/audio/speech", body=body,
                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    resp = conn.getresponse()
    t_headers = time.perf_counter()
    if resp.status != 200:
        detail = resp.read().decode(errors="replace")[:300]
        return {"ok": False, "status": resp.status, "error": detail}
    first = resp.read(HALF_SECOND)          # blocks until 0.5 s of audio, like Pipecat
    t_half = time.perf_counter()
    rest = resp.read()
    t_end = time.perf_counter()
    pcm = first + rest
    audio_s = len(pcm) / (2 * SAMPLE_RATE)
    return {"ok": True, "status": 200, "pcm": pcm, "audio_s": round(audio_s, 2),
            "first_audio_ms": round((t_headers - t0) * 1000, 1),
            "first_half_second_ms": round((t_half - t0) * 1000, 1) if len(first) >= HALF_SECOND else None,
            "total_ms": round((t_end - t0) * 1000, 1),
            "replica": resp.getheader("X-Navoiy-Replica"),
            "server_timing": resp.getheader("Server-Timing")}


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def summarize(values) -> str:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return "n/a"
    p95 = vals[min(len(vals) - 1, max(0, round(0.95 * len(vals)) - 1))]
    return f"p50 {statistics.median(vals):.0f} ms | p95 {p95:.0f} ms | max {vals[-1]:.0f} ms (n={len(vals)})"


def cmd_say(a):
    conn, path = connect(a.url, a.cafile, a.timeout)
    r = synth(conn, path, a.key, a.text, a.voice)
    if not r["ok"]:
        print(json.dumps(r, ensure_ascii=False)); return 1
    write_wav(Path(a.out), r.pop("pcm"))
    print(json.dumps(r, ensure_ascii=False, indent=2)); print(f"saved {a.out}")
    return 0


def cmd_suite(a):
    cases = json.loads(SUITE.read_text(encoding="utf-8"))["cases"]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    conn, path = connect(a.url, a.cafile, a.timeout)   # one keep-alive connection, like one call
    rows = []
    for c in cases:
        r = synth(conn, path, a.key, c["raw"])
        if r["ok"]:
            write_wav(out / f"{c['label']}.wav", r.pop("pcm"))
        rows.append({"label": c["label"], **r})
        print(json.dumps(rows[-1], ensure_ascii=False))
    ok = [r for r in rows if r["ok"]]
    print(f"\n{len(ok)}/{len(rows)} OK; WAVs in {out}")
    print("first audio :", summarize(r["first_audio_ms"] for r in ok))
    print("first 0.5 s :", summarize(r["first_half_second_ms"] for r in ok))
    return 0 if len(ok) == len(rows) else 1


def cmd_load(a):
    texts = [c["raw"] for c in json.loads(SUITE.read_text(encoding="utf-8"))["cases"]]

    def caller(i):
        conn, path = connect(a.url, a.cafile, a.timeout)
        results = []
        for j in range(a.requests):
            try:
                r = synth(conn, path, a.key, texts[(i * 7 + j) % len(texts)])
                r.pop("pcm", None)
            except Exception as exc:  # connection-level failure
                r = {"ok": False, "status": None, "error": f"{type(exc).__name__}: {exc}"}
                conn, path = connect(a.url, a.cafile, a.timeout)
            results.append(r)
        return results

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(a.callers) as pool:
        rows = [r for rs in pool.map(caller, range(a.callers)) for r in rs]
    wall = time.perf_counter() - t0
    ok = [r for r in rows if r["ok"]]
    print(f"{a.callers} parallel callers x {a.requests} sentences in {wall:.1f}s: "
          f"{len(ok)}/{len(rows)} OK")
    print("first audio :", summarize(r["first_audio_ms"] for r in ok))
    print("first 0.5 s :", summarize(r["first_half_second_ms"] for r in ok))
    replicas = {}
    for r in ok:
        replicas[r["replica"]] = replicas.get(r["replica"], 0) + 1
    print("per replica :", dict(sorted(replicas.items())))
    errors = {}
    for r in rows:
        if not r["ok"]:
            errors[f"{r['status']}: {r['error'][:80]}"] = errors.get(f"{r['status']}: {r['error'][:80]}", 0) + 1
    for e, n in errors.items():
        print(f"error x{n}: {e}")
    return 0 if len(ok) == len(rows) else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=os.environ.get("NAVOIY_URL"))
    p.add_argument("--key", default=os.environ.get("NAVOIY_API_KEY"))
    p.add_argument("--cafile", help="CA bundle, e.g. if a corporate proxy re-signs HTTPS")
    p.add_argument("--timeout", type=float, default=60)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("say"); s.add_argument("text"); s.add_argument("--voice", default="calm")
    s.add_argument("--out", default="navoiy.wav"); s.set_defaults(fn=cmd_say)
    s = sub.add_parser("suite"); s.add_argument("--out-dir", default="navoiy_suite"); s.set_defaults(fn=cmd_suite)
    s = sub.add_parser("load"); s.add_argument("--callers", type=int, default=4)
    s.add_argument("--requests", type=int, default=5); s.set_defaults(fn=cmd_load)
    a = p.parse_args(argv)
    if not a.url or not a.key:
        p.error("set --url/--key or NAVOIY_URL/NAVOIY_API_KEY")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
