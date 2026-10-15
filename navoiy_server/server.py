"""`serve`: start Triton replicas for the ready profile and run the HTTP API."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import threading
import time

from . import config, triton_runtime
from .api import build_app
from .artifacts import ensure_all
from .profile_build import SMOKE_SENTENCES, make_normalizer, ready_profile, smoke_test
from .replica_pool import ReplicaPool
from .triton_stream import SpeechStream

log = logging.getLogger("navoiy.serve")


def triton_ready_check(replicas: list[triton_runtime.Replica]):
    def check() -> bool:
        import tritonclient.grpc as grpc
        live = [r for r in replicas if r.alive()]
        if not live:
            return False
        for r in live:
            with grpc.InferenceServerClient(url=r.grpc_endpoint, verbose=False) as client:
                if not (client.is_server_ready()
                        and all(client.is_model_ready(m) for m in triton_runtime.MODELS)):
                    return False
        return True
    return check


async def watch_replicas(replicas, pool: ReplicaPool, interval_s: float = 2.0):
    reported = set()
    while True:
        await asyncio.sleep(interval_s)
        for r in replicas:
            if r.index not in reported and not r.alive():
                reported.add(r.index)
                log.error("Triton replica %d exited (code %s). Log tail:\n%s", r.index,
                          r.process.returncode, triton_runtime.tail(r.log_path, 40))
                await pool.mark_unhealthy(r.index)
        if len(reported) == len(replicas):
            log.critical("all Triton replicas are down; exiting so the container restarts")
            await asyncio.sleep(1)
            os._exit(70)


def _uvicorn_server(uvicorn_config):
    import uvicorn

    class Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):
            # Same as uvicorn's, minus the final re-raise of the signal: uvicorn
            # re-raises SIGTERM after shutdown, which kills the process before
            # in-flight sessions drain and Triton replicas are stopped.
            if threading.current_thread() is not threading.main_thread():
                yield
                return
            original = {sig: signal.signal(sig, self.handle_exit) for sig in (signal.SIGINT, signal.SIGTERM)}
            try:
                yield
            finally:
                for sig, handler in original.items():
                    signal.signal(sig, handler)

    return Server(uvicorn_config)


def serve(settings: config.Settings) -> None:
    import uvicorn

    arts = ensure_all(settings)
    profile_dir, ready = ready_profile(settings, arts)
    log.info("profile %s (%s)", profile_dir.name, ready["gpu"]["name"])
    normalizer = make_normalizer(arts)
    run_dir = settings.data_dir / "run" / time.strftime("serve-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    replicas: list[triton_runtime.Replica] = []
    try:
        # Sequential start: avoids N simultaneous engine loads competing for memory.
        for i in range(settings.replicas):
            r = triton_runtime.start_replica(i, arts, profile_dir, settings, run_dir)
            replicas.append(r)
            took = triton_runtime.wait_ready(r, timeout_s=900)
            warm = asyncio.run(smoke_test(r.grpc_endpoint, normalizer, SpeechStream,
                                          sentences=SMOKE_SENTENCES[:1]))
            log.info("replica %d ready in %.0fs, warm-up first audio %.0f ms", i, took,
                     warm[0]["first_audio_ms"])

        info = {"profile": profile_dir.name, "gpu": ready["gpu"]["name"],
                "replicas": settings.replicas, "llm_sha": arts.llm_sha}

        async def main():
            pool = ReplicaPool([r.grpc_endpoint for r in replicas],
                               settings.max_active_per_replica, settings.max_queue)
            app = build_app(settings=settings, pool=pool, normalizer=normalizer,
                            stream_factory=SpeechStream, ready_check=triton_ready_check(replicas),
                            info=info)
            watcher = asyncio.create_task(watch_replicas(replicas, pool))
            server = _uvicorn_server(uvicorn.Config(
                app, host=settings.host, port=settings.port, log_level="warning",
                access_log=False,
                # Pipecat keeps one connection per call and sends a request per
                # sentence; a longer keep-alive avoids reconnecting between them.
                timeout_keep_alive=75))
            log.info("Navoiy TTS %s listening on %s:%d (%d replica(s) x %d sessions, queue %d)",
                     config.VERSION, settings.host, settings.port, settings.replicas,
                     settings.max_active_per_replica, settings.max_queue)
            try:
                await server.serve()
            finally:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher
                # Let in-flight sessions finish briefly so streams close cleanly.
                pending = list(app.state.tasks)
                if pending:
                    await asyncio.wait(pending, timeout=10)

        asyncio.run(main())
    finally:
        for r in replicas:
            triton_runtime.stop_replica(r)
        log.info("stopped")


def doctor(settings: config.Settings) -> dict:
    """Report configuration, GPU and cache state without starting anything."""
    from . import hardware
    report = {"version": config.VERSION, "data_dir": str(settings.data_dir),
              "replicas": settings.replicas, "gpus": list(settings.gpus)}
    try:
        gpu = hardware.query_gpu(settings.gpus[0])
        report["gpu"] = hardware.describe(gpu)
        report["libraries"] = hardware.library_versions()
    except Exception as exc:
        report["gpu_error"] = str(exc)
    profiles = settings.profiles_dir
    report["profiles"] = sorted(p.name + (" (READY)" if (p / "READY.json").is_file() else " (incomplete)")
                                for p in profiles.iterdir() if p.is_dir()) if profiles.exists() else []
    return report
