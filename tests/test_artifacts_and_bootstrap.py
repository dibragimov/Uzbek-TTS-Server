import json, os, shlex, subprocess, tempfile, unittest
from pathlib import Path
from navoiy_server import artifacts, config
from tests.fakes import settings

ROOT = Path(__file__).resolve().parents[1]

def git(*a, cwd=None):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()

def make_repo(path: Path, files: dict) -> str:
    path.mkdir(parents=True, exist_ok=True)
    if not (path / ".git").exists():
        git("init", "-q", "-b", "main", cwd=path)
        git("config", "user.email", "t@t", cwd=path); git("config", "user.name", "t", cwd=path)
        git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=path)
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    git("add", "-A", cwd=path); git("commit", "-q", "-m", "c", cwd=path)
    return git("rev-parse", "HEAD", cwd=path)


class Snapshots(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.calls = []

    def downloader(self, fail=False):
        def dl(repo_id, revision, local_dir, token, allow_patterns=None):
            self.calls.append((repo_id, revision, allow_patterns))
            Path(local_dir, "config.json").write_text("{}")
            if fail:
                raise ConnectionError("network dropped")
        return dl

    def test_marker_only_after_success_and_cached_after(self):
        dest = self.tmp / "m@abc"
        with self.assertRaises(ConnectionError):
            artifacts.snapshot("org/m", "abc", dest, None, downloader=self.downloader(fail=True))
        self.assertFalse((dest / artifacts.MARKER).exists())   # interrupted -> not complete
        artifacts.snapshot("org/m", "abc", dest, None, downloader=self.downloader())
        artifacts.snapshot("org/m", "abc", dest, None, downloader=self.downloader())
        self.assertEqual(len(self.calls), 2)                   # third call used the cache

    def test_patterns_passed(self):
        artifacts.snapshot("org/src", "r", self.tmp / "s", None, patterns=["uztts/**"], downloader=self.downloader())
        self.assertEqual(self.calls[0][2], ["uztts/**"])

    def test_llm_sha_resolution_and_offline_fallback(self):
        s = settings(data_dir=self.tmp)
        ok = type("Api", (), {"model_info": lambda self, repo, revision, token: type("I", (), {"sha": "f" * 40})()})()
        self.assertEqual(artifacts.resolve_llm_sha(s, api=ok), "f" * 40)
        down = type("Api", (), {"model_info": lambda *a, **k: (_ for _ in ()).throw(ConnectionError("offline"))})()
        with self.assertRaises(RuntimeError) as ctx:
            artifacts.resolve_llm_sha(s, api=down)
        self.assertIn("HF_TOKEN", str(ctx.exception))
        cached = s.hf_dir / f"navoiy-llm@{'e' * 12}"
        cached.mkdir(parents=True)
        (cached / artifacts.MARKER).write_text(json.dumps({"repo": s.llm_repo, "sha": "e" * 40}))
        self.assertEqual(artifacts.resolve_llm_sha(s, api=down), "e" * 40)


class CosyVoiceCheckout(unittest.TestCase):
    def test_pinned_clone_cached_and_repaired(self):
        tmp = Path(tempfile.mkdtemp())
        sha = make_repo(tmp / "upstream", {"runtime/triton_trtllm/run.sh": "echo"})
        orig = (config.COSYVOICE_REPO_URL, config.COSYVOICE_REV)
        config.COSYVOICE_REPO_URL, config.COSYVOICE_REV = f"file://{tmp/'upstream'}", sha
        try:
            dest = tmp / "data/src/CosyVoice"
            artifacts.cosyvoice_source(dest)
            self.assertEqual(git("rev-parse", "HEAD", cwd=dest), sha)
            mtime = (dest / artifacts.MARKER).stat().st_mtime_ns
            artifacts.cosyvoice_source(dest)                    # cached: untouched
            self.assertEqual((dest / artifacts.MARKER).stat().st_mtime_ns, mtime)
            (dest / artifacts.MARKER).unlink()                  # simulate an interrupted clone
            artifacts.cosyvoice_source(dest)
            self.assertTrue((dest / artifacts.MARKER).is_file())
            self.assertFalse(list(dest.parent.glob("CosyVoice.tmp-*")))
        finally:
            config.COSYVOICE_REPO_URL, config.COSYVOICE_REV = orig


class RunPodStartCommand(unittest.TestCase):
    """Runs the exact start command the deploy guide tells you to paste into RunPod."""
    def run_command(self, env):
        cmd = (ROOT / "runpod/start-command.txt").read_text().strip()
        argv = shlex.split(cmd)   # RunPod splits the field like a shell, then execs it
        self.assertEqual(argv[:2], ["bash", "-c"])
        script = argv[2].replace("/workspace/navoiy-tts-server", str(self.workdir))
        return subprocess.run(["bash", "-c", script], env={**os.environ, **env},
                              capture_output=True, text=True, timeout=60)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.workdir = self.tmp / "workspace/navoiy-tts-server"
        self.repo = self.tmp / "remote"
        self.sha1 = make_repo(self.repo, {"runpod/start.sh": 'echo "STARTED $(git rev-parse HEAD)"\n'})

    def test_first_start_update_and_token_header(self):
        env = {"NAVOIY_GIT_URL": f"file://{self.repo}", "NAVOIY_GIT_REF": "main"}
        r = self.run_command(env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"STARTED {self.sha1}", r.stdout)
        sha2 = make_repo(self.repo, {"runpod/start.sh": 'echo "STARTED v2 $(git rev-parse HEAD)"\n'})
        r = self.run_command({**env, "GITHUB_TOKEN": "ghp_fake_token_value"})   # restart picks up new code
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"STARTED v2 {sha2}", r.stdout)
        self.assertNotIn("ghp_fake", (self.workdir / ".git/config").read_text(), "token persisted to disk")
        r = self.run_command({**env, "NAVOIY_GIT_REF": self.sha1})                # pin to an exact commit
        self.assertIn(f"STARTED {self.sha1}", r.stdout)

    def test_bad_ref_fails_loudly(self):
        r = self.run_command({"NAVOIY_GIT_URL": f"file://{self.repo}", "NAVOIY_GIT_REF": "no-such-branch"})
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
