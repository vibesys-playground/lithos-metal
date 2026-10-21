"""Published host shell: pinned cached checkpoint, HTTP and owned server lifetime.

Caller must hold `lockf -k /tmp/flint-gpu.lock` for this scope. No downloads,
MLX imports, alternate engine, or candidate-defined launch command are used.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from core import Turn, parse_stream

__all__ = [
    "MODEL",
    "REVISION",
    "PinnedTokenizer",
    "HttpChat",
    "model_path",
    "server_endpoint",
    "launch_argv",
]
MODEL = "RecursiveSI/flint-int4"
REVISION = "3b93f9053b530488405ae9fca51d4900d6ce21f0"


def model_path() -> Path:
    from huggingface_hub import snapshot_download

    cached = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
    path = Path(os.environ.get("FLINT_MODEL_PATH", cached)).expanduser().resolve()
    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "reference" / "checkpoint.json").read_text()
    )
    # Cache-only resolution above never downloads a second copy. A supplied
    # path must use these exact cached bytes, including tokenizer/template.
    for name in manifest["checkpoint_files"]:
        if Path(name).name != name or not (path / name).is_file():
            raise ValueError(f"Pinned checkpoint file missing: {name}")
        if not (path / name).samefile(cached / name):
            raise ValueError(f"FLINT_MODEL_PATH must reuse pinned HF cache files: {name}")
        with (path / name).open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != manifest["checkpoint_files"][name]:
            raise ValueError(f"Pinned checkpoint SHA-256 mismatch: {name}")
    if {p.name for p in path.glob("model-*.safetensors")} != {
        n
        for n in manifest["checkpoint_files"]
        if n.startswith("model-") and n.endswith(".safetensors")
    }:
        raise ValueError("Unexpected checkpoint shard set")
    return path


class PinnedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, path: Path):
        from transformers import AutoTokenizer

        return cls(
            AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        )

    def ids(self, messages: list[dict]) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def count(self, messages: list[dict]) -> int:
        return len(self.ids(messages))

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )


class HttpChat:
    def __init__(self, base_url: str, now=time.monotonic):
        self.base_url, self.now = base_url, now

    def request(self, messages: list[dict], max_tokens: int, *, stream: bool):
        body = {
            "model": "flint",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 0,
            "stream": stream,
        }
        if stream:
            body["stream_options"] = {"include_usage": True, "continuous_usage_stats": True}
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=600)

    def complete(self, messages: list[dict], max_tokens: int) -> Turn:
        started = self.now()
        with self.request(messages, max_tokens, stream=True) as response:
            return parse_stream(response, self.now, started)

    def greedy(self, messages: list[dict], max_tokens: int) -> dict:
        with self.request(messages, max_tokens, stream=False) as response:
            body = json.load(response)
        if "error" in body or len(body.get("choices", [])) != 1:
            raise ValueError(f"Invalid completion: {body}")
        return body


def launch_argv(path: Path, port: int, pack_path: Path | None = None) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "monolith.serve",
        "--model",
        str(path),
        "--revision",
        REVISION,
        "--local-files-only",
        "--no-draft",
        "--served-model-name",
        "flint",
        "--max-context",
        "32768",
        "--keep-warm",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if pack_path is not None:
        argv.extend(["--pack", str(pack_path)])
    return argv


@contextlib.contextmanager
def server_endpoint(workspace: Path, path: Path):
    """Own one server process group, including startup failure and caller failure.

    Startup compilation/packing is outside benchmark wall time. Temporary log
    content is surfaced on failure, then removed; no machine paths are saved.
    """
    with tempfile.TemporaryDirectory(prefix="flint-eval-") as temp:
        log_path = Path(temp) / "server.log"
        with log_path.open("w+") as log:
            # Port zero cannot be passed through to the external server's URL.
            # Bind/release a local ephemeral port; a collision fails readiness.
            import socket

            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            pack = os.environ.get("FLINT_PACK_PATH")
            proc = subprocess.Popen(
                launch_argv(path, port, Path(pack).expanduser() if pack else None),
                cwd=workspace,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            url = f"http://127.0.0.1:{port}"

            def terminate(signum, frame):
                raise KeyboardInterrupt("Evaluator terminated")

            prior_term = signal.signal(signal.SIGTERM, terminate)
            try:
                deadline = time.monotonic() + 900
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(
                            f"lithos-metal exited during startup ({proc.returncode})"
                        )
                    try:
                        with urllib.request.urlopen(url + "/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError("lithos-metal startup exceeded 900 seconds")
                    time.sleep(0.2)
                yield url
            except BaseException:
                log.flush()
                print(log_path.read_text()[-8000:], file=sys.stderr)
                raise
            finally:
                signal.signal(signal.SIGTERM, prior_term)
                # Kill the owned group even if its leader exited earlier.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
