# Operator setup

See `docs/flint-bundle-handoff.md` for operator setup and measured evidence.
The task contract is in `OBJECTIVE.md`.

## Confined probes use the prepared environment

Trusted operator setup and evaluation default to `FLINT_TASK_ENV_MODE=sync`:
check host prerequisites, then `uv sync --locked` into the cache addressed by
this candidate's exact `uv.lock` SHA-256. Before dispatching agents, run
`FLINT_TASK_ENV_MODE=sync .vibesys/tasks/flint-m3pro/setup_env.sh` outside
agent confinement for each changed lockfile. Setup prints the venv interpreter.

Agents explicitly set `FLINT_TASK_ENV_MODE=prepared`. Both setup and the
benchmark entrypoint then check host prerequisites and load that exact cache's
executable `bin/python` and `bin/cmake`, without invoking uv, acquiring the
setup lock, creating directories, or repairing the cache. Missing or partial
caches fail with an operator-setup instruction; another lock's cache is never
used. Unknown mode values fail. Keep this variable on probe commands only so
official evaluation keeps its existing sync path. No cache write allowance is
needed. Native builds still write only into the candidate checkout.

## Warm 16/544-token TTFT probe entrypoint

From the candidate root, save the Python below to an agent-writable artifact
path such as `/tmp/flint-warm-probe.py`, then run:

```sh
export FLINT_TASK_ENV_MODE=prepared
TASK_PYTHON=$(./.vibesys/tasks/flint-m3pro/setup_env.sh) || exit
/usr/bin/lockf -k /tmp/flint-gpu.lock "$TASK_PYTHON" /tmp/flint-warm-probe.py
```

The GPU lock encloses the entire server lifetime. Do not wrap the official
benchmark or accuracy adapter again: those already acquire this lock. This
probe owns its server through the published launcher context manager, builds
this checkout, performs one unscored warmup, then streams raw-token Completions
in counterbalanced order. TTFT ends at the first nonempty text event; empty
metadata events do not count. These are diagnostic observations, not official
RF scores. Prefix reuse can influence repeated requests; record the order.
The corpus is tokenized locally and each request sends exactly 16 or 544 IDs.

```python
import json
import sys
import time
import urllib.request
from pathlib import Path

workspace = Path.cwd()
benchmark = workspace / ".vibesys/tasks/flint-m3pro/benchmark"
sys.path.insert(0, str(benchmark))
from launcher import PinnedTokenizer, model_path, server_endpoint

path = model_path()
tokenizer = PinnedTokenizer.load(path)
tokens = tokenizer.encode((benchmark / "corpus.txt").read_text())
assert len(tokens) >= 544


def request(url, length):
    body = {"model": "flint", "prompt": tokens[:length], "max_tokens": 1,
            "temperature": 0, "stream": True}
    req = urllib.request.Request(
        url + "/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    start = time.monotonic()
    ttft = None
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            if ttft is None and any(c.get("text") for c in event.get("choices", [])):
                ttft = time.monotonic() - start
    if ttft is None:
        raise RuntimeError("No visible completion token")
    return {"prompt_tokens": length, "ttft_seconds": ttft}


with server_endpoint(workspace, path, max_context=2048) as url:
    request(url, 16)  # unscored server warmup
    for index, length in enumerate((16, 544, 544, 16)):
        print(json.dumps({"order": index, **request(url, length)}), flush=True)
```
