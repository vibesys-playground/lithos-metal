# Operator setup

This is a repository-native VibeSys task in the fork's `vibesys` branch.
The candidate is this entire lithos-metal tree. Model-port work arrives from
`flint-model` after `docs/flint-port-handoff.md` exists. Until then the
hermetic evaluator tests run, but missing reference pins fail the real gate.

Use Python 3.12 and an editable source installation (`pip install -e
'.[serve,dev]'`) so the native runtime is built for this checkout. The task
scripts use `python3` from the active environment. Build and install commands
that execute Metal work also need the GPU lock. No agent run is launched by
these commands. The manifest declares a CMake candidate build so every fresh
measurement checkout gets its own native extension. Install CMake and nanobind
for the same Python interpreter. Host candidate-build support is a VibeSys
prerequisite; a refusal must be fixed there, never bypassed in the trusted gate.

The model is pinned to `RecursiveSI/flint-int4` revision
`3b93f9053b530488405ae9fca51d4900d6ce21f0`. Default resolution uses
`huggingface_hub.snapshot_download(..., local_files_only=True)` and the normal
HF cache (`HF_HOME`/`HF_HUB_CACHE` are honored). It never downloads weights.
Wait for the shared download to finish. Do not create a second weight copy.
`FLINT_MODEL_PATH` may select that cached snapshot or a directory of symlinks
or hardlinks to its files. The evaluator requires identical cached files and
verifies every file's SHA-256 against `reference/checkpoint.json`, taken from
the pinned checkpoint's `flint-int4.json`. Personal paths are not committed.
Weight validation is outside measured benchmark wall time.

From the candidate repository root, with the active source-build environment:

```sh
python3 -m unittest discover -s .vibesys/tasks/flint-m3pro/tests -v
lockf -k /tmp/flint-gpu.lock python3 .vibesys/tasks/flint-m3pro/accuracy_checker/checker.py
lockf -k /tmp/flint-gpu.lock python3 .vibesys/tasks/flint-m3pro/benchmark/run.py \
  --vs-output /tmp/flint-results.jsonl --output-json /tmp/flint-metrics.json
```

If no model-port goldens exist, clone the independent rec reference outside
this repository at commit `e53663b262af4cb1ca498e596ca8b1a2d79e0800`, install
its plain-model dependencies in a reference-only environment, and generate:

```sh
lockf -k /tmp/flint-gpu.lock python3 .vibesys/tasks/flint-m3pro/reference/make_pins.py \
  --rec-repo /path/to/rec
```

Review and commit the generated pins before any experiment. Do not install rec
as a serving dependency. The reference never uses `Model.make_cache`, whose
full-attention defaults quantize KV. Pins have fixed provenance and thresholds;
the task offers no `--skip-pins`, external-server or custom-launch bypass.

VibeSys launch, from a VibeSys checkout with this external repository fetched:

```sh
uv run vibesys --project examples/model-serving/repositories/lithos-metal \
  --task flint-m3pro --outer-loop dynamic \
  --backend metal --local --cli-provider codex --model gpt-6.1-sol \
  --max-in-flight 1
```

`--local` keeps the experiment local and disables publication; it does not
select a run environment. Metal selects the host environment on macOS; do not pass
`--run-environment local` (it is not a supported flag value). Run the exact
same arguments with `vibesys preflight` first. If it returns
`dynamic_run_environment_unsupported`, stop and wait for the separate
host-only dynamic support change. Do not work around it by launching agents.

The task rules, hardware evidence, metric definitions, fixed workloads and
validity contract live in `OBJECTIVE.md`. Operator evidence and remaining work
are recorded outside the task in `docs/flint-bundle-handoff.md` so they do not
become extra optimization-agent instructions.
