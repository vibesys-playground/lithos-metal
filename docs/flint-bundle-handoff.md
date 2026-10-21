# Flint M3 Pro bundle operator handoff

This operator record is outside `.vibesys/tasks/flint-m3pro/`. It is not extra
task objective text. No VibeSys optimization agents have been launched.

## Implemented

- Repository-native task `flint-m3pro`, `metal`, service interface, dynamic
  launch instructions, pinned Recursive Flint INT4 identity and hardware facts.
- Two repeated six-turn coding conversations near 1k, 8k and 32k context;
  exactly 64 greedy tokens per turn, sequential requests and actual assistant
  history replay. Continuous usage measures committed-token arrival timing.
- Protocol v2 objectives: maximize mean decode tok/s across buckets and
  minimize mean TTFT on turns 2+. Per-bucket spread, overall ranges and
  standard deviation, whole-command and workload durations are reported.
  A score requires the whole benchmark to finish within 600 seconds.
- Self-contained evaluated launcher: build this checkout's native extension
  with CMake, resolve/hash existing cached checkpoint files, start the candidate
  module, and clean up its process group on normal/error/termination exits.
  There is no `[build]` manifest stage, per the host-run limitation.
- Four semantic probes and exact 32-token greedy pins replayed twice in
  opposite orders. No threshold/skip/reference override is candidate-controlled.
- Reference generator reuses template-identical model-port goldens, records
  early-EOS exclusions, then generates only additional pins needed to reach
  four. Plain rec `QATModel.native`, explicit unquantized KV caches, no rec
  megakernel. Reference dependencies match the model-port golden provenance.
- The VibeSys repository tracks the fork as a shallow `vibesys` submodule,
  registry `external_repo = true`, with the same lint boundary as other
  external engines. Integration includes dynamic host support at `7fd3fcf45`.

## Exact-ID server requirement

Flint emits special tokens, including a thinking marker. Existing lithos-metal
serving decodes with `skip_special_tokens=True`. Retokenizing response text
cannot establish equality of the raw greedy sequence. The immutable checker
therefore requires a top-level `completion_token_ids` array on non-streaming
Chat Completions responses. It compares all 32 raw IDs, completion usage and
visible decoded text. This diagnostic must report actual runtime outputs.
A missing field is a failure, not a text-only fallback. The model-port server
must implement this extension before the numerical baseline can be valid.
The bundle does not edit `monolith/` to add it.

The conservative baseline launcher also passes `--no-commute-norm`, exposed by
the model-port branch, to preserve the reference's BF16 rounding order.

## Verification completed

- 21 hermetic tests pass on Python 3.12: in-process scripted server, fake
  clock/tokenizer, all stream truncation positions, bad/missing/decreasing
  usage, dropped output, deadline, history/context and metric arithmetic,
  strict pin provenance, raw special-token equality, source scan and protocol.
- Task passes `vibesys validate` and the nine selected registered-example
  checks, including path references and evaluator-input trust policy.
- `./scripts/check_format.sh` and `./scripts/check_lint.sh` pass in VibeSys.
  Task Python passes its own Ruff import/error checks and formatting.
- Native CMake build succeeds on this Mac. It writes the extension to this
  checkout's `monolith/runtime`, not another installed source tree.
- The official VibeSys protocol reader accepts the writer's result row and
  both required objective names.
- Real pinned-tokenizer context counts for the targets were 1024, 1279,
  8191, 8448, 31744 and 31999 with representative previous assistant text.

## Preflight and launch

From the VibeSys checkout, the launch command is:

```sh
uv run vibesys --project examples/model-serving/repositories/lithos-metal \
  --task flint-m3pro --outer-loop dynamic --backend metal --local \
  --cli-provider codex --model gpt-6.1-sol --max-in-flight 1
```

For preflight, insert `preflight` immediately after `vibesys` and use the same
arguments. Preflight after merging `7fd3fcf45` reports settings valid,
backend `metal`, and all six dynamic roles; it does not return
`dynamic_run_environment_unsupported`. It still reports `uses_docker: true`.
That field does not confirm host execution; the separate dynamic-host handoff
records an actual scripted host/Seatbelt run. This bundle launches no agents
and applies no environment workaround.

Activate the Python 3.12 source-build environment first. `--local` keeps the
experiment local, not a run-environment selector. Do not pass the unsupported
`--run-environment local` value.

## Remaining integration and baseline

Do not merge `flint-model` until `docs/flint-port-handoff.md` exists. Then:

1. Merge it into fork branch `vibesys` and read its exact limitations and
   numerical qualification. Preserve this task's checker and thresholds.
2. Verify the raw-ID response extension and input-normalization option. The four
   reference pins are already committed with independent provenance; confirm
   the port uses the same chat template and raw-ID diagnostic contract.
3. Reuse a compatible existing pack with `FLINT_PACK_PATH`; check 32768-context
   compatibility. The observed model-port pack was prepared for 9216 context,
   so compatibility must not be assumed. Disk space was about 12 GiB after
   the shared checkpoint and model-port pack; do not create another full pack
   by default or download another checkpoint copy.
4. Run the real accuracy checker and benchmark under
   `lockf -k /tmp/flint-gpu.lock`, record exact failures or the baseline metric
   row, per-bucket spread, whole wall time and workload time. Never loosen a
   gate to obtain a baseline. Clean up every owned process.
5. Re-pin/push the fork SHA in VibeSys and merge/push the implementation branch
   to `exp/flint-lithos-metal`. Re-run preflight without launching agents.

Baseline numbers are not yet claimed: model-port handoff and exact-ID serving
are prerequisites. The first fallback reference attempt with MLX 0.32.0
stalled waiting for an MLX GPU event. It was terminated, escalated to SIGKILL,
and confirmed exited with the lock released. It produced no pins. A successful
attempt used the port's MLX 0.32.4 / mlx-lm 0.32.2 / Transformers 5.20.2 versions.
It completed on 2026-10-21 at about 22:48 UTC: four 32-token pins, two reused
from model-port goldens and two additional code continuations from rec plain
forward. The reused source SHA-256, versions, excluded early-EOS example and
independent reference commit are recorded in `reference/pins.json`.
All owned reference jobs have exited and released the GPU lock.

## Final handoff state

The model-port branch had no handoff at the final fetch (`6284f3b`), so it was
not merged. No numerical baseline, decode rate, TTFT or benchmark wall time is
claimed. The pre-decided next action is to merge that branch only after its
handoff exists, confirm `completion_token_ids`, and run both real gates with
a validated compatible pack under the GPU lock. Do not launch VibeSys agents
until those prerequisites are met.

The original plan was the first fork commit (`b257d4a`) and is removed at
handoff as requested. The first hermetically passing implementation was
`6677ce7`, within the initial 45-minute target.
