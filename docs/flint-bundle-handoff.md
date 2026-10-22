# Flint M3 Pro bundle operator handoff

Updated 2026-10-21 PDT (2026-10-22 UTC). No optimization agents have been
launched. The enforced gate is the interim gate below. It is replaced when
the port workstream publishes `tools/goldens/gate/flint_gate_v1.json`.

## Enforced accuracy gate and starting eligibility

**Interim gate v0. The starting candidate passes it. A no-sink mutation of
that candidate fails it.** Passing configuration preflight is not a passing
accuracy result. The real RF measurements here are direct evaluator-adapter
performance diagnostics, not framework-accepted baselines and not
optimization runs.

Both `flint-m3pro-quick` and `flint-m3pro` invoke the identical
`.vibesys/tasks/flint-m3pro/accuracy_checker/checker.py` command under the GPU
lock, with an 1800-second accuracy timeout. Both retain the schema's enforced
`benchmark.requires_accuracy=true` default. The checker enforces:

- The mechanical serving-source audit and pinned checkpoint/tokenizer/template provenance.
- Four semantic probes: arithmetic, Python list length and two conversation
  recall cases. Each asks for at most 64 tokens and checks its fixed regex.
- The same four pins: `port-1`, `port-2`, `code-0`, `code-1`. Each requests
  exactly 32 greedy tokens through the serving endpoint, in forward and
  reverse order. Prompt token IDs must match the pinned chat template, and
  the response must carry 32 raw completion ids whose visible text matches
  their decode. A token may differ from the pin only when the pinned
  reference top-1 minus top-2 margin at that index is below 0.3125. That pin
  then stops. A margin of 0.3125 or more fails. There is no short-output
  allowance.
- A teacher-forced check of 1024 positions. The protected driver
  `accuracy_checker/teacher_forced.py` runs this candidate's prefill engine
  and reads its logit buffer. It is not an HTTP route and it does not accept
  a logit file or a pass flag. Limits, against the pinned rec trace: max
  absolute error at most 22.5, relative L2 at most 0.19047058, cosine at
  least 0.99509772, and top-1 differences only where the reference margin is
  below 0.3125. A missing file, a hash mismatch, a missing row, a non-finite
  value, or a shape mismatch fails closed.

Thresholds and hashes live only in
`accuracy_checker/interim_gate_v0.json`. Each limit is twice a rec BF16
chunk-1 versus chunk-32 figure from the port handoff section "Semantic
mutation evidence":

| Check | Limit | Source measurement |
| --- | ---: | --- |
| Greedy and top-1 margin | 0.3125 | 2 x 0.15625 winner-change margin |
| Max logit absolute error | 22.5 | 2 x 11.25 |
| Relative L2 | 0.19047058 | 2 x 0.09523529 |
| Minimum cosine | 0.99509772 | 1 - 2 x (1 - 0.99754886) |

`1 - 0.99754886` is slightly above the rounded distance 0.00245114. The
pinned cosine limit equals `1 - 2 x (1 - 0.99754886)`, which is the approved
value 0.99509772.

Reference margins are `reference/pin_margins.json`, sha256
`38c73864ce1f6f78922d57899761fee677fdf11c1c1a7e373ed73712183dde35`. They were
recorded from rec commit `e53663b` with plain `QATModel.native`, BF16,
unquantized KV, prefill chunk 128, and the pinned tokens fed back in.
Reference top-1 matched every pinned token. `port-2` index 22 has margin
0.203125, so the previously blocking winner change is inside the interim
allowance. The 1024-id prefix is example 1 of the int4 golden, 72 ids
repeated and truncated; its sha256 is
`4930907e7e0ea1a8412c9649855ae8a2e7467c3a8951033e507b5c6ba4878e4b`. The rec
BF16 chunk-32 logit trace is stored outside git as
`flint-gate/rec-bf16-chunk32-logits-1024.npy` under the lithos-metal cache,
sha256 `591792617113cf245d10c9dc645b8a610d76f606bae8bad9cd432b7b54a8fd0e`,
shape `(1024, 150144)` float32.

The 1k driver uses 32-row prefill, `commute_norm` off, accelerator off, and
prefill optimizations off: the long-context schedule the port measured, not
the server `auto-exact` default. Greedy pins still use the real serving
endpoint. Residual trust gap: the 1k check does not exercise the HTTP
handler, and a candidate that special-cases this in-process session could
hide a serving-only numerical change. It cannot pass by returning a file.

On the unmodified starting candidate the full checker passed in 425 seconds
(19:46:33 to 19:53:38 PDT). Probes 4, pins 4, both orders. The only tolerated
stops were `port-2` token 22, margin 0.203125, on both passes. Logits: max
absolute error 11.625, relative L2 0.0826342471188159, minimum cosine
0.9976552267531771, 2 top-1 mismatches at positions 56 and 97, largest
mismatch margin 0.1328125.

Disabling attention sinks after weight binding, the same counterfactual as
the port handoff, failed the checker (exit 1, 19:57:17 to 20:01:31 PDT).
Semantic probe 0 failed before any pin was scored. The 1k check still ran:
max absolute error 69.25, relative L2 1.036409359474286, minimum cosine
0.9526978627987024, 323 top-1 mismatches, largest mismatch margin
20.2109375 at position 43. Those are the port's published no-sink figures.
The sink change was not committed.

This interim file is not `flint_gate_v1.json`. Do not retune these limits
when that frozen file arrives; replace the gate with it.

## Current integration

The fork branch is `vibesys`. The inherited uncommitted RF migration was
completed and first pushed as `2df13f8`. It retained the launcher, shared-pack
policy, accuracy checker, protocol-v2 writer and useful metric logic. The
bespoke `benchmark/core.py`, its streaming parser and workload-only tests were
removed. Stale bespoke operator commands were removed from the task README.
No inherited work was discarded without replacement or explanation.

The branch includes `flint-model` through `b161786`. Its changes remain owned
by the port workstream. This continuation changes no `monolith/` implementation
and requires no additional serving API extension. Existing raw-token
Completions and truthful prefix-cache telemetry support RF directly.

VibeSys's `exp/flint-bundle-2` pins the fork and is merged normally into
`exp/flint-lithos-metal`. The first RF experiment merge is `9689a3761`.
Only the example submodule pointer changes in VibeSys. Fetch before each merge;
never force push or push main. No PR was opened.

## Two tasks and switching

The input schema has one `benchmark: BenchmarkCommand` and rejects unknown
fields; it offers no list of benchmarks or staged benchmark selection.
There are therefore two tasks:

- `flint-m3pro-quick`: one RF-generated synthetic session with two rounds.
  Each adds 256 prompt tokens and requests 32 output tokens. Full prompt
  lengths are 256 and 544 tokens, served with 2048-token capacity. Full replay
  and the shared accuracy checker retain 32768-token capacity. It is a cheap bootstrap gradient,
  not evidence of representative coding traffic.
- `flint-m3pro`: first seven rounds of session 0 from RF's bundled TraceLab
  execution-v2 example. Full prompts are 14438, 16159, 16687, 18049, 27031,
  31294 and 32050 tokens; outputs are 157, 91, 79, 220, 242, 100 and 161.
  The bounded window retains source lengths. Subsequent rounds exceed the
  launcher's 32768 context and are explicitly excluded.

Both manifests call the same full-task RF adapter and accuracy checker. There
is no copied launcher, weight policy, checker or replay implementation. Agent
instructions live only in each task's `OBJECTIVE.md`.

Begin dynamic optimization on `flint-m3pro-quick` only after accuracy passes.
Stop at a round boundary, select the desired candidate commit, and start a
new run with `--task flint-m3pro`. The two tasks have different score
populations. Do not resume a quick run as a full run or compare their means
as if they were the same experiment.

## RF identity and replay policy

The existing evaluator package is `vibesys-evaluator-request-factory` version
`0.1.0`, entrypoint `request-factory-adapter`, result protocol 2. It injects
its trusted `session_runner` into the task adapter. RF owns request replay,
HTTP streaming, prefix construction, request logs and summaries. No replacement
runner was written and no RF source change is needed.

RF is pinned to `89dce4a64ae12e7084fbe464ddda882fa0a4c482`, matching VibeSys's
bundled evaluator. `benchmark/trace-identity.json` records the source URL,
source SHA-256, RF source manifest, synthetic generator settings, selection
policy and SHA-256 of both replay CSVs and the corpus. RF's bundled source
manifest reports 18579 folded prefix tokens across its complete source;
that is source canonicalization metadata, not a measured cache hit here.

Quick was generated through `launcher tracegen`, `generator.type: synthetic`,
one session, two rounds, fixed input 256, output 32, no compaction, constant
arrival rate 1 and seed 0. The pinned generator requires a positive tool-wait
distribution, so generation used 1 ms followed by the same explicit zero-wait
transformation as full. Full uses the already canonical RF bundled artifact;
no raw-session schema is reimplemented in this task.

Both replay `text-generation-session-execution-v2` with saturated arrivals,
`max_concurrency=1`, source order and prefix reuse. Every
`tool_wait_after_ms` is zero. This skips time spent waiting for tools and sends
the next round immediately after the preceding completion. Workload shapes
are replayed with synthetic token IDs, not original private prompt content.

The Flint revision remains `3b93f9053b530488405ae9fca51d4900d6ce21f0`.
`reference/checkpoint.json` pins the tokenizer and chat template along with
checkpoint files. Performance uses raw-token Completions, bypassing chat
framing; the independent accuracy checker still verifies chat framing.

The corpus has enough distinct source text for the full 65536-token pool.
Quick derives its 544-token pool from its largest prompt.
RF's mandatory cache probe is bounded by that pool, so quick does not pay an
8192-token probe. With the small quick pool the probe covers the entire pool
and can warm the first trace prompt on engines able to reuse shorter cached
prefixes. Actual reuse is reported rather than assumed: the current measured
first request is cold. Later rounds still append uncached tokens. Full retains RF's normal tail probe
of 8192 tokens from its larger pool. Neither probe is
scored, but both are included in startup and whole wall time.

The first calibration used three 256-input/16-output rounds and 32768 serving
capacity. It completed 3/3 requests but took 501.2804 seconds overall, including
262.2691 seconds startup and 238.6408 seconds scored replay. Its mean TTFT was
74113.16 ms, mean TPOT 324.952 ms and mean decode speed 6.7562 tokens/s. This
was too expensive for bootstrap. The current quick task replaces that shape
with two rounds and bounded 2048 capacity; the calibration remains evidence,
not the current bootstrap baseline. No full trace or accuracy threshold changed.

## Metrics and fixed budget

Scored replay has a fixed 600-second budget after RF preparation. The adapter
uses request-log creation for its watchdog and reconciles the final cutoff
against RF's first `submit_timestamp`, removing asynchronous writer scheduling
skew. Startup, scored duration and total wall duration are separate. Total
wall time includes model validation, native build, model load, cache probe,
replay and owned-server cleanup; lock queue time is excluded.

Objectives are:

- Minimize `ttft_ms_avg`: RF `first_token_id_ms`, averaged over planned rounds.
- Minimize `tpot_ms_avg`: RF `token_delivery_tpot_ms`, averaged over planned
  rounds. This divides first-to-last generated-token-event time by tokens
  delivered after the first event, excluding terminal usage and `[DONE]`.
- Maximize `completed_requests`: planned rounds completed by the cutoff.

For complete runs, objective means come from RF's summary after comparison
with its request log. No text-only TTFT fallback is accepted. A zero or
non-finite TPOT, short output, earlier HTTP error or inconsistent RF summary
invalidates evaluation.

Every unfinished planned request, including not-yet-submitted rounds, gets
`status=timed_out`, TTFT 600000 ms, TPOT 600000 ms and decode speed zero.
Late completions during cleanup receive the same censoring. The denominator
never shrinks. Completing more work improves the score. Quick must complete
both requests; a quick timeout is an invalid bootstrap measurement.

Diagnostic `decode_tokens_per_second` is the planned-round mean of
`1000 / token_delivery_tpot_ms`, with zero for censored rounds. The protocol
also reports completion/timeout counts, startup and scored duration. The JSON
sidecar preserves every planned outcome, RF's raw records, full summary and
input identity, including planned and measured prefix-cache statistics.
Accuracy remains an independent hard gate.

## Real RF quick measurement

The final quick configuration at candidate `cfcf5b0` completes **2/2 requests**,
**64/64 output tokens**, with **no failures or timeouts**, in **275.1683 seconds**
whole wall time (4.59 minutes), excluding the GPU lock queue. Startup is
**200.7767 seconds** and scored replay is **74.1973 seconds**, below its
600-second budget. Its observed metrics are:

- `ttft_ms_avg`: **35197.83 ms**.
- `tpot_ms_avg`: **55.031774 ms/token**.
- `decode_tokens_per_second`: **18.209088 tokens/s**, averaging per-round rates.

Planned prefix hit rate is **36.0%**; server-measured rate is **31.875%**
(255 cached tokens across 800 prompt tokens). The first request reports zero
cached tokens, the second 255. RF reports zero output mismatches and zero
telemetry drops. The real VibeSys protocol parser accepts the row without a
failure. The owned RF, benchmark and server processes have exited.

This is the current bootstrap baseline. The earlier three-round, 32k-capacity
calibration is a different benchmark configuration; its rates must not be
interpreted as an engine speedup comparison with this one.

## Real RF full measurement

Full replay at candidate `d214519` reached RF preparation and passed the
mandatory cache probe. Startup took **1622.2421 seconds**. The fixed scored
window expired at **600 seconds** while the first 14438-token request was
still prefilling. No trace request completed. RF flushed its own error outcome
and summary after owned-server shutdown.

The valid protocol-v2 row reports **0 completed / 7 timed out**,
`ttft_ms_avg=600000`, `tpot_ms_avg=600000`, and
`decode_tokens_per_second=0`. These are explicitly censored values, not
observed latency or decode rates. Whole wall time, including cleanup, was
**2237.4668 seconds** (37.29 minutes), excluding the GPU lock queue. The real
VibeSys protocol parser accepts this result without a failure.

The full runtime and trace configuration is unchanged by the later quick-only
capacity retuning. RF, benchmark and server processes exited, releasing their
lock. The revised quick command acquired it next. This measurement provides
performance diagnostics; it does not override the failing accuracy gate.

## Pack, engine and process policy

Use only the existing shared pack, read-only, resolved by `FLINT_PACK_DIR`
or its documented `$HOME/.cache/lithos-metal/flint-int4/REVISION/LAYOUT`
default. The pack key is
`65dcd02bfffafe7ef9c806b04a40f155ea4690d0111d77e03efff89a53c7ffe9`.
The launcher checks manifest identity and complete weight-file size, rejects
candidate-worktree aliases and requires 22736175104 bytes plus 2 GiB free.
It always passes `--pack PATH --pack-read-only`. No pack or weight copy was
created. Capacity-only rotary tables may extend in memory after exact
packed-prefix verification. KV remains BF16 and weights remain unchanged.

Every model/Metal command holds `lockf -k /tmp/flint-gpu.lock` for that command.
The accuracy manifest acquires it. The RF adapter re-executes itself under
`lockf`; do not double-lock it with another wrapper. Its hidden lock-held flag
is only for that internal re-exec. Each evaluator owns a fresh server process
group, terminates it on every exit path and releases the lock on completion.
Unrelated reference processes are never stopped by this task.

The task environment is the host equivalent of a task image. Docker cannot
reach Metal on macOS, and VibeSys has no Python-environment hook: `[build]`
runs inside the ambient run environment, and `[environment]` only selects
Modal or docker-in-docker. Accuracy and the benchmark therefore enter an
environment built from the candidate's own `uv.lock`.

`./.vibesys/tasks/flint-m3pro/setup_env.sh` is the one setup command, and it
is the accuracy command in both task manifests. The executable has to be a
path inside the project, so the command is not `/bin/sh`. It runs
`check_host.sh` first, then `uv sync --locked --extra serve
--no-install-project --python 3.12` into
`$HOME/.cache/lithos-metal/task-env/<sha256 of uv.lock>`. The interpreter is
that directory's `bin/python`, used as that path. It is a symlink, and
following it drops the environment. CMake is `bin/cmake` beside it. Task
commands invoke those paths. They do not use the VibeSys virtualenv or PATH. An
unchanged lock reuses the directory. A dependency change must be committed in
`uv.lock`. `--locked` fails with a message that says so, and does not update
the lock or resolve new dependencies. `uv sync --frozen` is not used: uv 0.8
rejects combining it with `--locked`, and `--frozen` alone installs a stale
lock without failing. Weights and the shared pack stay in their own caches
and are not installed into this environment. The native extension is still
compiled into the candidate checkout. A prebuilt extension from another
checkout is not a substitute.

`check_host.sh` is a standalone script. It checks the Xcode command line
tools, the Metal toolchain, macOS 26 or newer, the pinned weight snapshot, and
the shared pack. It prints each missing item and how to fix it, then exits
nonzero. Setup, accuracy, and the benchmark all run it before measuring.
`xcrun -f metal` is accepted when it prints a compiler. The Command Line
Tools on this host do not ship that shader compiler. The native target does
not invoke it: it compiles with `xcrun -f clang` and links `-framework Metal`,
so the check passes when those two are present.

macOS Seatbelt allows the agent to read and execute that cache: reads are
broad, `process-exec` and `file-map-executable` are allowed, and a candidate
worktree's two denied ancestors sit inside the project, not over
`$HOME/.cache`. Writes are denied outside the worktree, so `uv sync` cannot
create or refresh the cache from inside the agent confinement. Use `FLINT_TASK_ENV_MODE=prepared` for confined probes after trusted
operator setup. This validates host prerequisites and loads the exact lockfile
cache without invoking uv or writing shared caches. Missing environments fail
with an operator-setup instruction. The default sync path remains unchanged
for trusted setup and evaluation. See the task README for the warm 16/544-token
TTFT probe entrypoint. No cache write allowance was added.

The request-factory adapter starts `benchmark/run.py` on the evaluator's
interpreter. That file switches to `bin/python` before it imports the task
modules, so an older ambient interpreter (macOS `/usr/bin/python3` is 3.9)
never loads them. Accuracy goes through `setup_env.sh`, which makes the same
switch before the checker.

On 2026-10-21 a fresh clone of `vibesys` at `15e96f2` was run with
`PATH=/usr/bin:/bin`. That PATH has no cmake. Setup printed the cache
`bin/python`. Accuracy, including a cold native build, took 332.22 seconds
and passed the interim gate: 4 probes, 4 pins, 2 repeat passes, 1024 logit
rows, max abs logit 11.625, relative L2 0.082634, cosine 0.997655, two top-1
mismatches at margin 0.1328125. The quick benchmark, started by the
request-factory adapter on `/usr/bin/python3`, took 239.85 seconds, exit 0,
and completed 2 of 2 requests with no timeouts (decode 19.73 tokens/s,
startup 172.4 seconds, scored window 66.9 seconds).

To reproduce via the existing evaluator package, let `RF_ADAPTER` point to
its installed `adapter.py` and `RF_ENGINE` to its pinned tool binary:

```sh
python3 "$RF_ADAPTER" --engine "$RF_ENGINE" -- \
  .vibesys/tasks/flint-m3pro/benchmark/run.py --workload quick \
  --vs-output /tmp/flint-quick.jsonl --output-json /tmp/flint-quick.json
python3 "$RF_ADAPTER" --engine "$RF_ENGINE" -- \
  .vibesys/tasks/flint-m3pro/benchmark/run.py --workload full \
  --vs-output /tmp/flint-full.jsonl --output-json /tmp/flint-full.json
```

## Verification and qualification

54 hermetic task tests pass, covering the unchanged gate's injected wrong
output rejection, pack provenance, protocol v2, fixed denominators, late
completion censoring, pre-deadline failures, summary mapping and asynchronous
writer skew, plus tool resolution, the locked-environment failure, and
importing the benchmark on the system Python without loading task modules.
Tests use deterministic RF-shaped records, injected timer/transport
Fakes and temporary files; no sleeps or real model execution.

VibeSys format and lint scripts pass. Both tasks pass `vibesys validate`,
all six selected external-example registry checks pass, and fork hygiene plus
Ruff E9/F/I pass for the changed task modules. The two newly merged reference
provenance tests also pass without importing a real model.

Both requested preflights pass with `settings.valid=true`, `uses_docker=false`,
backend `metal`, an available evaluator and all six dynamic roles:

```sh
uv run vibesys preflight \
  --project examples/model-serving/repositories/lithos-metal \
  --task flint-m3pro-quick --outer-loop dynamic --backend metal --local \
  --cli-provider codex --model gpt-6.1-sol \
  --max-in-flight 2 --evaluation-slots 1
uv run vibesys preflight \
  --project examples/model-serving/repositories/lithos-metal \
  --task flint-m3pro --outer-loop dynamic --backend metal --local \
  --cli-provider codex --model gpt-6.1-sol \
  --max-in-flight 2 --evaluation-slots 1
```

The historical bespoke baseline completed its 1024-token request at
99.5051 seconds TTFT and 10.1565 decode tokens/s before timing out on the next
request. Whole wall time was 606.0507 seconds. It is superseded, not an RF score.
The interim gate above is what both tasks enforce. The old exact-pin checker
had failed `port-2` at token 22 (expected 3087, received 220) on a reference
margin of 0.203125. That exact rule is no longer the enforced gate. The
port's frozen `flint_gate_v1.json` is still pending and replaces this interim
file when it arrives. Performance diagnostics alone do not qualify a candidate.

[Structured measurements](flint-bundle-measurements.json) retain historical
observations, the superseded calibration and both final RF configurations.
Both requested real benchmarks have completed, and both protocol rows parse.
No RF change was needed, no weights or packs were copied, and all owned model
servers and benchmark processes have exited. Both configuration preflights
pass. The interim accuracy gate passes the starting candidate and fails
the uncommitted no-sink mutation, so an optimization run is no longer blocked
by the exact-pin near-tie. No optimization agents have been launched.

The next accuracy step belongs to the separate port workstream: publish
`flint_gate_v1.json` and replace `interim_gate_v0.json` with it. Do not
loosen the interim limits while waiting.
