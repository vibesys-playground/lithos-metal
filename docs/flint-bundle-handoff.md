# Flint M3 Pro bundle operator handoff

Updated 2026-10-21 PDT (2026-10-22 UTC). No optimization agents have been
launched. Accuracy qualification is still pending in the port workstream.

## Enforced accuracy gate and starting eligibility

**The current starting candidate fails the enforced gate. Neither task can
start an optimization run from this input today.** Passing configuration
preflight is not a passing accuracy result. The real RF measurements here are
direct evaluator-adapter performance diagnostics, not framework-accepted
baselines and not optimization runs.

Both `flint-m3pro-quick` and `flint-m3pro` invoke the identical
`.vibesys/tasks/flint-m3pro/accuracy_checker/checker.py` command under the GPU
lock, with an 1800-second accuracy timeout. Both retain the schema's enforced
`benchmark.requires_accuracy=true` default. The checker enforces:

- The mechanical serving-source audit and pinned checkpoint/tokenizer/template provenance.
- Four semantic probes: arithmetic, Python list length and two conversation
  recall cases. Each asks for at most 64 tokens and checks its fixed regex.
- Exactly four current reference pins: `port-1`, `port-2`, `code-0`, `code-1`.
  Each requests exactly 32 greedy tokens and must match every raw token ID,
  the 32-token usage count and the pinned visible text. Prompt token IDs must
  match the pinned chat template. Every pin must pass in forward and
  reverse order; the check stops at the first failure. There is no tolerance
  or short-output allowance.

The latest recorded real check passed all four semantic probes and `port-1`,
then failed `port-2` at zero-based token **22**, expected **3087**, actual
**220**. No model arithmetic changed during this RF continuation. This exact
pin failure therefore still blocks the starting candidate, even though the
port is investigating it as a reference-noise-level winner change.

The exact checker and reference pins are unchanged by this work. No threshold
has been changed or loosened. The separate port workstream is freezing a
teacher-forced gate and will supply a pinned threshold file later. That file
has **not** been received or wired into either task; the teacher-forced gate
is not the gate these tasks enforce today.

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

A source-build Python environment needs CMake, nanobind and the serving
extras. The native extension is built in the evaluated candidate checkout.
An initial attempt using the VibeSys environment failed before model startup:
`transformers 5.20.2` expected a Hub API absent from the VibeSys-locked
`huggingface_hub 1.10.2`. The VibeSys environment now has compatible
`transformers 5.5.4`, verified after `uv run` synchronizes dependencies.
The dedicated lithos source environment remains usable independently.

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

30 hermetic task tests pass, covering the unchanged gate's injected wrong
output rejection, pack provenance, protocol v2, fixed denominators, late
completion censoring, pre-deadline failures, summary mapping and asynchronous
writer skew. Tests use deterministic RF-shaped records, injected timer/transport
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
The unchanged exact-pin accuracy checker last passed all four semantic probes
and the first pin, then failed `port-2` at zero-based token 22: expected 3087,
received 220. No threshold was loosened. The port's long-context frozen gate
is still pending; performance diagnostics alone do not qualify a candidate.

[Structured measurements](flint-bundle-measurements.json) retain historical
observations, the superseded calibration and both final RF configurations.
Both requested real benchmarks have completed, and both protocol rows parse.
No RF change was needed, no weights or packs were copied, and all owned model
servers and benchmark processes have exited. Both configuration preflights
pass; the exact-pin accuracy gate still fails and blocks an optimization run.

The next accuracy step belongs to the separate port workstream: provide its
pinned, frozen teacher-forced threshold file and an explicit follow-up to wire
it into the shared checker. Until then both tasks continue to enforce the
unchanged exact 32-token pins. No optimization agents have been launched.
