# Flint single-user serving on M3 Pro

Improve lithos-metal serving of `RecursiveSI/flint-int4`, revision
`3b93f9053b530488405ae9fca51d4900d6ce21f0`, on an Apple M3 Pro with
18 GPU cores, 36 GB unified memory, about 150 GB/s memory bandwidth and
macOS 27. Optimize the engine in this candidate repository for one coding
assistant session, one request at a time. The run uses VibeSys's dynamic
strategy and host-only `metal` backend.

## Hardware evidence

`monolith/backends/metal/m3_pro/config.json` records historical probes on
macOS 26.6.2, battery power, measured 2026-09-19. These describe the chip,
not a baseline measured on the current OS:

- Nominal bandwidth 153.6 GB/s; measured raw streaming 124 to 134 GB/s,
  with about 12 cores needed to saturate the bus.
- One threadgroup per core, 384 threads, 12 SIMD groups per core at full speed.
- Dispatch boundary 1.6 to 2.7 microseconds; GPU dispatch 1.3 to 1.8
  microseconds; 1240-dispatch ICB replay 0.01 to 0.02 ms.
- Synchronization round trip 0.12 to 0.13 ms. Static in-kernel barrier
  2.1 to 2.6 microseconds, fenced 3.1 to 3.9 microseconds.
- Recommended command-buffer duration at most 16 ms. The profile has no
  measured decode GEMV cost table for this chip.

## Validity

The serving computation must use lithos-metal's compiler, runtime and Metal
kernels. Generated kernels and hand-written kernels in the engine's per-chip
override directories are allowed. At serve time do not import or execute
MLX, mlx-lm, rec, llama.cpp, Ollama, or another serving engine. Reading those
implementations as reference is allowed. Existing protocol adapters are not
an alternate engine. The source scan catches mechanical imports and literal
engine launches; the judge must also inspect dynamic imports, renamed copies,
and the executed numerical path. That review requires human/agent judgment
and is not claimed to be mechanically complete.

Use the pinned affine group-64 INT4 projection/expert/head weights and INT8
embeddings unchanged. Repacking exact codes/scales/biases is allowed;
requantization, alternate weights and KV quantization are forbidden. KV must
be BF16. The checkpoint's rec configuration contains quantized-KV defaults;
those defaults do not authorize using them here. No speculative draft model
is supplied. The baseline launcher preserves input normalization before
projections with `--no-commute-norm`, matching the reference's BF16 rounding. Changes must pass the interim accuracy gate below.

Do not modify `.vibesys/tasks/`, model weights, reference pins, the accuracy
gate, benchmark, objectives, or launcher. VibeSys protects task inputs. Do not
recognize evaluator prompts or special-case benchmark/probe outputs. Improve
real computation and caching. Do not drop requests, shorten generations,
return prerecorded replies, report invented usage, or move computation outside
the measured request. The checker and benchmark invalidate failures; the judge
reviews engine identity, numerics and benchmark-specific behavior.

Every command that loads a model or executes Metal work must be wrapped in
`lockf -k /tmp/flint-gpu.lock <command>`. Release it at command completion and
never hold it idle. The accuracy command acquires it in the manifest; the RF adapter acquires it
for its complete server scope.

## Workload and scoring

Request Factory at commit `89dce4a64ae12e7084fbe464ddda882fa0a4c482` owns
replay, HTTP streaming and measurement. The existing VibeSys RF evaluator
injects its pinned engine into the shared task adapter. It replays workload
shapes with synthetic token IDs, not original coding prompts or answers.
`benchmark/trace-identity.json` pins trace and corpus bytes. The checkpoint
manifest pins the Flint tokenizer and chat template. Raw-token Completions
bypasses chat framing during performance replay; accuracy still checks it.

Two tasks share one adapter, launcher, read-only pack policy and accuracy gate:

- `flint-m3pro-quick`: RF-generated synthetic session, two rounds, each
  appending 256 prompt tokens and requesting 32 outputs. Full prompt lengths
  are 256 and 544 tokens. It uses 2048-token serving capacity; full replay
  and the accuracy checker retain 32768-token capacity. This is a bootstrap gradient, not a production
  trace. It must complete all requests within the 600-second scored budget.
- `flint-m3pro`: the first seven rounds of the first session in RF's bundled
  TraceLab execution-v2 example. Input/output lengths remain unchanged. Later
  rounds are excluded because their full prompts exceed the launcher's 32768
  context. This bounded real-trace window is the later optimization target.

Both use `text-generation-session-execution-v2`, saturated arrival,
`max_concurrency=1`, source session order, greedy temperature zero and prefix
reuse. Every `tool_wait_after_ms` is zero: tool time is compressed so the next
round starts immediately after its predecessor. No concurrent users are added.
RF's mandatory cache-telemetry probe runs before scored replay.
Quick uses a 544-token pool to bound that probe. The probe covers this small
pool and can warm the first trace prompt; planned and actual cache hit rates
remain separate diagnostics. Later rounds still add uncached tokens.

The scored budget is 600 seconds, after startup and RF preparation. Every
planned round remains in the denominator. Requests missing at the deadline,
including rounds not yet submitted, are recorded as timed out with TTFT and
TPOT of 600000 ms and decode speed zero. A completion after the deadline is
also censored. Earlier HTTP errors and short outputs invalidate evaluation.
Accuracy remains a separate hard gate and is never inferred from performance.

Objectives minimize `tpot_ms_avg` and `ttft_ms_avg`, and maximize
`completed_requests`. Completed TTFT uses RF `first_token_id_ms`; completed
TPOT uses `token_delivery_tpot_ms`, the time from first to last output-token
event divided by tokens delivered after the first event. These are RF's
summary means when all requests finish, extended with the fixed worst-case
values for unfinished rounds. Diagnostic `decode_tokens_per_second` is the
arithmetic mean of `1000 / TPOT` over planned rounds, zero for timeouts.
Protocol v2 and a JSON sidecar retain per-round outcomes and RF's summary.
Startup, scored replay and whole wall duration are reported separately.

The manifest accepts one benchmark per task, with no staged benchmark switch.
Start dynamic optimization on `flint-m3pro-quick`, stop at a round boundary,
then relaunch on `flint-m3pro` using the selected candidate commit. Scores from
the two tasks are different experiments and must not be compared directly.

## Candidate contract

The trusted launcher starts from the candidate repository root. This example
shows full/accuracy capacity; quick supplies `--max-context 2048`:

```sh
python3 -m monolith.serve --model "$FLINT_MODEL_PATH" \
  --revision 3b93f9053b530488405ae9fca51d4900d6ce21f0 \
  --local-files-only --no-draft --no-commute-norm --served-model-name flint \
  --pack "$FLINT_PACK_DIR" --pack-read-only \
  --max-context 32768 --keep-warm 0 --host 127.0.0.1 --port PORT
```

The launcher resolves `FLINT_PACK_DIR`, defaulting to
`$HOME/.cache/lithos-metal/flint-int4/3b93f9053b530488405ae9fca51d4900d6ce21f0/block-r16-interleaved16-c9216`.
It always passes `--pack PATH --pack-read-only`. The path must be outside the
candidate worktree and already contain a complete pack. Missing or incompatible
packs fail closed, never trigger packing. The shared baseline pack key is
`65dcd02bfffafe7ef9c806b04a40f155ea4690d0111d77e03efff89a53c7ffe9`.
The launcher requires free disk of at least 22,736,175,104 bytes plus 2 GiB
before build or model load. It never copies or repacks weights in a worktree.

A storage-layout change may prepare an extra pack only through
`monolith.serving.pack_policy.SharedPackCache`, in a shared cache outside all
worktrees. Key it by the full SHA-256 storage-compatibility identity, including
checkpoint, quantization, row order, transforms and layout. Preserve the pinned
pack. The cache keeps at most one extra pack, removes least-recently-used extras
before building, admits one pack plus margin, publishes atomically and cleans
failed or interrupted builds. Hold the GPU lock for the entire build or use.
The current experiment authorizes only the existing pack; do not create an extra
pack until disk and operator authorization permit it. Context-only rotary tables
may be generated in memory without changing packed weights.

Preserve these CLI options and the existing serving endpoints:

- `GET /health`: HTTP 200 after startup. The launcher also checks child liveness.
- `GET /v1/models`: served alias `flint`.
- `POST /v1/chat/completions`: full message history, greedy decoding, a
  `max_tokens` budget, streaming and non-streaming responses.
- Streaming is SSE, terminated by `data: [DONE]`, with content in
  `choices[0].delta.content`, `finish_reason=length` at the output limit,
  and truthful `usage.prompt_tokens`/`completion_tokens`. Honor
  `stream_options.include_usage` and `continuous_usage_stats`; publish
  completion count 1 at first committed-token availability, then advance it
  through the requested output limit. Keep-alives and role-only deltas do not count as tokens.
- `POST /v1/completions`: raw token-array prompts, fixed requested output
  length, greedy decoding and incremental `choices[0].token_ids` in streaming
  SSE. Report actual `prompt_tokens_details.cached_tokens`; RF's two-request
  cache probe must succeed. This endpoint uses the same lithos engine.
- Non-streaming pins use 32 tokens and the same template and greedy path.
  Their requests set `return_token_ids=true`; responses must include a top-level `completion_token_ids` array with
  the raw generated IDs, including special tokens. This is a required
  diagnostic extension to Chat Completions: Flint emits special tokens that
  the existing server removes from visible text, so text cannot establish
  exact token equality. The extension must report actual runtime outputs,
  never retokenized visible text.

The evaluated launcher runs CMake in the measured checkout before starting
the server, producing its native extension in `monolith/runtime`. There is
no manifest `[build]` stage: host evaluation does not provide a confined build
runner. A preinstalled extension from another checkout is not a substitute.
Python, CMake and nanobind must be installed in the launch environment.
Build failures invalidate measurement and build time is included in benchmark
wall time.

A server is owned and stopped by each evaluator, including startup and
measurement failures. Startup does not use another installed lithos-metal
checkout; it executes this candidate's `monolith.serve` module.

## Accuracy

`accuracy_checker/checker.py` runs four semantic probes (arithmetic, Python
semantics and multi-turn recall) and four greedy pins. Pins use the serving
endpoint, both orders, and exactly 32 raw completion ids. Prompt IDs must
match the pinned chat template. A token may differ from the pin only when
the pinned reference top-1 minus top-2 margin at that position is below
0.3125. Comparison of that pin stops at the first such difference. A margin
of 0.3125 or more fails. Visible text must match the decode of the raw ids
returned. A short output fails. A server without `completion_token_ids`
fails closed. The gate never reconstructs missing special tokens or accepts
text as a substitute for raw IDs. Missing pins fail closed.

The same command then runs a protected teacher-forced check on a fixed
1024-token prefix. `accuracy_checker/teacher_forced.py` executes this
candidate's prefill engine and reads its logit buffer. There is no logits
HTTP route. The driver does not accept a logit file or a pass flag. Against
the pinned rec trace, max absolute error must be at most 22.5, relative L2
at most 0.19047058, and cosine at least 0.99509772. A top-1 difference is
allowed only where the reference margin is below 0.3125. A missing trace, a
hash mismatch, a missing row, a non-finite value, or a shape mismatch fails
closed.

Thresholds and trace hashes are fixed in
`accuracy_checker/interim_gate_v0.json`. Do not edit that file,
`reference/pin_margins.json`, or the checker. This interim gate is replaced
later by the frozen `flint_gate_v1.json` from the port workstream.

The independent reference uses the same pinned INT4 weights through rec's
plain `QATModel.native` MLX forward with explicit unquantized BF16 KV caches,
without its M4/M5-only megakernel. Reuse the model-port goldens when available.
`reference/make_pins.py` is an operator-only fallback, not candidate code and
never part of serving. Reference provenance is recorded with the pins.
