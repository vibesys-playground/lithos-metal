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
projections with `--no-commute-norm`, matching the reference's BF16 rounding. Changes must retain greedy output agreement.

Do not modify `.vibesys/tasks/`, model weights, reference pins, the accuracy
gate, benchmark, objectives, or launcher. VibeSys protects task inputs. Do not
recognize evaluator prompts or special-case benchmark/probe outputs. Improve
real computation and caching. Do not drop requests, shorten generations,
return prerecorded replies, report invented usage, or move computation outside
the measured request. The checker and benchmark invalidate failures; the judge
reviews engine identity, numerics and benchmark-specific behavior.

Every command that loads a model or executes Metal work must be wrapped in
`lockf -k /tmp/flint-gpu.lock <command>`. Release it at command completion and
never hold it idle. Both evaluator commands acquire it in the manifest.

## Workload and scoring

`benchmark/run.py` starts a fresh server, then sends one three-turn coding
conversation. Each request carries the actual earlier replies and full
accumulated history. The next request starts only after completion of the
previous request; there is no think time or concurrent load. The context
targets, counted with the pinned tokenizer and chat template, are 1024, 8192
and 32000 tokens, within 32 tokens below each target. Source-review material
grows the conversation. There is no persistent disk prefix cache.

The original baseline workload had two repetitions of six turns, with paired
turns near each bucket. A real reduced run measured 134.26 seconds TTFT at
1024 tokens before the larger buckets. The operator reduced repetitions to
one and turns to three, retaining all context buckets, to minimize work under
the 600-second total budget. A remaining timeout invalidates measurement;
candidates may not change this workload or omit the long-context request.

Each turn requests exactly 64 greedy output tokens, `temperature=0`, seed 0.
An early EOS is an invalid measurement; it is not silently padded or ignored.
The server must stream continuous usage so the first committed token and last
committed token can be timed even when text is held for UTF-8 decoding.

Two objectives in `objectives.toml` are ranked:

- Maximize `decode_tokens_per_second`: arithmetic mean of each turn's
  `(64 - 1) / (last token arrival - first token arrival)`, equally weighting
  the three context buckets. Prefill is excluded.
- Minimize `mean_ttft_turn2plus_ms`: mean request-to-first-committed-token
  latency of turns 2 and 3. It includes prefix
  lookup, uncached prefill, runtime and HTTP overhead.

The protocol-v2 result includes per-bucket means and ranges, overall decode
min/max/standard deviation, TTFT min/max and measured wall time. The JSON
sidecar records every turn's context length, rate and any reported cache-token
count. Cache counts are diagnostic and may be absent in the current server.
Metrics must not rely on a claimed cache hit instead of measured latency.
The whole benchmark command, including weight hashing, loading, packing,
startup warmup, requests and shutdown, must finish within 600 seconds to
produce a score. `wall_duration_s` reports that interval;
`workload_duration_s` reports the request workload alone. Throughput and
TTFT use per-request timings, so startup does not distort those objectives.
The launcher also has a separate 900-second startup hang guard for the
accuracy checker. A command-wide deadline interrupts blocking startup or requests at 600 seconds
and unwinds through owned-server cleanup. If baseline prefill still cannot meet
the budget, report that failure; do not silently retune the workload.

## Candidate contract

The trusted launcher starts from the candidate repository root:

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
  through count 64. Keep-alives and role-only deltas do not count as tokens.
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
semantics and multi-turn recall) plus at least four independent greedy pins.
Every 32-token pin must match exactly, on two passes in opposite order, with
no short-output tolerance or tunable threshold. Prompt IDs must also match
the pinned tokenizer's chat template. Raw completion IDs and visible decoded
text are both checked. A server without `completion_token_ids` fails closed;
the gate never reconstructs missing special tokens or accepts text as a
substitute for raw IDs.
Missing pins fail closed and never become a passing baseline.

The independent reference uses the same pinned INT4 weights through rec's
plain `QATModel.native` MLX forward with explicit unquantized BF16 KV caches,
without its M4/M5-only megakernel. Reuse the model-port goldens when available.
`reference/make_pins.py` is an operator-only fallback, not candidate code and
never part of serving. Reference provenance is recorded with the pins.
