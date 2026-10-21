# Flint port handoff

**Stopped at the exact-token gate; this candidate is not qualified for optimization.**
The implementation loads, packs, compiles and decodes on the M3 Pro, but native
continuation differs from the independent reference on the second fixture.
No numeric comparison was loosened. Serving smoke tests and performance
measurements were not run after this failure.

Implementation HEAD pushed to `origin/flint-model`:
`35eea5b30b357fadbed6259db074e842867535e6`.
The subsequent handoff-only commit is identified by the final task reply and
`git rev-parse origin/flint-model`.
Wall clock: **2026-10-21 15:56 PDT**, macOS 27.2 (26B118), Apple M3 Pro,
18 GPU cores, 36 GiB unified memory. Work began at approximately 15:14 PDT;
the numerical stop, rather than the four-hour limit, ended implementation.
No PR was opened; all pushes targeted only `flint-model`. `.vibesys/` was untouched.

## What works

- `monolith/models/flint/`: strict configuration validation, exact checkpoint
  byte views and weight map, an eager forward oracle, shared-layer lowering,
  architecture registration and the checkpoint chat template.
- Affine INT4 projections/experts/output head and unsigned affine INT8 embeddings,
  group 64, preserving checkpoint codes and BF16 scale/bias pairs. No re-quantization.
- Shared `CausalAttention`/`causal_attention`: parameterized causal window,
  optional RoPE, per-head learned softmax sinks, BF16 absolute-position K/V.
- Shared `SigmoidMoE`, `sigmoid_route`, `moe_sum`: top-k sigmoid routing,
  existing expert GEMVs, ungated shared expert and explicit BF16 boundaries.
- Package-owned native Metal emitters through `MetalEmitterContext`, declared
  cache capacity and writes. Static decoding reads live StepState.
- Shared serving parser for `<arg_key>` / `<arg_value>` calls and an explicit
  `--no-commute-norm` precision option. Template/tool-history unit checks pass.
- Real first-three-layer comparison: exact embedding and every checked residual
  passes cosine > 0.999 and max-absolute <= reference scale / 64.

No other model package was changed. The existing `gqa_decode` operation only
received a declarative cache-capacity input; its arithmetic stayed unchanged.
Shared `Experts` gained an optional separate-SiLU rounding flag, defaulting to
its prior behavior. No optimization, recipes tuning, fusion work, draft head or
KV quantization was added.

## Qualification failure

See [the machine-readable report](flint-port-qualification.json).

| Fixture | Result |
| --- | --- |
| Arithmetic prompt | All 64 native greedy IDs match exactly |
| Ocean sentence prompt | First 48 IDs match; first divergence at zero-based index 48 |
| Python expression prompt | Native check not attempted after the failure |

At the divergence:

- Expected token: **11**; native token: **501**.
- Reference top two: IDs `[11, 501]`, logits `[-0.92578125, -1.015625]`;
  margin **0.08984375**.
- Native top two: IDs `[501, 11]`, logits `[-0.96484375, -0.97265625]`;
  margin **0.0078125** favoring the other token.
- Replaying the same expected prefix through prefill also returns **501**.
  This is the remaining full-forward numerical-agreement gap; its exact cause
  was not established before stopping. No claim of unavoidable accumulation
  differences is made.

An earlier index-1 failure was a separate implementation bug: static attention
used parameter position zero. It was fixed before this report. The regression
preserves committed prefix K/V and requires new decode rows to be appended. It
fails at `6284f3b` and passes at `6e7eeaa`. The remaining index-48 failure persists
with that fix and through prefill replay.

## Goldens provenance

Checkpoint: `RecursiveSI/flint-int4` at
`3b93f9053b530488405ae9fca51d4900d6ce21f0`, reused from the default HF cache.
Reference: `recursive-oss/rec` commit
`e53663b262af4cb1ca498e596ca8b1a2d79e0800`,
`engine/src/rec_mlx/quantized.py:QATModel.native` and `model.py`.
Only the plain MLX path ran; no rec megakernel or `install.sh` was used.
Global caches were `KVCache`; sliding caches were `RotatingKVCache`, with no KV
quantization. Versions: MLX 0.32.4, mlx-lm 0.32.2, Transformers 5.20.2.
Tokenizer loading explicitly uses `fix_mistral_regex=False`, matching rec.

`tests/models/flint/goldens/int4.json` contains three prompt-ID arrays, 64 greedy
IDs each, top-10 logits for the first four steps and top-2 logits for every step.
A second reference run reproduced all 192 IDs and first-step top-k logits exactly.
The generator limits the MLX cache and wires its recommended working set to avoid
paging; these settings do not change arithmetic. Measured active model storage
was 22,568,789,640 bytes.

## One shared pack; reuse it

Set this in each consuming shell/worktree:

```sh
export FLINT_PACK_DIR="$HOME/.cache/lithos-metal/flint-int4/3b93f9053b530488405ae9fca51d4900d6ce21f0/block-r16-interleaved16-c9216"
```

The single full pack was moved there from `.agent-logs/flint-pack` by rename.
No second full pack or weights copy was created. No full or stale partial pack
remains under the clone's `.agent-logs/`. Do not repack in candidate worktrees.

- Size: **22,736,175,104 bytes**, 21.17 GiB; 339 slabs and 248 auxiliary tensors.
- Pack key: `65dcd02bfffafe7ef9c806b04a40f155ea4690d0111d77e03efff89a53c7ffe9`.
- Manifest SHA-256: `b01450fe47e91296e7cb7a57f94f795914d954ecbeec4188c9393d28cd0c390c`.
- `pack-key.json` beside the manifest records the identity and producer commit.
- Contract: affine-v1 with plain norm scales, INT4 projections/head, affine INT8
  embedding, group 64, 16 rows, interleaved16 lanes, block scales, lane scale order,
  shared scales and F32 rotary tables through position 9215.

Invalidate for a checkpoint/config/weight change; a bit-width/group/dtype change;
a weight-map, expert split, row permutation, auxiliary transform or rotary-table
change; a packed-layout change; or a context requirement beyond 9216 positions.
Smaller contexts can reuse this pack. Kernel/scheduling/decode changes that retain
the packed-weight contract do not require another pack. The key describes storage
compatibility, **not numerical qualification**.

## Verification evidence

Commands below use the current shared location. During the native runs the same
pack was still at `.agent-logs/flint-pack`; it was moved afterward.
All model-loading/Metal commands used `lockf -k /tmp/flint-gpu.lock`, released
between commands. No model server was started and no background model process remains.
Local raw logs remain in `.agent-logs/` and are not committed.

```sh
# Python 3.12 environment, per README; optional oracle tools installed separately.
.venv/bin/pip install -e '.[serve,dev]'
.venv/bin/python -m pytest tests/contract
# Before optional torch: 1141 passed, 23 skipped.
# With torch 2.14.1: 1161 passed, 16 skipped, 3 failures described below.

.venv/bin/python -m pytest tests/contract/test_flint_package.py \
  tests/contract/test_flint_chat.py tests/layers/test_flint_oracle.py
# 20 passed

.venv/bin/python -m pytest tests/contract/test_shared_causal_moe.py \
  tests/contract/test_ir_coverage.py tests/contract/test_static_geometry.py \
  tests/contract/test_attention_optional_norm.py \
  tests/contract/test_hybrid_moe_package.py tests/contract/test_qwen3_package.py
# 119 passed

lockf -k /tmp/flint-gpu.lock env MTL_SHADER_VALIDATION=1 \
  MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 .venv/bin/python -m pytest \
  tests/kernels/test_int8_affine.py
# 16 passed

lockf -k /tmp/flint-gpu.lock env MTL_SHADER_VALIDATION=1 \
  MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 .venv/bin/python -m pytest \
  tests/kernels/test_causal_decode_state.py tests/kernels/test_causal_attention.py \
  tests/kernels/test_sigmoid_moe.py
# 57 passed

lockf -k /tmp/flint-gpu.lock env MTL_SHADER_VALIDATION=1 \
  MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 .venv/bin/python -m pytest \
  tests/kernels/test_moe_ops.py tests/kernels/test_moe_program.py
# 33 passed
```

The three full-contract failures are in `test_streaming_moe_oracle.py`:
Transformers 5.20.2 no longer accepts `device=` in the reference rotary constructors.
All three reproduce unchanged at original commit `5485ad6bac0a`; no test was
skipped, patched or loosened to hide them.
Ruff checks and formatting pass for new Python modules/tests/tools. Core-error
Ruff checks pass for touched legacy modules; the repository's pre-existing broader
Ruff findings were not suppressed. `tools/ci/hygiene.py` and `git diff --check` pass.

Reference and native evidence entry points:

```sh
export MODEL="$HOME/.cache/huggingface/hub/models--RecursiveSI--flint-int4/snapshots/3b93f9053b530488405ae9fca51d4900d6ce21f0"
# REC_REFERENCE_DIR is the separate checkout of rec at the pinned commit above.
lockf -k /tmp/flint-gpu.lock .venv/bin/python tools/goldens/flint_golden.py \
  --model "$MODEL" --rec "$REC_REFERENCE_DIR" --out reference.json
# All 192 IDs/top-k values reproduced exactly on the second run.

lockf -k /tmp/flint-gpu.lock .venv/bin/python tools/goldens/flint_layers.py reference \
  --model "$MODEL" --rec "$REC_REFERENCE_DIR" --arrays reference-layers.npz
lockf -k /tmp/flint-gpu.lock env MTL_SHADER_VALIDATION=1 \
  MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 .venv/bin/python tools/goldens/flint_layers.py engine \
  --model "$MODEL" --pack "$FLINT_PACK_DIR" --arrays reference-layers.npz
# Embedding: cos=1, max_abs=0.
# Residual cosines: 0.9999876, 0.9999531, 0.9999324, 0.9999678, 0.9999677, 0.9999705.
# Every max_abs <= scale/64; no shader validation report.

lockf -k /tmp/flint-gpu.lock env MTL_SHADER_VALIDATION=1 \
  MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1 .venv/bin/python tools/goldens/check_flint.py \
  --model "$MODEL" --pack "$FLINT_PACK_DIR" --report qualification.json
# Exit 1: fixture 1 matches all 64; fixture 2 diverges at index 48.
```

## Remaining interfaces and baseline

1. Resolve native full-forward numerical agreement with the plain rec oracle,
   preserving the exact-token gate. Do not start optimization from this candidate.
2. Run the actual `lithos-metal serve` chat/tool round trip after qualification;
   only template/parser/history contracts have been verified so far. Use the
   existing shared pack, `--no-draft` and `--no-commute-norm` for this baseline.
3. **Baseline numbers: not measured.** Decode at approximately 1k and 8k context,
   and 8k prefill (three runs each), were withheld at the failed gate.
   `tools/bench/flint_baseline.py` implements the untuned measurement and refuses
   a qualification report whose `qualified` field is false; it has not run.
4. Shared attention intentionally rejects unsupported draft mixer modes. KV
   storage is full BF16 absolute-position storage, not a ring or quantized cache.

## Continuation plan — 2026-10-21

Starting at `664288cbfc8f` on `flint-model`; three-hour investigation budget.

1. Capture teacher-forced residuals, sub-operations, router sets and logits on
   all three fixed prefixes, including fixture 2 through divergence index 48.
   Compare per position, reporting absolute/relative error and cosine; localize
   changes against the steady accumulation trend.
2. Measure an independent reference arithmetic envelope using plain rec MLX
   with BF16 versus FP32 accumulation/ordering, preserving weights and inputs.
   Derive thresholds from this evidence before judging native results.
3. Fix any localized error outside that envelope, with a failing-before,
   passing-after kernel/layer regression and a sweep of sibling mechanisms.
4. Exercise teacher-forced 1k/4k/16k contexts, sliding window boundaries, full
   attention without rotary, and learned sinks. Reuse the existing weight pack;
   investigate extending only rotary auxiliary data in memory beyond 9216.
5. Publish measured gate data (position margins, bounded logit errors, routing
   statistics and greedy behavior), fixed outside candidate control, and prove
   sensitivity with an injected semantic bug. Never qualify by fitting native.
6. After qualification, smoke actual chat/tool serving and measure three runs
   each of 1k/8k decode and 8k prefill. Record failures and limitations explicitly.

Every model/Metal command holds `/tmp/flint-gpu.lock` for that command only.
No `.vibesys/` edits, repack, PR, or branch change. Small commits are pushed
immediately with the requested coauthor trailer. Diagnostic artifacts stay in
`.agent-logs/`; reproducible tools and compact evidence are committed.

### Initial teacher-forced evidence (continuation)

`tools/goldens/flint_trace.py` captures every attention/MLP residual, router
logit and selected expert set, and every vocabulary logit at every input
position. Input IDs are fixed reference prefixes, including all 64 next-token
predictions; native predictions never feed back. It supports an exact golden
schedule (prompt prefill then individual teacher-forced tokens), an alternate
chunk ordering, compact BF16 traces and streaming comparisons for long contexts.
The shared weight file is immutable. Beyond its original 9216-position capacity,
the diagnostic writes only small temporary computed rotary tables, compiling
against the original matrix slabs and an auxiliary-only table overlay.

Fixture 2 canonical reference reproduces all 64 committed golden next tokens,
including index 48 logits `[-0.92578125, -1.015625]`. Native teacher-forced
32-row execution reproduces the prior mismatch (`[-0.97265625, -0.96484375]`).
Across all 72 input positions its max vocabulary error is 2.0, relative L2
0.0206952, minimum cosine 0.9998507. Canonical-versus-32-row rec alone gives
2.25, 0.0175796, and 0.9998552 respectively. These are preliminary evidence,
not qualification thresholds. Other fixtures, isolated operations, long-context
checks and injected-bug sensitivity remain required before a gate is declared.

### Calibration rule (fixed before long-context native results)

Use only rec-to-rec measurements to calibrate: canonical BF16, BF16 at alternate
chunk orderings, and a widened FP32 reference (integer weight codes unchanged;
BF16 scale/bias, router and normalization parameters widen exactly). Keep all
positions, including prompt positions; never discard an outlier because native
misses it. Report each length separately as well as the suite extrema.

For each metric, the proposed conservative acceptance budget is twice the
largest observed reference-to-reference deviation: absolute and relative L2
errors, cosine distance `1-cos`, expert overlap loss and exact-set disagreement.
The factor two is chosen in advance for two independently ordered endpoints;
it is a measured engineering envelope, not a mathematical bound on arbitrary
implementations. The margin threshold is twice the maximum measured change in
the reference top-two logit gap, accompanied by the full-vocabulary error bound
and routing/hidden-state checks. A margin test alone cannot qualify a model.
Report the fraction of positions above this threshold so an overly ambiguous
suite is visible. Strict semantic leaf tests retain their existing two-ULP
contract independently of these whole-model accumulation budgets.

The final numbers and prompt IDs must be copied into the bundle-owned checker
and pinned outside the candidate checkout. A candidate must supply neither
thresholds nor reference traces, and a candidate-authored `qualified` boolean
is not evidence. The checker must compute metrics itself, reject missing rows,
non-finite values, changed routing cardinality and provenance/shape mismatches,
and run a known semantic mutation through the same frozen thresholds. This
continuation will record whether disabling attention sinks is rejected.

Layer-local qualification also replays each decoder block on its reference's
own previous residual, so an earlier routing flip cannot be mistaken for a bug
in a later operation. The short-prefix arithmetic calibration family is fixed
as BF16 chunk sizes `[1, 2, 4, 8, 16, 32, 64, 128]`, the canonical golden schedule,
and FP32. All configurations count; no favorable subset may be selected.
This covers the finite power-of-two GEMV/GEMM/attention ordering family used by
the diagnostic rather than treating one reference pair as an exhaustive bound.
Long-context comparisons use BF16 32-row chunks and FP32 32-row chunks, plus
the complete 1k BF16 one-row trace. All are exact widenings or execution-order
changes of the same checkpoint, never requantizations.
