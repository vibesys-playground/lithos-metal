# Flint port handoff

## Resume: 2026-10-21 20:03 PDT

The accuracy gate is frozen at `tools/goldens/gate/flint_gate_v1.json` (schema 1). Its thresholds, metrics, and calibration family stay as written. Native teacher-forced traces pass that gate at the short fixtures and at 1024 positions. They fail at 4096 and 16384 on one layer-local scalar. The port is unqualified. Serving smoke and `tools/bench/flint_baseline.py` stay unrun until a later change makes this same file pass at 1024, 4096, and 16384.

If work resumes:

1. Leave `flint_gate_v1.json` unchanged. A later arithmetic fix is justified only when an isolated replay of the suspected block, fed the reference residual, still exceeds a frozen limit.
2. Rescore native with `tools/goldens/gate` at 1024, 4096, and 16384, plus the three greedy continuations. The checker recomputes metrics and ignores any candidate `qualified` flag.
3. Run chat/tool smoke and the baseline only after that rescore passes. Baseline shape: decode near 1k and near 8k context, 8k prefill, three runs each, one `lockf -k /tmp/flint-gpu.lock` command per run.
4. Keep `.agent-logs/` under 15 GiB and stop new writes if home free space falls under 20 GiB. Current logs are about 845 MiB. The 3456-row replay arrays were deleted after `replay-layer1.json` was written.

### Frozen suite thresholds

Each suite threshold is twice the largest reference-to-reference deviation in the 31-comparison family. Per-length extrema sit beside those values in the file and are not a tighter gate. The stable margin is 0.59375: twice the winner-change margin 0.296875 from `short2-bf16-chunk2`, which is smaller than twice the top-two gap error 5.669410467147827 from `long16384-fp32-chunk32`.

| Scalar | Threshold | Deviation | Source |
| --- | --- | --- | --- |
| Logit max abs | 35.3990478515625 | 17.69952392578125 | `long16384-fp32-chunk32` |
| Logit relative L2 max | 0.34052122518820094 | 0.17026061259410047 | `short2-bf16-chunk64` |
| Logit cosine distance max | 0.00770418540439799 | 0.003852092702198995 | `short2-bf16-chunk1` |
| Logit top-1 error max | 4.28141975402832 | 2.14070987701416 | `long16384-fp32-chunk32` |
| Logit top-two gap error max | 11.338820934295654 | 5.669410467147827 | `long16384-fp32-chunk32` |
| `layers.1.self_attn.h` cosine distance max | 0.0020289048537087684 | 0.0010144524268543842 | `long4096-fp32-chunk32` |
| `layers.0.mlp.h` cosine distance max | 0.0008382156076864877 | 0.00041910780384324386 | `short1-bf16-chunk8` |

The file holds the same triple for every protected tensor (194 names) and for routing set disagreement and overlap loss. The largest routing set-disagreement threshold is 1.3189697265625 at `layers.26.mlp.ids`, from deviation 0.65948486328125 in `long16384-fp32-chunk32`. That value follows the precommitted two-times rule.

### Native score against the frozen gate

Whole-model logits and routing pass at every scored length. Stable top-1 has no failure on the teacher-forced traces.

| Trace | Rows | Result | Logit max abs | Relative L2 | Cosine distance | Top-1 error | Top-two gap | Worst set disagreement | Worst overlap loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short 0 | 76 | pass | 5.25 | 0.029680965730545135 | 0.0002886849896036603 | 1.34375 | 0.6015625 | 0.381578947368421 | 0.06578947368421051 |
| short 1 | 72 | pass | 2.0 | 0.020695215975646164 | 0.000149335148827201 | 0.84375 | 1.228515625 | 0.3055555555555556 | 0.05092592592592582 |
| short 2 | 81 | pass | 5.125 | 0.04307739883874989 | 0.0005237669957468549 | 0.71875 | 1.017578125 | 0.43209876543209874 | 0.09259259259259256 |
| long | 1024 | pass | 11.625 | 0.0826342471188159 | 0.0023447732468229177 | 1.359375 | 1.66015625 | 0.4365234375 | 0.08772786458333326 |
| long | 4096 | fail | 15.25 | 0.16313358073688608 | 0.002892295687629387 | 1.375 | 4.1240234375 | 0.56396484375 | 0.1346435546875 |
| long | 16384 | fail | 17.0 | 0.1643629910666138 | 0.002892295687629387 | 2.010009765625 | 5.486328125 | 0.66619873046875 | 0.1922200520833333 |

The only failure at 4096 and at 16384 is `layers.1.self_attn.h` cosine distance max 0.0022763134837696652, above 0.0020289048537087684. In the 16384-row trace that excess occurs at one position, 3452.

Greedy continuations use the frozen stable margin 0.59375. A first divergence is accepted when the committed reference margin at that step is at or below the margin.

| Example | First divergence | Reference margin | Mismatches of 64 | Result |
| --- | --- | --- | --- | --- |
| 0 | none | | 0 | pass |
| 1 | 48 | 0.08984375 | 16 | accepted |
| 2 | 22 | 0.203125 | 42 | accepted |

### No-sink mutation

Case `long1024-no-sinks` fails the frozen gate: 997 failures, `passed` false. Logit max abs is 69.25 (threshold 35.3990478515625), relative L2 max is 1.036409359474286 (threshold 0.34052122518820094), cosine distance max is 0.04730213720129761 (threshold 0.00770418540439799).

### Layer 1

Position 3452 is the only native row above the layer-1 attention cosine limit. The embedding there matches (max abs 0). The incoming `layers.0.mlp.h` residual differs by max abs 0.03125, relative L2 0.037205819374812134, cosine distance 0.0006034429787932227, all inside that tensor's frozen suite limits (0.046875, 0.05792637036077421, 0.0008382156076864877). Full-trace layer-1 attention at that row is max abs 0.03125, relative L2 0.0679894985919281, cosine distance 0.0022763134837696652. Logits at the same row stay small (max abs 2.5, cosine distance 4.7811311795653566e-05). The reference FP32-versus-BF16 comparison at that row has layer-1 attention cosine distance 0.000314903858680049.

Replaying native layer 1 alone, in 32-row prefill chunks, on the BF16 chunk-32 reference residual for the first 3456 positions of the repeated example-1 prompt, produces layer-1 attention cosine distance 7.86696941668108e-06 at index 3452. Over the whole prefix the maxima are max abs 0.00390625, relative L2 0.012001471204275518, cosine distance 7.025741904220784e-05. A single dispatch of all 3456 rows does not compile (the compute function exceeds available stack space); chunk 32 is the prefill width the long traces use.

The isolated block matches the reference. The full-trace cosine excess is amplification of an incoming residual that itself stays inside its frozen envelope. Layers 0 and 1 have no router, so this row is not an expert-set near tie. No arithmetic change follows from this measurement, and the frozen threshold stays in force. The 4096-row and 16384-row traces remain unqualified on that scalar.

## Sub-step plan — 2026-10-21 19:02 PDT

The calibration rule in "Calibration rule (fixed before long-context native results)" and the checker contract below it stay fixed. Native measurements do not enter any threshold. The interrupted 16k FP32 comparison stays ineligible; it is rerun from an empty output directory, not resumed. Each GPU command takes `lockf -k /tmp/flint-gpu.lock` alone. Check `df -h ~` before each batch and stop writing if home has under 20 GiB free or `.agent-logs/` would pass 15 GiB. After a comparison, keep the compact JSON or JSONL summary and delete raw arrays no later step reads.

Already eligible reference-only comparisons, kept as summaries: short examples 0, 1, and 2 at BF16 chunks 1 and 32 and FP32 chunk 32; example 0 BF16 chunks 2, 4, 8, and 16; long 1024 BF16 chunk 1 versus BF16 chunk 32; long 1024 and 4096 FP32 chunk 32 versus BF16 chunk 32. Free-running reference margins already kept: example 0 BF16 chunk 1 and FP32, example 1 BF16 chunk 1. Canonical short arrays and the 16k BF16 32-row archive stay until the comparisons that read them finish.

1. This plan (now).
2. About 25 minutes. One lock each: example 0 BF16 chunks 64 and 128, example 1 BF16 chunks 2 and 4. Compare with `--compare-to` against the retained canonical trace. Record wall time on the first command and use it to keep later steps inside their bounds. Write metrics only.
3. About 25 minutes. Example 1 BF16 chunks 8, 16, 64, and 128, same way.
4. About 25 minutes. Example 2 BF16 chunks 2, 4, 8, and 16.
5. About 25 minutes. Example 2 BF16 chunks 64 and 128. Then reference-only free runs that feed the margin rule: example 1 FP32 and example 2 BF16 chunk 1.
6. One command, which may run longer than 30 minutes: fresh FP32 32-row trace at 16384 positions compared with the retained BF16 32-row archive. Move the ineligible partial aside first. After `complete.json` exists and the row count is 16384 for every tensor, delete that archive. Then the last reference-only free run: example 2 FP32.
7. About 25 minutes. Reduce every complete reference-only comparison with the fixed rule: for each scalar, twice the largest reference-to-reference deviation, per tensor, with per-length extrema and suite extrema both recorded. Freeze `tools/goldens/gate/flint_gate_v1.json` with the fixed prompt token IDs, every threshold and the measurement it came from, reference and weight and pack SHA-256 identities, and the injected no-sink result. No native number is an input.
8. About 30 minutes. Add a checker that computes the metrics from native and reference traces and fails closed on missing rows, non-finite values, routing-cardinality changes, and shape or provenance mismatches. Hermetic tests cover those failures plus one pass and one fail. The stored no-sink trace must fail the frozen thresholds.
9. About 30 minutes. Score native at 1024, 4096, and 16384, plus the short native comparisons and the three greedy continuations (example 1 and example 2 native free runs if they are not already stored). If whole-model and routing metrics pass and only a layer-local criterion fails, replay that block on the reference residual. A real defect gets a fix and a failing-before, passing-after regression. Thresholds stay frozen.
10. Only if native qualifies: chat and tool serving smoke, then `tools/bench/flint_baseline.py` at about 1k decode, about 8k decode, and 8k prefill, three runs each, one GPU command per run.

## Time-boxed continuation handoff — 2026-10-21 18:58 PDT

**Still unqualified: the fixed reference calibration and final gate are incomplete.**
Native teacher-forced comparisons are complete at 1024, 4096 and 16384 positions.
All three canonical references reproduce all 192 committed greedy tokens and
stored top-two logits exactly. The complete native long-context measurements
and isolated-block evidence are recorded below.

Evidence points to BF16 rounding amplified by near-tie expert choices, but the
final bug/noise decision must wait for the complete independent noise envelope.
A partial comparison still fails layer-1 attention cosine distance:
**0.00227631348 observed versus 0.00202890485 allowed**. The fixed two-times
reference-only calibration rule has not been relaxed to make this pass.
Do not start optimization from a claimed qualified baseline yet.

No model arithmetic, kernel, recipe, fusion, prefill path or server-startup
behavior changed in this continuation. The later startup/prefill priority was
cancelled by the user; speed remains the optimization experiment's work.
Serving smoke and the requested three-run timing measurements were not run. No final numeric gate or qualified baseline is published.
The earlier bundle-reported TTFT 99.5 s / decode 10.2 tok/s are not new
measurements from this continuation.

Remaining work, in order:

1. Rerun the interrupted FP32 16k reference comparison to completion. Its partial metrics are not calibration-eligible.
2. Complete the predeclared BF16 short arithmetic family: example 0 chunks
   64/128, and examples 1/2 chunks 2/4/8/16/64/128. Compare directly against each
   retained canonical trace using `--compare-to`; retain metrics, not raw arrays.
3. Complete free-running example 1 FP32/native and example 2 BF16-1/FP32/native.
4. Reduce the entire fixed reference family, freeze the measured limits outside
   candidate ownership, evaluate all native cases and the no-sink mutation.
5. Only after qualification: CLI chat/tool smoke, then three decode/prefill
   measurements each at 1k and 8k. The helper supports one run per GPU command.

Local reduction state is retained in `.agent-logs/`: comparison JSON/JSONL and
capture metadata, `build_evidence.py`, `summarize_flint.py`, `export_gate.py`, and
`check_disk.py`. The reducer understands cached metrics after raw-array cleanup.
The 1k BF16 ordering trace's completion marker was reconstructed only after
validating all **198656** real records (194 tensors × 1024 one-row positions)
for contiguous, complete, finite measurements. No missing result was synthesized.

Disk policy is **20 GB maximum logs and at least 20 GiB free** before each batch.
Current logs are about **8.3 GiB**. Retain the 16k reference archive only until
its pending FP32 comparison completes, and canonical short arrays only until
all declared arithmetic comparisons finish. Other compared raw arrays were
removed; their metrics and provenance remain. Submit at most one GPU-lock
command from this thread at a time. All model/Metal work remains under
`lockf -k /tmp/flint-gpu.lock`.

## Historical handoff at 15:56 PDT

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

### Semantic mutation evidence

The full 1k no-sink counterfactual produces maximum logit error **69.25**,
relative L2 **1.03640936**, minimum logit cosine **0.95269786**, and **323/1024**
top-1 mismatches; a mismatching reference margin reaches **20.2109375**.
The unmodified port has **11.625**, **0.08263425**, **0.99765523**, and **2/1024**
mismatches, maximum margin **0.1328125**. Rec BF16 chunk-1 versus chunk-32 has
**11.25**, **0.09523529**, **0.99754886**, **2/1024**, and **0.15625**.

`tests/kernels/test_causal_boundary.py` adds deterministic histories at absolute
positions 513, 1026, 4096 and 16384. Sliding attention has a dominant key exactly
at p-512; full attention keeps the dominant key at zero and disables RoPE. A sink
has twice that key's exponential mass. The actual operation passes the existing
**2 BF16 ULP** oracle contract, while compiling with window 512 creates error
above **0.25**, and disabling sinks creates error above **0.5**. These are
constructed semantic checks, independent of whole-model noise tolerances.
All 65 boundary/attention/MoE/decode-state tests pass under Metal shader
validation (`.agent-logs/causal-boundary-validation.log`).

A real reference-only free-running check also diverges: fixture 1, BF16 prompt
ingested one token at a time instead of the canonical whole-prompt prefill,
first divergence **44/64**. Post-divergence token mismatch count is 20/64;
those later tokens are a different trajectory, not independent numerical errors.

The gate will additionally require top-1 equality above twice the largest
reference margin at an **observed reference-to-reference winner change**. This
is a stricter empirical stability check than the conservative top-two gap-error
bound; retain both measurements. For the complete 1k BF16 ordering comparison,
the observed maximum flip margin is 0.15625, giving an initial 0.3125 threshold
and 98.6328125% stable-position coverage. The final threshold uses the entire
fixed reference calibration family, including long contexts, rather than this
single preliminary pair. No native margin participates in setting it.

### Isolated-block localization

[Detailed localization measurements](flint-localization.json) replay all 49
blocks on the same 72-row reference inputs, resetting each block's cache. The
first measurable native difference is ordinary projection rounding: layer-0 QKV
max absolute error 0.03125, relative L2 0.00379304; reference FP32 widening gives
0.01597118 and 0.00316892. There is no initial large residual error. The cumulative
layer-0 MLP residual has max error 0.0234375 for both native and a BF16 reference
ordering change, relative L2 0.0288868 versus 0.0287299.

Isolated routing changes 59/3384 native sets versus 72/3384 under FP32 reference
arithmetic. All 3384 native selections are valid top-6 for their own BF16 router
logits. Every changed native set has a reference sixth/seventh logit gap no
greater than 0.015625. For example, layer 16 position 48 swaps experts 138/1
whose reference logits differ by 0.0078125. Its local MLP relative error rises
to 0.0636354, versus 0.00445331 for the FP32 comparison that did not cross this
particular boundary. Layer 28 similarly crosses a 0.0078125 cutoff. Thus a
pointwise ratio to one reference comparison is not an accumulation model: a
discrete near-tie route can amplify an ordinary BF16 rounding difference. The
fixed full reference family, all-layer routing statistics and whole-model
limits remain necessary; these local ratios do not adjust any threshold.

### Complete native long-context traces (18:10 PDT)

All positions, all 49 attention/MLP residuals, all 47 expert sets/router logits,
and the full vocabulary were compared with the same fixed token prefixes.
Native uses 32-row prompt execution throughout.

| Positions | Max logit absolute error | Max relative L2 | Minimum cosine | Top-1 differences | Largest differing reference margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 11.625 | 0.0826342471 | 0.9976552268 | 2 | 0.1328125 |
| 4096 | 15.25 | 0.1631335807 | 0.9971077043 | 2 | 0.1328125 |
| 16384 | 17.0 | 0.1643629911 | 0.9971077043 | 2 | 0.1328125 |

At 16k, pooled exact expert-set agreement is 0.552222719 and expert overlap is
0.889575931; the worst layer is 0.333801270 / 0.807779948. These require comparison
with the complete independent reference family; they are not standalone
acceptance thresholds. All 24,832 per-tensor chunk comparison records from the
first 4k positions match the separate 4k-capacity run exactly.

The fixed reference calibration family is still incomplete at this checkpoint.
In particular, the partial 4k layer-1 attention cosine budget is 0.00202890485,
while native is 0.00227631348. Do not qualify the port or increase that budget
from candidate measurements. The remaining predeclared reference configurations
and 16k FP32 comparison must decide the final bound.

Run resource policy: `.agent-logs` is capped at 20 GB, with at least 20 GiB free
before each new batch. Completed raw arrays are deleted after their final
comparison; JSON metrics and metadata remain. At 18:10 it occupies about 8.3 GiB.
The remaining canonical short traces and 16k reference archive are retained only
for pending comparisons. At most one GPU-lock command from this thread is
submitted at a time, allowing the other benchmark commands to interleave.

Canonical capture validation: all three prompts reproduce **192/192** committed
greedy predictions and every stored top-two logit exactly (maximum error zero).
The reference checkout is clean at its pinned commit; its `quantized.py` SHA-256
is `137cbdde8bc91e7148cf120dbabac9e6a209a9e490d3f1b869f51873c751a271`
and `model.py` is
`56363f7a1caea2b9db317feb192f4bfea32eccb2b6b190dffc3fa93ee09695b1`.
The trace loader now rejects tracked reference-source changes as well as a wrong
commit. The eventual bundle checker must own and pin its reference source too.

The short teacher-forced inputs are exactly each fixture's `prompt_ids +
greedy_ids[:-1]`, lengths **76, 72, 81**. Long inputs repeat the second fixture's
72 IDs, truncated to **1024, 4096, 16384**. No generated native ID enters these
comparisons. This deterministic corpus is public qualification data; it is not a
claim of accuracy over all inputs.

### Checker data contract (final numeric limits still pending)

The protected expected tensor set has **194** names: `embed_tokens.h`, `logits`,
49 each of `layers.i.self_attn.h` / `layers.i.mlp.h`, and 47 each of
`layers.i.mlp.router.gate.y` / `layers.i.mlp.ids` (layers 2 through 48). Widths
are respectively 2048, 150144, 2048, 2048, 256 and 6. Every tensor needs exactly
the protected input length; missing names/rows, non-finite values, duplicate or
out-of-range expert IDs, or changed expert cardinality fail closed.

For every position, compute full-vector maximum absolute error, relative
`L2(native-reference) / max(L2(reference), 1e-30)`, and cosine distance `1-cos`.
For every named numeric tensor, retain maximum, p99 (linear interpolation) and
mean of the per-position absolute and relative errors, plus maximum and mean
cosine distance. Here “mean absolute error” means the mean of **per-position
maximum component errors**, not an average that dilutes error over the vocabulary.
For each routing layer, use exact unordered set disagreement and mean
`1 - intersection_size/6`, weighting positions equally across chunk boundaries.

For each scalar independently, the fixed rule is **2 × the largest value from
any complete predeclared reference-only comparison**, pooled over the declared
lengths/configurations, separately for each tensor. Never pool layers to hide a
failing layer, average away a failing metric, or discard a prompt/outlier. The
margin rule is **min(2 × largest reference winner-change margin, 2 × largest
reference top-two-gap error)**, including the measured first-divergence margins
of reference-only free-running alternatives. Require native top-1 equality at
every position whose reference margin is strictly above that frozen threshold.

Run all three 64-token native greedy continuations. The first different token,
if any, must itself have a committed reference margin at or below the threshold;
matching a previous ambiguous position does not excuse a later stable error.
Later free-running tokens follow a different history, while the teacher-forced
checks continue to compare identical prefixes at every position.

Copy the final numeric limits, expected IDs/shapes, reference source and semantic
leaf checks into bundle-owned protected data and pin their hashes. The candidate
must neither choose the reference configurations nor supply thresholds or a
trusted `qualified` flag. Candidate checking and timing must use the same actual
arithmetic path; diagnostic copies may preserve intermediates, but the checker
must not select an accuracy-only fallback. The same frozen limits must reject
the measured whole-model no-sink mutation, alongside the independent strict
2-ULP window/sink/no-RoPE leaf contracts.

[The complete error trajectory at the reported divergence prefix](flint-divergence-prefix.json)
localizes the largest stage-to-stage gain at **layer 20 MoE MLP**: relative L2
rises about **4.074×**, reaching **0.0605904** natively; reference-only arithmetic
changes reach **0.0609752** at that same tensor and position. Layer 46 MLP later
reaches **0.0637177** native versus **0.0644898** reference-only. These measured
amplifications support numerical noise for the original short index-48 mismatch;
they do not waive the separate unresolved long-context cosine criterion. No
model arithmetic fix is justified by the reported short-prefix difference.

### Final stop and verification

The three-hour wall-clock box ended with the port **unqualified**. The 16k FP32
reference command waited approximately 45 minutes for the shared GPU and
acquired it during cleanup; it was interrupted, with no completion marker. Its
partial JSON is explicitly ineligible for calibration. All native long traces
are complete, but only **16/31** declared reference-only pairwise comparisons
and **4/9** planned free-running measurements are complete. No missing result or
final threshold was invented. The full reference family, final frozen numeric
limits, serving chat/tool smoke, and three-run 1k/8k baseline remain outstanding.

The short divergence is consistent with measured numerical noise: against the
same canonical trace, native logit maximum error / relative L2 are **2.0 /
0.0206952**, versus **2.5 / 0.0308360** from reference BF16 ordering alone. This
is evidence about that reported near-tie, not a claim that the unresolved
long-context cosine criterion passes. No model arithmetic was changed.

Verification: **65** attention/boundary/MoE/decode-state cases passed with Metal
shader validation; **2** dirty-reference CLI regression cases pass now and fail
against the pre-fix script. Changed Python files pass Ruff lint/format checks,
and repository hygiene passes. The shared pack manifest remains
`b01450fe47e91296e7cb7a57f94f795914d954ecbeec4188c9393d28cd0c390c`.
All owned model processes and lock waiters were stopped, with no lock retained.
Logs are approximately **8.25 GiB**; only raw canonical/16k reference arrays with
planned future readers remain. The final reply records the pushed HEAD and date.
