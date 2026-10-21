# Flint correctness port plan

Target: RecursiveSI/flint-int4 revision `3b93f9053b530488405ae9fca51d4900d6ce21f0`.
Starting engine commit: `5485ad6bac0a`. No optimization, speculative draft, or KV quantization.

## Work slices and gates

Each slice is intended to take at most 30 minutes; split further if its gate uncovers a larger problem.

1. Inspect engine/reference/config, publish this plan. Verify branch, extension contracts, and config facts.
2. Install Python 3.12 dependencies and reuse/download the pinned snapshot once. Verify shard metadata and disk use.
3. Establish plain rec MLX reference on M3 using unquantized KV caches. Verify short decode without importing the megakernel; record provenance and versions.
4. Generate small JSON goldens for several prompts: prompt IDs, 64 greedy IDs, first-step top logits. Verify regeneration agrees exactly.
5. Add strict config and checkpoint adapters. Verify accepted config, rejected unknown/unsupported keys, weight shapes and formats on synthetic contracts.
6. Add affine unsigned INT8 format using existing affine pack mechanisms. Verify exact code/scale/bias preservation and embedding decode against an independent oracle.
7. Add shared parameterized causal attention contract: optional RoPE, sliding window, learned per-head sinks. Verify shape/state/coverage contracts without GPU.
8. Implement conservative Metal attention binding; verify each behavior and combinations against independent oracle with shader validation.
9. Generalize shared MoE routing/combine for sigmoid scoring and ungated shared experts; support stacked expert checkpoint tensors. Verify routing, selected weights, expert addressing and combination independently.
10. Compose Flint package through shared layers, including forward oracle and chat template. Verify graph coverage and pack map on synthetic and real metadata.
11. Verify actual layers against plain MLX, then complete 64-token model continuations against fixtures. At first token divergence report position/margins and stop; never loosen comparisons.
12. Smoke test serve chat and tool round trip, cleaning up the server. Verify response text and tool history handling.
13. Measure untuned decode at roughly 1k/8k context and 8k prefill, three runs each. Record commands and raw results.
14. Commit handoff with provenance, passing checks, interface gaps, pushed revision, date and baseline.

All model-loading/Metal commands run under `lockf -k /tmp/flint-gpu.lock`; locks are released after each command.
Logs are local untracked `.agent-logs/`. Each passing commit is pushed only to `flint-model` with the requested coauthor trailer.

## Mechanism reuse

| Piece | Existing mechanism | Decision and reason |
| --- | --- | --- |
| Pre/post/final RMSNorm | `nn.RMSNorm` | Reuse plain learned scale |
| Per-head Q/K RMSNorm | `nn.GQAAttention` | Reuse plain-scale convention |
| Q/K/V and output projections | `nn.Linear` / affine INT4 | Reuse checkpoint codes, scale/bias and row permutations |
| Dense first layers | `nn.GatedMLP`, `nn.DecoderLayer` | Reuse SiLU gated MLP and residual order |
| Routed expert projections | `nn.Experts`, `moe_gemv` | Generalize checkpoint adapter for stacked tensors; same selected-expert mechanism |
| Expert selection | `moe_route` | Generalize score function to sigmoid with separate selection bias; retain existing softmax default |
| Shared expert | `nn.SparseMoE`, `moe_combine` | Generalize optional gate; Flint shared expert has none |
| Full/sliding attention | shared attention layer/operation | Generalize with window parameter; absolute-position cache first, no ring optimization |
| Learned sinks | shared attention operation | New explicit optional per-head sink input; sink contributes denominator only |
| Attention without RoPE | shared attention operation | Explicit disabled rotary contract; no fake rotary tables |
| Embedding | `nn.Embedding` | Reuse gather, add unsigned affine INT8 format because symmetric INT8 changes weights |
| Output head | `nn.LMHead` / affine INT4 | Reuse; head is quantized too |
| Chat | checkpoint/template and serving adapter | Model-owned Flint template; check tool serialization/parser compatibility |
| Lowering/runtime | registries, coverage guard, native Metal bindings | No model-name branches or runtime changes |

## Instance facts and general contracts

The pinned config is authoritative: hidden 2048, 49 layers, 32 query/4 KV heads, head dimension 128,
vocabulary 150144, full layers `[0,1,5,9,13,17,21,25,29,33,37,41,45,48]`; all others have window 513.
Full layers disable RoPE; sliding layers use default half-split RoPE at theta 10000. Q/K use per-head RMSNorm,
and every attention layer has a learned sink per query head. Dense layers `[0,1]` use intermediate 6144;
others select 6 of 256 experts (intermediate 512) and add an ungated shared expert (1024).
Quantization is affine, group 64: projections/experts/head 4-bit and embedding 8-bit.

Their general forms are dimensions, per-layer mixer/MLP policies, rotary enabled/disabled, a positive window
or unbounded causal history, sink input, selectable router scoring, optional shared gate, and bit-width-specific
formats. None of the numerical constants belongs in shared kernels as a Flint special case.
Config also records 8-bit K/V defaults; these describe rec's optional cache policy and are not enabled here.

## Design boundaries

Model package owns config validation, weight names/adaptations, layer schedule, template and model oracle.
Shared layers own attention/MoE composition; registered operations own arithmetic/state contracts and Metal
bindings own their implementation. Data flows checkpoint -> validated model -> graph -> compiler/runtime.
Alternative rejected: a standalone Flint lowering with embedded attention/router kernels; it hides reusable
mechanisms and violates extensions.md. Shared parameterized contracts keep backend policy outside the model.
No changes to other model packages are intended except mechanical migrations required by generalized APIs.

Correctness gates: contract tier, leaf kernel oracle comparisons and shader validation, real layer agreement,
then exact greedy golden tokens. No baseline or serving success is claimed before those gates pass.
