# Flint M3 Pro bundle operator handoff

Operator evidence outside `.vibesys/tasks/flint-m3pro/`. No VibeSys optimization
agents have been launched. Updated 2026-10-21; the final port gate is still pending.

## Integration

The bundle's fork branch is `vibesys`. The original bundle `37d1a9923e5f` now
includes the port through `d229770` from `flint-model`. The port remains
numerically unqualified; its current investigation and eventual gate belong to
`docs/flint-port-handoff.md`. No accuracy threshold has been changed here.

Implementation commits include:

- `c8aa948`: read-only pack reuse, bounded shared-cache policy and raw-ID API.
- `f7d0dd3`: computed position-table extension without repacking weights.
- `8b65602`, `9aa08f0`: reduced operator measurement, per-turn logging and
  failure diagnostics without publishing an invalid score.
- `d8616e6`: provenance rejection tests and current operator setup.

VibeSys uses a fresh worktree on `exp/flint-bundle-2`. A normal merge into
`exp/flint-lithos-metal` was pushed at `c071783bb62dd70cdfbd4177c2287d84fed45b68`.
A later pin update reached `5f9c4d2d1e` on the experiment branch, pointing to
`511f7e9`; the final measurement pin will advance.
No VibeSys `src/` or `libs/` file was changed. No main push, force push or PR.

## Disk policy

`FLINT_PACK_DIR` resolves the existing shared pack. Its documented default is
`$HOME/.cache/lithos-metal/flint-int4/3b93f9053b530488405ae9fca51d4900d6ce21f0/block-r16-interleaved16-c9216`.
The only authorized pack key is
`65dcd02bfffafe7ef9c806b04a40f155ea4690d0111d77e03efff89a53c7ffe9`.
The launcher checks the key, manifest SHA-256 and complete weight-file size,
rejects paths in the candidate worktree (including symlink aliases), and
requires at least 22,736,175,104 bytes plus 2 GiB free before build/model load.
It always supplies `--pack PATH --pack-read-only`; missing or incompatible
packs fail without creating directories or packed weights. Target and draft
paths honor the same read-only option.

For a future explicitly authorized storage-layout migration,
`monolith.serving.pack_policy.SharedPackCache` admits a full pack plus margin,
keys entries by full SHA-256 layout identity, keeps at most one extra pack,
evicts least-recently-used extras before building, publishes atomically and
cleans interrupted/failed builds. Its pinned pack is outside the eviction set.
Callers hold the model execution lock for build/use. No extra pack has been
created in this continuation, and current authorization still permits only the
existing pack. Policy and tests use small synthetic files, never real weights.

The existing pack's F32 rotary tables end at position 9215. The generic pack
API now permits model-generated F32 leading-axis extensions in memory, only
when every old value matches bit for bit and trailing dimensions agree. The
compiler binds new initialized table buffers while preserving read-only weight
windows. The real 32768-capacity server passed this check. No pack, manifest,
checkpoint or quantization was rewritten.

## Serving API

Non-streaming Chat Completions accepts optional `return_token_ids: true` and
returns top-level `completion_token_ids`, the actual committed runtime IDs,
including special/EOS tokens. Length equals completion usage. Stop-string
truncation applies consistently. Streaming with the option is rejected. It is
a model-independent diagnostic documented in `docs/design/serving.md`.
The immutable checker's HTTP client requests it explicitly. A callback carries
IDs from the real backend to serialization; no retokenization or shared mutable
last-response state is used.

## Verification

- The initial 158 affected fork contract tests passed. A later expanded run
  had 164 passes and exposed a pre-existing Ollama queue-duration flake, which
  assumed a thread started within a real sleep interval. That test now injects
  a clock and checks nine queue/execution combinations with no sleeps; eleven
  targeted duration/stream tests pass. The production default clock is unchanged.
- 30 hermetic bundle tests passed. Policy cases cover default/env selection,
  low disk, missing/incomplete packs, worktree aliases, altered identity,
  read-only reuse, one-extra eviction, builder failure and interrupted builds.
- Capacity regression fails at `c8aa948` with insufficient context and passes
  at head; it compiles static/dynamic programs from a tiny 8-position pack for
  32 and 32768 positions, checks actual rotary bindings target generated
  buffers and verifies packed bytes remain unchanged.
- Existing checker tests reject an injected wrong token even when the normal
  pins pass. No numerical threshold has been loosened.
- Touched fork Python passes Ruff E9/F; new/task modules pass import checks.
  Native CMake build and `tools/ci/hygiene.py` pass. Broader legacy packer lint
  has pre-existing unused imports, which were not suppressed or changed.
- VibeSys `check_format.sh`, `check_lint.sh`, task validation and six selected
  registered-example checks pass. `repoctl test --dry-run` cannot select checks
  because upstream `HANDOFF-dynamic-host.md` and `HANDOFF-preflight-docker.md`
  are unowned paths in the experiment's accumulated diff. They were not edited.
- Current VibeSys preflight reports valid settings, `uses_docker: false`,
  backend `metal` and all six dynamic roles. This comes from the other agent's
  merged preflight change. That agent's `HANDOFF-preflight-docker.md` recommends
  two coding workstreams and one evaluation slot; this task changes no policy.

## Real accuracy and performance

Every model/Metal command holds `lockf -k /tmp/flint-gpu.lock` for that command.
The first accuracy attempt failed before model startup because CMake was not
on PATH. Activating the source-build environment resolved that prerequisite.

The current unchanged accuracy checker then started the actual candidate server
at 32768 context through the trusted launcher and completed all four semantic
probes. The first 32-token pin (`port-1`, ocean sentence) passed. The next pin
(`port-2`, Python sum continuation) failed at zero-based token 22: expected
3087, received 220. Raw IDs were present and truthful; this is a numerical
failure, not a missing-extension fallback. The evaluator exited 1 and cleaned
its server process group. The current port-defined gate specification is still
pending, so this candidate is not qualified for optimization.

The initial reduced run used one repetition of the original six turns. It
completed only the 1024-token turn: TTFT **134.2618 s**, decode **10.5658 tok/s**,
64 committed output tokens. It was signalled at **629 s**, with approximately
**632.7 s** whole process wall time including cleanup and excluding the lock
queue. The second 1280-target turn was incomplete. This run had started before
the new command-wide deadline was loaded. Its server and process group exited,
and the lock was released. No score was published.

The scored workload is now one repetition of three turns at **1024, 8192 and
32000** tokens, still exactly 64 outputs each. Only repeats/turns were reduced;
all context buckets remain, including the largest target. The full run of that
minimum workload used the fresh VibeSys worktree's interpreter and installed
serving/build dependencies. It timed out during the 8192-target second turn.
The 32000-target third turn was not reached. Only the 1024-token first turn
completed: TTFT **99.5051 s**, decode **10.1565 tok/s**, 64 output tokens.
The deadline fired at 600 seconds; measured whole wall time including cleanup
was **606.0507 s**. The official protocol contains hello plus error, no result.
Both benchmark and owned server processes exited, releasing the GPU lock. The new 600-second command timer
interrupts blocking startup and requests, unwinds through server cleanup, and
writes an error row plus partial observations rather than a fabricated score.
The minimum three-bucket workload still cannot produce a valid baseline within
600 seconds. Further turn/repetition reductions would omit a required bucket;
the baseline prefill path needs work before this task is optimization-ready.

[Structured measurements](flint-bundle-measurements.json) records the completed
turns, the over-budget reduced attempt and the current accuracy failure. No
valid aggregate baseline or unmeasured long-context throughput is claimed.

## Design and next steps

Disk admission/location belongs to the trusted launcher; atomic shared-cache
lifetime belongs to the serving cache policy; computed-table compatibility
belongs to packs and binding belongs to the compiler. The existing dependency
direction is preserved, without a VibeSys graph edge. The mechanism fixed was
optional pack selection falling through to automatic unbounded preparation;
target and draft call sites were swept. Read-only explicit selection was chosen
over implicit cache fallback. Raw-ID callbacks were chosen over mutable
backend diagnostics to keep request ownership clear.

Both requested measurements have run. The scored workload is already reduced
to one turn in each of three retained buckets, and still times out. Fix or
optimize native prefill before retrying the full benchmark; do not drop a bucket,
hide startup work or publish partial observations as a score. Fetch and
merge `flint-model` again when it advances. When its handoff publishes the
accuracy gate, implement that exact specification in the immutable checker,
including an injected-bug rejection test, then rerun under the lock. Advance
the VibeSys submodule pin and merge/push normally after a fresh fetch. Keep all
servers owned and cleaned up; do not launch optimization from an unqualified
candidate.
