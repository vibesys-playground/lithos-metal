# Interim accuracy gate plan

Replace exact 32-token pin equality in the shared checker with interim gate v0.
Thresholds stay fixed. Provenance scan and the four semantic probes stay as they are.

## Pinned thresholds

`accuracy_checker/interim_gate_v0.json` is the only threshold file. Each number
is twice a rec-versus-rec deviation in the port handoff section "Semantic
mutation evidence" (BF16 chunk-1 versus chunk-32 on the 1024-token prefix):

- Greedy margin 0.3125, from winner-change margin 0.15625.
- Max logit absolute error 22.5, from 11.25.
- Relative L2 0.19047058, from 0.09523529.
- Minimum cosine 0.99509772, from cosine 0.99754886 (distance 0.00245114).

The same file pins the sha256 of the margin record and of the 1k reference
logits, plus the 1024 prefix ids: the second fixture's 72 ids repeated and
truncated, as in "Complete native long-context traces".

## Checker behavior

1. Margin-aware pins. Same four pins, through the serving endpoint, both
   orders. Require 32 raw completion ids. Compare in order. A mismatch is
   tolerated only when the pinned reference top-1 minus top-2 margin at that
   index is below 0.3125, and the pin then stops. A margin of 0.3125 or more
   fails. Visible text is checked on the matched prefix only.
2. Teacher-forced 1k. A driver in `accuracy_checker/` runs the candidate
   engine and returns next-token logits. The checker compares them to the
   pinned rec trace. It fails closed on a missing file, a hash mismatch,
   a short row count, a non-finite value, or a shape mismatch. Top-1
   mismatches are allowed only where the reference margin is below 0.3125.

The 1k driver is not an HTTP route. A route in candidate-editable `serve.py`
could return a stored array. Greedy pins still use the real endpoint. The
driver uses 32-row prefill, `commute_norm` off, and the read-only pack: the
long-context schedule the port measured, not the server `auto-exact` default.
That schedule gap is the residual trust gap.

## Reference recording

`reference/record_interim_gate.py` runs once under the GPU lock, rec commit
`e53663b`, plain `QATModel.native`:

- Pin margins use prefill chunk 128, then the pinned tokens, and record
  top-1 minus top-2 at each of the 32 steps into `reference/pin_margins.json`.
- The 1k trace is BF16 chunk-32 logits only, stored outside git at
  `~/.cache/lithos-metal/flint-gate/rec-bf16-chunk32-logits-1024.npy`.

## Proof

Hermetic tests use fakes and cover a full pass, a tolerated near-tie, an
intolerant mismatch, and each fail-closed case. A real run must pass the
starting candidate. An uncommitted no-sink mutation of that candidate must
fail. Thresholds are not adjusted if it does not.
