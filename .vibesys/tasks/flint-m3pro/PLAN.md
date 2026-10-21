# Flint M3 Pro input bundle plan

Each implementation step is at most 30 minutes; repeat a step with a smaller
scope if integration reveals more work. No VibeSys agents are launched.

1. Read repository guidance and serving contracts; commit this plan first.
   Verify: branch starts at upstream 5485ad6 and contains task files only.
2. Define the fixed single-session workload and immutable accuracy contract.
   Verify: manifest validation, protocol v2 schema and objective names agree.
3. Implement streaming benchmark, cache-aware conversation construction and
   launcher with deterministic seams. Verify: fake server/clock tests cover
   malformed streams, missing usage, short generations and metric arithmetic.
4. Implement probes and reference greedy pins gate; reuse model-port goldens
   when available. Verify: hermetic positive and negative gate tests; missing
   pins fail closed. No candidate-controlled threshold or skip option.
5. Add external-repository submodule and registry metadata in VibeSys.
   Verify: registered-example checks plus formatting and lint.
6. Fetch flint-model and merge only after docs/flint-port-handoff.md exists.
   Verify: port instructions, pinned weight identity and bf16 cache contract.
7. Run real gate and benchmark under lockf -k /tmp/flint-gpu.lock, release
   immediately after each command. Verify: numerical gate, repeated bucket
   results, spread and wall time at most about 600 seconds. Record failures
   exactly; do not weaken the gate or optimize the engine.
8. Run VibeSys preflight only with dynamic, metal and Codex gpt-6.1-sol.
   Verify: no agents started; record dynamic_run_environment_unsupported if
   returned. Document exact launch command and unresolved prerequisites.
9. Remove this plan, write operator handoff outside the task directory,
   push fork vibesys, pin it in VibeSys, merge the implementation branch into
   exp/flint-lithos-metal and push. Verify: both remote SHAs and clean task
   scope; never push main and never create a PR.

## Design

The task owns evaluation policy behind the HTTP serving contract. Pure
workload/measurement/gate functions consume typed observations. A thin shell
owns HTTP, model-cache resolution, monotonic timestamps and child lifetime.
An injected transport and clock let hermetic tests run without a model,
Metal, real time, sleeps, processes or sockets. The alternative of embedding
engine internals in the checker would couple correctness to the candidate;
HTTP plus independently produced reference pins hides more implementation.
No VibeSys product module or dependency edge changes.
