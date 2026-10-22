"""Diagnostic per-operation prefill costs, not an accuracy-qualified baseline.

Run under lockf -k /tmp/flint-gpu.lock. Reuses an existing pack without writing
weights. Profiles one 128-token chunk at position zero using the serving flags.
Timestamp-counter encoder overhead means these are shares, not end-to-end speed.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from monolith.generate import Session
from monolith.models.flint import FlintModel


def summarize(operations, intervals):
    if len(operations) != len(intervals):
        raise ValueError("Incomplete operation timing report")
    totals = defaultdict(float)
    counts = defaultdict(int)
    for op, (start, end) in zip(operations, intervals, strict=True):
        if not 0 <= start <= end:
            raise ValueError("Invalid operation timing interval")
        totals[op.name] += end - start
        counts[op.name] += 1
    duration = sum(totals.values())
    if duration <= 0:
        raise ValueError("Empty operation timing report")
    return [
        dict(
            operation=name,
            dispatches=counts[name],
            gpu_ms=milliseconds,
            share=milliseconds / duration,
        )
        for name, milliseconds in sorted(totals.items(), key=lambda item: -item[1])
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    fixture = json.loads(Path("tests/models/flint/goldens/int4.json").read_text())
    seed = fixture["examples"][2]["prompt_ids"]
    tokens = (seed * (1 + 128 // len(seed)))[:128]
    model = FlintModel.from_checkpoint(args.model, max_context=32768)
    session = Session(
        model,
        str(args.pack),
        eos=-1,
        autotune=False,
        commute_norm=False,
        prefill_chunk_size=128,
        prefill_optimizations=True,
        measured_choices=True,
    )
    try:
        engine = session.prefill_engine(rows=128)
        session.reset()
        state = engine.state()
        state.update(
            position=0,
            t_this_step=128,
            pending_tokens=tokens,
            prefill_left=1,
            stop_at=1,
        )
        engine.buffers[engine.program.step_state].write(
            engine.program.layout.pack(state), 0
        )
        # Compile/map first, then profile exactly one chunk. Each dispatch uses
        # a timestamp encoder; never compare its total with normal ICB timings.
        intervals = engine.profile(steps=1)[0]
        report = {
            "diagnostic_only": True,
            "qualified": False,
            "rows": 128,
            "position": 0,
            "context_capacity": 32768,
            "operations": summarize(engine.program.ops, intervals),
        }
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, allow_nan=False), flush=True)
    finally:
        session.release_engines()


if __name__ == "__main__":
    main()
