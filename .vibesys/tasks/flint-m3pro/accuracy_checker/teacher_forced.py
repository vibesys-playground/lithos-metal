"""Teacher-forced logits from the candidate engine.

This driver lives in the protected task directory. It reads prefill buffers
after the engine runs. It does not accept a logit file, a pass flag, or a
threshold. The schedule is 32-row BF16 prefill with commute_norm off, matching
the port's long-context native measurement rather than the server auto-exact
default. Greedy pins still go through the serving endpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

CHUNK = 32
MAX_CONTEXT = 9216


def bf16_bits_to_f32(raw: np.ndarray) -> np.ndarray:
    bits = np.asarray(raw, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def capture_logits(workspace: Path, model: Path, token_ids: list[int], *, destination: Path) -> None:
    """Run the candidate prefill engine and store float32 logits shaped (rows, vocab)."""
    if not token_ids or any(type(token) is not int or token < 0 for token in token_ids):
        raise ValueError("Teacher-forced token ids are invalid")
    workspace = workspace.resolve()
    sys.path.insert(0, str(workspace))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
    # Candidate imports stay after the path insert so this process loads the
    # checkout under test, not a previously installed copy.
    from monolith.generate import Session
    from monolith.models.flint import FlintModel
    from pack_policy import resolve_pack

    pack = resolve_pack(workspace, os.environ, Path.home())
    flint = FlintModel.from_checkpoint(
        model, max_context=max(MAX_CONTEXT, len(token_ids) + 1)
    )
    session = Session(
        flint,
        str(pack),
        eos=-1,
        autotune=False,
        commute_norm=False,
        accelerator="off",
        prefill_optimizations=False,
        prefill_chunk_size=CHUNK,
    )
    vocab = flint.config.vocab_size
    try:
        stored = np.lib.format.open_memmap(
            destination, mode="w+", dtype=np.float32, shape=(len(token_ids), vocab)
        )
        engine = session.prefill_engine(rows=CHUNK)
        session.reset()
        state_buffer = engine.buffers[engine.program.step_state]
        for start in range(0, len(token_ids), CHUNK):
            piece = token_ids[start : start + CHUNK]
            state = engine.state()
            state.update(
                t_this_step=len(piece),
                pending_tokens=piece,
                prefill_left=1,
                stop_at=1,
            )
            state_buffer.write(session.layout.pack(state), 0)
            engine.run(1, steps_per_cb=1, in_flight=1)
            nbytes = len(piece) * vocab * np.dtype(np.uint16).itemsize
            raw = np.frombuffer(engine.read("logits", nbytes), dtype=np.uint16).copy()
            rows = bf16_bits_to_f32(raw).reshape(len(piece), vocab)
            stored[start : start + len(piece)] = rows
            stored.flush()
            print(
                f"teacher-forced rows {start + len(piece)}/{len(token_ids)}",
                file=sys.stderr,
                flush=True,
            )
        stored.flush()
    finally:
        session.release_engines()
