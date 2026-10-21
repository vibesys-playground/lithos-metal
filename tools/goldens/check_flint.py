"""Exact greedy qualification of the unoptimized native Flint model.

Run under lockf. Stops at the first divergent fixture and records its first
mismatching position and the independent/native logit margins. No tolerances.
"""

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from monolith.formats.fp import bf16_to_f32
from monolith.generate import Session
from monolith.models.flint import FlintModel


@dataclass
class GoldenObserver:
    example: int
    expected: list[int]
    divergence: int | None = None

    def __call__(self, tokens):
        for i, (actual, golden) in enumerate(zip(tokens, self.expected, strict=False)):
            if actual != golden:
                self.divergence = i
                break
        print(
            json.dumps(
                {
                    "example": self.example,
                    "observed": len(tokens),
                    "first_divergence": self.divergence,
                }
            ),
            flush=True,
        )

    def cancelled(self):
        return self.divergence is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument(
        "--fixture", type=Path, default=Path("tests/models/flint/goldens/int4.json")
    )
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text())
    model = FlintModel.from_checkpoint(args.model, max_context=9216)
    session = Session(
        model,
        args.pack,
        eos=-1,
        autotune=False,
        commute_norm=False,
        accelerator="off",
        prefill_optimizations=False,
        prefill_chunk_size=32,
    )
    records = []
    status = 0
    try:
        for example_index, example in enumerate(fixture["examples"]):
            expected = example["greedy_ids"]
            observer = GoldenObserver(example_index, expected)

            result = session.generate(
                example["prompt_ids"],
                len(expected),
                steps_per_cb=1,
                in_flight=1,
                on_tokens=observer,
                cancelled=observer.cancelled,
            )
            record = {
                "example": example_index,
                "tokens": result.tokens,
                "expected": expected,
                "match": result.tokens == expected,
            }
            divergence = observer.divergence
            if divergence is not None:
                prefix = example["prompt_ids"] + expected[:divergence]
                replay = session.generate(prefix, 1, steps_per_cb=1, in_flight=1)
                vocab = model.config.vocab_size
                logits = bf16_to_f32(
                    np.frombuffer(session.read("logits"), np.uint16)
                ).reshape(-1, vocab)
                logits = logits[(len(prefix) - 1) % 32]
                top = np.argsort(-logits, kind="stable")[:10]
                golden = example.get("step_top2", example["top_logits"])[divergence]
                record.update(
                    first_divergence=divergence,
                    expected_id=expected[divergence],
                    actual_id=result.tokens[divergence],
                    replay_id=replay.tokens[0],
                    reference_top2=golden,
                    reference_margin=golden["logits"][0] - golden["logits"][1],
                    native_top_ids=top.tolist(),
                    native_top_logits=logits[top].tolist(),
                    native_margin=float(logits[top[0]] - logits[top[1]]),
                    native_expected_minus_actual=float(
                        logits[expected[divergence]] - logits[result.tokens[divergence]]
                    ),
                )
                status = 1
            records.append(record)
            args.report.write_text(
                json.dumps(
                    {
                        "qualified": status == 0
                        and len(records) == len(fixture["examples"]),
                        "records": records,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(record), flush=True)
            if status:
                break
    finally:
        session.release_engines()
        del session
        gc.collect()
    raise SystemExit(status)


if __name__ == "__main__":
    main()
