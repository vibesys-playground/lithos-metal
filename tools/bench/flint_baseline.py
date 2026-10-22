"""Three untuned end-to-end runs at 1k and 8k context; run under the GPU lock."""

import argparse
import gc
import json
from pathlib import Path

from monolith.generate import Session
from monolith.models.flint import FlintModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--qualification", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--context", type=int, choices=(1024, 8192))
    parser.add_argument("--runs", type=int, choices=(1, 3), default=3)
    args = parser.parse_args()
    qualification = json.loads(args.qualification.read_text())
    if not qualification.get("qualified"):
        raise ValueError(
            "baseline requires the completed teacher-forced accuracy qualification"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, fix_mistral_regex=False)
    seed = tokenizer.encode(
        "The ocean is blue and the sky is clear. ", add_special_tokens=False
    )
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
    try:
        session.generate(seed, 4, steps_per_cb=1, in_flight=1)
        for context in (args.context,) if args.context else (1024, 8192):
            prompt = (seed * (1 + context // len(seed)))[:context]
            for run in range(args.runs):
                result = session.generate(prompt, 33, steps_per_cb=1, in_flight=1)
                record = {
                    "context": context,
                    "run": run + 1,
                    "decode_tokens": result.decode_tokens,
                    "setup_ms": result.setup_ms,
                    "state_reset_ms": result.state_reset_ms,
                    "prefill_rows": [item["tokens"] for item in result.prefill_timings],
                    "decode_wall_ms": result.decode_wall_ms,
                    "decode_gpu_ms": result.decode_ms,
                    "decode_wall_tokens_per_second": result.decode_tokens
                    * 1000
                    / result.decode_wall_ms,
                    "prefill_wall_ms": result.prefill_wall_ms,
                    "prefill_gpu_ms": result.prefill_ms,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
                args.out.write_text(
                    json.dumps(
                        {
                            "settings": {
                                "autotune": False,
                                "commute_norm": False,
                                "accelerator": "off",
                                "prefill_optimizations": False,
                                "prefill_chunk_size": 32,
                                "steps_per_cb": 1,
                                "in_flight": 1,
                            },
                            "runs": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
    finally:
        session.release_engines()
        del session
        gc.collect()


if __name__ == "__main__":
    main()
