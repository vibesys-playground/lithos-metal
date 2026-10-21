#!/usr/bin/env python3
"""Trusted evaluator entry point. Run under lockf -k /tmp/flint-gpu.lock."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from core import REPEATS, measure, summarize
from launcher import HttpChat, PinnedTokenizer, model_path, server_endpoint
from vs_protocol import ResultWriter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vs-output", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Operator smoke run: one repetition, all context buckets; no score",
    )
    args = parser.parse_args()
    if args.reduced and args.vs_output is not None:
        parser.error("--reduced cannot publish a VibeSys score")
    if args.vs_output is None and args.output_json is None:
        parser.error("--vs-output or --output-json is required")
    repeats = 1 if args.reduced else REPEATS
    writer = ResultWriter(args.vs_output) if args.vs_output else None
    started = time.monotonic()
    output = args.output_json or Path(str(args.vs_output) + ".metrics.json")
    observed = []
    turn_log = output.with_suffix(".turns.jsonl")
    try:
        turn_log.write_text("")
        path = model_path()
        tokenizer = PinnedTokenizer.load(path)
        with server_endpoint(Path.cwd(), path) as url:

            def observe(row):
                print(json.dumps({"turn": row}, allow_nan=False), flush=True)
                observed.append(row)
                with turn_log.open("a") as log:
                    log.write(json.dumps(row, allow_nan=False) + "\n")

            rows, wall = measure(
                HttpChat(url),
                tokenizer,
                time.monotonic,
                repeats=repeats,
                observe=observe,
            )
        values = summarize(rows, wall, repeats=repeats)
        total_wall = time.monotonic() - started
        if total_wall > 600:
            raise ValueError(
                f"Whole benchmark exceeded 600 seconds ({total_wall:.3f}); workload={wall:.3f}"
            )
        values["wall_duration_s"] = total_wall
        output.write_text(
            json.dumps(
                {"metrics": values, "turns": rows, "reduced": args.reduced},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        if writer:
            writer.result(values)
        print(json.dumps(values, allow_nan=False))
        return 0
    except Exception as exc:
        output.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_duration_s": time.monotonic() - started,
                    "turns": observed,
                    "reduced": args.reduced,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        if writer:
            writer.error(f"{type(exc).__name__}: {exc}")
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
