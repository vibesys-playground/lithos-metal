#!/usr/bin/env python3
"""Trusted evaluator entry point. Run under lockf -k /tmp/flint-gpu.lock."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from core import measure, summarize
from launcher import HttpChat, PinnedTokenizer, model_path, server_endpoint
from vs_protocol import ResultWriter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vs-output", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.vs_output is None and args.output_json is None:
        parser.error("--vs-output or --output-json is required")
    writer = ResultWriter(args.vs_output) if args.vs_output else None
    try:
        path = model_path()
        tokenizer = PinnedTokenizer(path)
        with server_endpoint(Path.cwd(), path) as url:
            rows, wall = measure(HttpChat(url), tokenizer, time.monotonic)
        values = summarize(rows, wall)
        output = args.output_json or Path(str(args.vs_output) + ".metrics.json")
        output.write_text(
            json.dumps({"metrics": values, "turns": rows}, indent=2, allow_nan=False) + "\n"
        )
        if writer:
            writer.result(values)
        print(json.dumps(values, allow_nan=False))
        return 0
    except Exception as exc:
        if writer:
            writer.error(f"{type(exc).__name__}: {exc}")
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if writer:
            writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
