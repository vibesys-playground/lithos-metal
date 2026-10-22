#!/usr/bin/env python3
"""VibeSys Request Factory adapter: own server lifetime, not replay execution."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from budget import deadline_guard
from launcher import model_path, server_endpoint
from replay_metrics import BUDGET_SECONDS, replay_window, summarize
from vs_protocol import ResultWriter

STARTUP_SECONDS = 1800
CONTEXT_LIMITS = {"quick": 2048, "full": 32768}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-factory-engine", required=True)
    parser.add_argument("--vs-output", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--workload", choices=tuple(CONTEXT_LIMITS), default="full")
    parser.add_argument("--gpu-lock-held", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.gpu_lock_held:
        os.execvp(
            "lockf",
            [
                "lockf",
                "-k",
                "/tmp/flint-gpu.lock",
                sys.executable,
                str(Path(__file__).resolve()),
                *sys.argv[1:],
                "--gpu-lock-held",
            ],
        )
    root = Path(__file__).resolve().parent
    output = args.output_json or Path(str(args.vs_output) + ".metrics.json")
    writer = ResultWriter(args.vs_output)
    started = time.monotonic()
    started_timestamp = time.time()
    rf = None
    candidate_commit = None
    try:
        candidate_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
        with Path(args.request_factory_engine).open("rb") as engine_file:
            engine_sha256 = hashlib.file_digest(engine_file, "sha256").hexdigest()
        identity = json.loads((root / "trace-identity.json").read_text())
        for name, digest in identity["files"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Pinned RF replay input changed: {name}")
        with (root / f"{args.workload}.csv").open() as trace:
            planned = list(csv.DictReader(trace))
        max_context = CONTEXT_LIMITS[args.workload]
        pool_limit = (
            max(int(row["prefix_len"]) + int(row["input_len"]) for row in planned)
            if args.workload == "quick"
            else 2 * max_context
        )
        with (
            tempfile.TemporaryDirectory(prefix="flint-rf-") as temp,
            contextlib.ExitStack() as owner,
        ):
            temporary = Path(temp)
            log, summary = temporary / "requests.jsonl", temporary / "summary.json"
            terminal = owner.enter_context((temporary / "terminal.log").open("w+"))
            with deadline_guard(STARTUP_SECONDS):
                path = model_path()
                url = owner.enter_context(
                    server_endpoint(Path.cwd(), path, max_context=max_context)
                )
                command = [
                    args.request_factory_engine,
                    "--trace",
                    str(root / f"{args.workload}.csv"),
                    "--input-file-format",
                    "text-generation-session-execution-v2",
                    "--text-file",
                    str(root / "corpus.txt"),
                    "--tokenizer",
                    str(path / "tokenizer.json"),
                    "--token-pool-limit",
                    str(pool_limit),
                    "--model",
                    "flint",
                    "--backend",
                    "openai",
                    "--dialect",
                    "openai",
                    "--base-url",
                    url + "/v1",
                    "--temperature",
                    "0",
                    "--arrival-mode",
                    "saturated",
                    "--max-concurrency",
                    "1",
                    "--runtime-worker-threads",
                    "1",
                    "--stop-session-on-error",
                    "true",
                    "--request-log",
                    "true",
                    "--log-path",
                    str(log),
                    "--timeline",
                    "false",
                    "--summary-path",
                    str(summary),
                ]
                rf = subprocess.Popen(
                    command, stdout=terminal, stderr=subprocess.STDOUT
                )
                # The writer starts after RF's mandatory cache probe. Its file
                # birth time provides a watchdog boundary; final scoring uses
                # RF's own first-submit timestamp to remove scheduling skew.
                while not log.exists():
                    if rf.poll() is not None:
                        raise RuntimeError(
                            "RF preparation failed: "
                            + (temporary / "terminal.log").read_text()[-4000:]
                        )
                    time.sleep(0.02)
            startup = time.monotonic() - started
            # macOS records file birth time when RF opens its request log, just
            # before replay. Include polling delay in the fixed scored budget.
            replay_started = log.stat().st_birthtime
            deadline_timestamp = replay_started + BUDGET_SECONDS
            remaining = max(0, deadline_timestamp - time.time())
            print(
                json.dumps(
                    {"startup_duration_s": startup, "scored_budget_s": BUDGET_SECONDS}
                ),
                flush=True,
            )
            timed_out = False
            try:
                rf.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Closing our server ends the in-flight HTTP request. RF then
                # flushes its own buffered completed-request log and summary.
                owner.close()
                rf.wait(timeout=30)
            if rf.returncode != 0:
                raise RuntimeError(
                    f"RF exited {rf.returncode}: "
                    + (temporary / "terminal.log").read_text()[-4000:]
                )
            records = [json.loads(line) for line in log.read_text().splitlines()]
            rf_summary = json.loads(summary.read_text())
            replay_started, deadline_timestamp, timed_out = replay_window(
                records,
                log_created_timestamp=replay_started,
                watchdog_timed_out=timed_out,
            )
            startup = replay_started - started_timestamp
            scored_duration = min(time.time() - replay_started, BUDGET_SECONDS)
            values, turns = summarize(
                planned,
                records,
                rf_summary=rf_summary,
                timed_out=timed_out,
                deadline_timestamp=deadline_timestamp if timed_out else None,
            )
            if args.workload == "quick" and values["timed_out_requests"]:
                raise ValueError("Bootstrap must complete every planned request")
            values.update(startup_duration_s=startup, scored_duration_s=scored_duration)
            result = dict(
                workload=args.workload,
                max_context=max_context,
                token_pool_limit=pool_limit,
                candidate_commit=candidate_commit,
                request_factory_engine_sha256=engine_sha256,
                metrics=values,
                turns=turns,
                timed_out=timed_out,
                rf_summary=rf_summary,
                rf_records=records,
                identity=identity,
                total_wall_duration_s=time.monotonic() - started,
            )
        # Publish only after the owned server exits, so whole wall time includes
        # cleanup on both completed and timed-out paths.
        result["total_wall_duration_s"] = time.monotonic() - started
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        writer.result(values)
        print(json.dumps(result, allow_nan=False), flush=True)
        return 0
    except Exception as exc:
        writer.error(f"{type(exc).__name__}: {exc}")
        output.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "workload": args.workload,
                    "candidate_commit": candidate_commit,
                    "error": f"{type(exc).__name__}: {exc}",
                    "total_wall_duration_s": time.monotonic() - started,
                }
            )
            + "\n"
        )
        print(f"RF benchmark failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if rf is not None and rf.poll() is None:
            rf.terminate()
            try:
                rf.wait(timeout=10)
            except subprocess.TimeoutExpired:
                rf.kill()
                rf.wait(timeout=10)
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
