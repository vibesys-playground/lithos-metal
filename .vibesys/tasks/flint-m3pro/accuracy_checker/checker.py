#!/usr/bin/env python3
"""Immutable accuracy gate: source audit, semantic probes and exact greedy pins.

Published functions load_pins/check_accuracy accept a faithful in-process Chat
and pinned tokenizer. No threshold, alternate reference or skip CLI is offered.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from launcher import MODEL, REVISION, HttpChat, PinnedTokenizer, model_path, server_endpoint

__all__ = ["load_pins", "check_accuracy", "scan_engine", "PROBES", "PinChat"]
PIN_LENGTH = 32
MIN_PINS = 4
FORBIDDEN = {"mlx", "mlx_lm", "rec", "rec_mlx", "llama_cpp", "ollama", "vllm", "sglang"}
PROBES = (
    ([{"role": "user", "content": "What is 17 + 25? Reply with only the integer."}], r"\b42\b"),
    (
        [
            {
                "role": "user",
                "content": "In Python, what does len([2, 4, 6, 8]) return? Reply with only the integer.",
            }
        ],
        r"\b4\b",
    ),
    (
        [
            {"role": "user", "content": "Remember the build marker: cobalt-7319."},
            {"role": "assistant", "content": "I will remember the marker."},
            {"role": "user", "content": "What was the exact build marker?"},
        ],
        r"cobalt-7319",
    ),
    (
        [
            {
                "role": "user",
                "content": "The deployment region is cedar-west and the shard count is 13.",
            },
            {"role": "assistant", "content": "Recorded."},
            {"role": "user", "content": "Repeat the deployment region and shard count."},
        ],
        r"(?s)(?=.*cedar-west)(?=.*\b13\b)",
    ),
)


class PinChat(Protocol):
    def greedy(self, messages: list[dict], max_tokens: int) -> dict: ...


def scan_engine(workspace: Path) -> None:
    """Reject mechanical forbidden imports/execs; renamed engines need judge review.

    Scans candidate Python outside tests/tools and model oracle.py references.
    Existing lazy Torch layer oracles are allowed as reference code. Whether
    served execution uses the compiler/runtime is additionally judge-reviewed.
    """
    for path in workspace.rglob("*.py"):
        relative = path.relative_to(workspace)
        if any(
            p.startswith(".") or p in {"tests", "tools", "build", "dist", "__pycache__"}
            for p in relative.parts
        ):
            continue
        if path.name == "oracle.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(relative))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or "", *(n.name for n in node.names)]
            if any(n.split(".")[0] in FORBIDDEN for n in names):
                raise ValueError(f"Forbidden serving import in {relative}:{node.lineno}: {names}")
            if isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id in {"__import__", "exec", "eval"})
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr
                    in {
                        "import_module",
                        "run",
                        "Popen",
                        "call",
                        "system",
                        "check_call",
                        "check_output",
                    }
                )
            ):
                literals = [
                    n.value
                    for n in ast.walk(node)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                ]
                if any(
                    re.search(r"\b(mlx[-_]lm|rec_mlx|llama[._-]cpp|ollama)\b", s) for s in literals
                ):
                    raise ValueError(
                        f"Possible forbidden engine exec/import in {relative}:{node.lineno}"
                    )


def load_pins(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(
            "Missing reference/pins.json; reuse flint-model goldens or run reference/make_pins.py under GPU lock"
        )
    data = json.loads(path.read_text())
    if set(data) != {"version", "model", "revision", "reference", "kv_format", "pins"}:
        raise ValueError("Unknown/missing pins keys")
    if (data["version"], data["model"], data["revision"], data["kv_format"]) != (
        1,
        MODEL,
        REVISION,
        "bf16",
    ):
        raise ValueError("Pins provenance does not match pinned int4 weights and bf16 KV")
    if not isinstance(data["reference"], dict) or not data["reference"].get("commit"):
        raise ValueError("Pins require independent reference commit provenance")
    pins = data["pins"]
    if not isinstance(pins, list) or len(pins) < MIN_PINS:
        raise ValueError(f"At least {MIN_PINS} independent pins required")
    seen = set()
    for pin in pins:
        if set(pin) != {"id", "messages", "prompt_token_ids", "expected_token_ids"}:
            raise ValueError("Unknown/missing pin keys")
        if not isinstance(pin["id"], str) or pin["id"] in seen:
            raise ValueError("Duplicate/invalid pin id")
        seen.add(pin["id"])
        for name in ("prompt_token_ids", "expected_token_ids"):
            if (
                not isinstance(pin[name], list)
                or not pin[name]
                or any(type(n) is not int or n < 0 for n in pin[name])
            ):
                raise ValueError(f"Invalid pin {pin['id']} {name}")
        if len(pin["expected_token_ids"]) != PIN_LENGTH:
            raise ValueError(f"Pin {pin['id']} must contain exactly {PIN_LENGTH} tokens")
        if not isinstance(pin["messages"], list) or not pin["messages"]:
            raise ValueError("Pin messages must be nonempty")
        for message in pin["messages"]:
            if (
                set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)
            ):
                raise ValueError("Invalid pin messages")
    if len({json.dumps(p["messages"], sort_keys=True) for p in pins}) != len(pins):
        raise ValueError("Pins must use distinct prompts")
    return pins


def response_text(body: dict) -> str:
    if "error" in body or len(body.get("choices", [])) != 1:
        raise ValueError("Invalid completion response")
    choice = body["choices"][0]
    if choice["message"].get("tool_calls"):
        raise ValueError("Expected text, received tool calls")
    text = choice["message"].get("content")
    if not isinstance(text, str) or not text:
        raise ValueError("Empty completion")
    return text


def check_accuracy(chat: PinChat, tokenizer: PinnedTokenizer, pins: list[dict]) -> dict:
    for index, (messages, pattern) in enumerate(PROBES):
        body = chat.greedy(messages, 64)
        if not re.search(pattern, response_text(body), re.IGNORECASE):
            raise ValueError(f"Semantic probe {index} failed: {response_text(body)!r}")
    # Reverse order on the second pass exercises session replacement and stale
    # prefix state as well as repeated deterministic decoding.
    for pin in [*pins, *reversed(pins)]:
        if tokenizer.ids(pin["messages"]) != pin["prompt_token_ids"]:
            raise ValueError(f"Chat-template mismatch for pin {pin['id']}")
        expected = pin["expected_token_ids"]
        if tokenizer.encode(tokenizer.decode(expected)) != expected:
            raise ValueError(f"Pin {pin['id']} is not losslessly observable via text API")
        body = chat.greedy(pin["messages"], PIN_LENGTH)
        got = tokenizer.encode(response_text(body))
        if got != expected or body.get("usage", {}).get("completion_tokens") != PIN_LENGTH:
            first = next(
                (i for i, (a, b) in enumerate(zip(got, expected)) if a != b),
                min(len(got), len(expected)),
            )
            raise ValueError(
                f"Greedy pin {pin['id']} diverged at token {first}: {got[:PIN_LENGTH]}"
            )
    return {"probes_passed": len(PROBES), "pins_passed": len(pins), "repeat_passes": 2}


def main() -> int:
    task = Path(__file__).resolve().parents[1]
    try:
        scan_engine(Path.cwd())
        pins = load_pins(task / "reference" / "pins.json")
        path = model_path()
        with server_endpoint(Path.cwd(), path) as url:
            result = check_accuracy(HttpChat(url), PinnedTokenizer.load(path), pins)
        print(json.dumps(result))
        return 0
    except Exception as exc:
        print(f"Accuracy failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
