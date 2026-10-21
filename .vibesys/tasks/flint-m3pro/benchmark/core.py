"""Published evaluator core: sequential workload, strict SSE observations and scoring.

No engine imports or I/O. HTTP/clock/tokenizer implementations are injected by
run.py; tests call this same surface. Errors invalidate the entire measurement.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

__all__ = [
    "TARGETS",
    "REPEATS",
    "OUTPUT_TOKENS",
    "Turn",
    "Chat",
    "Tokenizer",
    "measure",
    "summarize",
    "extend_history",
    "parse_stream",
]
TARGETS = (1024, 8192, 32000)
REPEATS = 1
OUTPUT_TOKENS = 64


class Tokenizer(Protocol):
    def count(self, messages: list[dict]) -> int: ...


class Chat(Protocol):
    def complete(self, messages: list[dict], max_tokens: int) -> "Turn": ...


@dataclass(frozen=True)
class Turn:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    decode_s: float
    cached_tokens: int | None = None


def parse_stream(
    lines: Iterable[bytes], now: Callable[[], float], started: float
) -> Turn:
    """Parse one streamed completion, requiring completion, usage and exactly 64 tokens.

    Continuous usage timestamps committed token availability, including tokens
    held by the UTF-8 decoder. Content-only streams cannot produce a score.
    """
    text, first, last, usage, done, finish = [], None, None, None, False, None
    previous = 0
    for raw in lines:
        line = raw.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data: "):
            raise ValueError(f"Unexpected SSE line: {line[:100]}")
        data = line[6:]
        if data == "[DONE]":
            done = True
            break
        frame = json.loads(data)
        if "error" in frame:
            raise ValueError(f"Server error: {frame['error']}")
        for choice in frame.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("tool_calls"):
                raise ValueError("Benchmark requests text, received tool calls")
            content = delta.get("content")
            if content:
                text.append(content)
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
        current = frame.get("usage")
        if current is not None:
            n = current.get("completion_tokens")
            p = current.get("prompt_tokens")
            if type(n) is not int or type(p) is not int or n < previous or p < 1:
                raise ValueError("Invalid or decreasing usage token counts")
            if n > previous:
                # A jump on the first frame makes first-token timing unknowable.
                if first is None and n != 1:
                    raise ValueError("Missing continuous first-token usage")
                timestamp = now()
                if first is None:
                    first = timestamp
                last = timestamp
            previous, usage = n, current
    if not done or finish != "length" or not text or usage is None:
        raise ValueError("Incomplete/empty stream, missing usage or early EOS")
    if usage["completion_tokens"] != OUTPUT_TOKENS or first is None or last <= first:
        raise ValueError("Wrong output budget or unmeasurable decode interval")
    if first < started or not all(math.isfinite(v) for v in (first, last, started)):
        raise ValueError("Invalid clock observations")
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens")
    if cached is not None and (
        type(cached) is not int or not 0 <= cached <= usage["prompt_tokens"]
    ):
        raise ValueError("Invalid cached_tokens")
    return Turn(
        "".join(text),
        usage["prompt_tokens"],
        OUTPUT_TOKENS,
        first - started,
        last - first,
        cached,
    )


def extend_history(
    history: list[dict], target: int, tokenizer: Tokenizer, repeat: int, turn: int
) -> list[dict]:
    """Append deterministic code-review material to a real conversation near target.

    Count the pinned model's actual chat template, not words or server claims.
    Earlier user and assistant messages are preserved byte for byte.
    """
    intro = f"Review batch {repeat}, turn {turn}. Continue a detailed code review of this module.\n"
    code = "".join(
        f"def transform_{i}(records):\n    return [item.get('value', {i}) + {i} for item in records if item]\n\n"
        for i in range(6000)
    )
    ending = "\nExplain correctness, edge cases, and propose tests. Give a detailed review with code examples."
    low, high, best = 0, len(code), None
    while low <= high:
        middle = (low + high) // 2
        candidate = [
            *history,
            {"role": "user", "content": intro + code[:middle] + ending},
        ]
        count = tokenizer.count(candidate)
        if count <= target:
            best, low = candidate, middle + 1
        else:
            high = middle - 1
    if best is None or not target - 32 <= tokenizer.count(best) <= target:
        raise ValueError(f"Cannot construct context bucket {target}")
    return best


def measure(
    chat: Chat,
    tokenizer: Tokenizer,
    now: Callable[[], float],
    *,
    repeats: int = REPEATS,
    observe: Callable[[dict], None] | None = None,
) -> tuple[list[dict], float]:
    """One sequential growing session retaining all three context buckets."""
    if not 1 <= repeats <= REPEATS:
        raise ValueError("Invalid benchmark repeat count")
    rows, started = [], now()
    for repeat in range(repeats):
        history = [
            {
                "role": "system",
                "content": "You are a coding assistant. Review code carefully and explain your reasoning.",
            }
        ]
        for index, target in enumerate(TARGETS):
            if now() - started >= 600:
                raise ValueError("Benchmark exceeded 600 seconds")
            history = extend_history(history, target, tokenizer, repeat, index)
            expected = tokenizer.count(history)
            result = chat.complete(history, OUTPUT_TOKENS)
            if result.prompt_tokens != expected:
                raise ValueError(
                    f"prompt_tokens={result.prompt_tokens}, pinned tokenizer={expected}"
                )
            if (
                result.completion_tokens != OUTPUT_TOKENS
                or not result.text
                or result.decode_s <= 0
            ):
                raise ValueError("Incomplete generation")
            rows.append(
                {
                    "repeat": repeat,
                    "turn": index + 1,
                    "bucket": index,
                    "target": target,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "ttft_s": result.ttft_s,
                    "decode_tokens_per_second": (OUTPUT_TOKENS - 1) / result.decode_s,
                    "cached_tokens": result.cached_tokens,
                }
            )
            if observe is not None:
                observe(rows[-1])
            history.append({"role": "assistant", "content": result.text})
    wall = now() - started
    if wall > 600:
        raise ValueError(f"Benchmark exceeded 600 seconds ({wall:.3f})")
    return rows, wall


def summarize(
    rows: list[dict], wall: float, *, repeats: int = REPEATS
) -> dict[str, float]:
    if len(rows) != repeats * len(TARGETS):
        raise ValueError("Missing turns")
    rates = [r["decode_tokens_per_second"] for r in rows]
    latencies = [r["ttft_s"] * 1000 for r in rows if r["turn"] > 1]
    values = {
        "decode_tokens_per_second": statistics.mean(rates),
        "mean_ttft_turn2plus_ms": statistics.mean(latencies),
        "decode_min_tokens_per_second": min(rates),
        "decode_max_tokens_per_second": max(rates),
        "decode_stdev_tokens_per_second": statistics.stdev(rates),
        "ttft_turn2plus_min_ms": min(latencies),
        "ttft_turn2plus_max_ms": max(latencies),
        "wall_duration_s": wall,
        "workload_duration_s": wall,
    }
    for bucket in range(3):
        group = [r["decode_tokens_per_second"] for r in rows if r["bucket"] == bucket]
        values[f"bucket_{bucket}_decode_tokens_per_second"] = statistics.mean(group)
        values[f"bucket_{bucket}_decode_spread_tokens_per_second"] = max(group) - min(
            group
        )
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0
        for v in values.values()
    ):
        raise ValueError("Nonfinite or negative metric")
    return values
