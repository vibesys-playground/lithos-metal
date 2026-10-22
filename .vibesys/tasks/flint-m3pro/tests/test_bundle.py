"""Hermetic evaluator tests: only injected clocks/transports and temporary files.

Run: python3 -m unittest discover -s .vibesys/tasks/flint-m3pro/tests -v
No model, Metal, real time, sleep, process, thread or socket is used.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accuracy_checker"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from checker import PROBES, check_accuracy, load_pins, scan_engine
from launcher import MODEL, REVISION, PinnedTokenizer, launch_argv
from make_pins import REC_COMMIT, reuse_port_goldens
from vs_protocol import METRICS, ResultWriter


class FakeTokenizer:
    """Reversible text codec; chat framing uses one synthetic token per four characters."""

    def ids(self, messages):
        return [ord(c) for c in json.dumps(messages)[::4]]

    def count(self, messages):
        return len(self.ids(messages))

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def visible(self, ids):
        return self.decode(ids)


class LauncherTests(unittest.TestCase):
    def test_template_adapter_returns_token_ids_with_thinking_disabled(self):
        class TemplateTokenizer:
            def apply_chat_template(
                self,
                messages,
                *,
                tokenize,
                add_generation_prompt,
                return_dict=True,
                enable_thinking=True,
            ):
                if not tokenize or not add_generation_prompt:
                    raise ValueError("This fixture requires a generation prompt")
                tokens = [1, 2, 3, 4] if enable_thinking else [1, 2, 3]
                return {"input_ids": tokens} if return_dict else tokens

        tokenizer = PinnedTokenizer(TemplateTokenizer())
        self.assertEqual(
            tokenizer.ids([{"role": "user", "content": "review code"}]), [1, 2, 3]
        )
        self.assertEqual(
            tokenizer.count([{"role": "user", "content": "review code"}]), 3
        )

    def test_launcher_uses_candidate_and_fixed_numerics(self):
        argv = launch_argv(Path("checkpoint"), 1234, Path("shared-pack"))
        self.assertIn("monolith.serve", argv)
        for flag in (
            "--no-draft",
            "--no-commute-norm",
            "--local-files-only",
            "--max-context",
            "--keep-warm",
        ):
            self.assertIn(flag, argv)
        self.assertNotIn("mlx", " ".join(argv))
        cached = launch_argv(Path("checkpoint"), 1234, Path("shared-pack"))
        self.assertEqual(cached[-2:], ["--pack", "shared-pack"])


class PinTests(unittest.TestCase):
    def payload(self):
        tokenizer = FakeTokenizer()
        pins = [
            {
                "id": str(i),
                "messages": [{"role": "user", "content": f"code {i}"}],
                "prompt_token_ids": tokenizer.ids(
                    [{"role": "user", "content": f"code {i}"}]
                ),
                "expected_token_ids": tokenizer.encode("x" * 32),
            }
            for i in range(4)
        ]
        return {
            "version": 1,
            "model": MODEL,
            "revision": REVISION,
            "kv_format": "bf16",
            "reference": {"commit": "reference-commit"},
            "pins": pins,
        }

    def load(self, payload):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pins.json"
            path.write_text(json.dumps(payload))
            return load_pins(path)

    def server(self, pins, corrupt=False):
        replies = {
            json.dumps(messages): answer
            for (messages, _), answer in zip(
                PROBES, ("42", "4", "cobalt-7319", "cedar-west 13")
            )
        }
        replies.update({json.dumps(p["messages"]): "x" * 32 for p in pins})

        class PinServer:
            def greedy(self, messages, max_tokens):
                content = replies[json.dumps(messages)]
                if corrupt and messages == pins[-1]["messages"]:
                    content = "y" + content[1:]
                return {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"completion_tokens": len(content)},
                    "completion_token_ids": [ord(c) for c in content],
                }

        return PinServer()

    def test_pins_and_probes_pass_and_any_divergence_fails(self):
        pins = self.load(self.payload())
        result = check_accuracy(self.server(pins), FakeTokenizer(), pins)
        self.assertEqual(result["pins_passed"], 4)
        with self.assertRaisesRegex(ValueError, "diverged at token 0"):
            check_accuracy(self.server(pins, corrupt=True), FakeTokenizer(), pins)

    def test_special_tokens_require_raw_ids_even_when_visible_text_matches(self):
        pins = self.load(self.payload())
        pins[0]["expected_token_ids"] = [0, *([ord("x")] * 31)]
        base = self.server(pins)

        class SpecialTokenizer(FakeTokenizer):
            def visible(self, ids):
                return self.decode([n for n in ids if n > 1])

        class RawServer:
            missing = False
            corrupt = False

            def greedy(self, messages, max_tokens):
                body = base.greedy(messages, max_tokens)
                for pin in pins:
                    if messages == pin["messages"]:
                        ids = list(pin["expected_token_ids"])
                        body["completion_token_ids"] = ids
                        body["choices"][0]["message"]["content"] = (
                            SpecialTokenizer().visible(ids)
                        )
                        body["usage"]["completion_tokens"] = 32
                        if self.corrupt and ids[0] == 0:
                            body["completion_token_ids"] = [1, *ids[1:]]
                        if self.missing:
                            body.pop("completion_token_ids")
                return body

        server = RawServer()
        check_accuracy(server, SpecialTokenizer(), pins)
        server.corrupt = True
        with self.assertRaisesRegex(ValueError, "diverged at token 0"):
            check_accuracy(server, SpecialTokenizer(), pins)
        server.corrupt, server.missing = False, True
        with self.assertRaisesRegex(ValueError, "requires raw completion_token_ids"):
            check_accuracy(server, SpecialTokenizer(), pins)

    def test_pin_provenance_and_schema_fail_closed(self):
        for field, value in (
            ("revision", "wrong"),
            ("kv_format", "int8"),
            ("version", 2),
            ("pins", []),
            ("unknown", 1),
        ):
            payload = self.payload()
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.load(payload)
        with self.assertRaisesRegex(ValueError, "Missing reference/pins.json"):
            load_pins(Path("missing/pins.json"))

    def test_duplicate_and_short_and_bool_ids_fail(self):
        for mutate in (
            lambda p: p[0].update(expected_token_ids=[1] * 31),
            lambda p: p[0].update(expected_token_ids=[True] * 32),
            lambda p: p[0].update(id=p[1]["id"]),
            lambda p: p[0].update(messages=p[1]["messages"]),
        ):
            payload = self.payload()
            mutate(payload["pins"])
            with self.assertRaises(ValueError):
                self.load(payload)

    def test_template_mismatch_fails(self):
        pins = self.load(self.payload())
        pins[0]["prompt_token_ids"] = [0]
        with self.assertRaisesRegex(ValueError, "Chat-template mismatch"):
            check_accuracy(self.server(pins), FakeTokenizer(), pins)

    def test_source_scan(self):
        for source in (
            "import mlx.core",
            "from mlx_lm import load",
            "import rec_mlx",
            "import importlib\nimportlib.import_module('mlx_lm')",
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "server.py").write_text(source)
                with self.subTest(source=source), self.assertRaises(ValueError):
                    scan_engine(path)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "server.py").write_text(
                '"MLX reference may be read."\nimport monolith'
            )
            scan_engine(path)


class GoldenReuseTests(unittest.TestCase):
    def test_reuses_template_identical_pins_and_records_early_eos(self):
        tokenizer = FakeTokenizer()
        examples = []
        for i in range(3):
            content = f"code {i}"
            examples.append(
                {
                    "prompt": f"<|user|>{content}<|assistant|>",
                    "prompt_ids": tokenizer.ids([{"role": "user", "content": content}]),
                    "greedy_ids": [0 if i == 0 else 120] * 64,
                    "top_logits": [],
                    "step_top2": [],
                }
            )
        data = {
            "model": MODEL,
            "revision": REVISION,
            "reference": {"commit": REC_COMMIT, "cache": "unquantized KVCache"},
            "examples": examples,
        }
        pins, excluded = reuse_port_goldens(data, tokenizer, [0])
        self.assertEqual(len(pins), 2)
        self.assertEqual(excluded, [0])
        data["examples"][1]["prompt_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "chat template"):
            reuse_port_goldens(data, tokenizer, [0])


class ProtocolTests(unittest.TestCase):
    def test_schema_flushed_before_result_and_failure_has_no_result(self):
        for fail in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "result.jsonl"
                writer = ResultWriter(path)
                self.assertEqual(json.loads(path.read_text())["kind"], "hello")
                if fail:
                    writer.error("missing weights")
                else:
                    writer.result(dict.fromkeys(METRICS, 1.0))
                writer.close()
                records = [json.loads(s) for s in path.read_text().splitlines()]
                self.assertEqual(
                    [r["kind"] for r in records],
                    ["hello", "error" if fail else "result"],
                )

    def test_nonfinite_bool_and_unknown_metrics_rejected(self):
        for value in (float("nan"), float("inf"), True):
            with tempfile.TemporaryDirectory() as directory:
                writer = ResultWriter(Path(directory) / "result")
                row = dict.fromkeys(METRICS, 1.0)
                row["decode_tokens_per_second"] = value
                with self.assertRaises(ValueError):
                    writer.result(row)
                writer.close()


if __name__ == "__main__":
    unittest.main()
