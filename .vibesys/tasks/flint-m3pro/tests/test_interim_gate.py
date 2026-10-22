"""Hermetic interim-gate tests. No model, Metal, clock, sleep, process or socket."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accuracy_checker"))
from checker import (
    APPROVED_THRESHOLDS,
    PREFIX_IDS_SHA256,
    compare_logits,
    load_gate,
    load_pin_margins,
    load_reference_logits,
    check_accuracy,
)
from test_bundle import FakeTokenizer, PinTests


def prefix_ids() -> list[int]:
    root = Path(__file__).resolve().parents[4]
    example = json.loads(
        (root / "tests/models/flint/goldens/int4.json").read_text()
    )["examples"][1]
    base = example["prompt_ids"] + example["greedy_ids"][:-1]
    return (base * ((1024 + len(base) - 1) // len(base)))[:1024]


def gate_payload(**overrides) -> dict:
    ids = prefix_ids()
    payload = {
        "version": 0,
        "name": "interim_gate_v0",
        "thresholds": {
            key: {"value": value, "source": f"approved {key}"}
            for key, value in APPROVED_THRESHOLDS.items()
        },
        "prefix": {
            "length": 1024,
            "source": "second fixture repeated to 1024",
            "ids_sha256": PREFIX_IDS_SHA256,
            "ids": ids,
        },
        "reference_logits": {
            "file": "rec-bf16-chunk32-logits-1024.npy",
            "schedule": "bf16 chunk 32",
            "sha256": "0" * 64,
            "shape": [1024, 150144],
            "dtype": "float32",
        },
        "pin_margins": {
            "file": "pin_margins.json",
            "schedule": "prefill chunk 128",
            "sha256": "0" * 64,
        },
    }
    payload.update(overrides)
    return payload


def limits(**overrides):
    values = {
        "margin_limit": 0.3125,
        "max_logit_abs": 22.5,
        "relative_l2_max": 0.19047058,
        "min_cosine": 0.99509772,
    }
    values.update(overrides)
    return values


class GateFileTests(unittest.TestCase):
    def test_approved_thresholds_and_prefix_load(self):
        ids = prefix_ids()
        self.assertEqual(hashlib.sha256(json.dumps(ids).encode()).hexdigest(), PREFIX_IDS_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interim_gate_v0.json"
            path.write_text(json.dumps(gate_payload()))
            loaded = load_gate(path)
        self.assertEqual(loaded["thresholds"]["margin"]["value"], 0.3125)
        self.assertEqual(loaded["prefix"]["ids"], ids)

    def test_changed_threshold_fails_closed(self):
        payload = gate_payload()
        payload["thresholds"]["margin"]["value"] = 0.5
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "margin"):
                load_gate(path)


class PinToleranceTests(unittest.TestCase):
    def host(self):
        return PinTests()

    def test_near_tie_stops_and_ignores_the_later_trajectory(self):
        host = self.host()
        pins = host.load(host.payload())
        margins = host.margins(pins, 1.0)
        margins[pins[0]["id"]][0] = 0.1
        margins[pins[0]["id"]][5] = 1.0
        base = host.server(pins)

        class Server:
            def greedy(self, messages, max_tokens):
                body = base.greedy(messages, max_tokens)
                if messages == pins[0]["messages"]:
                    ids = list(pins[0]["expected_token_ids"])
                    ids[0] = ids[0] + 1
                    ids[5] = ids[5] + 1
                    body["completion_token_ids"] = ids
                    body["choices"][0]["message"]["content"] = FakeTokenizer().visible(ids)
                return body

        result = check_accuracy(Server(), FakeTokenizer(), pins, margins)
        self.assertEqual(
            [stop["token"] for stop in result["tolerated_pin_stops"]],
            [0, 0],
        )

    def test_margin_at_the_limit_is_intolerant(self):
        host = self.host()
        pins = host.load(host.payload())
        margins = host.margins(pins, 0.3125)
        with self.assertRaisesRegex(ValueError, "diverged at token 0"):
            check_accuracy(host.server(pins, corrupt=True), FakeTokenizer(), pins, margins)

    def test_margin_file_hash_rows_and_nonfinite_fail_closed(self):
        host = self.host()
        pins = host.load(host.payload())
        gate = gate_payload()
        payload = {
            "version": 1,
            "reference_commit": "abc",
            "schedule": "prefill chunk 128",
            "pins": {
                pin["id"]: {"margins": [0.1] * 32, "top1": [1] * 32} for pin in pins
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pin_margins.json"
            path.write_text(json.dumps(payload))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            gate["pin_margins"]["sha256"] = digest
            self.assertEqual(set(load_pin_margins(path, gate, pins)), {pin["id"] for pin in pins})
            gate["pin_margins"]["sha256"] = "1" * 64
            with self.assertRaisesRegex(ValueError, "hash"):
                load_pin_margins(path, gate, pins)
            payload["pins"][pins[0]["id"]]["margins"][0] = float("nan")
            path.write_text(json.dumps(payload))
            gate["pin_margins"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "margin"):
                load_pin_margins(path, gate, pins)
            payload["pins"][pins[0]["id"]]["margins"] = [0.1] * 31
            payload["pins"][pins[0]["id"]]["top1"] = [1] * 31
            path.write_text(json.dumps(payload))
            gate["pin_margins"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_pin_margins(path, gate, pins)
        with self.assertRaisesRegex(ValueError, "Missing"):
            load_pin_margins(Path("missing-margins.json"), gate, pins)


class LogitTests(unittest.TestCase):
    def test_exact_rows_pass_and_a_near_tie_is_tolerated(self):
        reference = np.array([[0.0, 1.0, 0.9]], dtype=np.float32)
        passed = compare_logits(reference, reference.copy(), **limits())
        self.assertEqual(passed["top1_mismatches"], 0)
        flipped = np.array([[0.0, 0.9, 1.0]], dtype=np.float32)
        tolerated = compare_logits(
            reference,
            flipped,
            **limits(max_logit_abs=1.0, relative_l2_max=1.0, min_cosine=0.9),
        )
        self.assertEqual(tolerated["top1_mismatches"], 1)
        self.assertLess(tolerated["mismatch_margin_max"], 0.3125)

    def test_intolerant_mismatch_and_metric_bounds_fail(self):
        reference = np.array([[0.0, 5.0, 0.0]], dtype=np.float32)
        flipped = np.array([[5.0, 0.0, 0.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "position 0"):
            compare_logits(reference, flipped, **limits(max_logit_abs=100, relative_l2_max=100, min_cosine=0))
        grown = np.array([[30.0, 1.0]], dtype=np.float32)
        base = np.array([[3.0, 1.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "max_abs"):
            compare_logits(base, grown, **limits(relative_l2_max=100, min_cosine=0))
        shifted = np.array([[10.0, 1.0]], dtype=np.float32)
        aligned = np.array([[10.0, 0.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "cosine"):
            compare_logits(
                aligned,
                shifted,
                **limits(max_logit_abs=22.5, relative_l2_max=1.0, min_cosine=0.999),
            )
        scaled = aligned * 2
        with self.assertRaisesRegex(ValueError, "relative_l2"):
            compare_logits(
                aligned,
                scaled,
                **limits(max_logit_abs=100, relative_l2_max=0.1, min_cosine=0),
            )

    def test_shape_nonfinite_and_trace_file_fail_closed(self):
        row = np.ones((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            compare_logits(row, row[:1], **limits())
        broken = row.copy()
        broken[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            compare_logits(row, broken, **limits())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.npy"
            gate = gate_payload()
            with self.assertRaisesRegex(ValueError, "Missing"):
                load_reference_logits(path, gate)
            np.save(path, row)
            gate["reference_logits"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "shape"):
                load_reference_logits(path, gate)
            gate["reference_logits"]["sha256"] = "2" * 64
            with self.assertRaisesRegex(ValueError, "hash"):
                load_reference_logits(path, gate)


if __name__ == "__main__":
    unittest.main()
