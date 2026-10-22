"""Hermetic launch policy: no model, process or Metal work."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
from pack_policy import (
    ENVIRONMENT_REFUSED_EXIT,
    MARGIN_BYTES,
    PACK_BYTES,
    PACK_KEY,
    HostRefusal,
    default_pack_dir,
    failure_exit_status,
    resolve_pack,
)


def write_identity(pack):
    (pack / "pack-key.json").write_text(
        json.dumps(
            {
                "key": PACK_KEY,
                "identity": {
                    "manifest_sha256": hashlib.sha256(
                        (pack / "manifest.json").read_bytes()
                    ).hexdigest()
                },
            }
        )
    )


class PackPolicyTests(unittest.TestCase):
    def test_default_and_env_reuse_without_writing(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            pack = default_pack_dir(home).resolve()
            pack.mkdir(parents=True)
            (pack / "manifest.json").write_text(json.dumps({"nbytes": 3}))
            (pack / "weights.pack").write_bytes(b"abc")
            write_identity(pack)
            before = {
                p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in pack.iterdir()
            }
            for env in ({}, {"FLINT_PACK_DIR": str(pack)}):
                self.assertEqual(
                    resolve_pack(
                        home / "worktree",
                        env,
                        home,
                        free_bytes=PACK_BYTES + MARGIN_BYTES,
                    ),
                    pack,
                )
            after = {
                p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in pack.iterdir()
            }
            self.assertEqual(before, after)
            with self.assertRaisesRegex(HostRefusal, "Insufficient free disk"):
                resolve_pack(
                    home / "worktree",
                    {},
                    home,
                    free_bytes=PACK_BYTES + MARGIN_BYTES - 1,
                )

    def test_missing_pack_never_creates_any_files(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            with self.assertRaisesRegex(ValueError, "automatic packing is disabled"):
                resolve_pack(home / "worktree", {}, home, free_bytes=2**60)
            self.assertEqual(list(home.iterdir()), [])

    def test_worktree_pack_and_symlink_into_worktree_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            worktree = home / "worktree"
            pack = worktree / "pack"
            pack.mkdir(parents=True)
            alias = home / "alias"
            alias.symlink_to(pack, target_is_directory=True)
            for path in (pack, alias):
                with self.assertRaisesRegex(
                    ValueError, "outside the candidate worktree"
                ):
                    resolve_pack(
                        worktree, {"FLINT_PACK_DIR": str(path)}, home, free_bytes=2**60
                    )

    def test_incomplete_pack_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            pack = default_pack_dir(home).resolve()
            pack.mkdir(parents=True)
            (pack / "manifest.json").write_text(json.dumps({"nbytes": 4}))
            (pack / "weights.pack").write_bytes(b"abc")
            write_identity(pack)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                resolve_pack(home / "worktree", {}, home, free_bytes=2**60)

    def test_storage_identity_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            pack = default_pack_dir(home).resolve()
            pack.mkdir(parents=True)
            (pack / "manifest.json").write_text(json.dumps({"nbytes": 3}))
            (pack / "weights.pack").write_bytes(b"abc")
            for metadata, message in (
                ({"key": "0" * 64}, "authorized shared pack key"),
                (
                    {"key": PACK_KEY, "identity": {"manifest_sha256": "0" * 64}},
                    "storage identity",
                ),
            ):
                (pack / "pack-key.json").write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, message):
                    resolve_pack(home / "worktree", {}, home, free_bytes=2**60)
            (pack / "pack-key.json").unlink()
            with self.assertRaisesRegex(ValueError, "pack-key.json"):
                resolve_pack(home / "worktree", {}, home, free_bytes=2**60)

    def test_only_a_host_refusal_alone_exits_with_the_refusal_status(self):
        refusal = HostRefusal("Insufficient free disk")
        self.assertEqual(failure_exit_status([refusal]), ENVIRONMENT_REFUSED_EXIT)
        self.assertEqual(failure_exit_status([refusal, refusal]), ENVIRONMENT_REFUSED_EXIT)
        # A candidate failure next to a refusal is still the candidate's.
        self.assertEqual(failure_exit_status([refusal, ValueError("logits mismatch")]), 1)
        self.assertEqual(failure_exit_status([ValueError("Insufficient free disk")]), 1)
        self.assertEqual(failure_exit_status([]), 1)
