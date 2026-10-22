"""Trusted pack-location and disk-admission policy. Never writes packed weights."""

import hashlib
import json
import shutil
from pathlib import Path

__all__ = [
    "ENVIRONMENT_REFUSED_EXIT",
    "PACK_BYTES",
    "MARGIN_BYTES",
    "PACK_KEY",
    "HostRefusal",
    "default_pack_dir",
    "failure_exit_status",
    "resolve_pack",
]
PACK_BYTES = 22_736_175_104
MARGIN_BYTES = 2 * 1024**3
PACK_KEY = "65dcd02bfffafe7ef9c806b04a40f155ea4690d0111d77e03efff89a53c7ffe9"


# VibeSys evaluator protocol: a trusted evaluator command that exits 75 says its host
# refused the measurement before judging the candidate, so the run measures again later.
ENVIRONMENT_REFUSED_EXIT = 75


class HostRefusal(ValueError):
    """The host cannot admit this measurement; nothing of the candidate was judged."""


def failure_exit_status(failures):
    """75 when every failure is the host's refusal, else 1 (the candidate failed)."""
    if failures and all(isinstance(failure, HostRefusal) for failure in failures):
        return ENVIRONMENT_REFUSED_EXIT
    return 1


def default_pack_dir(home):
    return (
        Path(home)
        / ".cache/lithos-metal/flint-int4/3b93f9053b530488405ae9fca51d4900d6ce21f0/block-r16-interleaved16-c9216"
    )


def resolve_pack(workspace, env, home, *, free_bytes=None):
    path = (
        Path(env.get("FLINT_PACK_DIR", default_pack_dir(home))).expanduser().resolve()
    )
    workspace = Path(workspace).resolve()
    if path == workspace or workspace in path.parents:
        raise ValueError("FLINT_PACK_DIR must be outside the candidate worktree")
    if not (path / "manifest.json").is_file() or not (path / "weights.pack").is_file():
        raise ValueError(
            "FLINT_PACK_DIR must name an existing shared pack; automatic packing is disabled"
        )
    manifest_bytes = (path / "manifest.json").read_bytes()
    key_file = path / "pack-key.json"
    if not key_file.is_file():
        raise ValueError("Shared pack is missing pack-key.json storage provenance")
    metadata = json.loads(key_file.read_text())
    if metadata.get("key") != PACK_KEY:
        raise ValueError("FLINT_PACK_DIR does not name the authorized shared pack key")
    if (
        metadata.get("identity", {}).get("manifest_sha256")
        != hashlib.sha256(manifest_bytes).hexdigest()
    ):
        raise ValueError("Shared pack manifest does not match its storage identity")
    manifest = json.loads(manifest_bytes)
    if (path / "weights.pack").stat().st_size != manifest["nbytes"]:
        raise ValueError("Shared weights.pack is incomplete")
    # Reserve space even on cache hits: a future explicit layout migration must
    # never silently consume the last free bytes. Existing packs are never evicted.
    available = shutil.disk_usage(path).free if free_bytes is None else free_bytes
    required = PACK_BYTES + MARGIN_BYTES
    if available < required:
        raise HostRefusal(
            f"Insufficient free disk: {available} bytes available; need {required} "
            "bytes (one pack plus 2 GiB margin). Do not repack or copy weights."
        )
    return path
