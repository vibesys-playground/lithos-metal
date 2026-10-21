"""Trusted pack-location and disk-admission policy. Never writes packed weights."""

import json
import shutil
from pathlib import Path

__all__ = ["PACK_BYTES", "MARGIN_BYTES", "PACK_KEY", "default_pack_dir", "resolve_pack"]
PACK_BYTES = 22_736_175_104
MARGIN_BYTES = 2 * 1024**3
PACK_KEY = "65dcd02bfffafe7ef9c806b04a40f155ea4690d0111d77e03efff89a53c7ffe9"


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
    manifest = json.loads((path / "manifest.json").read_text())
    if (path / "weights.pack").stat().st_size != manifest["nbytes"]:
        raise ValueError("Shared weights.pack is incomplete")
    # Reserve space even on cache hits: a future explicit layout migration must
    # never silently consume the last free bytes. Existing packs are never evicted.
    available = shutil.disk_usage(path).free if free_bytes is None else free_bytes
    required = PACK_BYTES + MARGIN_BYTES
    if available < required:
        raise ValueError(
            f"Insufficient free disk: {available} bytes available; need {required} "
            "bytes (one pack plus 2 GiB margin). Do not repack or copy weights."
        )
    return path
