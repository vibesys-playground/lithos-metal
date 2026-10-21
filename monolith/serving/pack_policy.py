"""Disk admission and a serialized, bounded shared cache for explicit repacking.

The pinned pack is read-only and outside the eviction set. Callers hold their
model execution lock for the whole use, so no cached pack is evicted in use.
No model code is imported here. Builders must write only into their destination.
"""

import fcntl
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

__all__ = ["require_pack_space", "SharedPackCache"]


def require_pack_space(free_bytes, pack_bytes, margin_bytes):
    """Fail before loading/building when one full pack plus margin cannot fit."""
    if pack_bytes <= 0 or margin_bytes < 0 or free_bytes < 0:
        raise ValueError(
            "Pack disk admission requires nonnegative sizes and a positive pack size"
        )
    required = pack_bytes + margin_bytes
    if free_bytes < required:
        raise ValueError(
            f"Insufficient free disk: {free_bytes} bytes available; "
            f"need {required} bytes (one {pack_bytes}-byte pack plus "
            f"{margin_bytes}-byte margin). Remove unused data; do not copy weights or repack in a worktree."
        )


@contextmanager
def _locked(root):
    with (root / ".cache.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class SharedPackCache:
    """One extra layout-keyed pack, never the pinned pack; atomic publish/cleanup.

    ``build(destination)`` constructs and validates a complete pack, returning
    only after success. Lookup updates recency. Before a miss, least-recently-used
    extras (including interrupted builds) are removed before disk admission.
    The cache owns its root exclusively; unknown files/directories fail closed.
    A process crash can leave one partial build, reclaimed on the next lookup.
    """

    def __init__(self, root, *, pinned, pack_bytes, margin_bytes):
        self._root = Path(root).expanduser().resolve()
        self._pinned = Path(pinned).expanduser().resolve()
        if (
            self._root == self._pinned
            or self._root in self._pinned.parents
            or self._pinned in self._root.parents
        ):
            raise ValueError("Extra pack cache must be separate from the pinned pack")
        self._pack_bytes, self._margin_bytes = pack_bytes, margin_bytes

    def resolve(self, layout_key, build):
        if not re.fullmatch(r"[0-9a-f]{64}", layout_key):
            raise ValueError(
                "layout_key must be a full SHA-256 storage-compatibility key"
            )
        self._root.mkdir(parents=True, exist_ok=True)
        with _locked(self._root):
            entries = []
            for path in self._root.iterdir():
                if path.name == ".cache.lock":
                    continue
                if (
                    path.is_symlink()
                    or not path.is_dir()
                    or not (
                        re.fullmatch(r"[0-9a-f]{64}", path.name)
                        or path.name.startswith(".building-")
                    )
                ):
                    raise ValueError(f"Unexpected shared pack cache entry: {path.name}")
                entries.append(path)
            destination = self._root / layout_key
            hit = destination in entries
            # Keep only the requested hit. With a one-extra bound all older
            # entries must leave on a miss; sorting also recovers an overfull cache.
            for path in sorted(entries, key=lambda p: p.stat().st_mtime_ns):
                if path != destination:
                    shutil.rmtree(path)
            if hit:
                if (
                    not (destination / "manifest.json").is_file()
                    or not (destination / "weights.pack").is_file()
                ):
                    raise ValueError("Incomplete shared pack cache entry")
                destination.touch()
                return destination
            require_pack_space(
                shutil.disk_usage(self._root).free, self._pack_bytes, self._margin_bytes
            )
            temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=self._root))
            try:
                build(temporary)
                if (
                    not (temporary / "manifest.json").is_file()
                    or not (temporary / "weights.pack").is_file()
                ):
                    raise ValueError("Builder did not produce a complete shared pack")
                total_bytes = sum(
                    p.stat().st_size for p in temporary.rglob("*") if p.is_file()
                )
                if (
                    (temporary / "weights.pack").stat().st_size > self._pack_bytes
                    or total_bytes > self._pack_bytes + self._margin_bytes
                ):
                    raise ValueError("Built pack exceeds admitted size")
                temporary.rename(destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return destination
