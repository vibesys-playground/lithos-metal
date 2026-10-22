#!/bin/sh
# Host prerequisites that a lockfile cannot pin. Prints every miss and how to
# fix it, then exits nonzero. No package install and no PATH lookup.
set -u

TASK_PATH=$(dirname "$0")
HERE=$(CDPATH= cd -- "$TASK_PATH" && pwd) || {
  printf 'Cannot enter task directory: %s\n' "$TASK_PATH" >&2
  exit 1
}
ROOT=$(CDPATH= cd -- "$HERE/../../.." && pwd) || {
  printf 'Cannot resolve project root for task directory: %s\n' "$HERE" >&2
  exit 1
}
failures=""

note() {
  failures="${failures}
$1"
}

if ! devdir=$(/usr/bin/xcode-select -p 2>/dev/null) || [ ! -d "$devdir" ]; then
  note "missing: Xcode command line tools (xcode-select -p failed).
fix: run xcode-select --install and wait until the developer directory exists."
fi

# The native target compiles with clang and links -framework Metal. It does not
# invoke the metal shader compiler. Command Line Tools ship clang and the
# framework; xcrun -f metal needs a full Xcode and is accepted when present.
metal=$(/usr/bin/xcrun -f metal 2>/dev/null || true)
clang=$(/usr/bin/xcrun -f clang 2>/dev/null || true)
sdk=$(/usr/bin/xcrun --sdk macosx --show-sdk-path 2>/dev/null || true)
framework="${sdk}/System/Library/Frameworks/Metal.framework"
if [ -n "$metal" ] && [ -x "$metal" ]; then
  :
elif [ -n "$clang" ] && [ -x "$clang" ] && [ -d "$framework" ]; then
  :
else
  note "missing: Metal compiler (xcrun -f metal failed, and xcrun -f clang or Metal.framework is absent).
fix: install the Xcode command line tools (xcode-select --install). xcrun -f metal must print a compiler, or xcrun -f clang must print clang and the macOS SDK must contain Metal.framework."
fi

version=$(/usr/bin/sw_vers -productVersion 2>/dev/null || true)
major=${version%%.*}
case "$major" in
  ''|*[!0-9]*)
    note "missing: macOS version (sw_vers -productVersion failed).
fix: this task requires macOS 26 or newer."
    ;;
  *)
    if [ "$major" -lt 26 ]; then
      note "missing: macOS 26 or newer (this host reports ${version}).
fix: lithos-metal's Metal runtime requires macOS 26 or newer."
    fi
    ;;
esac

if [ -n "${HF_HUB_CACHE:-}" ]; then
  hub="$HF_HUB_CACHE"
elif [ -n "${HF_HOME:-}" ]; then
  hub="$HF_HOME/hub"
else
  hub="${HOME:?HOME is required}/.cache/huggingface/hub"
fi
if [ -n "${FLINT_MODEL_PATH:-}" ]; then
  snapshot=$FLINT_MODEL_PATH
else
  snapshot="$hub/models--RecursiveSI--flint-int4/snapshots/3b93f9053b530488405ae9fca51d4900d6ce21f0"
fi

if ! weight_note=$(/usr/bin/python3 - "$snapshot" "$HERE/reference/checkpoint.json" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

snapshot, manifest_path = Path(sys.argv[1]), Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text())
missing = [name for name in manifest["checkpoint_files"] if not (snapshot / name).is_file()]
if not missing:
    sys.exit(0)
print(
    "missing: pinned Flint weights under "
    + str(snapshot)
    + " ("
    + ", ".join(missing)
    + ")."
)
print(
    "fix: the task does not download weights. Restore the local Hugging Face "
    "snapshot for RecursiveSI/flint-int4 revision "
    "3b93f9053b530488405ae9fca51d4900d6ce21f0, or set FLINT_MODEL_PATH to that "
    "snapshot. Do not copy weights into the candidate checkout."
)
sys.exit(1)
PY
); then
  note "$weight_note"
fi

if [ -n "${FLINT_PACK_DIR:-}" ]; then
  pack=$FLINT_PACK_DIR
else
  pack="${HOME:?HOME is required}/.cache/lithos-metal/flint-int4/3b93f9053b530488405ae9fca51d4900d6ce21f0/block-r16-interleaved16-c9216"
fi

case "$pack" in
  "$ROOT"|"$ROOT"/*)
    note "missing: shared pack outside the candidate checkout (FLINT_PACK_DIR is inside it).
fix: point FLINT_PACK_DIR at the existing shared pack. The task does not copy or pack weights."
    ;;
esac

if ! pack_note=$(/usr/bin/python3 - "$pack" 2>&1 <<'PY'
import hashlib
import json
import sys
from pathlib import Path

pack = Path(sys.argv[1])
key = "65dcd02bfffafe7ef9c806b04a40f155ea4690d0111d77e03efff89a53c7ffe9"
problems = []
for name in ("manifest.json", "weights.pack", "pack-key.json"):
    if not (pack / name).is_file():
        problems.append(f"{name} is absent")
if not problems:
    manifest_bytes = (pack / "manifest.json").read_bytes()
    metadata = json.loads((pack / "pack-key.json").read_text())
    manifest = json.loads(manifest_bytes)
    if metadata.get("key") != key:
        problems.append("pack-key.json does not name the pinned pack")
    identity = metadata.get("identity", {}).get("manifest_sha256")
    if identity != hashlib.sha256(manifest_bytes).hexdigest():
        problems.append("manifest.json does not match pack-key.json")
    size = (pack / "weights.pack").stat().st_size
    if size != manifest.get("nbytes"):
        problems.append("weights.pack size does not match its manifest")
if not problems:
    sys.exit(0)
print("missing: shared pack at " + str(pack) + " (" + "; ".join(problems) + ").")
print(
    "fix: reuse the existing read-only pack at "
    "$HOME/.cache/lithos-metal/flint-int4/3b93f9053b530488405ae9fca51d4900d6ce21f0/"
    "block-r16-interleaved16-c9216 or set FLINT_PACK_DIR to it. "
    "The task does not pack or copy weights."
)
sys.exit(1)
PY
); then
  note "$pack_note"
fi

if [ -n "$failures" ]; then
  printf 'Host check failed:%s\n' "$failures" >&2
  exit 1
fi

echo "Host check passed."
