#!/bin/sh
# One setup command for the agent and the evaluator. check_host.sh runs first.
set -eu
TASK_PATH=$(dirname "$0")
HERE=$(CDPATH= cd -- "$TASK_PATH" && pwd) || {
  printf 'Cannot enter task directory: %s\n' "$TASK_PATH" >&2
  exit 1
}
"$HERE/check_host.sh"
if [ "$#" -eq 0 ]; then
  set -- setup
fi
exec /usr/bin/python3 "$HERE/benchmark/task_env.py" "$@"
