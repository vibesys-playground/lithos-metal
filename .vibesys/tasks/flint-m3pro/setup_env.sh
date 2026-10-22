#!/bin/sh
# One setup command for the agent and the evaluator. check_host.sh runs first.
set -eu
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
"$HERE/check_host.sh"
if [ "$#" -eq 0 ]; then
  set -- setup
fi
exec /usr/bin/python3 "$HERE/benchmark/task_env.py" "$@"
