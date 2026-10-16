#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Dry-run needs only stdlib Python; never checks CUDA or starts training.
for behavior_arg in "$@"; do
  if [[ "$behavior_arg" == --dry-run ]]; then
    exec python3 tools/diagnostics/scbs_behavior_four_queue.py "$@"
  fi
done
behavior_conda="${SCBS_BEHAVIOR_CONDA:-${CONDA_EXE:-conda}}"
exec "$behavior_conda" run --no-capture-output -n "${SCBS_BEHAVIOR_ENV:-wq}" \
  python -u tools/diagnostics/scbs_behavior_four_queue.py "$@"
