#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for serf_arg in "$@"; do
  if [[ "$serf_arg" == --dry-run ]]; then
    exec python3 tools/diagnostics/scbs_serf_queue.py "$@"
  fi
done
serf_conda="${SCBS_SERF_CONDA:-${CONDA_EXE:-conda}}"
exec "$serf_conda" run --no-capture-output -n "${SCBS_SERF_ENV:-wq}" \
  python -u tools/diagnostics/scbs_serf_queue.py "$@"
