#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for box_arg in "$@"; do
  if [[ "$box_arg" == --dry-run ]]; then
    exec python3 tools/diagnostics/scbs_box_local_sweep.py "$@"
  fi
done
box_conda="${SCBS_BOX_CONDA:-${CONDA_EXE:-conda}}"
exec "$box_conda" run --no-capture-output -n "${SCBS_BOX_ENV:-wq}" \
  python -u tools/diagnostics/scbs_box_local_sweep.py "$@"
