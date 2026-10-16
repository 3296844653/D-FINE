#!/usr/bin/env bash
# One command: independent training runs, then best-EMA metrics/confusion export.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for box_arg in "$@"; do
  case "$box_arg" in
    --dry-run|--help|-h)
      exec "${SCBS_BOXL_PYTHON:-python3}" -u tools/diagnostics/scbs_boxlocal_five.py "$@" ;;
  esac
done
box_conda="${SCBS_BOXL_CONDA:-${CONDA_EXE:-conda}}"
command -v "$box_conda" >/dev/null 2>&1 || { printf '找不到Conda。\n' >&2; exit 1; }
exec "$box_conda" run --no-capture-output -n "${SCBS_BOXL_ENV:-wq}" \
  python -u tools/diagnostics/scbs_boxlocal_five.py "$@"
