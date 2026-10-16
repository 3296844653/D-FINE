#!/usr/bin/env bash
# Plain D-FINE + original GQC. No AQS, BoxLocal, synthetic suite or ON/OFF.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for gqc_arg in "$@"; do
  case "$gqc_arg" in
    --dry-run|--help|-h)
      exec "${SCBS_GQC_PYTHON:-python3}" -u tools/diagnostics/scbs_gqc_only.py "$@" ;;
  esac
done
gqc_conda="${SCBS_GQC_CONDA:-${CONDA_EXE:-conda}}"
command -v "$gqc_conda" >/dev/null 2>&1 || { printf '找不到Conda。\n' >&2; exit 1; }
exec "$gqc_conda" run --no-capture-output -n "${SCBS_GQC_ENV:-wq}" \
  python -u tools/diagnostics/scbs_gqc_only.py "$@"
