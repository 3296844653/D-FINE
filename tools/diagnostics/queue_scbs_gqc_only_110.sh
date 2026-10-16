#!/usr/bin/env bash
# Upload a staged payload, then queue it in SERVER tmux. No live-source sync.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[[ "$(pwd -P)" == '/Users/wq/Desktop/D-FINE' ]] || { printf '请从Mac桌面D-FINE提交。\n' >&2; exit 1; }
exec "${SCBS_GQC_PYTHON:-python3}" -u tools/diagnostics/scbs_gqc_only_queue.py submit "$@"
