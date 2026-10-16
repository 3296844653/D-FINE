#!/usr/bin/env bash
# Same protocol as the serial control; only select parallel residual fusion.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_scbs_aqs_box_local_l2_110.sh" --parallel "$@"
