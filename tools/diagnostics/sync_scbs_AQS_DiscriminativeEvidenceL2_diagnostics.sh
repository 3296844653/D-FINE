#!/usr/bin/env bash
# This sync changes diagnostic TOOLS ONLY, never the trained detector/config.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
der_diag_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$der_diag_local" ]] || { printf '只允许从桌面D-FINE同步。\n' >&2; exit 1; }
[[ "$#" -le 1 ]] || { printf '用法：[--dry-run]\n' >&2; exit 1; }
der_diag_dry=0
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数。\n' >&2; exit 1; }; der_diag_dry=1
fi
der_diag_target="${SCBS_DER_SSH_TARGET:-a5@100.76.151.67}"
der_diag_remote='/home/a5/MyProject/wq_project/D-FINE'
der_diag_manifest="$der_diag_local/tools/diagnostics/scbs_AQS_DiscriminativeEvidenceL2_diagnostics_sync_files.txt"
while IFS= read -r der_diag_file; do
  [[ -z "$der_diag_file" ]] || test -f "$der_diag_local/$der_diag_file"
done < "$der_diag_manifest"
ssh "$der_diag_target" bash -s -- "$der_diag_remote" <<'REMOTE'
set -euo pipefail
cd "$1"
for dependency in train.py configs/dfine/dfine_s_scbs_hrw_110_AQS_DiscriminativeEvidenceL2.yml \
    src/zoo/dfine/discriminative_evidence.py tools/diagnostics/scbs_discriminative_evidence_experiment.py \
    tools/diagnostics/scbs_instance_evidence_experiment.py tools/diagnostics/analyze_detection_diagnostics.py \
    tools/diagnostics/export_aqs_parameters.py; do
  test -f "$dependency" || { printf '缺少已训练项目的依赖：%s\n' "$dependency" >&2; exit 1; }
done
REMOTE
der_diag_backup="/home/a5/MyProject/wq_project/D-FINE-DER-diagnostics-tools-backup-$(date +%Y%m%dT%H%M%S)-$$"
der_diag_command=(rsync -av --relative --files-from="$der_diag_manifest" --backup --backup-dir="$der_diag_backup")
[[ "$der_diag_dry" == 0 ]] || der_diag_command+=(--dry-run)
"${der_diag_command[@]}" "$der_diag_local/" "$der_diag_target:$der_diag_remote/"
printf '诊断工具同步%s完成；未同步/修改模型、配置、数据或权重。\n' "$( [[ "$der_diag_dry" == 1 ]] && printf '演练' || printf '')"
[[ "$der_diag_dry" == 1 ]] || printf '被替换工具备份：%s\n' "$der_diag_backup"
