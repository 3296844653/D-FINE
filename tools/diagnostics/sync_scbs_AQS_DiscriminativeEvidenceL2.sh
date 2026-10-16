#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
der_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$der_local" ]] || { printf '只允许从桌面D-FINE同步。\n' >&2; exit 1; }
[[ "$#" -le 1 ]] || { printf '用法：[--dry-run]\n' >&2; exit 1; }
der_dry=0
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数。\n' >&2; exit 1; }; der_dry=1
fi
der_target="${SCBS_DER_SSH_TARGET:-a5@100.76.151.67}"
der_remote='/home/a5/MyProject/wq_project/D-FINE'
der_manifest="$der_local/tools/diagnostics/scbs_AQS_DiscriminativeEvidenceL2_sync_files.txt"
while IFS= read -r der_file; do
  [[ -z "$der_file" ]] || test -f "$der_local/$der_file"
done < "$der_manifest"
ssh "$der_target" bash -s -- "$der_remote" <<'REMOTE'
set -euo pipefail
cd "$1"
for dep in train.py configs/dfine/dfine_s_scbs_hrw_110.yml src/zoo/dfine/hybrid_encoder.py \
    src/zoo/dfine/medium_finegrained.py src/zoo/dfine/scale_aligned_detail_fusion.py \
    src/zoo/dfine/semantic_differential_relation_fusion.py \
    src/solver/query_diagnostics.py src/solver/aqs_parameters.py \
    tools/diagnostics/analyze_detection_diagnostics.py tools/diagnostics/export_aqs_parameters.py; do
  test -f "$dep" || { printf '缺少现有服务器依赖：%s\n' "$dep" >&2; exit 1; }
done
REMOTE
der_backup="/home/a5/MyProject/wq_project/D-FINE-AQS-DiscriminativeEvidenceL2-backup-$(date +%Y%m%dT%H%M%S)-$$"
der_command=(rsync -av --relative --files-from="$der_manifest" --backup --backup-dir="$der_backup")
[[ "$der_dry" == 0 ]] || der_command+=(--dry-run)
"${der_command[@]}" "$der_local/" "$der_target:$der_remote/"
printf '桌面源码同步%s完成；不修改数据、基线配置、权重或旧结果。\n' "$( [[ "$der_dry" == 1 ]] && printf '演练' || printf '')"
[[ "$der_dry" == 1 ]] || printf '被替换源码备份：%s\n' "$der_backup"
