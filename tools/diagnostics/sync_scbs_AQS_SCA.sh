#!/usr/bin/env bash
# SCA changes configuration only; NEVER sync neural source or baseline YAMLs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
sca_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$sca_local" ]] || { printf '只允许从桌面D-FINE同步。\n' >&2; exit 1; }
[[ "$#" -le 1 ]] || { printf '用法：[--dry-run]\n' >&2; exit 1; }
sca_dry=0
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数。\n' >&2; exit 1; }; sca_dry=1
fi
sca_target="${SCBS_SCA_SSH_TARGET:-a5@100.76.151.67}"
sca_remote='/home/a5/MyProject/wq_project/D-FINE'
sca_manifest="$sca_local/tools/diagnostics/scbs_AQS_SCA_sync_files.txt"
while IFS= read -r sca_file; do
  [[ -z "$sca_file" ]] || test -f "$sca_local/$sca_file"
done < "$sca_manifest"
ssh "$sca_target" bash -s -- "$sca_remote" <<'REMOTE'
set -euo pipefail
cd "$1"
for dep in train.py configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml \
    tools/diagnostics/scbs_instance_evidence_experiment.py \
    tools/diagnostics/analyze_detection_diagnostics.py tools/diagnostics/export_aqs_parameters.py; do
  test -f "$dep" || { printf '缺少现有AQS项目依赖：%s\n' "$dep" >&2; exit 1; }
done
if ps -ww -u "$(id -un)" -o args= | grep -E '[t]orchrun .*dfine_s_scbs_hrw_110_AQS_SCA|[p]ython[^ ]* .*dfine_s_scbs_hrw_110_AQS_SCA' >/dev/null; then
  printf 'AQS_SCA正在运行，暂不替换其配置/工具。\n' >&2; exit 1
fi
REMOTE
sca_backup="/home/a5/MyProject/wq_project/D-FINE-AQS-SCA-config-backup-$(date +%Y%m%dT%H%M%S)-$$"
sca_cmd=(rsync -av --relative --files-from="$sca_manifest" --backup --backup-dir="$sca_backup")
[[ "$sca_dry" == 0 ]] || sca_cmd+=(--dry-run)
"${sca_cmd[@]}" "$sca_local/" "$sca_target:$sca_remote/"
printf 'AQS+SCA配置/工具同步完成%s；不同步模型源码、原基线配置、数据、权重或旧输出。\n' "$( [[ "$sca_dry" == 1 ]] && printf '（演练）' || printf '')"
[[ "$sca_dry" == 1 ]] || printf '被替换文件备份：%s\n' "$sca_backup"
