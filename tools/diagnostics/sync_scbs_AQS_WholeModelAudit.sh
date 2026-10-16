#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
audit_local='/Users/wq/Desktop/D-FINE'
[[ "$(pwd -P)" == "$audit_local" ]] || { printf '只允许从桌面D-FINE同步。\n' >&2; exit 1; }
[[ "$#" -le 1 && ( "$#" == 0 || "${1:-}" == --dry-run ) ]] || { printf '用法：[--dry-run]\n' >&2; exit 1; }
audit_target="${AQS_AUDIT_SSH_TARGET:-a5@100.76.151.67}"
audit_remote='/home/a5/MyProject/wq_project/D-FINE'
audit_manifest="$audit_local/tools/diagnostics/scbs_AQS_WholeModelAudit_sync_files.txt"
while IFS= read -r audit_file; do
  [[ -z "$audit_file" ]] || test -f "$audit_local/$audit_file"
done < "$audit_manifest"
ssh "$audit_target" bash -s -- "$audit_remote" <<'REMOTE'
set -euo pipefail
cd "$1"
test -f src/zoo/dfine/dfine_decoder.py
test -f configs/dfine/dfine_s_scbs_hrw_110_aqs_layer2_r010.yml
if ps -ww -u "$(id -un)" -o args= | grep -E '[p]ython[^ ]* .*tools/diagnostics/(scbs_|stage_feature_probe|gt_box_classifier|wait_for_gpu)' >/dev/null; then
  printf '相关诊断正在运行，暂不替换工具。\n' >&2; exit 1
fi
REMOTE
audit_backup="/home/a5/MyProject/wq_project/D-FINE-WholeModelAudit-backup-$(date +%Y%m%dT%H%M%S)-$$"
audit_command=(rsync -av --relative --files-from="$audit_manifest" --backup --backup-dir="$audit_backup")
[[ "$#" == 0 ]] || audit_command+=(--dry-run)
"${audit_command[@]}" "$audit_local/" "$audit_target:$audit_remote/"
printf 'WholeModelAudit诊断文件同步完成；替换文件备份目录：%s\n' "$audit_backup"
