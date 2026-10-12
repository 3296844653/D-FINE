#!/usr/bin/env bash
# Sync diagnostics ONLY; never replace model/training configs or old outputs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
audit_sync_dry=0
[[ "$#" -le 1 ]] || { printf '用法：bash tools/diagnostics/sync_scbs_SADF_runtime_audit.sh [--dry-run]\n' >&2; exit 1; }
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数：%s\n' "$1" >&2; exit 1; }
  audit_sync_dry=1
fi
audit_sync_target="${SCBS_SADF_AUDIT_SSH_TARGET:-a5@100.76.151.67}"
audit_sync_remote='/home/a5/MyProject/wq_project/D-FINE'
audit_sync_manifest='tools/diagnostics/scbs_SADF_runtime_audit_sync_files.txt'
while IFS= read -r audit_sync_file; do
  [[ -z "$audit_sync_file" ]] && continue
  [[ "$audit_sync_file" == tools/diagnostics/* && -f "$audit_sync_file" ]] || {
    printf '非法或缺少诊断文件：%s\n' "$audit_sync_file" >&2; exit 1;
  }
done < "$audit_sync_manifest"
ssh "$audit_sync_target" "test -f '$audit_sync_remote/configs/dfine/dfine_s_scbs_hrw_110_SADF.yml' && test -f '$audit_sync_remote/configs/dfine/dfine_s_scbs_hrw_110.yml' && test -f '$audit_sync_remote/src/zoo/dfine/scale_aligned_detail_fusion.py' && test -f '$audit_sync_remote/src/solver/confusion_export.py'" || {
  printf '服务器未连接或缺少现有SADF/诊断依赖；未同步。\n' >&2; exit 1;
}
audit_sync_backup="/home/a5/MyProject/wq_project/D-FINE-SADF-runtime-audit-backup-$(date +%Y%m%dT%H%M%S)"
audit_sync_command=(rsync -av --relative --files-from="$audit_sync_manifest" --backup --backup-dir="$audit_sync_backup")
[[ "$audit_sync_dry" == 0 ]] || audit_sync_command+=(--dry-run)
"${audit_sync_command[@]}" ./ "$audit_sync_target:$audit_sync_remote/"
printf '仅同步SADF诊断脚本/说明，不修改模型源码、训练配置、数据或权重。\n'
[[ "$audit_sync_dry" == 1 ]] || printf '被替换文件备份：%s\n' "$audit_sync_backup"
