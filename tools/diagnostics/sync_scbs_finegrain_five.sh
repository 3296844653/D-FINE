#!/usr/bin/env bash
# Explicit code/config allowlist. Preserve remote base configs, data and results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
fg_sync_dry=0
[[ "$#" -le 1 ]] || { printf '用法：bash tools/diagnostics/sync_scbs_finegrain_five.sh [--dry-run]\n' >&2; exit 1; }
if [[ "$#" == 1 ]]; then
  [[ "$1" == --dry-run ]] || { printf '未知参数：%s\n' "$1" >&2; exit 1; }
  fg_sync_dry=1
fi
# The Desktop checkout is the user's sync source. This task's worktree also
# contains an unrelated unfinished literature branch: do not export that code.
fg_sync_search=(grep -q)
command -v rg >/dev/null 2>&1 && fg_sync_search=(rg -q)
if "${fg_sync_search[@]}" 'from \.literature_modules import' src/zoo/dfine/dfine.py src/zoo/dfine/dfine_decoder.py; then
  printf '此目录包含未完成的literature分支；请从 /Users/wq/Desktop/D-FINE 执行同步。\n' >&2
  exit 1
fi
fg_sync_target="${FINEGRAIN_SSH_TARGET:-a5@100.76.151.67}"
fg_sync_remote='/home/a5/MyProject/wq_project/D-FINE'
fg_sync_manifest='tools/diagnostics/scbs_finegrain_five_sync_files.txt'
while IFS= read -r fg_sync_file; do
  [[ -z "$fg_sync_file" ]] && continue
  case "$fg_sync_file" in src/zoo/dfine/*|configs/dfine/dfine_s_scbs_hrw_110_*|tools/diagnostics/*) ;;
    *) printf '非法白名单项：%s\n' "$fg_sync_file" >&2; exit 1 ;; esac
  [[ -f "$fg_sync_file" ]] || { printf '缺少文件：%s\n' "$fg_sync_file" >&2; exit 1; }
done < "$fg_sync_manifest"
ssh "$fg_sync_target" "test -f '$fg_sync_remote/train.py' && test -f '$fg_sync_remote/configs/dfine/dfine_s_scbs_hrw_110.yml' && test -f '$fg_sync_remote/src/solver/confusion_export.py' && test -f '$fg_sync_remote/src/solver/query_diagnostics.py'" || {
  printf '服务器未连接或缺少原基线/诊断支持；未同步。\n' >&2; exit 1;
}
fg_sync_backup="/home/a5/MyProject/wq_project/D-FINE-finegrain-backup-$(date +%Y%m%dT%H%M%S)"
fg_sync_cmd=(rsync -av --relative --files-from="$fg_sync_manifest" --backup --backup-dir="$fg_sync_backup")
[[ "$fg_sync_dry" == 0 ]] || fg_sync_cmd+=(--dry-run)
"${fg_sync_cmd[@]}" ./ "$fg_sync_target:$fg_sync_remote/"
if [[ "$fg_sync_dry" == 0 ]]; then
  printf '白名单同步完成；被替换的旧文件备份：%s\n' "$fg_sync_backup"
else
  printf '同步演练完成，未更改服务器文件。\n'
fi
printf '未同步数据/权重/结果，未改服务器110轮基线配置，未启动训练。\n'
